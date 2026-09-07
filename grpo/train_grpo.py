"""GRPO-style training for SMB 8-4 (group-relative, critic-free, outcome reward).

A "prompt" is a start state (the 8-4 door or an archive cell); a "group" is
G envs loaded from that exact state. Each env plays a segment of H agent
steps (masked after its first terminal). The rollout's outcome R is the sum
of the env's own rewards over the segment; the advantage is group-relative,
A = (R - mean_g) / (std_g + eps), shared by every action of the rollout
(no critic, no bootstrapping). Update = PPO-clip on per-action log-ratios,
several epochs over the batch, plus a small entropy bonus.

Analogue of TRPO's "vine" sampling / VinePPO: identical restarts from
savestates give a zero-variance baseline, and Go-Explore's archive supplies
the prompts so segments stay short while the level stays long.

Usage:
  venv_retro/bin/python grpo/train_grpo.py --config configs/mario_ppo_native_84.yaml \
      --init CHECKPOINT.pth --run-name Mario_GRPO84 [--hours 4]
"""
import argparse, copy, glob, os, pickle, sys, threading, time, math
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rl_games.algos_torch import model_builder
from mario_native_vecenv import MarioNativeVecEnv, FRAME_STACK
from tensorboardX import SummaryWriter


def _clip_worker(m_cpu, cfg, run_dir, step, level):
    """Best-of-2 door episodes from `level` with the CPU model copy, hack-free
    (transitions + ending), published as mp4/npz + TensorBoard GIF."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from clip_watcher import record, publish
        torch.set_num_threads(2)
        mx, frames, acts, start, total, info, pfr = record(m_cpu, cfg, level, 2, 3000, seed=step)
        publish(run_dir, step, frames, acts, start, mx, total, info, level, pfr)
        print(f'  [clip] it {step} {level}: max x {mx}, R {total:.0f}', flush=True)
    except Exception as e:
        print(f'  [clip] failed: {e}', flush=True)


def build_model(params, cfg, obs_shape, init=None):
    net = model_builder.ModelBuilder().load(params)
    model = net.build({'actions_num': 12, 'input_shape': obs_shape, 'num_seqs': 1, 'value_size': 1,
                       'normalize_value': cfg['normalize_value'], 'normalize_input': cfg['normalize_input']})
    if init:
        ck = torch.load(init, map_location='cpu', weights_only=False)
        sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[init] loaded {init} (epoch {ck.get("epoch")}); missing {len(missing)} unexpected {len(unexpected)}')
    return model


def logits_of(model, obs_t):
    return model({'obs': obs_t, 'is_train': False})['logits']


class Prompts:
    """Start states: the door + archive cells, sampled by a learnability score
    (last group std of outcomes) so groups with zero variance are avoided."""

    demo_share = 0.0
    round_robin = True

    def __init__(self, env, archive_path, door_share):
        # one door state per configured level (8-4 only, or the whole route)
        self.levels = [l for l in env.stages if l != 'FullGame']
        self.doors = {l: env.states[l] for l in self.levels}
        self.door_share = door_share
        self.clear = {l: 0.0 for l in self.levels}      # EMA of eval clear rate
        self.cells, self.states = [], []
        # demonstrations recorded from the explorers' random walks:
        # cell -> (start state, actions that reached it). A demo prompt
        # starts at the start state, forces all but the last `tail`
        # actions and leaves the rest to the policy; the tail grows as the
        # policy succeeds (Go-Explore phase 2 / the backward algorithm,
        # from the run's OWN trajectories, no human data)
        self.demos, self.tail, self.hist = {}, {}, {}
        arch = pickle.load(open(archive_path, 'rb')) if archive_path and os.path.exists(archive_path) else {}
        for k, e in arch.items():
            if k[0] not in self.doors:
                continue
            sts = e[0] if isinstance(e[0], list) else [e[0]]
            self.cells.append(k); self.states.append(sts)
        self.score = np.ones(len(self.cells)) * 5.0     # (rescaled to the running max as scores arrive)
        self.uses = np.zeros(len(self.cells), int)
        print(f'[prompts] door + {len(self.cells)} archive cells')

    def record_demo(self, cell, start, acts, start_cell=None):
        # a demo is a walk from another cell into this one: the explorer's
        # own start cell (re-entered on its first step) and 1-2 step hops
        # are not demonstrations of anything (they used to overwrite real
        # ones because the shortest demo per cell wins)
        if cell == start_cell or len(acts) < 3:
            return
        # only demos in the direction of progress: a later x-bin, or higher
        # up within the same bin. Every block-top demo used to be a DROP
        # from the pipe top (explorers start there too, and falling is the
        # shortest way in), so the trainer taught descending, not climbing.
        if start_cell is not None and not (cell[2] > start_cell[2] or (cell[2] == start_cell[2] and cell[3] < start_cell[3])):
            return
        if cell not in self.demos or len(acts) < len(self.demos[cell][1]):
            self.demos[cell] = (start, list(acts)); self.tail.setdefault(cell, 4)

    def demo_prompt(self, rng):
        live = [c for c in self.demos if self.tail[c] < len(self.demos[c][1])]
        if not live:
            return None
        idx = {c: i for i, c in enumerate(self.cells)}
        w = np.array([(self.score[idx[c]] if c in idx else 5.0) + 0.5 for c in live]); w = w / w.sum()
        c = live[rng.choice(len(live), p=w)]; start, acts = self.demos[c]
        return ('demo', c, start, acts[:len(acts) - self.tail[c]])

    def demo_result(self, cell, frac_reached):
        if cell not in self.demos:          # graduated by another group this iteration
            return
        self.hist.setdefault(cell, []).append(round(frac_reached, 2))
        n = len(self.demos[cell][1])
        if frac_reached >= 0.5:
            self.tail[cell] = min(n, self.tail[cell] + 4)     # the policy handles more of it
        elif frac_reached == 0.0:
            self.tail[cell] = max(2, self.tail[cell] - 1)
        if self.tail[cell] >= n:
            self.demos.pop(cell, None); self.tail.pop(cell, None)   # graduated
            self.graduated = getattr(self, 'graduated', 0) + 1

    def frontier(self, k, rng):
        """k least-visited cells (Go-Explore's exploration rule) for the
        archive-growing random walkers; door states when there are no cells."""
        if not self.cells:
            return [self.doors[self.levels[rng.randint(len(self.levels))]] for _ in range(k)]
        w = 1.0 / (1.0 + self.uses); w = w / w.sum()
        out = []
        for _ in range(k):
            i = rng.choice(len(self.cells), p=w)
            out.append(self.states[i][rng.randint(len(self.states[i]))])
        return out

    def sample(self, k, rng):
        if len(self.levels) > 1 and self.round_robin:
            return self.sample_round_robin(k, rng)
        out = []
        n_door = max(1, int(round(k * self.door_share))) if self.cells else k
        # mastered doors fade (0.15 floor keeps every level in rotation)
        w = np.array([0.15 + (1.0 - self.clear[l]) for l in self.levels]); w = w / w.sum()
        for j in range(n_door):
            lvl = self.levels[rng.choice(len(self.levels), p=w)]
            out.append(('door:' + lvl, self.doors[lvl]))
        # level first (mastered levels fade), then a cell of that level by
        # learnability: a level with many cells must not dominate the pool
        by_level = {}
        for i, c in enumerate(self.cells):
            by_level.setdefault(c[0], []).append(i)
        lv = [l for l in self.levels if l in by_level]
        wl = np.array([0.15 + (1.0 - self.clear[l]) for l in lv]); wl = wl / wl.sum()
        for _ in range(k - n_door):
            if self.demos and rng.random_sample() < self.demo_share:
                dp = self.demo_prompt(rng)
                if dp is not None:
                    out.append(dp); continue
            l = lv[rng.choice(len(lv), p=wl)]
            ids = by_level[l]
            w = self.score[ids] + 0.5; w = w / w.sum()
            i = ids[rng.choice(len(ids), p=w)]
            out.append((i, self.states[i][rng.randint(len(self.states[i]))]))
        return out

    def sample_round_robin(self, k, rng):
        """Retention: every level gets a group each iteration (cyclic if
        k != levels), so a mastered level keeps being trained instead of
        drifting while its learnability is zero. Within a level: a door
        prompt with prob door_share, else a demo (demo_share) or a cell by
        learnability."""
        out = []
        by_level = {}
        for i, c in enumerate(self.cells):
            by_level.setdefault(c[0], []).append(i)
        order = list(self.levels); rng.shuffle(order)
        for g in range(k):
            l = order[g % len(order)]
            ids = by_level.get(l, [])
            if not ids or rng.random_sample() < self.door_share:
                out.append(('door:' + l, self.doors[l])); continue
            if self.demos and rng.random_sample() < self.demo_share:
                live = [c for c in self.demos if c[0] == l and self.tail[c] < len(self.demos[c][1])]
                if live:
                    c = live[rng.randint(len(live))]; start, acts = self.demos[c]
                    out.append(('demo', c, start, acts[:len(acts) - self.tail[c]])); continue
            w = self.score[ids] + 0.5; w = w / w.sum()
            i = ids[rng.choice(len(ids), p=w)]
            out.append((i, self.states[i][rng.randint(len(self.states[i]))]))
        return out

    def update(self, idx, group_std):
        if isinstance(idx, str):
            return
        self.uses[idx] += 1
        self.score[idx] = 0.7 * self.score[idx] + 0.3 * group_std

    def note_clears(self, ev):
        for l in self.levels:
            k = 'clear_' + l if len(self.levels) > 1 else 'clear'
            if k in ev:
                self.clear[l] = 0.7 * self.clear[l] + 0.3 * ev[k]

    def refresh(self, archive):
        """Adopt cells the env archived during rollouts (new prompts start
        optimistic so they get sampled soon)."""
        known = set(self.cells); added = 0
        for k, e in archive.items():
            if k[0] not in self.doors or k in known:
                continue
            sts = e[0] if isinstance(e[0], list) else [e[0]]
            self.cells.append(k); self.states.append(list(sts)); added += 1
        if added:
            # optimistic for real: a new (frontier) cell starts at the best
            # current learnability score, so it is sampled before the
            # established cells (a fixed 5.0 was ~30x below typical outcome
            # stds and the frontier was almost never practised)
            init = float(self.score.max()) if len(self.score) else 5.0
            self.score = np.concatenate([self.score, np.full(added, max(init, 5.0))])
            self.uses = np.concatenate([self.uses, np.zeros(added, int)])
        # keep state variants fresh for cells the env re-saved
        for i, k in enumerate(self.cells):
            e = archive.get(k)
            if e is not None:
                self.states[i] = list(e[0] if isinstance(e[0], list) else [e[0]])
        return added


def load_states(env, states):
    """Put every env at its given savestate and rebuild all Python trackers
    and the frame ring (what reset() does, with our states)."""
    for i, st in enumerate(states):
        env.lib.benv_load(env.env, i, st)
        env._fetch_obs(i)
    env._post_reset_init(range(env.num_actors), env.ram)
    f = env.obs_u8.astype(np.float32) / 255.0
    env._ring[:] = f[..., None]
    if env.u8_obs:
        env._ring_u8[:] = env.obs_u8[..., None]
    for i in range(env.num_actors):
        env.start_cell[i] = None
        env.ep_cells[i] = set()
    env.ep_steps[:] = 0
    return env.obs_u8_stack() if env.u8_obs else env._obs()


@torch.no_grad()
def full_game_eval(model, cfg, device, episodes, max_steps=6000, n_threads=8, seed=1):
    """The real objective: sequential game from the first configured level
    with 3 lives, no noise. Reports the level index reached (0-31), mean and
    max, and the victory rate."""
    ec = dict(cfg['env_config']); [ec.pop(k, None) for k in ('name', 'action_type', 'archive_path')]
    first = [l for l in ec.get('random_stages') or ['1-1']][0]
    ec.update(random_stages=[first], route_levels=list(ec.get('random_stages') or [first]), episode_life=False,
              sticky_actions=0.0, explore_eps=0.0, self_restart_prob=0.0, explore_episode_prob=0.0,
              reset_noops=0, n_threads=n_threads, dense_infos=False, seed=seed)
    env = MarioNativeVecEnv('fullgame', episodes, **ec); obs = env.reset(); n = episodes
    done_m = np.zeros(n, bool); gp = np.zeros(n, int); vic = np.zeros(n, bool)
    for _ in range(max_steps):
        lg = logits_of(model, torch.from_numpy(obs).to(device))
        a = torch.distributions.Categorical(logits=lg).sample().cpu().numpy()
        obs, r, d, inf = env.step(a)
        gp = np.where(~done_m, np.maximum(gp, env.progress), gp)
        for i in np.nonzero(d & ~done_m)[0]:
            done_m[i] = True; vic[i] = bool(inf[i].get('victory', False)); gp[i] = max(gp[i], inf[i].get('game_progress', 0))
        if done_m.all():
            break
    env.close()
    return dict(level_mean=float(gp.mean()), level_max=int(gp.max()), victory=float(vic.mean()))


@torch.no_grad()
def clean_door_eval(model, env, device, episodes, max_steps=1500):
    """Sampled policy, no exploration noise, full episodes from each level's
    door (the n eval envs are split evenly over the configured levels)."""
    n = env.num_actors
    levels = [l for l in env.stages if l != 'FullGame']
    lv = [levels[i % len(levels)] for i in range(n)]
    obs = load_states(env, [env.states[l] for l in lv])
    maxx = np.zeros(n); done_m = np.zeros(n, bool); vic = np.zeros(n, bool); loops = np.zeros(n, bool); clear = np.zeros(n, bool)
    for _ in range(max_steps):
        lg = logits_of(model, torch.from_numpy(obs).to(device))
        a = torch.distributions.Categorical(logits=lg).sample().cpu().numpy()
        obs, r, d, inf = env.step(a)
        for i in range(n):
            if done_m[i]:
                continue
            maxx[i] = max(maxx[i], inf[i]['max_x_pos'])
            clear[i] = clear[i] or inf[i].get('stages_cleared', 0) > 0 or bool(inf[i].get('victory', False))
            if d[i]:
                done_m[i] = True; vic[i] = bool(inf[i].get('victory', False)); loops[i] = bool(inf[i].get('page_reset', inf[i].get('looped', False)))
        if done_m.all():
            break
    out = dict(mean_x=float(maxx.mean()), max_x=float(maxx.max()), victory=float(vic.mean()), loop=float(loops.mean()),
               clear=float(clear.mean()))
    if len(levels) > 1:
        for l in levels:
            m = np.array([x == l for x in lv])
            out['mean_x_' + l] = float(maxx[m].mean()); out['clear_' + l] = float(clear[m].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mario_ppo_native_84.yaml')
    ap.add_argument('--init', default='')
    ap.add_argument('--run-name', default='Mario_GRPO84')
    ap.add_argument('--archive', default='native/archive_84.pkl')
    ap.add_argument('--group', type=int, default=16)          # G rollouts per prompt
    ap.add_argument('--groups', type=int, default=8)          # K prompts per iteration
    ap.add_argument('--horizon', type=int, default=256)       # segment length (agent steps)
    ap.add_argument('--door-share', type=float, default=0.25)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--minibatch', type=int, default=4096)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--entropy', type=float, default=0.005)
    ap.add_argument('--no-std', action='store_true', help='Dr.GRPO: do not divide by the group std')
    ap.add_argument('--hours', type=float, default=1e9)
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--eval-episodes', type=int, default=32)
    ap.add_argument('--n-threads', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grow-archive', action='store_true', help='record rollout cells into the archive and use them as prompts')
    ap.add_argument('--max-cells', type=int, default=1024)
    ap.add_argument('--clip-every', type=int, default=100, help='iterations between gameplay clips (background thread, CPU); 0 = off')
    ap.add_argument('--fullgame-every', type=int, default=100, help='iterations between sequential full-game evals (3 lives from the first level); 0 = off')
    ap.add_argument('--rtg', action='store_true', help='per-step advantages from discounted reward-to-go, group-normalised at each step (temporal credit) instead of one outcome per rollout')
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--cell-variants', type=int, default=3, help='max tile-signature variants per spatial archive cell')
    ap.add_argument('--explorers', type=int, default=0, help='extra envs per iteration that random-walk from least-visited cells ONLY to grow the archive (never in the update)')
    ap.add_argument('--demo-share', type=float, default=0.0, help='fraction of cell groups started from an explorer demo with a forced prefix (backward chaining); needs --explorers')
    a = ap.parse_args()

    params = yaml.safe_load(open(a.config))['params']; cfg = params['config']
    ec = dict(cfg['env_config']); ec.pop('name', None); ec.pop('action_type', None)
    # the env provides observations, dynamics and the per-step reward, and
    # (grow_archive) records a Go-Explore cell archive from the rollouts:
    # new cells become prompts, so groups start where the policy's outcomes
    # still vary (from scratch, door-only prompts collapse to zero variance
    # at the first obstacle). No exploration noise beyond the policy's own
    # sampling; the tiny restart prob only switches the env's archiving on
    # (dead rollouts are masked, so its resets are never trained on).
    ec.update(dict(self_restart_prob=1e-6 if a.grow_archive else 0.0, explore_eps=0.0,
                   explore_episode_prob=0.0, archive_path=a.archive if a.grow_archive else None,
                   self_restart_cells=a.max_cells, cell_tiles=True, cell_y_band=32, cell_max_variants=a.cell_variants,
                   sticky_actions=0.0, n_threads=a.n_threads, dense_infos=True, seed=a.seed))
    N = a.group * a.groups
    NX = N + a.explorers              # explorers ride along in the same batch, outside the buffers
    env = MarioNativeVecEnv('grpo', NX, **dict(ec, dense_infos=False, explore_pure=True)); env.reset(); env.enable_u8_obs()
    eval_env = MarioNativeVecEnv('grpo_eval', a.eval_episodes, **dict(ec, sticky_actions=0.0, n_threads=8, seed=a.seed + 1))
    eval_env.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(params, cfg, env.observation_space.shape, a.init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    prompts = Prompts(env, a.archive, a.door_share); prompts.demo_share = a.demo_share
    rng = np.random.RandomState(a.seed)

    run_dir = os.path.join('runs', f'{a.run_name}_{time.strftime("%d-%H-%M-%S")}')
    os.makedirs(os.path.join(run_dir, 'nn'), exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, 'summaries'))
    print(f'[grpo] {N} envs = {a.groups} groups x {a.group}; horizon {a.horizon}; run {run_dir}')

    H, t0, it, frames = a.horizon, time.time(), 0, 0
    clip_thread = None
    obs_buf = np.zeros((H, N, *env.observation_space.shape), np.uint8)
    act_buf = np.zeros((H, N), np.int64); logp_buf = np.zeros((H, N), np.float32)
    mask_buf = np.zeros((H, N), np.float32); rew_buf = np.zeros((H, N), np.float32)
    while time.time() - t0 < a.hours * 3600:
        it += 1
        chosen = prompts.sample(a.groups, rng)
        # demo prompts: ('demo', cell, start_state, forced_prefix)
        state_of = lambda ch: ch[2] if ch[0] == 'demo' else ch[1]
        xstates = prompts.frontier(a.explorers, rng)
        states = [state_of(chosen[i // a.group]) for i in range(N)] + xstates
        obs = load_states(env, states)[:N]
        forced = np.full((H, N), -1, np.int64)
        for g in range(a.groups):
            if chosen[g][0] == 'demo':
                pre = chosen[g][3][:H]
                forced[:len(pre), g * a.group:(g + 1) * a.group] = np.array(pre)[:, None]
        if a.explorers:
            env.explorer[N:] = H          # env substitutes persistent random actions for these
            xacts = [[] for _ in range(a.explorers)]
            # start cell from the loaded state itself (a first-step jump
            # left it unknown before, and unknown bypassed the direction
            # filter: that is how "drop from the pipe top" demos got in)
            xcell = [env.cell_of(N + j) for j in range(a.explorers)]
        alive = np.ones(N, bool); maxx = np.zeros(N); loops = np.zeros(N, bool); vics = np.zeros(N, bool)
        model.eval()
        for t in range(H):
            with torch.no_grad():
                lg = logits_of(model, torch.from_numpy(obs).to(device).float().div_(255.0))
                dist = torch.distributions.Categorical(logits=lg)
                act = dist.sample(); lp = dist.log_prob(act)
            obs_buf[t] = obs                          # already uint8
            act_np = act.cpu().numpy(); fz = forced[t]
            act_buf[t] = np.where(fz >= 0, fz, act_np); logp_buf[t] = lp.cpu().numpy()
            mask_buf[t] = alive & (fz < 0)             # forced steps are never trained on
            _, r, d, inf = env.step(np.concatenate([act_buf[t], np.zeros(a.explorers, np.int64)]) if a.explorers else act_buf[t])
            obs = env.obs_u8_stack()[:N]; r = r[:N]; d = d[:N]
            if a.explorers:
                for j in range(a.explorers):
                    xacts[j].append(int(env.last_action[N + j]))
                    c = env.entered_cell[N + j]
                    if c is not None and len(xacts[j]) <= 96:
                        prompts.record_demo(c, xstates[j], xacts[j], start_cell=xcell[j])
            rew_buf[t] = np.where(alive, r, 0.0)
            # bookkeeping from env arrays; the env resets finished envs
            # inside step(), so their pre-reset values come from the info
            # dicts it still fills for done envs
            maxx = np.where(alive & ~d, np.maximum(maxx, env.max_x[:N]), maxx)
            for i in np.nonzero(alive & d)[0]:
                maxx[i] = max(maxx[i], inf[i]['max_x_pos'])
                loops[i] = bool(inf[i].get('page_reset', False)); vics[i] = bool(inf[i].get('victory', False))
            alive &= ~d
            if not alive.any():
                break
        frames += NX * H * 4
        if a.grow_archive:
            prompts.refresh(env.archive)
        R = rew_buf.sum(0)                                        # outcome per rollout
        adv = np.zeros(N, np.float32); n_live_groups = 0
        for g in range(a.groups):
            sl = slice(g * a.group, (g + 1) * a.group)
            m, s = R[sl].mean(), R[sl].std()
            if chosen[g][0] == 'demo':
                cell = chosen[g][1]
                # success = the rollout ENTERED the target cell (same key,
                # incl. y-band and tile signature); an x-only test let
                # floor rollouts "reach" the block top by standing there
                fr = float(np.mean([cell in env.ep_cells[i] for i in range(sl.start, sl.stop)]))
                prompts.demo_result(cell, fr)
                if cell[2] >= 17 and cell[3] <= 3:
                    print(f'  [demo] it {it} target {cell[2]}/{cell[3]}/{cell[6]} len {len(prompts.demos.get(cell, (None, []))[1]) if cell in prompts.demos else "grad"} prefix {len(chosen[g][3])} reached {fr:.2f} maxx {maxx[sl].mean():.0f}', flush=True)
            else:
                prompts.update(chosen[g][0], float(s))
            if s > 1e-6:
                n_live_groups += 1
                adv[sl] = (R[sl] - m) / (1.0 if a.no_std else s + 1e-6)
        if a.rtg:
            # temporal credit: discounted reward-to-go per step, normalised
            # within the group AT THAT STEP (all rollouts of a group share
            # the start, so the group's per-step spread is a valid baseline
            # early on and a conservative one later). An action late in a
            # rollout is no longer blamed for what happened before it.
            G = np.zeros_like(rew_buf); run = np.zeros(N, np.float32)
            for t in range(H - 1, -1, -1):
                run = rew_buf[t] + a.gamma * run * mask_buf[t]
                G[t] = run
            Gg = G.reshape(H, a.groups, a.group)
            mu = Gg.mean(2, keepdims=True); sd = Gg.std(2, keepdims=True)
            adv_t = np.where(sd > 1e-6, (Gg - mu) / (sd + 1e-6), 0.0).reshape(H, N).astype(np.float32)
        # ---- update ----
        model.train()
        T = H * N
        o = torch.from_numpy(obs_buf.reshape(T, *env.observation_space.shape)).to(device)   # uint8, ~1 GB at 256x128
        ac = torch.from_numpy(act_buf.reshape(T)).to(device); olp = torch.from_numpy(logp_buf.reshape(T)).to(device)
        mk = torch.from_numpy(mask_buf.reshape(T)).to(device)
        ad = torch.from_numpy((adv_t if a.rtg else np.repeat(adv[None], H, 0)).reshape(T)).to(device)
        valid = torch.nonzero(mk > 0).squeeze(1)
        stats = dict(loss=0.0, kl=0.0, ent=0.0, clipfrac=0.0, n=0)
        for ep in range(a.epochs):
            perm = valid[torch.randperm(len(valid), device=device)]
            for s0 in range(0, len(perm), a.minibatch):
                idx = perm[s0:s0 + a.minibatch]
                ob = o[idx].float().div_(255.0)
                lg = logits_of(model, ob); dist = torch.distributions.Categorical(logits=lg)
                lp = dist.log_prob(ac[idx]); ratio = torch.exp(lp - olp[idx])
                A = ad[idx]
                pg = -torch.min(ratio * A, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A).mean()
                ent = dist.entropy().mean()
                loss = pg - a.entropy * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    stats['loss'] += pg.item(); stats['ent'] += ent.item()
                    stats['kl'] += (olp[idx] - lp).mean().item()
                    stats['clipfrac'] += ((ratio - 1).abs() > a.clip).float().mean().item(); stats['n'] += 1
        n_upd = max(stats['n'], 1)
        door = np.array([isinstance(chosen[i // a.group][0], str) and chosen[i // a.group][0].startswith('door') for i in range(N)])
        el = time.time() - t0
        writer.add_scalar('rewards/step', float(R.mean()), it)
        writer.add_scalar('grpo/live_groups', n_live_groups / a.groups, it)
        writer.add_scalar('grpo/entropy', stats['ent'] / n_upd, it)
        writer.add_scalar('grpo/kl', stats['kl'] / n_upd, it)
        writer.add_scalar('grpo/clipfrac', stats['clipfrac'] / n_upd, it)
        writer.add_scalar('grpo/frames', frames, it)
        if door.any():
            writer.add_scalar('mario/door_mean_x', float(maxx[door].mean()), it)
            writer.add_scalar('mario/door_max_x', float(maxx[door].max()), it)
            writer.add_scalar('mario/door_reward', float(R[door].mean()), it)
        writer.add_scalar('mario/loop_rate', float(loops.mean()), it)
        writer.add_scalar('mario/victory_rate', float(vics.mean()), it)
        writer.add_scalar('mario/cell_max_x', float(maxx[~door].mean()) if (~door).any() else 0.0, it)
        writer.add_scalar('grpo/prompt_cells', len(prompts.cells), it)
        writer.add_scalar('grpo/demos', len(prompts.demos), it)
        writer.add_scalar('grpo/demos_graduated', getattr(prompts, 'graduated', 0), it)
        if it % 5 == 0:
            print(f'it {it} {el/60:5.1f}min fps {frames/el:5.0f} R {R.mean():7.1f} door_x {maxx[door].mean() if door.any() else 0:6.0f} '
                  f'live_groups {n_live_groups}/{a.groups} loops {loops.mean():.2f} ent {stats["ent"]/n_upd:.3f} kl {stats["kl"]/n_upd:.4f} cells {len(prompts.cells)} demos {len(prompts.demos)} grad {getattr(prompts, "graduated", 0)}', flush=True)
        if a.clip_every and it % a.clip_every == 0 and not (clip_thread and clip_thread.is_alive()):
            m_cpu = copy.deepcopy(model).cpu().eval()
            lvl = prompts.levels[rng.randint(len(prompts.levels))]
            clip_thread = threading.Thread(target=_clip_worker, args=(m_cpu, cfg, run_dir, it, lvl), daemon=True)
            clip_thread.start()
        if it % a.eval_every == 0:
            model.eval()
            ev = clean_door_eval(model, eval_env, device, a.eval_episodes)
            for k, v in ev.items():
                writer.add_scalar(f'eval/door_{k}', v, it)
            prompts.note_clears(ev)
            extra = ' '.join(f'{k[6:]}:{ev[k]:.2f}' for k in ev if k.startswith('clear_'))
            print(f'  [eval] door: mean_x {ev["mean_x"]:.0f} max_x {ev["max_x"]:.0f} victory {ev["victory"]:.3f} clear {ev["clear"]:.2f} loops {ev["loop"]:.2f} {extra}', flush=True)
            if a.fullgame_every and it % a.fullgame_every == 0:
                fg = full_game_eval(model, cfg, device, 16, seed=a.seed + 2)
                for k, v in fg.items():
                    writer.add_scalar(f'eval/fullgame_{k}', v, it)
                lv = lambda g: '%d-%d' % (g // 4 + 1, g % 4 + 1)
                print(f'  [fullgame] 3 lives from the start: level reached mean {fg["level_mean"]:.1f} ({lv(int(round(fg["level_mean"])))}) max {lv(fg["level_max"])} victory {fg["victory"]:.3f}', flush=True)
            # prompt state on disk for inspection (demos, tails, per-demo
            # success history, learnability scores)
            with open(os.path.join(run_dir, 'nn', 'prompts.pkl'), 'wb') as f:
                pickle.dump({'cells': prompts.cells, 'score': prompts.score, 'uses': prompts.uses,
                             'demos': {c: (len(v[1]), v[1], v[0]) for c, v in prompts.demos.items()},   # (len, actions, start state)
                             'tail': prompts.tail, 'hist': prompts.hist, 'graduated': getattr(prompts, 'graduated', 0)}, f)
            ck = {'model': model.state_dict(), 'iter': it, 'frames': frames}
            torch.save(ck, os.path.join(run_dir, 'nn', 'grpo_last.pth'))
            # numbered copy per eval: clips / ghosts for ANY step can be
            # rendered later (tools/clip_watcher.py, tools/ghosts.py)
            torch.save(ck, os.path.join(run_dir, 'nn', 'grpo_it%06d.pth' % it))
        writer.flush()
    torch.save({'model': model.state_dict(), 'iter': it, 'frames': frames}, os.path.join(run_dir, 'nn', 'grpo_last.pth'))
    env.close(); eval_env.close()


if __name__ == '__main__':
    main()
