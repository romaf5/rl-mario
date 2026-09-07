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
import argparse, glob, os, pickle, sys, time, math
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rl_games.algos_torch import model_builder
from mario_native_vecenv import MarioNativeVecEnv, FRAME_STACK
from tensorboardX import SummaryWriter


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

    def __init__(self, env, archive_path, door_share):
        # one door state per configured level (8-4 only, or the whole route)
        self.levels = [l for l in env.stages if l != 'FullGame']
        self.doors = {l: env.states[l] for l in self.levels}
        self.door_share = door_share
        self.cells, self.states = [], []
        arch = pickle.load(open(archive_path, 'rb')) if archive_path and os.path.exists(archive_path) else {}
        for k, e in arch.items():
            if k[0] not in self.doors:
                continue
            sts = e[0] if isinstance(e[0], list) else [e[0]]
            self.cells.append(k); self.states.append(sts)
        self.score = np.ones(len(self.cells)) * 5.0     # optimistic: unsampled cells first
        self.uses = np.zeros(len(self.cells), int)
        print(f'[prompts] door + {len(self.cells)} archive cells')

    def sample(self, k, rng):
        out = []
        n_door = max(1, int(round(k * self.door_share))) if self.cells else k
        for j in range(n_door):
            lvl = self.levels[rng.randint(len(self.levels))]
            out.append(('door:' + lvl, self.doors[lvl]))
        w = self.score + 0.5
        w = w / w.sum()
        for _ in range(k - n_door):
            i = rng.choice(len(self.cells), p=w)
            out.append((i, self.states[i][rng.randint(len(self.states[i]))]))
        return out

    def update(self, idx, group_std):
        if isinstance(idx, str):
            return
        self.uses[idx] += 1
        self.score[idx] = 0.7 * self.score[idx] + 0.3 * group_std

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
            self.score = np.concatenate([self.score, np.full(added, 5.0)])
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
    for i in range(env.num_actors):
        env.start_cell[i] = None
    env.ep_steps[:] = 0
    return env._obs()


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
                   self_restart_cells=a.max_cells, cell_tiles=True, cell_y_band=32,
                   n_threads=a.n_threads, dense_infos=True, seed=a.seed))
    N = a.group * a.groups
    env = MarioNativeVecEnv('grpo', N, **ec); env.reset()
    eval_env = MarioNativeVecEnv('grpo_eval', a.eval_episodes, **dict(ec, sticky_actions=0.0, n_threads=8, seed=a.seed + 1))
    eval_env.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(params, cfg, env.observation_space.shape, a.init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    prompts = Prompts(env, a.archive, a.door_share)
    rng = np.random.RandomState(a.seed)

    run_dir = os.path.join('runs', f'{a.run_name}_{time.strftime("%d-%H-%M-%S")}')
    os.makedirs(os.path.join(run_dir, 'nn'), exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, 'summaries'))
    print(f'[grpo] {N} envs = {a.groups} groups x {a.group}; horizon {a.horizon}; run {run_dir}')

    H, t0, it, frames = a.horizon, time.time(), 0, 0
    obs_buf = np.zeros((H, N, *env.observation_space.shape), np.uint8)
    act_buf = np.zeros((H, N), np.int64); logp_buf = np.zeros((H, N), np.float32)
    mask_buf = np.zeros((H, N), np.float32); rew_buf = np.zeros((H, N), np.float32)
    while time.time() - t0 < a.hours * 3600:
        it += 1
        chosen = prompts.sample(a.groups, rng)
        states = [chosen[i // a.group][1] for i in range(N)]
        obs = load_states(env, states)
        alive = np.ones(N, bool); maxx = np.zeros(N); loops = np.zeros(N, bool); vics = np.zeros(N, bool)
        model.eval()
        for t in range(H):
            with torch.no_grad():
                lg = logits_of(model, torch.from_numpy(obs).to(device))
                dist = torch.distributions.Categorical(logits=lg)
                act = dist.sample(); lp = dist.log_prob(act)
            obs_buf[t] = np.clip(obs * 255.0, 0, 255).astype(np.uint8)
            act_buf[t] = act.cpu().numpy(); logp_buf[t] = lp.cpu().numpy(); mask_buf[t] = alive
            obs, r, d, inf = env.step(act_buf[t])
            rew_buf[t] = np.where(alive, r, 0.0)
            for i in np.nonzero(alive)[0]:
                maxx[i] = max(maxx[i], inf[i]['max_x_pos'])
                if d[i]:
                    alive[i] = False; loops[i] = bool(inf[i].get('page_reset', inf[i].get('looped', False))); vics[i] = bool(inf[i].get('victory', False))
            if not alive.any():
                break
        frames += N * H * 4
        if a.grow_archive:
            prompts.refresh(env.archive)
        R = rew_buf.sum(0)                                        # outcome per rollout
        adv = np.zeros(N, np.float32); n_live_groups = 0
        for g in range(a.groups):
            sl = slice(g * a.group, (g + 1) * a.group)
            m, s = R[sl].mean(), R[sl].std()
            prompts.update(chosen[g][0], float(s))
            if s > 1e-6:
                n_live_groups += 1
                adv[sl] = (R[sl] - m) / (1.0 if a.no_std else s + 1e-6)
        # ---- update ----
        model.train()
        T = H * N
        o = torch.from_numpy(obs_buf.reshape(T, *env.observation_space.shape))
        ac = torch.from_numpy(act_buf.reshape(T)).to(device); olp = torch.from_numpy(logp_buf.reshape(T)).to(device)
        mk = torch.from_numpy(mask_buf.reshape(T)).to(device); ad = torch.from_numpy(np.repeat(adv[None], H, 0).reshape(T)).to(device)
        valid = torch.nonzero(mk > 0).squeeze(1)
        stats = dict(loss=0.0, kl=0.0, ent=0.0, clipfrac=0.0, n=0)
        for ep in range(a.epochs):
            perm = valid[torch.randperm(len(valid), device=device)]
            for s0 in range(0, len(perm), a.minibatch):
                idx = perm[s0:s0 + a.minibatch]
                ob = o[idx.cpu()].to(device).float() / 255.0
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
        door = np.array([isinstance(chosen[i // a.group][0], str) for i in range(N)])
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
        if it % 5 == 0:
            print(f'it {it} {el/60:5.1f}min fps {frames/el:5.0f} R {R.mean():7.1f} door_x {maxx[door].mean() if door.any() else 0:6.0f} '
                  f'live_groups {n_live_groups}/{a.groups} loops {loops.mean():.2f} ent {stats["ent"]/n_upd:.3f} kl {stats["kl"]/n_upd:.4f} cells {len(prompts.cells)}', flush=True)
        if it % a.eval_every == 0:
            model.eval()
            ev = clean_door_eval(model, eval_env, device, a.eval_episodes)
            for k, v in ev.items():
                writer.add_scalar(f'eval/door_{k}', v, it)
            extra = ' '.join(f'{k[6:]}:{ev[k]:.2f}' for k in ev if k.startswith('clear_'))
            print(f'  [eval] door: mean_x {ev["mean_x"]:.0f} max_x {ev["max_x"]:.0f} victory {ev["victory"]:.3f} clear {ev["clear"]:.2f} loops {ev["loop"]:.2f} {extra}', flush=True)
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
