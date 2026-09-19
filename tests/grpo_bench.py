"""Checks for the GRPO trainer's update math and bookkeeping (no training).
Run: venv_retro/bin/python tests/grpo_bench.py

Properties (2026-09-11 review):
  * the stored rollout log-prob is the true log-prob (an asymmetric clamp
    against the fresh log-prob turned the ratio of every rare forced /
    hinted action into ~0, so no gradient could lift it);
  * the PPO ratio is bounded by clamping the DIFFERENCE, so lp == olp gives
    ratio 1 however small both are;
  * the self-imitation loss keeps a gradient below p = e^-20;
  * zero-advantage samples (dead groups) are not trained on (they only
    added entropy and flattened mastered levels);
  * --no-std applies in the reward-to-go path as well;
  * CLI reward knobs default to the config's env_config, and the launch is
    recorded in the run dir;
  * loop metrics read the flag that is actually set at the done step;
  * prompts pruned from the archive are pruned from the prompt pool;
  * a train-mode forward for logits does not update the value normaliser.
2026-09-18 review:
  * reward-to-go statistics only over rollouts still live at that step, and
    float-noise spreads are dead groups (relative threshold);
  * clipfrac uses each sample's own clip range;
  * never-tried prompt cells are sampled optimistically; credit is by cell
    key (survives a prune); a winners-only pool stays winners-only on refresh;
    --door-share 0 means no door group; loaded states carry no explorer walk;
  * main() on a fake env: config archive caps, the input archive is copied to
    the run dir, torch is seeded, the config's device, frames count the steps
    taken, the full-game eval has its own schedule and follows the config's
    route, live groups under --rtg, the outcome is zeroed on leaving steps,
    warnings for knobs that need --grow-archive.
"""
import contextlib, io, json, os, pickle, sys, tempfile, types
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grpo import train_grpo as G

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


# ---------------------------------------------------------------- log-probs / ratio / BC
logits = torch.zeros(1, 12); logits[0, 0] = 30.0            # action 0 near-certain; the rest ~e^-30
dist = torch.distributions.Categorical(logits=logits)
act = torch.tensor([5])
true_lp = dist.log_prob(act)
check('rollout log-prob is the true log-prob (%.1f), not a clamp' % true_lp.item(),
      torch.allclose(G.rollout_logp(dist, act), true_lp), G.rollout_logp(dist, act).item())
lp = torch.tensor([-25.0], requires_grad=True); olp = torch.tensor([-25.0])
ratio = torch.exp(G.log_ratio(lp, olp))
check('ratio is 1 when the fresh and stored log-probs agree at -25', torch.allclose(ratio, torch.ones(1)), ratio.item())
(ratio * 1.0).sum().backward()
check('policy gradient at a rare action is nominal (d ratio / d lp = 1)', abs(lp.grad.item() - 1.0) < 1e-5, lp.grad.item())
check('log-ratio is bounded', G.log_ratio(torch.tensor([0.0]), torch.tensor([-100.0])).item() <= 10.0 + 1e-6)
lps = torch.tensor([-30.0, -30.0], requires_grad=True)
G.bc_loss(lps).backward()
check('self-imitation loss keeps a gradient at log-prob -30', float(lps.grad.abs().sum()) > 0, lps.grad)

# ---------------------------------------------------------------- valid samples
mask = torch.tensor([1.0, 1.0, 0.0, 1.0]); adv = torch.tensor([0.5, 0.0, 0.7, -0.2])
vi = G.valid_indices(mask, adv).tolist()
check('valid samples: masked-out and zero-advantage samples are excluded', vi == [0, 3], vi)

# ---------------------------------------------------------------- advantages
R = np.array([1.0, 3.0, 5.0, 7.0, 2.0, 2.0, 2.0, 2.0], np.float32)     # 2 groups of 4; group 2 has zero variance
adv, live = G.outcome_advantages(R, groups=2, group=4, no_std=False)
check('outcome advantages: zero-variance group -> 0, live groups counted', live == 1 and np.all(adv[4:] == 0) and abs(adv[:4].mean()) < 1e-6, (live, adv))
adv_ns, _ = G.outcome_advantages(R, groups=2, group=4, no_std=True)
check('outcome advantages: --no-std leaves the centred returns undivided', np.allclose(adv_ns[:4], R[:4] - 4.0), adv_ns)
Gt = np.stack([R, R * 2.0])                                          # H=2
a_std = G.rtg_advantages(Gt, np.ones_like(Gt), groups=2, group=4, no_std=False)
a_ns = G.rtg_advantages(Gt, np.ones_like(Gt), groups=2, group=4, no_std=True)
check('rtg advantages: --no-std applies (centred, undivided) and dead groups are 0',
      np.allclose(a_ns[0, :4], R[:4] - 4.0) and np.allclose(a_ns[1, :4], 2 * R[:4] - 8.0) and np.all(a_ns[:, 4:] == 0)
      and not np.allclose(a_std[0, :4], a_ns[0, :4]), (a_std[0], a_ns[0]))
Rn = np.array([200.0 + 3e-5, 200.0 - 3e-5] * 2, np.float32)             # float32 rounding of --outcome-progress
an, ln = G.outcome_advantages(Rn, groups=1, group=4, no_std=False)
check('outcome advantages: a float-noise spread (std %.1e on R 200) is a dead group' % Rn.std(), Rn.std() > 1e-6 and ln == 0 and np.all(an == 0), (ln, an))

# ---------------------------------------------------------------- rtg: statistics over the live rollouts only
Gl = np.array([[12.0, 7.0, 1.0, 1.0], [10.0, 5.0, 0.0, 0.0]], np.float32)    # 1 group of 4; rollouts 2, 3 ended at t=0
al = G.rtg_advantages(Gl, np.array([[1, 1, 1, 1], [1, 1, 0, 0]], np.float32), groups=1, group=4, no_std=False)
check('rtg advantages: the weaker of two survivors is negative once the others ended (ended rollouts are not in the baseline)',
      al[1, 1] < 0 < al[1, 0] and np.all(al[1, 2:] == 0) and al[0, 0] > 0, al)
a1 = G.rtg_advantages(Gl, np.array([[1, 1, 1, 1], [1, 0, 0, 0]], np.float32), groups=1, group=4, no_std=False)
check('rtg advantages: a lone survivor gets 0 (not +sqrt(G-1) whatever its return)', np.all(a1[1] == 0), a1[1])
ar = G.rtg_advantages(np.stack([Rn, Rn]), np.ones((2, 4), np.float32), groups=1, group=4, no_std=False)
check('rtg advantages: a float-noise spread is 0', np.all(ar == 0), ar)

# ---------------------------------------------------------------- clipfrac against each sample's own clip
pg, cf = G.clip_objective(torch.tensor([1.5, 1.5]), torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.2]))
check('clip objective: clipfrac counts each sample against its own clip (demo 1.0 inside, 0.2 clipped)',
      abs(cf.item() - 0.5) < 1e-6 and abs(pg.item() + 1.35) < 1e-6, (pg.item(), cf.item()))

# ---------------------------------------------------------------- CLI defaults follow the config
cfg_ec = {'cell_bonus': 100, 'cell_x_bin': 64}
ov = G.resolve_env_overrides(cell_bonus=None, cell_x_bin=None, env_config=cfg_ec)
check('CLI reward knobs default to the config (bonus 100, x-bin 64)', ov == {'cell_bonus': 100.0, 'cell_x_bin': 64}, ov)
ov = G.resolve_env_overrides(cell_bonus=0.0, cell_x_bin=None, env_config=cfg_ec)
check('an explicit CLI value still overrides the config', ov == {'cell_bonus': 0.0, 'cell_x_bin': 64}, ov)
ov = G.resolve_env_overrides(cell_bonus=None, cell_x_bin=None, env_config={})
check('without config values the old defaults apply (0, 128)', ov == {'cell_bonus': 0.0, 'cell_x_bin': 128}, ov)
ck = G.resolve_cell_keys({'cell_y_band': 16, 'cell_screen_bin': 64, 'cell_tiles': True})
check('cell-key layout follows the config (y-band 16, screen bin 64, tiles on) so finisher cells match the PPO archive', ck == {'cell_y_band': 16, 'cell_screen_bin': 64, 'cell_tiles': True}, ck)
ck = G.resolve_cell_keys({})
check('cell-key layout defaults (y-band 32, no screen bin, tiles on) when the config is silent', ck == {'cell_y_band': 32, 'cell_screen_bin': 0, 'cell_tiles': True}, ck)
d = tempfile.mkdtemp()
G.write_launch_record(d, ['train_grpo.py', '--rtg'], {'cell_bonus': 100.0}, {'rtg': True})
rec = json.load(open(os.path.join(d, 'launch.json')))
check('launch record holds argv, resolved env config and args', rec['argv'][1] == '--rtg' and rec['env_config']['cell_bonus'] == 100.0 and rec['args']['rtg'] is True, rec)

# ---------------------------------------------------------------- loop flag
check('looped(): reads page_resets / loop_timeout, not the drop-step flag',
      G.looped({'page_resets': 1, 'page_reset': False}) and G.looped({'loop_timeout': True}) and not G.looped({'page_reset': False, 'page_resets': 0}))

# ---------------------------------------------------------------- prompt pruning
class _Env:
    stages = ['8-4']; states = {'8-4': b'door'}; cell_x_bin = 64
p = G.Prompts(_Env(), None, 0.25)
K = lambda b: ('8-4', 3, b, 2, 0, 3, 0)
p.refresh({K(10): [[b'a'], 0, 300], K(11): [[b'b'], 0, 300], K(12): [[b'c'], 0, 300]})
p.score[:] = [1.0, 2.0, 3.0]; p.uses[:] = [4, 5, 6]
p.refresh({K(10): [[b'a'], 0, 300], K(12): [[b'c2'], 0, 300]})
check('prompts: a cell evicted from the archive leaves the prompt pool (scores/uses stay aligned)',
      p.cells == [K(10), K(12)] and list(p.score) == [1.0, 3.0] and list(p.uses) == [4, 6] and p.states[1] == [b'c2'],
      (p.cells, list(p.score), list(p.uses)))

# ---------------------------------------------------------------- loaded cells get tried, credit by key, door share 0
apath = os.path.join(tempfile.mkdtemp(), 'arch.pkl')
pickle.dump({K(b): [[b'%d' % b], 0, 300, 1, 0, 0] for b in (10, 11, 12)}, open(apath, 'wb'))
cell_of = lambda p, e: e if isinstance(e, tuple) else p.cells[e]      # prompt id: cell key (or an index, pre-fix)
p = G.Prompts(_Env(), apath, 0.0); p.uses[2] = 10; p.score[2] = 400.0   # one tried cell at a typical outcome std
rng = np.random.RandomState(0)
draws = [cell_of(p, e[0]) for _ in range(50) for e in p.sample(4, rng) if not isinstance(e[0], str)]
fr = [round(draws.count(K(b)) / max(len(draws), 1), 3) for b in (10, 11, 12)]
check('prompts: never-tried loaded cells are sampled like the best tried cell (they sat at 5.0 against 400)', min(fr[:2]) > 0.2, fr)
ch = G.Prompts(_Env(), apath, 0.0).sample(4, np.random.RandomState(1))
check('prompts: --door-share 0 gives no door group', not any(isinstance(e[0], str) for e in ch), [e[0] for e in ch])
p = G.Prompts(_Env(), apath, 0.0)
e = next((e for _ in range(200) for e in p.sample(4, rng) if not isinstance(e[0], str) and cell_of(p, e[0]) == K(11)), None)
p.refresh({K(11): [[b'11'], 0, 300], K(12): [[b'12'], 0, 300]})       # K(10) pruned between sampling and crediting
p.update(e[0], 9.0)
check('prompts: a prune between sampling and crediting still credits the sampled cell (by key)',
      dict(zip(p.cells, p.uses.tolist())) == {K(11): 1, K(12): 0}, dict(zip(p.cells, p.uses.tolist())))

# ---------------------------------------------------------------- a winners-only pool stays winners-only
p = G.Prompts(_Env(), None, 0.25, winners_only=True)
arch = {K(20): [[b'w'], 0, 300, 2, 5, 0], K(21): [[b'l'], 0, 300, 0, 5, 0], K(22): [[b'x'], 0, 300, 0, 0, 1]}
p.refresh(arch)
check('prompts: winners-only refresh adopts only cells with saved policy / explorer wins', p.cells == [K(20), K(22)], p.cells)
p.refresh({**arch, K(23): [[b'n'], 0, 300], K(24): [[b'm'], 0, 300]}, won=lambda c: c == K(24))
check('prompts: winners-only refresh asks the env\'s live win counts (won predicate) for new cells', p.cells == [K(20), K(22), K(24)], p.cells)

# ---------------------------------------------------------------- value normaliser untouched by logits
params = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'mario_ppo_native_84.yaml')))['params']
torch.manual_seed(0)              # a random-init policy that happens to walk left makes the sampling checks meaningless
model = G.build_model(params, params['config'], (84, 84, 4))
model.train()
before = model.value_mean_std.count.clone()
G.logits_of(model, torch.zeros(2, 84, 84, 4))
check('logits_of in train mode leaves the value normaliser count unchanged', torch.equal(model.value_mean_std.count, before),
      (before.item(), model.value_mean_std.count.item()))

# ---------------------------------------------------------------- the trainer's evals sample the policy, seeded
from mario_native_vecenv import MarioNativeVecEnv
rp = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'mario_ppo_native_routeDet.yaml')))['params']
rc = rp['config']; ec = dict(rc['env_config']); [ec.pop(k, None) for k in ('name', 'action_type', 'archive_path', 'video_levels')]
ec.update(sticky_actions=0.0, explore_eps=0.0, self_restart_prob=0.0, explore_episode_prob=0.0, reset_noops=0, n_threads=2,
          dense_infos=True, seed=1, random_stages=['1-1'], route_levels=list(rc['env_config']['random_stages']))
eval_env = MarioNativeVecEnv('t', 4, **ec); eval_env.reset()
model.eval(); dev = torch.device('cpu')
ev1 = G.clean_door_eval(model, eval_env, dev, 4, max_steps=30, seed=5)
ev2 = G.clean_door_eval(model, eval_env, dev, 4, max_steps=30, seed=5)
ev3 = G.clean_door_eval(model, eval_env, dev, 4, max_steps=30, seed=6)
eval_env.explorer[:] = 7                          # a restart walk the env armed inside step()
G.load_states(eval_env, [eval_env.states['1-1']] * 4)
xw = eval_env.explorer.copy()
eval_env.close()
check('load_states: no explorer walk carries over into the loaded rollouts', not xw.any(), xw)
check('door eval samples the policy (per-episode max x are not all equal)', len(set(ev1['max_x_all'])) > 1, ev1.get('max_x_all'))
check('door eval is reproducible for a fixed seed', ev1 == ev2, (ev1, ev2))
check('door eval differs for another seed', ev1 != ev3, (ev1, ev3))
fg1 = G.full_game_eval(model, rc, dev, 2, max_steps=30, n_threads=2, seed=3)
fg2 = G.full_game_eval(model, rc, dev, 2, max_steps=30, n_threads=2, seed=3)
check('full-game eval is reproducible for a fixed seed', fg1 == fg2, (fg1, fg2))

# ---------------------------------------------------------------- route-aware full-game progress
check('route_progress: an off-route exit (4-2 flag -> 4-3) counts as the last on-route level',
      G.route_progress(14, {0, 1, 12, 13, 28, 29, 30, 31}) == 13 and G.route_progress(28, {0, 1, 12, 13, 28}) == 28 and G.route_progress(5, set()) == 5)

# ---------------------------------------------------------------- main() and full_game_eval() on a fake env
class _Stop(Exception):
    pass


class _FakeEnv:
    """MarioNativeVecEnv stand-in (no emulator): records its kwargs; every
    rollout ends at step die_at; reward r_fn(t, n); every step is a leaving
    step whose raw progress term is 10 (the env pays 0 there); the
    max_iters+1-th load_states stops main()."""
    made, die_at, max_iters = [], 4, 2
    def __init__(self, name, n, **kw):
        _FakeEnv.made.append(self); self.name, self.num_actors, self.kw = name, n, kw
        self.stages = list(kw.get('random_stages') or ['FullGame']); self.states = {l: b'' for l in self.stages}
        self.cell_x_bin = kw.get('cell_x_bin', 128); self.observation_space = types.SimpleNamespace(shape=(84, 84, 4))
        ap_ = kw.get('archive_path'); self.archive = pickle.load(open(ap_, 'rb')) if ap_ and os.path.exists(ap_) else {}
        z = lambda dt: np.zeros(n, dt)
        self.obs_u8 = np.zeros((n, 84, 84), np.uint8); self._ring = np.zeros((n, 84, 84, 4), np.float32)
        self.u8_obs, self._ring_u8, self.ram, self.loads, self.t = False, None, None, 0, 0
        self.start_cell, self.entered_cell = [None] * n, [None] * n
        self.is_door, self.explorer, self.ep_steps = z(bool), z(np.int32), z(np.int32)
        self.max_x, self.last_action, self.progress = z(np.int64), z(np.int64), z(np.int32)
        self.last_terms, self.last_leaving = {}, z(bool)
    def reset(self):
        self.t = 0; return self._obs()
    def enable_u8_obs(self):
        self.u8_obs = True; self._ring_u8 = np.zeros((self.num_actors, 84, 84, 4), np.uint8)
    def _post_reset_init(self, idx, ram):
        self.loads += 1; self.t = 0
        if self.loads > self.max_iters:
            raise _Stop()
    load_state = _fetch_obs = lambda self, i, *st: None
    _seed_cells = freeze_door_seen = lambda self, x: None
    cell_of = lambda self, i: None
    _won = lambda self, c: c in self.archive and self.archive[c][3] > 0
    close = lambda self: None
    _obs = lambda self: np.zeros((self.num_actors, 84, 84, 4), np.float32)
    obs_u8_stack = lambda self: np.zeros((self.num_actors, 84, 84, 4), np.uint8)
    def step(self, act):
        self.t += 1; n = self.num_actors
        self.last_terms = {'clear': np.zeros(n, np.float32), 'progress': np.full(n, 10.0, np.float32)}
        self.last_leaving = np.ones(n, bool)
        return self._obs(), self.r_fn(self.t, n), np.full(n, self.t >= self.die_at), [{'max_x_pos': 0} for _ in range(n)]


import device_support
CFG42 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'mario_ppo_native_42.yaml')
cfg42 = yaml.safe_load(open(CFG42))['params']['config']


def run_main(argv, max_iters=2, r_fn=None):
    """grpo main() on _FakeEnv, CPU, --clip-every 0: stdout, envs made,
    TensorBoard scalars {tag: {it: v}}, full-game eval calls, the device
    requested from resolve_device, the run dir, an unexpected exception."""
    _FakeEnv.made, _FakeEnv.max_iters = [], max_iters
    _FakeEnv.r_fn = staticmethod(r_fn or (lambda t, n: (np.arange(n) % 2).astype(np.float32)))
    res = dict(sc={}, fg=[], dev=[], logdir=[], err=None)
    class W:
        def __init__(self, d): res['logdir'].append(d)
        def add_scalar(self, k, v, it): res['sc'].setdefault(k, {})[it] = float(v)
        def flush(self): pass
    saved = (G.MarioNativeVecEnv, G.SummaryWriter, G.full_game_eval, G.clean_door_eval, device_support.resolve_device, sys.argv)
    G.MarioNativeVecEnv, G.SummaryWriter = _FakeEnv, W
    G.full_game_eval = lambda *x, **k: (res['fg'].append(k.get('seed')), dict(level_mean=0.0, level_max=0, victory=0.0, off_route=0.0))[1]
    G.clean_door_eval = lambda *x, **k: dict(mean_x=0.0, max_x=0.0, victory=0.0, loop=0.0, clear=0.0, max_x_all=[0])
    device_support.resolve_device = lambda d=None: (res['dev'].append(d), 'cpu')[1]
    sys.argv = ['train_grpo.py', '--config', CFG42, '--clip-every', '0'] + list(argv)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            G.main()
    except (_Stop, SystemExit):
        pass
    except Exception as ex:
        res['err'] = repr(ex)
    finally:
        G.MarioNativeVecEnv, G.SummaryWriter, G.full_game_eval, G.clean_door_eval, device_support.resolve_device, sys.argv = saved
    res.update(out=out.getvalue(), envs=list(_FakeEnv.made), run_dir=os.path.dirname(res['logdir'][0]) if res['logdir'] else None)
    return res


tmp = tempfile.mkdtemp()
_FakeEnv.r_fn = staticmethod(lambda t, n: np.zeros(n, np.float32))
G.MarioNativeVecEnv, saved_env = _FakeEnv, G.MarioNativeVecEnv
_FakeEnv.made = []
try:
    G.full_game_eval(model, {'env_config': dict(cfg42['env_config'])}, dev, 2, max_steps=8, n_threads=1, seed=1)
finally:
    G.MarioNativeVecEnv = saved_env
rl = _FakeEnv.made[0].kw.get('route_levels') if _FakeEnv.made else None
check('full-game eval: exits count by the config route (4-2 run: the warp into 8-1 is on route)', rl == cfg42['env_config']['route_levels'], rl)

# run A: --grow-archive from a winners-only PPO archive, --rtg; every rollout ends at step 4 of 8
src = os.path.join(tmp, 'archive_in.pkl'); K42 = lambda b: ('4-2', 0, b, 5, 0, 1, 0, 3)
pickle.dump({K42(b): [[b'%d' % b], 0, 300, int(b < 12), 0, 0] for b in (10, 11, 12)}, open(src, 'wb'))
src_bytes = open(src, 'rb').read()
torch.manual_seed(12345)
# same summed outcome in every rollout, different timing: R is dead, reward-to-go is not
rA = run_main(['--run-name', os.path.join(tmp, 'A'), '--archive', src, '--grow-archive', '--winners-only', '--rtg', '--seed', '3',
               '--groups', '2', '--group', '2', '--horizon', '8', '--eval-every', '3', '--fullgame-every', '2'],
              r_fn=lambda t, n: np.where(np.arange(n) % 2 == 0, float(t == 1), float(t == 3)).astype(np.float32))
seedA = torch.initial_seed()
check('main (fake env) runs two iterations', rA['err'] is None and rA['run_dir'] and 2 in rA['sc'].get('grpo/frames', {}), (rA['err'], rA['out'][-800:]))
kwA = next((e.kw for e in rA['envs'] if e.name == 'grpo'), {})
check('main: archive caps default to the config (self_restart_cells 16384, cell_max_variants 8), not 1024 / 3',
      (kwA.get('self_restart_cells'), kwA.get('cell_max_variants')) == (16384, 8), (kwA.get('self_restart_cells'), kwA.get('cell_max_variants')))
recA = json.load(open(os.path.join(rA['run_dir'], 'launch.json'))) if rA['run_dir'] and os.path.exists(os.path.join(rA['run_dir'], 'launch.json')) else {}
check('main: launch.json records the resolved archive caps',
      (recA.get('env_config', {}).get('self_restart_cells'), recA.get('env_config', {}).get('cell_max_variants')) == (16384, 8), recA.get('env_config'))
apA = kwA.get('archive_path')
check('main: the env grows a copy of the input archive in the run dir (the input is never its archive_path)',
      apA and os.path.abspath(apA) != os.path.abspath(src) and os.path.dirname(apA) == rA['run_dir']
      and open(apA, 'rb').read() == src_bytes and open(src, 'rb').read() == src_bytes, apA)
check('main: --seed seeds torch too', seedA == 3, seedA)
check('main: the device defaults to the config\'s (cuda:0, device_support falls back)', rA['dev'] == [cfg42['device']], rA['dev'])
frA = rA['sc'].get('grpo/frames', {})
check('main: frames count the steps actually taken (4 of 8 per rollout: 2 its x 4 envs x 4 steps x 4 frames)', frA.get(2) == 2 * 4 * 4 * 4, frA)
check('main: the full-game eval runs on its own schedule (--fullgame-every 2, --eval-every 3: at it 2)', len(rA['fg']) == 1, rA['fg'])
lgA = rA['sc'].get('grpo/live_groups', {})
check('main: --rtg live groups count non-zero per-step advantages (summed outcomes are all equal here)', lgA.get(1) == 1.0, lgA)
pcA = rA['sc'].get('grpo/prompt_cells', {})
check('main: --winners-only --grow-archive keeps the non-winning archive cell out of the pool', pcA.get(1) == 2.0, pcA)

# run B: no --grow-archive, explorers and the config's novelty bonus; --outcome clear with progress
rB = run_main(['--run-name', os.path.join(tmp, 'B'), '--archive', os.path.join(tmp, 'none.pkl'), '--explorers', '1',
               '--outcome', 'clear', '--outcome-progress', '0.2', '--groups', '1', '--group', '2', '--horizon', '8', '--eval-every', '100'], max_iters=1)
outB = ' '.join(rB['out'].split())
check('main: warns that the novelty bonus (config 100) is never paid without --grow-archive', 'never paid without --grow-archive' in outB, rB['out'][:600])
check('main: warns that --explorers does nothing without --grow-archive', '--explorers does nothing without --grow-archive' in outB, rB['out'][:600])
check('main: --outcome-progress pays nothing on leaving steps (env.last_leaving)', rB['sc'].get('rewards/step', {}).get(1) == 0.0, (rB['err'], rB['sc'].get('rewards/step')))

rC = run_main(['--run-name', os.path.join(tmp, 'C'), '--archive', os.path.join(tmp, 'none.pkl'), '--device', 'cpu'], max_iters=0)
check('main: --device overrides the config device', rC['dev'] == ['cpu'], (rC['dev'], rC['out'][-300:]))
rD = run_main(['--help'])
hD = ' '.join(rD['out'].split())
check('main: --hint help says every remaining demo step is hinted', 'every remaining demo step' in hD and 'first free step' not in hD, hD[hD.find('--hint'):][:200])

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
