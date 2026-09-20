"""Backward-curriculum checks (Go-Explore phase 2 on the agent's own route).
Run: venv_retro/bin/python tests/demo_bench.py

Properties:
  * every archived state variant carries its action prefix from the level's
    door, and replaying that prefix from the door reproduces the saved state
    byte for byte (door episodes, restarts, explorer walks; through reservoir
    rotation); archives without prefixes still load;
  * the first on-route clear with a known prefix becomes the route, verified
    by replay; a wrong exit never does; the route and tau* survive a save and
    reload, and a fresh run refuses the route sidecar;
  * demo_start_prob of the resets start on the route inside [tau*, tau*+W];
    tau 0 is a door episode, other route starts are neither door nor restart
    and credit no cell; tau* moves back demo_step at a 20% band success rate
    (13/64) and not below (12/64); a route start that plays the rest of the
    route warps and counts as a band success;
"""
import os, sys, tempfile
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import mario_native_vecenv as mnv
from mario_native_vecenv import MarioNativeVecEnv

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'mario_ppo_native_42.yaml')))['params']['config']
EC = dict(cfg['env_config']); [EC.pop(k, None) for k in ('name', 'action_type')]
EC.update(archive_path=None, explorer_envs=0, n_threads=1, dense_infos=True, seed=0,
          demo_start_prob=0.75, demo_window=32, demo_step=16, demo_success=0.2, demo_success_n=64)
VINE = np.load(os.path.join(ROOT, 'tests', 'data', 'vine_route_4-2.npy')).astype(np.int8)


def make(n=1, **kw):
    env = MarioNativeVecEnv('dbench', n, **dict(EC, **kw))
    return env, env.reset()


def replay_state(env, actions):
    """Independent replay: the level's door state, raw batch steps on a fresh
    1-core env, the savestate at the end."""
    lib = env.lib; rom = open(mnv.ROM, 'rb').read()
    e1 = lib.benv_create(rom, len(rom), 1, 1, int(env.single_stage))
    buf = __import__('ctypes').create_string_buffer(env.state_size)
    a = np.zeros(1, np.int32); o = np.zeros((1, 84, 84), np.uint8); r = np.zeros((1, 0x800), np.uint8)
    lib.benv_load(e1, 0, env.states[env.stages[0]])
    for x in np.frombuffer(actions, np.int8):
        a[0] = mnv._ACTION_BYTES[int(x)]
        lib.benv_step(e1, a.ctypes.data, o.ctypes.data, r.ctypes.data)
    lib.benv_save(e1, 0, buf); s = bytes(buf.raw); lib.benv_destroy(e1)
    return s


class _Always:                         # every refresh chance taken (rotations happen)
    def __init__(self, r): self.r = r
    def random_sample(self, *a): return np.zeros(a[0]) if a else 0.0
    def __getattr__(self, k): return getattr(self.r, k)


# ---------------------------------------------------------------- prefixes replay exactly
env, obs = make(n=4, self_restart_prob=0.6, explorer_envs=4, explore_pure=True)
rs = np.random.RandomState(1)
for s in range(300):
    env.step(rs.randint(0, 12, size=4))
env.rng = _Always(env.rng)             # force reservoir refreshes and rotations
for s in range(200):
    env.step(rs.randint(0, 12, size=4))
pairs = [(c, k, st, p) for c, e in env.archive.items() for k, (st, p) in enumerate(zip(e[0], env._prefixes(e)))]
known = [(c, k, st, p) for c, k, st, p in pairs if p is not None]
full = sum(len(e[0]) == 4 for e in env.archive.values())
bad = [(c, k) for c, k, st, p in known[:60] if replay_state(env, p) != st]
check('prefix: every archived variant has a prefix (%d of %d)' % (len(known), len(pairs)),
      len(pairs) >= 20 and len(known) == len(pairs), (len(known), len(pairs)))
check('prefix: %d variants replay byte-identically from the door (%d cells at 4 variants: rotated)'
      % (min(60, len(known)), full), not bad and full > 0, bad[:3])
check('prefix: variants and prefixes stay aligned (%d cells)' % len(env.archive),
      all(len(e[0]) == len(e[8]) for e in env.archive.values()))
env.close()

# ---------------------------------------------------------------- feature off = no prefixes, no extra draws
env0, _ = make(n=2, self_restart_prob=0.6, demo_start_prob=0.0)
for s in range(120):
    env0.step(rs.randint(0, 12, size=2))
check('off: demo_start_prob 0 stores no prefixes (entries of %d slots)' % max(len(e) for e in env0.archive.values()),
      not env0.demo_on and all(len(e) < 9 for e in env0.archive.values()))
env0.close()

# ---------------------------------------------------------------- old archives load
env, _ = make(n=1, self_restart_prob=0.6)
env.lib.benv_save(env.env, 0, env._sbuf)
old = [[bytes(env._sbuf.raw)], 0, 400, 0, 0, 0, 0, 0]
check('old entry: no [8] -> prefixes None, padded and aligned', env._prefixes(old) == [None] and len(old) == 9)
env.close()

# ---------------------------------------------------------------- the first clear becomes the route
env, obs = make(n=1, self_restart_prob=1e-6)
k_done = None
for k, a in enumerate(VINE):
    obs, r, d, inf = env.step(np.array([a]))
    if d[0]:
        k_done = k; break
R = env.route
check('route: the vine run from the door becomes the route (%s actions, done at step %s)'
      % (None if R is None else len(R['actions']), k_done),
      R is not None and R['actions'] == VINE[:k_done + 1].tobytes())
check('route: start states stop before the level change (last %s, %d states, gp of last = 4-2)'
      % (None if R is None else R['last'], 0 if R is None else len(R['states'])),
      R is not None and len(R['states']) == R['last'] + 1 and R['last'] <= k_done - 1)
check('route: tau* starts demo_step before the last start state (%s)' % (None if R is None else R['tau_star']),
      R is not None and R['tau_star'] == R['last'] - 16)
check('route: state 0 is the door state', R is not None and R['states'][0] == env.states['4-2'])
env.close()
env, obs = make(n=1, self_restart_prob=1e-6)
flag = np.load(os.path.join(ROOT, 'tests', 'data', 'exit_flag_4-2.npy')).astype(np.int8)
check('route: a wrong exit (flag into 4-3) is rejected', not env.set_route(flag.tobytes()) and env.route is None)
check('route: set_route accepts the vine run', env.set_route(VINE.tobytes()) and env.route is not None)
env.close()

# ---------------------------------------------------------------- persistence
path = os.path.join(tempfile.mkdtemp(), 'archive.pkl')
env, obs = make(n=1, self_restart_prob=1e-6, archive_path=path)
for s in range(30):
    env.step(np.array([3]))
env.set_route(VINE.tobytes()); env.route['tau_star'] = 123
env.close()
env, obs = make(n=1, self_restart_prob=1e-6, archive_path=path)
check('persistence: route and tau* reload from the sidecar (tau* %s)' % (env.route and env.route['tau_star']),
      env.route is not None and env.route['tau_star'] == 123 and env.route['actions'] == VINE.tobytes())
check('persistence: archive entries keep their prefixes', all(len(e) == 9 for e in env.archive.values()))
env.close()
import train
os.remove(path)
cfgp = {'params': {'config': {'env_config': {'archive_path': path}}}}
check('persistence: a fresh run refuses an existing route sidecar', train.fresh_archive_conflict(cfgp, None, False))

# ---------------------------------------------------------------- curriculum draws
env, obs = make(n=1, self_restart_prob=0.6)
env.set_route(VINE.tobytes()); R = env.route
taus = []
for k in range(2000):
    env._reset_env(0); taus.append(int(env.demo_tau[0]))
taus = np.array(taus); on = taus >= 0
check('draws: %.3f of resets start on the route (0.75)' % on.mean(), abs(on.mean() - 0.75) < 0.04)
check('draws: every route start is inside [tau*, tau*+W] (%d..%d, tau* %d)' % (taus[on].min(), taus[on].max(), R['tau_star']),
      taus[on].min() >= R['tau_star'] and taus[on].max() <= min(R['tau_star'] + 32, R['last']))
env._reset_env(0)
while env.demo_tau[0] < 0:
    env._reset_env(0)
check('draws: a route start is neither door nor restart and credits no cell',
      not env.is_door[0] and not env.was_restart[0] and env.start_cell[0] is None)
R['tau_star'] = 0
while env.demo_tau[0] != 0:
    env._reset_env(0)
check('draws: tau 0 is a door episode', env.is_door[0] and not env.was_restart[0])
env.close()

# ---------------------------------------------------------------- tau* moves at 20%, not below
env, obs = make(n=1, self_restart_prob=0.6)
env.set_route(VINE.tobytes()); R = env.route; t0 = R['tau_star']
for j in range(64):
    env._demo_outcome(t0, j >= 52)     # 52 failures then 12 successes (the window slides off a failure next)
check('tau*: 12/64 band successes keep tau* (%d)' % R['tau_star'], R['tau_star'] == t0 and abs(R['rate'] - 12 / 64) < 1e-9)
env._demo_outcome(t0 + 40, True)       # outside the band: ignored
check('tau*: outcomes outside the band are ignored', R['tau_star'] == t0)
env._demo_outcome(t0, True)            # sliding window: 13/64
check('tau*: 13/64 moves tau* back by demo_step (%d -> %d)' % (t0, R['tau_star']),
      R['tau_star'] == t0 - 16 and R['band'] == [] and R['moves'] == 1)
env.close()

# ---------------------------------------------------------------- a route start that plays the rest clears
env, obs = make(n=1, self_restart_prob=0.6, demo_start_prob=0.999999)
env.set_route(VINE.tobytes()); R = env.route
env._reset_env(0); env._fetch_obs(0); env._post_reset_init([0], env.ram)
tau = int(env.demo_tau[0]); res = None
for a in VINE[tau:]:
    obs, r, d, inf = env.step(np.array([a]))
    if d[0]:
        res = (float(r[0]), inf[0]['stages_cleared'], inf[0]['demo_start'], inf[0]['demo_tau']); break
check('suffix: from route state %d the rest of the route warps (%s)' % (tau, res),
      res is not None and res[0] >= 4700 and res[1] == 1 and res[2] and res[3] == tau)
check('suffix: the clear is a band success (%s)' % R['band'], (tau >= R['tau_star'] + 16) or R['band'][-1:] == [True])
env.close()

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
