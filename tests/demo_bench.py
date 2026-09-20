"""Backward-curriculum checks (Go-Explore phase 2 on the agent's own route).
Run: venv_retro/bin/python tests/demo_bench.py

Properties:
  * every archived state variant carries its action prefix from the level's
    door, and replaying that prefix from the door reproduces the saved state
    byte for byte (door episodes, restarts, explorer walks; through reservoir
    rotation); archives without prefixes still load;
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

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
