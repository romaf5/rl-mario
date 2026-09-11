"""Env checks for the relative novelty bonus and the eval env defaults.
Run: venv_retro/bin/python tests/novelty_bench.py

Properties (2026-09-11 review):
  * the bonus is a function of the state, not of the env index: two door
    envs that enter the same new cell in the same step are paid alike;
  * while frozen (a GRPO rollout), the door counts neither grow nor decay,
    so the same walk pays the same at any point of the horizon; unfreezing
    merges what the rollout saw;
  * the single-env eval adapter steps hack-free by default (deaths, pipes
    and the ending are visible, and the video comes from the same core the
    timer hack is applied to).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mario_native_vecenv import MarioNativeVecEnv, NativeEvalEnv

ROUTE = ['1-1', '1-2', '4-1', '4-2', '8-1', '8-2', '8-3', '8-4']
OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


def make(n, **kw):
    ec = dict(full_game=True, random_stages=['1-1'], route_levels=ROUTE, sticky_actions=0,
              explore_eps=0, self_restart_prob=1e-6, reset_noops=0, n_threads=1, dense_infos=True,
              cell_bonus=100.0, cell_bonus_relative=True, cell_x_bin=64, seed=0, episode_life=True)
    ec.update(kw)
    env = MarioNativeVecEnv('bench', n, **ec); env.reset()
    return env


# ---------------------------------------------------------------- order independence
env = make(2)
bonus = [[], []]
for s in range(40):
    env.step(np.array([3, 3]))
    b = env.last_terms['cell_bonus']
    bonus[0].append(float(b[0])); bonus[1].append(float(b[1]))
env.close()
check('novelty: two identical door envs are paid the same bonus per step (%.0f vs %.0f)'
      % (sum(bonus[0]), sum(bonus[1])), bonus[0] == bonus[1] and sum(bonus[0]) > 0, (sum(bonus[0]), sum(bonus[1])))

# ---------------------------------------------------------------- frozen counts
env = make(1)
env.door_seen = {'marker': 1.0}
env.freeze_door_seen(True)
for s in range(70):
    env.step(np.array([3]))
check('novelty: frozen counts do not decay (marker still 1.0 after 70 steps)', env.door_seen.get('marker') == 1.0, env.door_seen.get('marker'))
check('novelty: frozen counts do not grow (only the marker in door_seen)', set(env.door_seen) == {'marker'}, list(env.door_seen)[:5])
check('novelty: entries seen while frozen are pending', len(env._door_seen_pending) > 0, len(env._door_seen_pending))
env.freeze_door_seen(False)
check('novelty: unfreezing merges the pending entries', len(env.door_seen) > 1, len(env.door_seen))
env.close()

# ---------------------------------------------------------------- eval adapter default
ev = NativeEvalEnv(full_game=True, random_stages=['1-1'], route_levels=ROUTE, n_threads=1)
check('eval env: NativeEvalEnv steps hack-free by default', ev.v._raw_steps is True, ev.v._raw_steps)
ev.close()

# ---------------------------------------------------------------- defensive checks (C++ review 2026-09-11)
env = make(1)
try:
    env.load_state(0, b'\x00' * 100); bad = False
except ValueError:
    bad = True
check('env: loading a savestate of the wrong size raises instead of over-reading', bad)
env.close()
env = make(1, n_threads=0)
env.step(np.array([3])); x1 = int(env.last_signals.x[0]); env.step(np.array([3])); x2 = int(env.last_signals.x[0])
check('env: n_threads=0 is clamped to 1 (the step still advances the game)', x2 > x1, (x1, x2))
env.close()

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
