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

# ---------------------------------------------------------------- frontier restarts (2026-09-12)
import collections
def frontier_env(**kw):
    env = make(1, random_stages=['1-1', '1-2'], self_restart_prob=1.0, self_restart_frontier_prob=1.0,
               self_restart_frontier_k=16, **kw)
    st = env.states['1-1']
    K = lambda lvl, b: (lvl, 2, b, 2, 0, 1, 0)
    env.archive = {}
    for b in range(30):                       # level A: 30 heavily tried cells that NEVER won (dead ends)
        env.archive[K('1-1', b)] = [[st], 500, 300]; env.cell_tries[K('1-1', b)] = 500
    for b in (44, 45):                        # level B: 2 cells that win 5% of the time
        env.archive[K('1-2', b)] = [[st], 100, 300]; env.cell_tries[K('1-2', b)] = 100; env.cell_wins[K('1-2', b)] = 5
    return env
def draws(env, n=300):
    c = collections.Counter()
    for _ in range(n):
        env._reset_env(0); c[env.start_cell[0][0]] += 1
    return c
env = frontier_env(); d = draws(env); env.close()
check('frontier: draws go to cells that have WON, never to never-winning dead ends (1-2 winners %d/300, 1-1 dead ends %d/300)' % (d['1-2'], d['1-1']),
      d['1-1'] == 0 and d['1-2'] == 300, dict(d))
env = frontier_env()
for b in range(30):                           # now level A's cells also win, at a worse rate (harder): global top-k would take them all
    env.cell_wins[('1-1', 2, b, 2, 0, 1, 0)] = 10
d = draws(env); env.close()
check('frontier (global): the 16 hardest winners crowd out the other level (1-2 gets %d/300)' % d['1-2'], d['1-2'] == 0, dict(d))
env = frontier_env(frontier_per_level=True)
for b in range(30):
    env.cell_wins[('1-1', 2, b, 2, 0, 1, 0)] = 10
d = draws(env); env.close()
check('frontier per level: each level keeps its share of the draws (1-2 gets %d/300, 1-1 %d/300)' % (d['1-2'], d['1-1']),
      100 <= d['1-2'] <= 200 and 100 <= d['1-1'] <= 200, dict(d))
env = frontier_env(frontier_per_level=True)
for c in list(env.cell_wins): env.cell_wins.pop(c)      # no winners anywhere: fall back to least-practised cells of the chosen level
env.archive[('1-2', 2, 46, 2, 0, 1, 0)] = [[env.states['1-1']], 0, 300]     # a fresh, never-tried 1-2 cell
d = draws(env); fresh = sum(1 for _ in range(0))
cnt = collections.Counter()
for _ in range(300):
    env._reset_env(0); cnt[env.start_cell[0]] += 1
env.close()
lv12 = {c: v for c, v in cnt.items() if c[0] == '1-2'}
check('frontier per level, no winners: the least-practised cell of the level gets the most draws (fresh %d vs %s; uses are balanced as they accrue)'
      % (cnt[('1-2', 2, 46, 2, 0, 1, 0)], sorted(v for c, v in lv12.items() if c[2] != 46)),
      cnt[('1-2', 2, 46, 2, 0, 1, 0)] == max(lv12.values()) and sum(lv12.values()) > 100, dict(lv12))

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
