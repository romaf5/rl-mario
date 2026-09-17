"""Env checks for the invisible explorers, the trained-level archive gate, the
tile-variant cap key and end_on_stage_exit (2026-09-16).
Run: venv_retro/bin/python tests/explorer_bench.py

Properties:
  * explorer cores step in the batch but the trainer only ever sees the
    training envs (obs / reward / done / infos / time_outs), training envs
    never take a random walk, a walk ends at its budget, and walks count
    neither as policy restarts (uses / tries) nor as door episodes;
  * the archive saves states of the TRAINED levels only (a 4-2 run with the
    full route used to archive 8-x states after the warp);
  * the tile-variant cap counts signatures of one spot; camera bins are part
    of the spot, not variants;
  * end_on_stage_exit ends the episode at the paid route exit.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mario_native_vecenv import MarioNativeVecEnv, _load_state

ROUTE = ['1-1', '1-2', '4-1', '4-2', '8-1', '8-2', '8-3', '8-4']
OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


def make(n, **kw):
    ec = dict(full_game=True, random_stages=['4-2'], route_levels=ROUTE, sticky_actions=0,
              explore_eps=0, self_restart_prob=1e-6, reset_noops=0, n_threads=2, dense_infos=True,
              cell_tiles=True, cell_x_bin=32, cell_y_band=16, cell_screen_bin=64,
              cell_max_variants=3, explore_pure=True, seed=0, episode_life=True)
    ec.update(kw)
    env = MarioNativeVecEnv('xbench', n, **ec)
    return env, env.reset()


# ---------------------------------------------------------------- invisible explorers
env, obs = make(3, explorer_envs=8, explore_episode_steps=40)
check('explorers: reset returns the training envs only', obs.shape == (3, 84, 84, 4), obs.shape)
shapes_ok, max_train_walk, max_walk_steps, door_explorer = True, 0, 0, False
for s in range(400):
    obs, r, d, inf = env.step(np.full(3, 3))
    shapes_ok &= (obs.shape[0] == 3 and r.shape == (3,) and d.shape == (3,)
                  and len(inf) == 3 and len(inf.time_outs) == 3)
    max_train_walk = max(max_train_walk, int(env.explorer[:3].max()))
    max_walk_steps = max(max_walk_steps, int(env.ep_steps[3:].max()))
    door_explorer |= bool(env.is_door[3:].any())
check('explorers: step returns obs/reward/done/infos/time_outs of the training envs only', shapes_ok)
check('explorers: training envs never take a random walk', max_train_walk == 0, max_train_walk)
check('explorers: a walk ends at its step budget (40)', max_walk_steps <= 40, max_walk_steps)
check('explorers: explorer episodes are never door episodes', not door_explorer)
check('explorers: walks restart from archive cells (%d walks from %d cells)' % (env.n_walks, len(env.explore_walks)),
      env.n_walks > 8 and len(env.explore_walks) > 0, (env.n_walks, len(env.explore_walks)))
check('explorers: walks count neither as policy uses nor as tries',
      len(env.archive) > 0 and all(e[1] == 0 for e in env.archive.values()) and not env.cell_tries,
      (len(env.archive), sum(e[1] for e in env.archive.values()), len(env.cell_tries)))
env.close()

env, obs = make(2, explorer_envs=8, self_restart_prob=0.0)
check('explorers: off without an archive (self_restart_prob 0)', env.n_explorers == 0 and obs.shape[0] == 2, env.n_explorers)
env.close()
env, obs = make(2, explorer_envs=8, play_mode=True)
check('explorers: off in play mode', env.n_explorers == 0 and obs.shape[0] == 2, env.n_explorers)
env.close()
env, obs = make(2)
check('explorers: default (explorer_envs 0) leaves the batch unchanged', env.num_actors == 2 and env.n_explorers == 0)
env.close()

# ---------------------------------------------------------------- archive gate: trained levels only
env, obs = make(2)                  # trains 4-2 on the full route
st = _load_state('8-1')
for i in range(2):
    env.load_state(i, st); env._fetch_obs(i)
env._post_reset_init(range(2), env.ram)
for s in range(150):
    env.step(np.array([3, 4]))
check('archive gate: a 4-2 run playing 8-1 archives no 8-1 state', not any(c[0] == '8-1' for c in env.archive),
      sorted({c[0] for c in env.archive}))
env.close()
env, obs = make(2, random_stages=['8-1'])
for s in range(150):
    env.step(np.array([3, 4]))
check('archive gate: an 8-1 run archives 8-1 states', any(c[0] == '8-1' for c in env.archive), len(env.archive))
env.close()

# ---------------------------------------------------------------- tile-variant cap key
env, obs = make(1)
K = lambda sig, screen: ('4-2', 2, 32, 5, 0, 2, sig, screen)
for scr in (14, 15, 16):
    env.archive[K(111, scr)] = [[b'x'], 0, 300]
check('variant cap: camera bins of one spot are not tile variants of each other (1)',
      env._tile_variants(K(222, 14)) == 1, env._tile_variants(K(222, 14)))
env.archive[K(222, 14)] = [[b'x'], 0, 300]; env.archive[K(333, 14)] = [[b'x'], 0, 300]
check('variant cap: tile variants of one spot and camera bin are counted (3)',
      env._tile_variants(K(444, 14)) == 3, env._tile_variants(K(444, 14)))
env.archive.clear()
env.close()


# ---------------------------------------------------------------- end_on_stage_exit
def exit_probe(world, level, **kw):
    """Door of 4-2, then the level bytes are set to another level (what the
    game does on a pipe / vine warp) and the env scores the next steps."""
    env, obs = make(1, **kw)
    for s in range(5):
        env.step(np.array([0]))
    env.ram[0, 0x75F] = world; env.ram[0, 0x75C] = level
    env.lib.benv_set_ram(env.env, 0, env.ram[0].tobytes())
    out = []
    for s in range(4):
        obs, r, d, inf = env.step(np.array([0]))
        out.append((float(r[0]), bool(d[0]), bool(env.last_signals.wrong_exit[0])))
        if d[0]:
            break
    env.close()
    return out


on = exit_probe(7, 0, end_on_stage_exit=True)
off = exit_probe(7, 0)
check('end_on_stage_exit: the paid warp into 8-1 ends the episode on the clear step (+1900)',
      on[-1][1] and on[-1][0] >= 1900 and not on[-1][2], on)
check('end_on_stage_exit: off by default -- the episode plays on after the clear',
      any(r >= 1900 for r, _, _ in off) and not any(d for _, d, _ in off), off)
wrong = exit_probe(3, 2, end_on_stage_exit=True)
check('end_on_stage_exit: a wrong exit (4-3) still ends the episode unpaid',
      wrong[-1][1] and wrong[-1][2] and wrong[-1][0] < 500, wrong)

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
