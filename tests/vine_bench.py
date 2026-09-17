"""4-2 vine warp end to end on the training path (verification outside training).
Run: venv_retro/bin/python tests/vine_bench.py

Replays input sequences found by a savestate search from the Level4-2 door
(COMPLEX_MOVEMENT indices, 8 NOOPs appended so the level change confirms)
through the 4-2 training config, explorers off, no archive file:
  * tests/data/vine_route_4-2.npy: bump the vine brick, climb, vine into the
    bonus area, DOWN into the world-8 pipe;
  * tests/data/exit_flag_4-2.npy: the flag into 4-3 (a wrong exit).
Properties (2026-09-16 review):
  * the vine lift is a new frame (no hold, page reset, time-out or done) and
    the bonus area pays progress;
  * the world-8 pipe pays 500 + per_extra * 14 on its confirm step and ends
    the episode as a real terminal (end_on_stage_exit), not a time-out;
  * with the config's gamma the vine route is worth more than the flag run
    from the decision point (x >= 1000), novelty bonus excluded;
  * the flag into 4-3 is a wrong exit: done, and the leaving steps pay 0;
  * no archived state was saved mid-bump ($23 in its tile columns).
"""
import os, sys
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from mario_native_vecenv import MarioNativeVecEnv

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'mario_ppo_native_42.yaml')))['params']['config']
ec = dict(cfg['env_config']); [ec.pop(k, None) for k in ('name', 'action_type')]
ec.update(archive_path=None, explorer_envs=0, n_threads=1, dense_infos=True, seed=0)
per_extra = next(t['per_extra'] for t in ec['reward'] if t['type'] == 'level_clear')
gamma = cfg['gamma']


def replay(name):
    acts = np.load(os.path.join(ROOT, 'tests', 'data', name)).astype(int)
    env = MarioNativeVecEnv('vine', 1, **ec); env.reset()
    log = []
    for a in acts:
        obs, r, d, inf = env.step(np.array([a])); S = env.last_signals
        log.append(dict(r=float(r[0]), d=bool(d[0]), x=int(S.x[0]), atype=int(S.atype[0]), gp=int(S.gp[0]),
                        frame=bool(S.frame_change[0]), hold=bool(S.hold[0]), reset=bool(S.page_reset[0]),
                        timeout=bool(S.timeout[0]), wrong=bool(S.wrong_exit[0]), delta=int(S.level_delta[0]),
                        tout=bool(inf.time_outs[0]),
                        bonus=float(env.last_terms.get('cell_bonus', np.zeros(1))[0])))
        if d[0]:
            break
    # archived states: load each into a probe core and read its tile columns
    probe = MarioNativeVecEnv('probe', 1, **dict(ec, self_restart_prob=0.0)); probe.reset()
    bounce = []
    for c, e in env.archive.items():
        st = e[0][0] if isinstance(e[0], list) else e[0]
        probe.load_state(0, st); probe._fetch_obs(0)
        x = int(probe.ram[0, 0x6D]) * 256 + int(probe.ram[0, 0x86])
        if (probe._tile_grid(0, x) == 0x23).any():
            bounce.append(c)
    n_cells = len(env.archive)
    probe.close(); env.close()
    return log, n_cells, bounce


def value_from(log, x0=1000):
    d = next(i for i, l in enumerate(log) if l['x'] >= x0 and l['atype'] == 2)
    r = np.array([l['r'] - l['bonus'] for l in log[d:]])
    return float((r * gamma ** np.arange(len(r))).sum())


vine, n_vine, bounce_vine = replay('vine_route_4-2.npy')
lift = next((i for i, l in enumerate(vine) if l['atype'] == 1 and l['x'] < 200), None)
check('vine: the replay reaches the bonus area (AreaType 1, x < 200)', lift is not None, [(l['x'], l['atype']) for l in vine[-5:]])
if lift is not None:
    l = vine[lift]
    check('vine: the lift into the bonus area is a new frame, not a hold / page reset / time-out / done',
          l['frame'] and not (l['hold'] or l['reset'] or l['timeout'] or l['d']), l)
    check('vine: the bonus area pays progress', sum(v['r'] for v in vine[lift:lift + 60]) > 0)
check('vine: no page reset, time-out or wrong exit anywhere on the route',
      not any(v['reset'] or v['timeout'] or v['wrong'] for v in vine), [i for i, v in enumerate(vine) if v['reset'] or v['timeout'] or v['wrong']])
last = vine[-1]
clear = 500 + per_extra * 14
check('vine: the world-8 pipe confirms 8-1 (+15 levels), pays the clear %d and ends the episode' % clear,
      last['d'] and last['delta'] == 15 and last['r'] >= clear and not last['wrong'], last)
check('vine: the warp is a real terminal, not a time-out (no bootstrap)', last['d'] and not last['tout'], last)
check('vine: the pending step inside 8-1 pays nothing', len(vine) > 1 and vine[-2]['gp'] == 28 and vine[-2]['r'] == 0, vine[-2])

flag, n_flag, bounce_flag = replay('exit_flag_4-2.npy')
fl = flag[-1]
check('flag: the flag into 4-3 is a wrong exit that ends the episode', fl['d'] and fl['wrong'], fl)
check('flag: the leaving steps (inside 4-3) pay 0', all(v['r'] == 0 for v in flag if v['gp'] != 13), [v['r'] for v in flag if v['gp'] != 13])
v_vine, v_flag = value_from(vine), value_from(flag)
check('valuation: with gamma %.3f / per_extra %d the vine route is worth more than the flag run from x >= 1000 (%.0f vs %.0f)'
      % (gamma, per_extra, v_vine, v_flag), v_vine > 1.25 * v_flag, (v_vine, v_flag))
check('archive: no state saved mid-bump ($23) on either route (%d + %d cells)' % (n_vine, n_flag),
      not bounce_vine and not bounce_flag, (bounce_vine[:3], bounce_flag[:3]))

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
