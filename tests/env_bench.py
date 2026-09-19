"""Env checks for the 2026-09-19 review fixes (training path of the 4-2 config).
Run: venv_retro/bin/python tests/env_bench.py

Properties:
  * the step's Signals survive the resets done inside that step (the trackers
    used to alias them: a reset rewrote .t / .frame of the step that ended);
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
EC = dict(cfg['env_config']); [EC.pop(k, None) for k in ('name', 'action_type')]
EC.update(archive_path=None, explorer_envs=0, n_threads=1, dense_infos=True, seed=0)


def make(n=1, **kw):
    env = MarioNativeVecEnv('ebench', n, **dict(EC, **kw))
    return env, env.reset()


def kill(env, i=0):
    """Drop Mario below the screen: the game kills him on the next frame."""
    env.ram[i, 0xB5] = 2; env.ram[i, 0xCE] = 0xF0
    env.lib.benv_set_ram(env.env, int(i), env.ram[i].tobytes())


def replay(env, name):
    for a in np.load(os.path.join(ROOT, 'tests', 'data', name)).astype(int):
        obs, r, d, inf = env.step(np.array([a]))
        if d[0]:
            return obs, r, d, inf
    return None


# ---------------------------------------------------------------- signals are not aliased by resets
env, obs = make(self_restart_prob=0.0)
out = replay(env, 'exit_flag_4-2.npy')
sg = env.last_signals
check('signals: the wrong-exit step keeps its own timer (sig.t %d = info time %d, not the reset door timer %d)'
      % (sg.t[0], out[3][0]['time'], env.time_last[0]) if out else 'signals: flag run replayed',
      out is not None and int(sg.t[0]) == out[3][0]['time'] != int(env.time_last[0]),
      None if out is None else (int(sg.t[0]), out[3][0]['time'], int(env.time_last[0])))
check('signals: the wrong-exit step keeps its own frame (4-3), not the reset door frame',
      out is not None and bool(sg.wrong_exit[0]) and sg.frame[0] != env.prev_frame[0],
      None if out is None else (int(sg.frame[0]), int(env.prev_frame[0])))
env.close()

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
