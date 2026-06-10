#!/usr/bin/env python3
"""Validation suite for the stable-retro Mario backend (mario_env.py).

Run from venv_retro (CPU only; never touches the GPU):

    CUDA_VISIBLE_DEVICES= ./venv_retro/bin/python validate_retro.py
    CUDA_VISIBLE_DEVICES= ./venv_retro/bin/python validate_retro.py --skip-bench-old --skip-train

Checks:
    1. Single-env throughput benchmark, retro backend vs nes_py backend
       (the nes_py env runs in a subprocess using the original venv/).
    2. Correctness probe: obs pipeline, RAM variables, x_pos movement,
       death/life handling, EpisodicLife behavior, reward shaping math
       (progress_reward, idle_penalty, stage_bonus), flag address sanity.
    3. Random-stage reset distribution (50 resets hit multiple stages).
    4. End-to-end CPU smoke training (10 epochs, configs/mario_ppo_cpu_smoke.yaml).
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_PYTHON = os.path.join(HERE, 'venv', 'bin', 'python')
NEW_PYTHON = os.path.join(HERE, 'venv_retro', 'bin', 'python')

RESULTS = []


def record(name, passed, detail=''):
    status = 'PASS' if passed else 'FAIL'
    RESULTS.append((name, passed, detail))
    print(f'  [{status}] {name}' + (f' -- {detail}' if detail else ''))


def check(name, cond, detail=''):
    record(name, bool(cond), detail)
    return bool(cond)


# ---------------------------------------------------------------------------
# 1. Benchmark
# ---------------------------------------------------------------------------

BENCH_CODE = r'''
import time, numpy as np
from {module} import create_mario_env
env = create_mario_env(name='SuperMarioBros-1-1-v0')
env.reset()
n = {steps}
# warmup
for i in range(50):
    _, _, done, _ = env.step(i % 12)
    if done: env.reset()
t0 = time.perf_counter()
for i in range(n):
    _, _, done, _ = env.step(3)  # right + B
    if done: env.reset()
dt = time.perf_counter() - t0
print('BENCH steps_per_sec=%.1f frames_per_sec=%.1f' % (n / dt, 4 * n / dt))
'''


VEC_BENCH_CODE = r'''
import time, numpy as np
from {vecmod} import {veccls}
venv = {veccls}('bench', {workers}, name='SuperMarioBros-1-1-v0')
actions = np.full({workers}, 3, dtype=np.int64)
for _ in range(20):
    venv.step(actions)
n = {steps}
t0 = time.perf_counter()
for _ in range(n):
    venv.step(actions)
dt = time.perf_counter() - t0
venv.close()
total = n * {workers}
print('BENCH steps_per_sec=%.1f frames_per_sec=%.1f' % (total / dt, 4 * total / dt))
'''


def run_bench(python, code):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    out = subprocess.run(
        [python, '-c', code],
        capture_output=True, text=True, cwd=HERE, env=env, timeout=1800)
    m = re.search(r'steps_per_sec=([\d.]+) frames_per_sec=([\d.]+)', out.stdout)
    if not m:
        print(out.stdout[-2000:])
        print(out.stderr[-2000:])
        return None, None
    return float(m.group(1)), float(m.group(2))


def bench(args):
    print('\n=== 1. Single-env benchmark (env steps/sec; 1 env step = 4 frames) ===')
    new_sps, new_fps = run_bench(
        NEW_PYTHON, BENCH_CODE.format(module='mario_env',
                                      steps=args.bench_steps))
    check('retro benchmark ran', new_sps is not None,
          f'{new_sps:.0f} steps/s = {new_fps:.0f} frames/s' if new_sps else '')

    if args.skip_bench_old:
        print('  (skipping nes_py benchmark)')
        return
    old_sps, old_fps = run_bench(
        OLD_PYTHON, BENCH_CODE.format(module='mario_env',
                                      steps=args.bench_steps_old))
    check('nes_py benchmark ran', old_sps is not None,
          f'{old_sps:.0f} steps/s = {old_fps:.0f} frames/s' if old_sps else '')
    if new_sps and old_sps:
        ratio = new_sps / old_sps
        check('single-env speedup >= 5x (target)', ratio >= 5.0,
              f'retro is {ratio:.1f}x faster')

    if args.vec_workers > 0:
        w = args.vec_workers
        print(f'  --- aggregate vecenv benchmark ({w} workers, informational) ---')
        new_v, _ = run_bench(NEW_PYTHON, VEC_BENCH_CODE.format(
            vecmod='mario_vecenv', veccls='MarioRetroVecEnv',
            workers=w, steps=args.bench_steps))
        old_v, _ = run_bench(OLD_PYTHON, VEC_BENCH_CODE.format(
            vecmod='mario_vecenv', veccls='MarioVecEnv',
            workers=w, steps=args.bench_steps_old))
        if new_v and old_v:
            print(f'  vecenv aggregate: retro {new_v:.0f} steps/s vs '
                  f'nes_py {old_v:.0f} steps/s ({new_v / old_v:.1f}x)')


# ---------------------------------------------------------------------------
# 2. Correctness probe
# ---------------------------------------------------------------------------

def probe():
    print('\n=== 2. Correctness probe ===')
    from mario_env import (create_mario_env, RetroMarioEnv,
                                 MarioProgressWrapper, Wrapper)

    # --- 2a. full pipeline obs ---
    env = create_mario_env(name='SuperMarioBros-1-1-v0')
    obs = env.reset()
    check('obs shape (84,84,4)', obs.shape == (84, 84, 4), str(obs.shape))
    check('obs dtype float32', obs.dtype == np.float32, str(obs.dtype))
    check('obs range [0,1]', 0.0 <= obs.min() and obs.max() <= 1.0,
          f'[{obs.min():.3f}, {obs.max():.3f}]')
    check('action space Discrete(12)', env.action_space.n == 12)

    # --- 2b. RAM variables at a known state (Level1-1 start) ---
    base = env.unwrapped
    info0 = base._cur
    check('world/stage = 1-1',
          info0['world0'] == 0 and info0['stage0'] == 0)
    check('life = 2 at start', info0['life'] == 2)
    check('time = 400 at start', info0['time'] == 400)
    check('x_pos = 40 at start', base._x_position == 40, str(base._x_position))
    check('flag address sane (flag_get False at start, float_state/enemies readable)',
          (not base._flag_get) and 0 <= info0['float_state'] <= 255
          and all(0 <= info0[f'enemy_type{i}'] <= 255 for i in range(5)))

    # --- 2c. x_pos increases when running right ---
    x0 = base._x_position
    for _ in range(40):
        obs, r, done, info = env.step(3)  # right + B
        if done:
            break
    check('x_pos increases running right', info['x_pos'] > x0 + 100,
          f"{x0} -> {info['x_pos']}")
    check('info keys present', all(
        k in info for k in ('x_pos', 'flag_get', 'life', 'world', 'stage',
                            'game_progress', 'max_x_pos', 'time', 'y_pos')))

    # --- 2d. idle penalty: stand still ---
    env.reset()
    rewards = []
    for _ in range(20):
        obs, r, done, info = env.step(0)  # NOOP
        rewards.append(r)
    # after idle_threshold=10 base frames (2.5 outer steps), each outer step
    # accumulates 4 * -0.5 = -2.0 plus occasional -1 time penalty
    tail = rewards[5:]
    check('idle penalty kicks in when standing still',
          np.mean(tail) <= -1.9, f'mean tail reward {np.mean(tail):.2f}')

    # --- 2d2. synthetic flag_get: write the flagpole pattern into RAM
    # (enemy slot 0 = 0x31 flagpole, float_state = 3 sliding down the pole)
    # and verify the detection logic + RAM plumbing fires. Real level
    # completion is impractical with scripted actions; the actual flag
    # transition is exercised in training. ---
    base = env.unwrapped
    base._assign(0x0016, 0x31)
    base._assign(0x001D, 3)
    base._data.update_ram()
    base._cur = base._data.lookup_all()
    check('flag_get fires on synthetic flagpole RAM pattern', base._flag_get)
    env.close()

    # --- 2e. death decrements life (full game, base env: per-frame steps,
    # no reward shaping, so the death frame must be exactly -15 = clipped
    # x_reward + time_penalty + death_penalty(-25), like nes_py) ---
    env = RetroMarioEnv()  # full game, raw
    env.reset()
    min_r, died, done = 0.0, False, False
    for _ in range(1600):
        obs, r, done, info = env.step(3)  # run right into the first goomba
        min_r = min(min_r, r)
        if info['life'] == 1:
            died = True
            break
    check('death decrements life (3 lives: 2 -> 1)', died)
    check('death penalty frame == -15 (clipped -25)', min_r == -15.0,
          f'min frame reward {min_r:.1f}')
    check('post-death not done (full game continues)', not done)
    env.close()

    # --- 2f. EpisodicLife: death => done, reset continues same game ---
    env = create_mario_env(name='SuperMarioBros-v0', episode_life=True)
    env.reset()
    got_done = False
    for _ in range(400):
        obs, r, done, info = env.step(3)
        if done:
            got_done = True
            break
    obs = env.reset()  # should continue from life 1, not restart game
    check('episodic life: death -> done, reset continues with life 1',
          got_done and env.unwrapped._life == 1,
          f'life after reset = {env.unwrapped._life}')
    env.close()

    # --- 2g. reward shaping math on a stub env ---
    class StubEnv:
        observation_space = None
        action_space = None

        def __init__(self, xs, flags):
            self.xs, self.flags, self.i = xs, flags, 0
            self._life = 2

        @property
        def unwrapped(self):
            return self

        def reset(self, **kw):
            self.i = 0
            return None

        def step(self, action):
            x, f = self.xs[self.i], self.flags[self.i]
            self.i += 1
            return None, 0.0, False, {'x_pos': x, 'flag_get': f,
                                      'world': 1, 'stage': 1, 'life': 2}

    # progress reward: x 0 -> 1000 (delta capped at 20): 20 * 0.001 * 1000 = 20
    stub = MarioProgressWrapper(StubEnv([1000, 1005, 1005], [False, False, True]),
                                stage_bonus=500.0, idle_penalty=0.5,
                                idle_threshold=10, progress_reward=0.001)
    stub.reset()
    _, r1, _, _ = stub.step(0)
    check('progress_reward math: capped delta 20 * 0.001 * x 1000 = 20',
          math.isclose(r1, 20.0), f'{r1:.3f}')
    _, r2, _, _ = stub.step(0)
    check('progress_reward math: delta 5 * 0.001 * x 1005 = 5.025',
          math.isclose(r2, 5 * 0.001 * 1005), f'{r2:.3f}')
    _, r3, _, _ = stub.step(0)
    check('stage_bonus on flag_get transition (+500)',
          math.isclose(r3, 500.0 - 0.0), f'{r3:.3f}')

    # idle penalty: x constant; first 10 idle steps free, then -0.5 each
    stub = MarioProgressWrapper(StubEnv([100] * 15, [False] * 15),
                                stage_bonus=500.0, idle_penalty=0.5,
                                idle_threshold=10, progress_reward=0.001)
    stub.reset()
    rs = [stub.step(0)[1] for _ in range(15)]
    # rs[0]: first move 0->100, delta capped at 20 -> 20 * 0.001 * 100 = 2.0
    # rs[1..10]: idle steps 1-10 within threshold -> 0.0
    # rs[11..]: idle steps 11+ -> -0.5 each
    check('idle_penalty math: free for 10 steps, then -0.5/step',
          math.isclose(rs[0], 2.0)
          and all(math.isclose(r, 0.0) for r in rs[1:11])
          and all(math.isclose(r, -0.5) for r in rs[11:]),
          f'rs[0]={rs[0]}, rs[1:11]={rs[1:11]}, rs[11:]={rs[11:]}')


# ---------------------------------------------------------------------------
# 3. Random stage distribution
# ---------------------------------------------------------------------------

def stage_distribution():
    print('\n=== 3. Random-stage reset distribution (50 resets) ===')
    from mario_env import create_mario_env
    stages = ['1-1', '1-2', '1-3', '1-4', '2-1', '2-2', '2-3', '2-4']
    # episode_life=False so every reset() is a real reset (with episode_life,
    # EpisodicLife continues the same game between lives, as intended; stage
    # re-randomization happens on real done, i.e. on every death in
    # single-stage mode)
    env = create_mario_env(name='SuperMarioBros-v0', random_stages=stages,
                           episode_life=False)
    env.seed(123)
    counts = {}
    for _ in range(50):
        env.reset()
        _, _, _, info = env.step(0)
        key = f"{info['world']}-{info['stage']}"
        counts[key] = counts.get(key, 0) + 1
        check_ok = key in stages
        if not check_ok:
            record('reset landed on configured stage', False, key)
            env.close()
            return
    env.close()
    print(f'  distribution: {dict(sorted(counts.items()))}')
    check('all resets land on configured stages',
          set(counts) <= set(stages), str(sorted(counts)))
    check('>= 4 distinct stages over 50 resets', len(counts) >= 4,
          f'{len(counts)} distinct')


# ---------------------------------------------------------------------------
# 4. CPU smoke training
# ---------------------------------------------------------------------------

def smoke_train():
    print('\n=== 4. End-to-end CPU smoke training (10 epochs) ===')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    t0 = time.perf_counter()
    out = subprocess.run(
        [NEW_PYTHON, 'train.py',
         '--config', 'configs/mario_ppo_cpu_smoke.yaml',
         '--video-freq', '0'],
        capture_output=True, text=True, cwd=HERE, env=env, timeout=3600)
    dt = time.perf_counter() - t0
    tail = (out.stdout + out.stderr)[-3000:]
    ok = out.returncode == 0 and 'MAX EPOCHS NUM!' in out.stdout
    check('smoke training completed 10 epochs', ok,
          f'returncode={out.returncode}, {dt:.0f}s')
    if not ok:
        print(tail)
        return
    fps = re.findall(r'fps total:\s*([\d.]+)', out.stdout)
    if fps:
        print(f'  training fps total (last): {fps[-1]} (agent steps/s, '
              f'1 agent step = 4 emulated frames)')

    # Verify all logged TensorBoard scalars (rewards, losses) are finite
    import glob
    runs = sorted(glob.glob(os.path.join(HERE, 'runs', 'Mario_RetroCpuSmoke*')),
                  key=os.path.getmtime)
    if not runs:
        record('smoke run dir found', False)
        return
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator)
    acc = EventAccumulator(os.path.join(runs[-1], 'summaries'))
    acc.Reload()
    tags = acc.Tags().get('scalars', [])
    bad, n_vals = [], 0
    reward_tags = [t for t in tags if 'reward' in t.lower()]
    last_rewards = {}
    for tag in tags:
        vals = [e.value for e in acc.Scalars(tag)]
        n_vals += len(vals)
        if not np.all(np.isfinite(vals)):
            bad.append(tag)
        if tag in reward_tags and vals:
            last_rewards[tag] = round(vals[-1], 2)
    check('training scalars logged and finite',
          n_vals > 0 and not bad,
          f'{n_vals} values across {len(tags)} tags'
          + (f', non-finite: {bad}' if bad else ''))
    check('episode rewards were logged', len(last_rewards) > 0,
          str(last_rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench-steps', type=int, default=3000,
                        help='env steps for the retro benchmark')
    parser.add_argument('--bench-steps-old', type=int, default=600,
                        help='env steps for the (slow) nes_py benchmark')
    parser.add_argument('--vec-workers', type=int, default=4,
                        help='workers for the aggregate vecenv benchmark '
                             '(0 to disable)')
    parser.add_argument('--skip-bench-old', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    bench(args)
    probe()
    stage_distribution()
    if not args.skip_train:
        smoke_train()

    print('\n=== Summary ===')
    n_pass = sum(1 for _, p, _ in RESULTS if p)
    for name, passed, detail in RESULTS:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f'{n_pass}/{len(RESULTS)} checks passed')
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == '__main__':
    main()
