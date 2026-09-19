"""Checks for the retro (stable-retro) reference chain and its vecenv
(2026-09-18 review). CPU only, ~1 min.
Run: venv_retro/bin/python tests/retro_bench.py [--skip-vecenv]

Properties:
  * MarioVecEnv works under BOTH start methods (fork: Linux; spawn: the
    macOS default): the shared arrays are really shared (pickled numpy
    views were private copies under spawn), a host without
    sched_setaffinity runs unpinned, payload commands (stage weights, seeds)
    complete while the workers sleep, a dead worker raises instead of
    hanging the master, and a second close() is a no-op;
  * every life decrement ends an EpisodicLife episode (the byte goes
    2, 1, 0, 0xFF: 0 is the last playable life);
  * info['life'] of the death step is the post-death count;
  * a self-restart's first observation is the restored game's frame, not
    the previous episode's last one, and its first step pays no progress
    for standing where the state was saved;
  * sticky actions draw from a per-env stream, never the global np.random;
  * MaxAndSkip with skip=1 returns the current frame;
  * in single-stage mode a warp (level change without a flag) is terminal
    and paid per stage, as the MarioProgressWrapper docstring says.
Tools:
  * a play-tool csv replays its FINAL timeline (rewound branches dropped);
  * checkpoint steps read 'epoch' (rl_games) as well as 'iter' (GRPO);
  * clips keep the config's route when they start on one level, and a
    config that does not match the checkpoint fails to load;
  * archived eval traces (door episodes, and clips whose stuck lives were
    ended by a forced time-up) replay exactly with the run's config
    (--traces GLOB; skipped when nothing matches).

RETRO_BENCH_PATH=<dir> puts <dir> first on sys.path (and the vecenv
subprocesses'), to run the same checks against another copy of the modules.
"""
import argparse, os, subprocess, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA = os.environ.get('RETRO_BENCH_PATH')
sys.path.insert(0, ROOT)
if EXTRA:
    sys.path.insert(0, EXTRA)

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)), flush=True)


# ----------------------------------------------------------------- vecenv
VEC_CODE = r'''
import os, sys, time, signal, threading
sys.path[:0] = [p for p in {paths!r} if p]
import multiprocessing as mp
def watchdog(sec, what):
    def fire():
        print('HANG ' + what[0], flush=True)
        for p in mp.active_children():         # they hold our stdout open
            p.kill()
        os._exit(3)
    t = threading.Timer(sec, fire)
    t.daemon = True; t.start(); return t
if __name__ == '__main__':
    mp.set_start_method({method!r}, force=True)
    import numpy as np
    from mario_vecenv import MarioVecEnv
    what = ['construct + first reset']
    wd = watchdog(60, what)
    stages = ['1-1', '1-2']
    v = MarioVecEnv('bench', 4, name='SuperMarioBros-v0', random_stages=stages)
    obs = v.reset()
    print('RES reset', obs.shape == (4, 84, 84, 4) and float(obs.max()) > 0, obs.shape, flush=True)
    what[0] = 'steps'
    xs0 = None
    for t in range(30):
        obs, r, d, infos = v.step(np.full(4, 3))
        if xs0 is None:
            xs0 = [i['x_pos'] for i in infos]
    moved = sum(i['x_pos'] > x0 for i, x0 in zip(infos, xs0))
    print('RES steps', moved >= 3, 'moved %d/4' % moved, flush=True)
    ss = sorted({{i.get('start_stage') for i in infos}})
    print('RES start_stage', set(ss) <= set(stages), ss, flush=True)
    ok = True
    for k in range(4):
        time.sleep(0.3)                          # every worker falls asleep
        what[0] = 'set_stage_weights after idle #%d' % k
        t0 = time.time(); v.set_stage_weights({{'1-1': 1.0, '1-2': float(k % 2)}})
        what[0] = 'set_seeds after idle #%d' % k
        time.sleep(0.3); v.set_seeds([k, k + 1, k + 2, k + 3])
        what[0] = 'step after payload #%d' % k
        v.step(np.zeros(4, dtype=np.int64))
    print('RES payload_after_idle', True, flush=True)
    what[0] = 'dead worker'
    os.kill(v.processes[1].pid, signal.SIGKILL); v.processes[1].join(5)
    t0 = time.time()
    try:
        v.step(np.zeros(4, dtype=np.int64)); raised = None
    except Exception as e:
        raised = type(e).__name__
    print('RES dead_worker_raises', raised is not None and time.time() - t0 < 10,
          'raised %s after %.1fs' % (raised, time.time() - t0), flush=True)
    what[0] = 'close'
    v.close(); t0 = time.time(); v.close()
    print('RES second_close_noop', time.time() - t0 < 1.0, '%.2fs' % (time.time() - t0), flush=True)
    print('RES workers_gone', all(not p.is_alive() for p in v.processes), flush=True)
    wd.cancel()
'''


def vecenv_checks():
    for method in ('spawn', 'fork'):
        code = VEC_CODE.format(paths=[EXTRA, ROOT], method=method)
        try:
            out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                 cwd=ROOT, timeout=180)
        except subprocess.TimeoutExpired as e:
            check('vecenv[%s] completed' % method, False, 'no exit within 180 s: '
                  + (e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout))[-300:])
            continue
        lines = out.stdout.splitlines()
        res = [l.split(' ', 3) for l in lines if l.startswith('RES ')]
        hang = [l for l in lines if l.startswith('HANG ')]
        for r in res:
            check('vecenv[%s] %s' % (method, r[1]), r[2] == 'True', ' '.join(r[3:]))
        if hang or out.returncode != 0 or len(res) < 7:
            err = [l for l in out.stderr.splitlines() if 'Error' in l][-2:]
            check('vecenv[%s] completed' % method, False,
                  (hang[0] if hang else 'exit %d' % out.returncode) + ' ' + ' | '.join(err))


# ----------------------------------------------------------------- env chain
def episodic_life_boundaries():
    from mario_env import create_mario_env
    env = create_mario_env(name='SuperMarioBros-v0', episode_life=True)
    env.reset()
    lives = []
    for _ in range(4000):
        _, _, done, info = env.step(3)            # run into the first goomba
        if done:
            lives.append(int(info['life']))
            if env.unwrapped.life == 0xFF:
                break
            env.reset()
    env.close()
    check('EpisodicLife: one episode per life, lives at the ends [1, 0, 255]',
          lives == [1, 0, 255], lives)


def life_on_death_step():
    from mario_env import RetroMarioEnv
    env = RetroMarioEnv()
    env.reset()
    for _ in range(1600):
        _, r, done, info = env.step(3)
        if r <= -15:                               # the death frame
            break
    check("info['life'] on the death step is the post-death count (2 -> 1)",
          info['life'] == 1 and env.life == 1, (info['life'], env.life))
    env.close()


def self_restart():
    from mario_env import RetroMarioEnv, MarioProgressWrapper
    base = RetroMarioEnv(target=(1, 1), self_restart_prob=1.0)
    shots = {}
    orig = base._maybe_archive

    def archive_and_shoot():                       # screen at archive time
        n = len(base._archive); orig()
        if len(base._archive) > n:
            cell = [c for c in base._archive if c not in shots][0]
            shots[cell] = base._em.get_screen().astype(np.int16)
    base._maybe_archive = archive_and_shoot
    env = MarioProgressWrapper(base)
    env.reset()
    for _ in range(1600):
        _, _, done, _ = env.step(3)
        if done:                                   # died at the first goomba
            break
    last = base._em.get_screen().astype(np.int16)
    near = min(shots, key=lambda c: c[2])          # the start: other camera than the death
    far = max(shots, key=lambda c: c[2])
    entries = dict(base._archive)
    base._archive = {near: entries[near]}
    obs = env.reset().astype(np.int16)
    d_saved = np.abs(obs - shots[near]).mean(); d_last = np.abs(obs - last).mean()
    check('self-restart: first obs is the restored frame (mean |d| vs saved %.1f, vs the previous '
          'episode\'s last %.1f)' % (d_saved, d_last),
          base.last_reset_was_restart and d_saved < 2.0 and d_saved < d_last / 4, (d_saved, d_last))
    base._archive = {far: entries[far]}
    env.reset()
    x0 = base._x_position
    _, r, _, info = env.step(0)
    # the saved state is mid-run: momentum carries Mario dx px on this frame,
    # which pays dx (gym x-delta) + dx * 0.001 * x; anything above that is the
    # spike of a tracker that started at x 0 (min(x, 20) * 0.001 * x)
    dx = info['x_pos'] - x0
    excess = r - (dx + max(dx, 0) * 0.001 * info['x_pos'])
    check('self-restart at x %d: the first NOOP frame pays only its momentum (dx %d, excess %.2f)'
          % (x0, dx, excess), base.last_reset_was_restart and -1.01 <= excess <= 1e-6, (r, dx, excess))
    base.close()


def sticky_rng():
    from mario_env import StickyActionWrapper

    class Stub:
        def __init__(self): self.seen = []
        unwrapped = property(lambda self: self)
        def seed(self, seed=None): return [seed]
        def reset(self, **kw): return None
        def step(self, a): self.seen.append(a); return None, 0.0, False, {}

    def run(seed):
        s = Stub(); w = StickyActionWrapper(s, p=0.5); w.seed(seed)
        for t in range(200):
            w.step(t % 12)
        return s.seen
    np.random.seed(123); before = np.random.get_state()[1].copy()
    a, b, c = run(1), run(1), run(2)
    after = np.random.get_state()[1]
    check('sticky actions leave the global np.random untouched', np.array_equal(before, after))
    check('sticky actions: same seed -> same repeats, other seed -> other repeats',
          a == b and a != c)


def skip1():
    from gymnasium import spaces
    from mario_env import MaxAndSkipEnv

    class Base:
        observation_space = spaces.Box(0, 255, (2, 2, 3), np.uint8)
        want_obs = True
        def __init__(self, seq): self.seq = seq; self.i = 0
        unwrapped = property(lambda self: self)
        def reset(self, **kw): return np.zeros((2, 2, 3), np.uint8)
        def step(self, a):
            v, d = self.seq[self.i]; self.i += 1
            return np.full((2, 2, 3), v, np.uint8), 0.0, d, {}
    env = MaxAndSkipEnv(Base([(200, True), (10, False), (20, False)]), skip=1)
    o1 = env.step(0)[0]; env.reset(); o2 = env.step(0)[0]; o3 = env.step(0)[0]
    check('MaxAndSkip skip=1 returns the current frame (not max with a stale one)',
          o1.max() == 200 and o2.max() == 10 and o3.max() == 20,
          (int(o1.max()), int(o2.max()), int(o3.max())))


def single_stage_warp():
    from mario_env import create_mario_env
    env = create_mario_env(name='SuperMarioBros-v0', random_stages=['1-2'], episode_life=True,
                           stage_bonus=500.0)
    env.reset()
    base = env.unwrapped
    for _ in range(5):
        env.step(0)
    base._assign(0x075F, 3); base._assign(0x075C, 0)   # the warp zone's 4-1
    total, done, info, steps = 0.0, False, {}, 0
    for steps in range(1, 11):
        _, r, done, info = env.step(0); total += r
        if done:
            break
    check('single-stage warp 1-2 -> 4-1 is terminal and pays 11 x stage_bonus '
          '(done after %d steps, reward %.0f)' % (steps, total),
          done and steps <= 2 and total >= 5500 - 50 and info.get('warped')
          and info.get('stages_cleared') == 1, (done, steps, total, info.get('warped')))
    env.close()


# ----------------------------------------------------------------- tools
def tools_checks():
    import tempfile, yaml
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    from play import final_branch            # tools/play.py (pygame loads lazily)
    rows = [(1, 'R', ''), (2, 'R', ''), (3, 'A', ''), (4, 'L', ''), (5, 'L', ''),
            (3, 'A', 'rewound to here'), (4, 'R+B', ''), (5, 'R+B', ''), (6, 'NOOP', ''),
            (6, 'NOOP', 'rewound to here')]
    f = tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False)
    f.write('step,action,note\n' + ''.join('%d,%s,%s\n' % r for r in rows)); f.close()
    acts = final_branch(f.name); os.remove(f.name)
    check('play-tool csv replay = the final timeline (R R A R+B R+B NOOP)', acts == [1, 1, 5, 3, 3, 0], acts)

    import torch
    from render_ckpt import build, ckpt_step, clip_env_config
    check("checkpoint step: rl_games 'epoch', GRPO 'iter'",
          ckpt_step({'epoch': 3500}) == 3500 and ckpt_step({'iter': 75}) == 75)
    route_cfg = os.path.join(ROOT, 'configs', 'mario_ppo_native_routeDet.yaml')
    rc = yaml.safe_load(open(route_cfg))['params']['config']
    ec = clip_env_config(rc, '1-2')
    check('clip from one level keeps the route (%s)' % ec.get('route_levels'),
          ec['random_stages'] == ['1-2'] and ec['route_levels'] == list(
              rc['env_config'].get('route_levels') or rc['env_config']['random_stages']))
    params = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'mario_ppo_native_42.yaml')))['params']
    from rl_games.algos_torch import model_builder
    net = model_builder.ModelBuilder().load(params).build(
        {'actions_num': 12, 'input_shape': (84, 84, 4), 'num_seqs': 1, 'value_size': 1,
         'normalize_value': True, 'normalize_input': False})
    d = tempfile.mkdtemp()
    ck = os.path.join(d, 'ck.pth'); torch.save({'model': net.state_dict(), 'epoch': 7}, ck)
    params['config']['normalize_input'] = True           # a config that does not match
    bad = os.path.join(d, 'bad.yaml'); yaml.safe_dump({'params': params}, open(bad, 'w'))
    try:
        build(bad, ck); loaded = True
    except RuntimeError:
        loaded = False
    check('a checkpoint does not load silently under a mismatching config', not loaded)


def trace_replays(pattern):
    import glob, yaml
    paths = sorted(glob.glob(pattern))
    if not paths:
        print('skip trace replays: nothing matches %s' % pattern); return
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import ghosts
    ec = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'mario_ppo_native_42.yaml')))['params']['config']['env_config']
    traces = [ghosts.load_trace(p) for p in paths]
    bad = ghosts.check_traces(traces, ec, None)
    check('%d archived traces replay exactly with the run config (x in play, lives)' % len(traces), bad == 0,
          '%d diverged' % bad)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--skip-vecenv', action='store_true')
    ap.add_argument('--traces', default=os.path.join(
        ROOT, 'runs_archive', 'Mario_PPO42i_*', 'eval_traces', 'epoch_1000', '*.npz'),
        help='archived eval traces of a run trained with configs/mario_ppo_native_42.yaml')
    a = ap.parse_args()
    if not a.skip_vecenv:
        vecenv_checks()
    for fn in (episodic_life_boundaries, life_on_death_step, self_restart, sticky_rng, skip1,
               single_stage_warp, tools_checks, lambda: trace_replays(a.traces)):
        try:
            fn()
        except Exception as e:
            check(getattr(fn, '__name__', 'check') + ' ran', False, '%s: %s' % (type(e).__name__, e))
    print('%d/%d checks passed' % (sum(OK), len(OK)))
    sys.exit(0 if all(OK) else 1)


if __name__ == '__main__':
    main()
