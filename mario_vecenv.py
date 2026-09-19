"""Multiprocessing vectorized environment for Mario.

One env per worker process. Two reasons this exists instead of rl_games'
RayVecEnv:
- Ray workers cannot see custom env registrations made in the main process.
- stable-retro allows only ONE emulator per process, so one-env-per-process
  is required, not just faster.

Transport is shared memory, not pipes: on WSL2 a pipe roundtrip costs ~85us
in syscall/wakeup latency, which at 48-64 workers dominated the whole
vec-step. The master writes one control word per worker (generation,
command and action packed into a single aligned 8-byte store, so no worker
can see a new generation with a stale command or action); workers spin on
theirs (each is pinned to its own CPU where the OS allows it), step, and
write frame/reward/done/info-scalars back into shared arrays, then their
done word behind a memory fence. Workers fall back to blocking on their
pipe after a spin budget, so they sleep through the learner's train phase;
the sleep is a timed poll, so a lost wake-up costs latency, never a hang.
Pipes also carry the rare payload commands (seed, stage weights) and the
startup handshake.

Workers run the env chain up to WarpFrame (frame_only=True) and share raw
(84, 84) uint8 frames; scaling to [0, 1] and 4-frame stacking happen
master-side. Each worker auto-resets its env when an episode ends, returning
the first frame of the next episode (the master refills that env's stack).
"""

import ctypes
import os
import platform
import sys
import time

import numpy as np
from gymnasium import spaces
from multiprocessing import Pipe, Process
from multiprocessing.sharedctypes import RawArray

from rl_games.common import vecenv
from rl_games.common.ivecenv import IVecEnv

FRAME_STACK = 4
FRAME_SHAPE = (84, 84)

# Broadcast commands, packed with the generation into each worker's control
# word: ctl = gen << 8 | cmd << 4 | action
CMD_STEP = 0
CMD_RESET = 1
CMD_SEED = 2      # payload per worker via pipe
CMD_WEIGHTS = 3   # payload broadcast via pipe
CMD_CLOSE = 4

# Fixed schema for info scalars shared per step (master rebuilds dicts)
INFO_KEYS = ['x_pos', 'max_x_pos', 'game_progress', 'progress_gain',
             'stages_cleared', 'warped', 'victory', 'looped', 'flag_get',
             'life', 'world', 'stage', 'time', 'score', 'coins']
N_INFO = len(INFO_KEYS) + 1  # + start stage as a level index (-1 = none)

# Spin ~8ms before sleeping: covers the master's inter-step work during
# rollout; the 0.5s+ train phase sends workers to a blocking pipe recv.
SPIN_ITERS = 4000
# A sleeping worker re-checks its control word this often even without a
# wake-up message: the sleeping-flag / control-word handshake is a
# store-then-load pattern on both sides that the CPU may reorder (x86 too),
# so a wake-up can be lost; it then costs this much latency, not a hang.
SLEEP_POLL_S = 0.05
# The master checks that every worker is still alive this often while it
# waits for a broadcast (a dead worker used to make it spin forever).
LIVENESS_S = 0.5


def _make_fence():
    """Full memory barrier for the shared-memory hand-offs. x86 is TSO
    (stores are seen in program order, loads are not reordered with older
    loads), so the result publication needs none there. ARM64 is weakly
    ordered: a worker's frame stores may become visible after its done word,
    and the master's frame loads may be satisfied before its done-word load;
    libSystem's OSMemoryBarrier is a full `dmb ish` on Apple silicon. Other
    weakly ordered hosts have no portable fence reachable from Python and
    are not supported."""
    if platform.machine().lower() in ('x86_64', 'amd64', 'i386', 'i686'):
        return lambda: None
    if sys.platform == 'darwin':
        fn = ctypes.CDLL('/usr/lib/libSystem.B.dylib').OSMemoryBarrier
        fn.restype = None
        fn.argtypes = []
        return fn
    return lambda: None


_fence = _make_fence()


def _stage_index(name):
    """'4-1' -> 12 (world-major level index); -1 if not a 'W-S' name. Any
    stage is representable, so a start stage outside random_stages (a
    continuation life after a warp, a restart state) keeps its own name
    instead of being filed under the list's first stage."""
    try:
        w, s = str(name).split('-')
        return (int(w) - 1) * 4 + (int(s) - 1)
    except (ValueError, AttributeError):
        return -1


class _Shared:
    """Numpy views over the RawArrays shared between master and workers.

    Pickling (spawn / forkserver start methods: macOS and Windows default to
    spawn) carries the RawArray objects, which multiprocessing passes to the
    child as the same shared memory, and the views are rebuilt on the other
    side. Pickling the numpy views themselves copied them BY VALUE: every
    worker wrote into a private copy and the first broadcast never
    completed."""

    def __init__(self, num_actors):
        n = num_actors
        h, w = FRAME_SHAPE
        self._n = n
        self._raw = dict(
            ctl=RawArray('q', n), done_gen=RawArray('q', n),
            sleeping=RawArray('q', n), frames=RawArray('B', n * h * w),
            rewards=RawArray('f', n), dones=RawArray('B', n),
            infos=RawArray('f', n * N_INFO))
        self._make_views()

    def _make_views(self):
        n, (h, w), r = self._n, FRAME_SHAPE, self._raw
        self.ctl = np.frombuffer(r['ctl'], dtype=np.int64)
        self.done_gen = np.frombuffer(r['done_gen'], dtype=np.int64)
        self.sleeping = np.frombuffer(r['sleeping'], dtype=np.int64)
        self.frames = np.frombuffer(r['frames'], dtype=np.uint8).reshape(n, h, w)
        self.rewards = np.frombuffer(r['rewards'], dtype=np.float32)
        self.dones = np.frombuffer(r['dones'], dtype=np.uint8)
        self.infos = np.frombuffer(r['infos'], dtype=np.float32).reshape(n, N_INFO)

    def __getstate__(self):
        return {'n': self._n, 'raw': self._raw}

    def __setstate__(self, state):
        self._n, self._raw = state['n'], state['raw']
        self._make_views()


def _worker(remote, parent_remote, env_kwargs, worker_idx, shm):
    """Run a single environment, serving broadcast commands from shm."""
    parent_remote.close()
    seed = env_kwargs.pop('seed', None)
    # Pin to one CPU: the host has 32 physical cores / 64 SMT threads and
    # WSL2 migration thrash otherwise eats much of the parallel speedup.
    # MARIO_CPU_BASE offsets the pin range so two runs can share the box.
    # (Linux only: macOS has no sched_setaffinity and no pinning API.)
    base = int(os.environ.get('MARIO_CPU_BASE', '0'))
    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, {(base + worker_idx) % os.cpu_count()})
        except OSError:
            pass

    from mario_env import create_mario_env
    env = create_mario_env(**env_kwargs)
    if seed is not None:
        env.seed(int(seed) + worker_idx)  # decorrelate workers

    i = worker_idx
    my_gen = 0
    pending_payload = None
    remote.send((env.observation_space, env.action_space))  # ready handshake

    def write_result(obs, reward, done, info):
        shm.frames[i] = obs[..., 0]
        shm.rewards[i] = reward
        shm.dones[i] = done
        row = shm.infos[i]
        for k, key in enumerate(INFO_KEYS):
            row[k] = float(info.get(key, 0))
        row[N_INFO - 1] = _stage_index(info.get('start_stage'))

    try:
        while True:
            # Wait for the next generation: spin, then sleep on the pipe.
            spins = 0
            while True:
                word = int(shm.ctl[i])       # one atomic 8-byte load
                if word >> 8 != my_gen:
                    break
                spins += 1
                if spins <= SPIN_ITERS:
                    os.sched_yield()
                    continue
                shm.sleeping[i] = 1
                _fence()                     # flag visible before the re-check
                if int(shm.ctl[i]) >> 8 != my_gen:   # avoid a lost wakeup
                    shm.sleeping[i] = 0
                    continue
                # 'wake' or ('payload', data); timed, see SLEEP_POLL_S
                if remote.poll(SLEEP_POLL_S):
                    msg = remote.recv()
                    if isinstance(msg, tuple) and msg[0] == 'payload':
                        pending_payload = msg[1]
                shm.sleeping[i] = 0
            my_gen = word >> 8
            cmd = (word >> 4) & 0xF

            if cmd == CMD_STEP:
                obs, reward, done, info = env.step(word & 0xF)
                if done:
                    obs = env.reset()
                write_result(obs, reward, done, info)
            elif cmd == CMD_RESET:
                shm.frames[i] = env.reset()[..., 0]
            elif cmd in (CMD_SEED, CMD_WEIGHTS):
                # the master sends the payload right after the generation
                # bump (it doubles as the wake-up of a sleeping worker);
                # drain stale 'wake' messages until it arrives
                while pending_payload is None:
                    msg = remote.recv()
                    if isinstance(msg, tuple) and msg[0] == 'payload':
                        pending_payload = msg[1]
                if cmd == CMD_SEED:
                    env.seed(pending_payload)
                else:
                    env.unwrapped.set_stage_weights(pending_payload)
                pending_payload = None
            elif cmd == CMD_CLOSE:
                env.close()
                _fence()
                shm.done_gen[i] = my_gen
                break
            _fence()                         # results visible before done
            shm.done_gen[i] = my_gen
    except (EOFError, KeyboardInterrupt):
        pass


class MarioVecEnv(IVecEnv):
    """rl_games IVecEnv over `num_actors` pinned worker processes."""

    def __init__(self, config_name, num_actors, **env_kwargs):
        self.num_actors = num_actors
        env_kwargs = dict(env_kwargs, frame_only=True)
        self._stages = list(env_kwargs.get('random_stages') or [])
        self._shm = _Shared(num_actors)

        self.remotes, work_remotes = zip(*[Pipe() for _ in range(num_actors)])
        self.processes = []
        for idx, (work_remote, remote) in enumerate(
                zip(work_remotes, self.remotes)):
            p = Process(target=_worker,
                        args=(work_remote, remote, env_kwargs, idx,
                              self._shm),
                        daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()

        self._gen = 0
        self._closed = False
        spaces_ = []
        for idx, r in enumerate(self.remotes):     # ready handshake
            try:
                spaces_.append(r.recv())
            except EOFError:
                self.processes[idx].join(timeout=5)
                self._terminate()
                raise RuntimeError(
                    'MarioVecEnv worker %d died during startup (exit code %s); '
                    'its traceback is above' % (idx, self.processes[idx].exitcode))
        frame_space, self.action_space = spaces_[0]
        assert self.action_space.n <= 16, 'actions are packed into 4 bits'
        h, w = frame_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(h, w, FRAME_STACK), dtype=np.float32)

        # Frame stacks as a ring buffer: _ring[..., ptr] is the newest frame;
        # the ordered stack is materialized once per step into a fresh array.
        self._ring = np.zeros((num_actors, h, w, FRAME_STACK),
                              dtype=np.float32)
        self._ptr = 0

    # -- broadcast machinery --------------------------------------------------

    def _broadcast(self, cmd, payloads=None, actions=None):
        """Publish (gen, cmd, action) to every worker, wake sleepers, wait
        for all workers.

        Payload commands send the payload AFTER the generation bump: a
        sleeping worker is woken by the payload itself, a spinning one sees
        the new generation and blocks on its pipe until the payload is
        there. (Sent before the bump, the payload woke a sleeper that went
        straight back to recv() on the old generation, and no 'wake' ever
        followed: set_stage_weights hung with idle workers.)"""
        shm = self._shm
        gen = self._gen + 1
        self._gen = gen
        word = (gen << 8) | (int(cmd) << 4)
        if actions is not None:
            shm.ctl[:] = word | np.asarray(actions, dtype=np.int64).ravel()
        else:
            shm.ctl[:] = word
        _fence()                    # control words visible before the flag read
        if payloads is not None:
            for remote, payload in zip(self.remotes, payloads):
                remote.send(('payload', payload))
        else:
            # Snapshot before nonzero: workers flip their sleeping flag
            # concurrently and np.nonzero's two-pass scan crashes if the
            # count changes mid-call. A stale wake is harmless (workers
            # re-check their word before sleeping; extra wakes are drained).
            for i in np.nonzero(shm.sleeping.copy())[0]:
                self.remotes[i].send('wake')
        next_check = time.monotonic() + LIVENESS_S
        while not (shm.done_gen == gen).all():
            os.sched_yield()
            if time.monotonic() >= next_check:
                next_check = time.monotonic() + LIVENESS_S
                dead = [k for k, p in enumerate(self.processes)
                        if not p.is_alive() and shm.done_gen[k] != gen]
                if dead:
                    raise RuntimeError(
                        'MarioVecEnv worker(s) %s died (exit codes %s)'
                        % (dead, [self.processes[k].exitcode for k in dead]))
        _fence()                    # results read after every done word

    def _push_frames(self, dones=None):
        """Append the shared uint8 frames to the per-env ring."""
        f = self._shm.frames.astype(np.float32)
        f /= 255.0
        self._ptr = (self._ptr + 1) % FRAME_STACK
        self._ring[..., self._ptr] = f
        if dones is None:
            self._ring[:] = f[..., None]
        elif dones.any():
            self._ring[dones] = f[dones][..., None]
        # Materialize oldest -> newest into a fresh array (callers keep it)
        order = [(self._ptr + 1 + j) % FRAME_STACK for j in range(FRAME_STACK)]
        return self._ring[..., order]

    def _build_infos(self):
        out = []
        for row in self._shm.infos:
            info = {key: row[k] for k, key in enumerate(INFO_KEYS)}
            info['flag_get'] = bool(info['flag_get'])
            info['warped'] = bool(info['warped'])
            info['victory'] = bool(info['victory'])
            idx = int(row[N_INFO - 1])
            if self._stages and idx >= 0:
                info['start_stage'] = '%d-%d' % (idx // 4 + 1, idx % 4 + 1)
            out.append(info)
        return out

    # -- IVecEnv API ----------------------------------------------------------

    def step(self, actions):
        shm = self._shm
        self._broadcast(CMD_STEP, actions=actions)
        dones = shm.dones.astype(bool)
        return (
            self._push_frames(dones),
            shm.rewards.copy(),
            dones,
            self._build_infos(),
        )

    def reset(self):
        self._broadcast(CMD_RESET)
        return self._push_frames()

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'agents': 1,
            'value_size': 1,
        }

    def set_seeds(self, seeds):
        self._broadcast(CMD_SEED, payloads=list(seeds))

    def set_stage_weights(self, weights):
        """Broadcast random-stage sampling weights (dict stage -> weight)."""
        self._broadcast(CMD_WEIGHTS, payloads=[weights] * self.num_actors)

    def has_action_masks(self):
        return False

    def close(self):
        if self._closed:            # a second close() is a no-op
            return
        self._closed = True
        try:
            self._broadcast(CMD_CLOSE)
        except (BrokenPipeError, OSError, RuntimeError):
            pass                    # a dead worker: terminate the rest
        for p in self.processes:
            p.join(timeout=5)
        self._terminate()

    def _terminate(self):
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


def register_mario_vecenv():
    """Register the MARIO vecenv type with rl_games."""
    vecenv.register(
        'MARIO',
        lambda config_name, num_actors, **kwargs: MarioVecEnv(
            config_name, num_actors, **kwargs),
    )
