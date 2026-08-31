"""Multiprocessing vectorized environment for Mario.

One env per worker process. Two reasons this exists instead of rl_games'
RayVecEnv:
- Ray workers cannot see custom env registrations made in the main process.
- stable-retro allows only ONE emulator per process, so one-env-per-process
  is required, not just faster.

Transport is shared memory, not pipes: on WSL2 a pipe roundtrip costs ~85us
in syscall/wakeup latency, which at 48-64 workers dominated the whole
vec-step. The master writes actions into a shared array and bumps a
generation counter; workers spin on it (each is pinned to its own CPU),
step, and write frame/reward/done/info-scalars back into shared arrays.
Workers fall back to blocking on their pipe after a spin budget, so they
sleep through the learner's train phase; pipes also carry rare commands
(seed, stage weights) and the startup handshake.

Workers run the env chain up to WarpFrame (frame_only=True) and share raw
(84, 84) uint8 frames; scaling to [0, 1] and 4-frame stacking happen
master-side. Each worker auto-resets its env when an episode ends, returning
the first frame of the next episode (the master refills that env's stack).
"""

import os
import time

import numpy as np
from gymnasium import spaces
from multiprocessing import Pipe, Process
from multiprocessing.sharedctypes import RawArray

from rl_games.common import vecenv
from rl_games.common.ivecenv import IVecEnv

FRAME_STACK = 4
FRAME_SHAPE = (84, 84)

# Broadcast commands (master bumps `gen` after setting `cmd`)
CMD_STEP = 0
CMD_RESET = 1
CMD_SEED = 2      # payload per worker via pipe
CMD_WEIGHTS = 3   # payload broadcast via pipe
CMD_CLOSE = 4

# Fixed schema for info scalars shared per step (master rebuilds dicts)
INFO_KEYS = ['x_pos', 'max_x_pos', 'game_progress', 'progress_gain',
             'stages_cleared', 'warped', 'victory', 'flag_get', 'life',
             'world', 'stage', 'time', 'score', 'coins']
N_INFO = len(INFO_KEYS) + 1  # + start_stage index

# Spin ~8ms before sleeping: covers the master's inter-step work during
# rollout; the 0.5s+ train phase sends workers to a blocking pipe recv.
SPIN_ITERS = 4000


class _Shared:
    """Numpy views over the RawArrays shared between master and workers."""

    def __init__(self, num_actors):
        n = num_actors
        h, w = FRAME_SHAPE
        self.gen = np.frombuffer(RawArray('q', 1), dtype=np.int64)
        self.cmd = np.frombuffer(RawArray('q', 1), dtype=np.int64)
        self.done_gen = np.frombuffer(RawArray('q', n), dtype=np.int64)
        self.sleeping = np.frombuffer(RawArray('q', n), dtype=np.int64)
        self.actions = np.frombuffer(RawArray('i', n), dtype=np.int32)
        self.frames = np.frombuffer(
            RawArray('B', n * h * w), dtype=np.uint8).reshape(n, h, w)
        self.rewards = np.frombuffer(RawArray('f', n), dtype=np.float32)
        self.dones = np.frombuffer(RawArray('B', n), dtype=np.uint8)
        self.infos = np.frombuffer(
            RawArray('f', n * N_INFO), dtype=np.float32).reshape(n, N_INFO)


def _worker(remote, parent_remote, env_kwargs, worker_idx, shm):
    """Run a single environment, serving broadcast commands from shm."""
    parent_remote.close()
    seed = env_kwargs.pop('seed', None)
    # Pin to one CPU: the host has 32 physical cores / 64 SMT threads and
    # WSL2 migration thrash otherwise eats much of the parallel speedup.
    # MARIO_CPU_BASE offsets the pin range so two runs can share the box.
    base = int(os.environ.get('MARIO_CPU_BASE', '0'))
    try:
        os.sched_setaffinity(0, {(base + worker_idx) % os.cpu_count()})
    except OSError:
        pass

    from mario_env import create_mario_env
    env = create_mario_env(**env_kwargs)
    if seed is not None:
        env.seed(int(seed) + worker_idx)  # decorrelate workers

    stages = env_kwargs.get('random_stages') or []
    stage_idx = {s: i for i, s in enumerate(stages)}
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
        row[N_INFO - 1] = stage_idx.get(info.get('start_stage'), 0)

    try:
        while True:
            # Wait for the next generation: spin, then block on the pipe.
            spins = 0
            while shm.gen[0] == my_gen:
                spins += 1
                if spins <= SPIN_ITERS:
                    os.sched_yield()
                    continue
                shm.sleeping[i] = 1
                if shm.gen[0] != my_gen:     # re-check to avoid lost wakeup
                    shm.sleeping[i] = 0
                    break
                msg = remote.recv()          # 'wake' or ('payload', data)
                shm.sleeping[i] = 0
                if isinstance(msg, tuple) and msg[0] == 'payload':
                    pending_payload = msg[1]
            my_gen = int(shm.gen[0])
            cmd = int(shm.cmd[0])

            if cmd == CMD_STEP:
                obs, reward, done, info = env.step(int(shm.actions[i]))
                if done:
                    obs = env.reset()
                write_result(obs, reward, done, info)
            elif cmd == CMD_RESET:
                shm.frames[i] = env.reset()[..., 0]
            elif cmd in (CMD_SEED, CMD_WEIGHTS):
                if pending_payload is None:
                    msg = remote.recv()
                    pending_payload = msg[1]
                if cmd == CMD_SEED:
                    env.seed(pending_payload)
                else:
                    env.unwrapped.set_stage_weights(pending_payload)
                pending_payload = None
            elif cmd == CMD_CLOSE:
                env.close()
                shm.done_gen[i] = my_gen
                break
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

        spaces_ = [r.recv() for r in self.remotes]  # ready handshake
        frame_space, self.action_space = spaces_[0]
        h, w = frame_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(h, w, FRAME_STACK), dtype=np.float32)

        # Frame stacks as a ring buffer: _ring[..., ptr] is the newest frame;
        # the ordered stack is materialized once per step into a fresh array.
        self._ring = np.zeros((num_actors, h, w, FRAME_STACK),
                              dtype=np.float32)
        self._ptr = 0

    # -- broadcast machinery --------------------------------------------------

    def _broadcast(self, cmd, payloads=None):
        """Set cmd, bump gen, wake sleepers, wait for all workers."""
        shm = self._shm
        if payloads is not None:
            for remote, payload in zip(self.remotes, payloads):
                remote.send(('payload', payload))
        shm.cmd[0] = cmd
        gen = int(shm.gen[0]) + 1
        shm.gen[0] = gen
        if payloads is None:
            # Snapshot before nonzero: workers flip their sleeping flag
            # concurrently and np.nonzero's two-pass scan crashes if the
            # count changes mid-call. A stale wake is harmless (workers
            # re-check gen before blocking; extra wakes are drained).
            for i in np.nonzero(shm.sleeping.copy())[0]:
                self.remotes[i].send('wake')
        while not (shm.done_gen == gen).all():
            os.sched_yield()

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
            if self._stages:
                info['start_stage'] = self._stages[int(row[N_INFO - 1])]
            out.append(info)
        return out

    # -- IVecEnv API ----------------------------------------------------------

    def step(self, actions):
        shm = self._shm
        shm.actions[:] = np.asarray(actions, dtype=np.int32)
        self._broadcast(CMD_STEP)
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
        try:
            self._broadcast(CMD_CLOSE)
        except (BrokenPipeError, OSError):
            pass
        for p in self.processes:
            p.join(timeout=5)


def register_mario_vecenv():
    """Register the MARIO vecenv type with rl_games."""
    vecenv.register(
        'MARIO',
        lambda config_name, num_actors, **kwargs: MarioVecEnv(
            config_name, num_actors, **kwargs),
    )
