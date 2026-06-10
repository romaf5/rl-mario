"""Multiprocessing vectorized environment for Mario.

One env per worker process, plain pipes for IPC. Two reasons this exists
instead of rl_games' RayVecEnv:
- Ray workers cannot see custom env registrations made in the main process.
- stable-retro allows only ONE emulator per process, so one-env-per-process
  is required, not just faster.

Each worker imports create_mario_env locally and auto-resets its env when an
episode ends, returning the first observation of the next episode.
"""

import numpy as np
from multiprocessing import Pipe, Process

from rl_games.common import vecenv
from rl_games.common.ivecenv import IVecEnv


def _worker(remote, parent_remote, env_kwargs):
    """Run a single environment, serving commands from the master process."""
    parent_remote.close()
    from mario_env import create_mario_env
    env = create_mario_env(**env_kwargs)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'seed':
                env.seed(data)
                remote.send(None)
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
    except EOFError:
        pass


class MarioVecEnv(IVecEnv):
    """rl_games IVecEnv over `num_actors` worker processes in lockstep."""

    def __init__(self, config_name, num_actors, **env_kwargs):
        self.num_actors = num_actors

        self.remotes, work_remotes = zip(*[Pipe() for _ in range(num_actors)])
        self.processes = []
        for work_remote, remote in zip(work_remotes, self.remotes):
            p = Process(target=_worker, args=(work_remote, remote, env_kwargs),
                        daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        obs, rewards, dones, infos = zip(*[r.recv() for r in self.remotes])
        return (
            np.array(obs, dtype=self.observation_space.dtype),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.array([r.recv() for r in self.remotes],
                        dtype=self.observation_space.dtype)

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
        for remote, seed in zip(self.remotes, seeds):
            remote.send(('seed', seed))
        for remote in self.remotes:
            remote.recv()

    def has_action_masks(self):
        return False

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except BrokenPipeError:
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
