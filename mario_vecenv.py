"""Vectorized environments for Mario training without Ray dependency."""

import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
from rl_games.common.ivecenv import IVecEnv
from rl_games.common import vecenv


def _worker(remote, parent_remote, env_kwargs):
    """Worker process that runs a single environment."""
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
                obs = env.reset()
                remote.send(obs)
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
    """Multiprocessing vectorized environment for Mario.

    Each environment runs in a separate process for true parallelism,
    which significantly improves throughput over synchronous stepping.
    """

    def __init__(self, config_name, num_actors, **kwargs):
        self.num_actors = num_actors
        self.waiting = False

        # Start worker processes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_actors)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = Process(target=_worker, args=(work_remote, remote, kwargs), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()

        # Get env info from first worker
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

        obs = self.reset()
        self._obs_dtype = obs.dtype
        self._obs_shape = obs.shape[1:]  # per-env shape

    def step(self, actions):
        # Send actions to all workers
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        # Collect results
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)

        return (
            np.array(obs, dtype=self._obs_dtype),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.array(obs)

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
            config_name, num_actors, **kwargs
        ),
    )
