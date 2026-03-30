import numpy as np
import gym
from gym import spaces
from collections import deque


class EpisodicLifeMarioEnv(gym.Wrapper):
    """Treat loss of life as end-of-episode for better value estimation.
    The real environment only resets when all lives are lost.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped._life
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs


class MarioProgressWrapper(gym.Wrapper):
    """Reward shaping for game completion (not just running forward).
    - Big bonus for completing a stage (flag_get)
    - Penalty for standing still (idle_penalty)
    - Tracks overall game progress as a metric
    """
    def __init__(self, env, stage_bonus=500.0, idle_penalty=0.5,
                 idle_threshold=10, progress_scale=1.0):
        gym.Wrapper.__init__(self, env)
        self.stage_bonus = stage_bonus
        self.idle_penalty = idle_penalty
        self.idle_threshold = idle_threshold
        self.progress_scale = progress_scale
        self._prev_flag_get = False
        self._prev_x_pos = 0
        self._idle_steps = 0
        self._max_x_pos = 0
        self._stage_progress = 0  # 0-31, tracking which stage we're on

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_flag_get = False
        self._prev_x_pos = 0
        self._idle_steps = 0
        self._max_x_pos = 0
        self._stage_progress = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Stage completion bonus
        flag_get = info.get('flag_get', False)
        if flag_get and not self._prev_flag_get:
            reward += self.stage_bonus
        self._prev_flag_get = flag_get

        # Idle penalty: only after idle_threshold consecutive idle steps
        # Brief pauses (jumping, waiting for enemies) are fine
        x_pos = info.get('x_pos', 0)
        if x_pos <= self._prev_x_pos:
            self._idle_steps += 1
            if self._idle_steps > self.idle_threshold:
                reward -= self.idle_penalty
        else:
            self._idle_steps = 0
        self._prev_x_pos = x_pos

        # Track stage progress (world 1-8, stage 1-4 -> 0-31)
        world = info.get('world', 1)
        stage = info.get('stage', 1)
        current_progress = (world - 1) * 4 + (stage - 1)
        if current_progress > self._stage_progress:
            self._stage_progress = current_progress
        info['game_progress'] = self._stage_progress
        info['game_progress_pct'] = self._stage_progress / 31.0

        # Track x_pos within stage
        if x_pos > self._max_x_pos:
            self._max_x_pos = x_pos
        info['max_x_pos'] = self._max_x_pos

        # Scale the base reward
        reward *= self.progress_scale

        return obs, reward, done, info


class StickyActionWrapper(gym.Wrapper):
    """Repeat the previous action with some probability for robustness."""
    def __init__(self, env, p=0.25):
        gym.Wrapper.__init__(self, env)
        self.p = p
        self._last_action = 0

    def step(self, action):
        if np.random.random() < self.p:
            action = self._last_action
        self._last_action = action
        return self.env.step(action)

    def reset(self, **kwargs):
        self._last_action = 0
        return self.env.reset(**kwargs)


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every skip-th frame, max-pool last 2 observations."""
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    """Resize frames to 84x84 grayscale."""
    def __init__(self, env, width=84, height=84, grayscale=True):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        import cv2
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values to [0, 1]."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class FrameStack(gym.Wrapper):
    """Stack k last frames along the last axis."""
    def __init__(self, env, k=4):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.concatenate(list(self.frames), axis=-1)


class RandomStageWrapper(gym.Wrapper):
    """On each reset, randomly pick a stage from a pool of pre-created envs."""
    def __init__(self, env, stages, action_space_wrapper):
        gym.Wrapper.__init__(self, env)
        import gym_super_mario_bros
        self._stages = stages
        self._envs = {}
        for stage in stages:
            name = f'SuperMarioBros-{stage}-v0'
            e = gym_super_mario_bros.make(name)
            e = action_space_wrapper(e)
            self._envs[stage] = e
        self._current_stage = None

    def reset(self, **kwargs):
        stage = np.random.choice(self._stages)
        self.env = self._envs[stage]
        self._current_stage = stage
        return self.env.reset(**kwargs)

    def close(self):
        for e in self._envs.values():
            e.close()


def create_mario_env(**kwargs):
    """Factory function for rl_games environment registration.

    kwargs:
        name: environment id (default: SuperMarioBros-v0)
        action_type: 'simple' or 'complex' (default: 'complex')
        episode_life: treat each life as episode (default: True)
        stage_bonus: reward for completing a stage (default: 500)
        idle_penalty: per-step penalty after idle_threshold consecutive idle steps (default: 0.5)
        idle_threshold: steps of no progress before penalty kicks in (default: 10)
        skip: frame skip (default: 4)
        sticky_actions: probability of repeating previous action (default: 0)
        random_stages: list of stages to randomize, e.g. ['1-1','1-2','1-3','1-4']
                       if empty/None, uses the single env from 'name' (default: None)
    """
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

    name = kwargs.pop('name', 'SuperMarioBros-v0')
    action_type = kwargs.pop('action_type', 'complex')
    episode_life = kwargs.pop('episode_life', True)
    stage_bonus = kwargs.pop('stage_bonus', 500.0)
    idle_penalty = kwargs.pop('idle_penalty', 0.5)
    idle_threshold = kwargs.pop('idle_threshold', 10)
    skip = kwargs.pop('skip', 4)
    sticky_prob = kwargs.pop('sticky_actions', 0.0)
    random_stages = kwargs.pop('random_stages', None)

    actions = COMPLEX_MOVEMENT if action_type == 'complex' else SIMPLE_MOVEMENT
    wrap_actions = lambda e: JoypadSpace(e, actions)

    env = gym_super_mario_bros.make(name)
    env = wrap_actions(env)

    if random_stages:
        env = RandomStageWrapper(env, random_stages, wrap_actions)

    if sticky_prob > 0:
        env = StickyActionWrapper(env, p=sticky_prob)

    if episode_life:
        env = EpisodicLifeMarioEnv(env)

    env = MarioProgressWrapper(env, stage_bonus=stage_bonus, idle_penalty=idle_penalty,
                               idle_threshold=idle_threshold)
    env = MaxAndSkipEnv(env, skip=skip)
    env = WarpFrame(env, width=84, height=84, grayscale=True)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, k=4)

    return env
