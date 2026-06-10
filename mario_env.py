"""stable-retro (C-core libretro) backend for Super Mario Bros.

Drop-in replacement for mario_env.create_mario_env() with ~10x throughput.
Replicates gym_super_mario_bros semantics (reward function, RAM hacks, info
dict) on top of stable-retro's NES core, using the custom integration in
retro_integration/SuperMarioBros-Nes-v0 (extended data.json + generated
Level<W>-<S>.state files for all 32 levels, see gen_states.py).

This module intentionally does NOT import mario_env.py (which needs old gym
0.25 / nes_py); the frame wrappers are re-implemented verbatim on a minimal
Wrapper base so the file runs inside venv_retro (gymnasium only).
"""

import gzip
import os
from collections import defaultdict, deque

import numpy as np
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(HERE, 'retro_integration')
GAME = 'SuperMarioBros-Nes-v0'

# NES button order in stable-retro
_BUTTONS = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
_BUTTON_INDEX = {'B': 0, 'select': 2, 'start': 3, 'up': 4, 'down': 5,
                 'left': 6, 'right': 7, 'A': 8}

# Same action sets as gym_super_mario_bros.actions
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]

COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]

# Constants from gym_super_mario_bros.smb_env
_STATUS_MAP = defaultdict(lambda: 'fireball', {0: 'small', 1: 'tall'})
_BUSY_STATES = (0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07)
_STAGE_OVER_ENEMIES = (0x2D, 0x31)  # Bowser, Flagpole

# RAM addresses (for writes; reads come from data.json variables)
_ADDR_PLAYER_STATE = 0x000E
_ADDR_CHANGE_AREA_TIMER = 0x06DE
_ADDR_PRELEVEL_TIMER = 0x07A0

_integration_registered = False


def _register_integration():
    global _integration_registered
    if not _integration_registered:
        import stable_retro as retro
        retro.data.Integrations.add_custom_path(INTEGRATION_PATH)
        _integration_registered = True


def _load_state_bytes(stage_name):
    """Read raw (un-gzipped) emulator state bytes for e.g. '1-1' or 'FullGame'."""
    if stage_name == 'FullGame':
        fname = 'FullGame.state'
    else:
        fname = f'Level{stage_name}.state'
    path = os.path.join(INTEGRATION_PATH, GAME, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No savestate for stage {stage_name!r} at {path}. '
            f'Run gen_states.py to generate it.')
    with gzip.open(path, 'rb') as f:
        return f.read()


class RetroMarioEnv:
    """SMB on stable-retro with gym_super_mario_bros-compatible semantics.

    - Discrete action space (COMPLEX_MOVEMENT by default), one action per frame.
    - Reward per frame: x_pos delta (|delta|>5 -> 0) + time penalty + death
      penalty (-25), clipped to [-15, 15] like nes_py's reward_range.
    - Same RAM hacks after each step: kill Mario during the dying animation,
      skip end-of-world cutscenes (full game), skip area-change animations and
      "occupied" states (black inter-life screens).
    - done: single stage -> dying/dead/flag_get; full game -> game over.
    - Old gym API: reset() -> obs, step() -> (obs, reward, done, info).
    """

    reward_range = (-15, 15)

    def __init__(self, target=None, random_stages=None,
                 actions=COMPLEX_MOVEMENT):
        _register_integration()
        import stable_retro as retro

        self._retro = retro.make(
            GAME, state=retro.State.NONE,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            use_restricted_actions=retro.Actions.ALL,
            render_mode='rgb_array')

        self._random_stages = list(random_stages) if random_stages else None
        self._target = target
        self.is_single_stage_env = (target is not None
                                    or self._random_stages is not None)

        # Pre-load savestate bytes (load_state gunzips from disk every call)
        if self._random_stages:
            self._states = {s: _load_state_bytes(s) for s in self._random_stages}
        elif target is not None:
            world, stage = target
            self._states = {'_': _load_state_bytes(f'{world}-{stage}')}
        else:
            self._states = {'_': _load_state_bytes('FullGame')}

        # Discrete action -> button mask
        self._actions = actions
        self._masks = []
        for combo in actions:
            mask = np.zeros(len(_BUTTONS), dtype=np.uint8)
            for b in combo:
                if b != 'NOOP':
                    mask[_BUTTON_INDEX[b]] = 1
            self._masks.append(mask)
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = self._retro.observation_space  # (224,240,3) u8

        self._rng = np.random.RandomState()
        self._cur = {}            # latest data.json variable values
        self._time_last = 0
        self._x_last = 0
        self._noop_mask = np.zeros(len(_BUTTONS), dtype=np.uint8)

        # Fast path: step the emulator directly, bypassing RetroEnv.step()
        # (action_to_array, scenario reward/done evaluation, movie hooks).
        self._em = self._retro.em
        self._data = self._retro.data
        # When False, step() returns obs=None (cheap). MaxAndSkipEnv only
        # consumes the last 2 of every `skip` frames, so it toggles this to
        # avoid 2 unused framebuffer copies per agent step.
        self.want_obs = True

    # -- emulator helpers ---------------------------------------------------

    def _frame(self):
        """Advance one frame with no buttons pressed (outside RetroEnv.step)."""
        em = self._em
        em.set_button_mask(self._noop_mask, 0)
        em.step()
        self._data.update_ram()
        self._cur = self._data.lookup_all()

    def _assign(self, addr, value):
        self._data.memory.assign(addr, '|u1', value)

    # -- RAM-derived state (mirrors smb_env.py properties) -------------------

    @property
    def _x_position(self):
        return self._cur['xpos_hi'] * 0x100 + self._cur['xpos_lo']

    @property
    def _life(self):
        return self._cur['life']

    @property
    def _time(self):
        return self._cur['time']

    @property
    def _y_position(self):
        if self._cur['y_viewport'] < 1:
            return 255 + (255 - self._cur['y_pixel'])
        return 255 - self._cur['y_pixel']

    @property
    def _is_dying(self):
        return (self._cur['player_state'] == 0x0B
                or self._cur['y_viewport'] > 1)

    @property
    def _is_dead(self):
        return self._cur['player_state'] == 0x06

    @property
    def _is_game_over(self):
        return self._cur['life'] == 0xFF

    @property
    def _is_busy(self):
        return self._cur['player_state'] in _BUSY_STATES

    @property
    def _is_world_over(self):
        return self._cur['gameplay_mode'] == 2

    @property
    def _is_stage_over(self):
        c = self._cur
        for key in ('enemy_type0', 'enemy_type1', 'enemy_type2',
                    'enemy_type3', 'enemy_type4'):
            if c[key] in _STAGE_OVER_ENEMIES:
                # player float state is 3 while sliding down the flag pole
                return c['float_state'] == 3
        return False

    @property
    def _flag_get(self):
        return self._is_world_over or self._is_stage_over

    @property
    def screen(self):
        """Current RGB frame (compat with nes_py's env.screen)."""
        return self._retro.get_screen()

    @property
    def unwrapped(self):
        return self

    # -- reward (mirrors smb_env.py) -----------------------------------------

    def _x_reward(self):
        x = self._x_position
        reward = x - self._x_last
        self._x_last = x
        if reward < -5 or reward > 5:
            return 0
        return reward

    def _time_penalty(self):
        t = self._time
        reward = t - self._time_last
        self._time_last = t
        if reward > 0:
            return 0
        return reward

    def _death_penalty(self):
        if self._is_dying or self._is_dead:
            return -25
        return 0

    # -- RAM hacks (mirrors smb_env.py::_did_step) ----------------------------

    def _kill_mario(self):
        self._assign(_ADDR_PLAYER_STATE, 0x06)
        self._frame()

    def _skip_end_of_world(self):
        if self._is_world_over:
            time = self._time
            for _ in range(5000):
                if self._time != time:
                    break
                self._frame()

    def _skip_change_area(self):
        timer = self._cur['change_area_timer']
        if 1 < timer < 255:
            self._assign(_ADDR_CHANGE_AREA_TIMER, 1)

    def _skip_occupied_states(self):
        for _ in range(5000):
            if not (self._is_busy or self._is_world_over):
                break
            self._assign(_ADDR_PRELEVEL_TIMER, 0)
            self._frame()

    def _did_step(self, done):
        if done:
            return False
        stepped = False
        if self._is_dying:
            self._kill_mario()
            stepped = True
        if not self.is_single_stage_env and self._is_world_over:
            self._skip_end_of_world()
            stepped = True
        self._skip_change_area()
        if self._is_busy or self._is_world_over:
            self._skip_occupied_states()
            stepped = True
        return stepped

    # -- gym (old) API --------------------------------------------------------

    def seed(self, seed=None):
        self._rng.seed(seed)
        return [seed]

    def reset(self, **kwargs):
        if self._random_stages:
            stage = self._rng.choice(self._random_stages)
            self._retro.initial_state = self._states[stage]
        else:
            self._retro.initial_state = self._states['_']
        obs, _ = self._retro.reset()
        self._cur = dict(self._retro.data.lookup_all())
        self._time_last = self._time
        self._x_last = self._x_position
        return obs

    def step(self, action):
        em = self._em
        em.set_button_mask(self._masks[action], 0)
        em.step()
        self._data.update_ram()
        self._cur = self._data.lookup_all()

        reward = self._x_reward() + self._time_penalty() + self._death_penalty()
        if reward < self.reward_range[0]:
            reward = self.reward_range[0]
        elif reward > self.reward_range[1]:
            reward = self.reward_range[1]

        if self.is_single_stage_env:
            done = self._is_dying or self._is_dead or self._flag_get
        else:
            done = self._is_game_over

        out_info = dict(
            coins=self._cur['coins'],
            flag_get=self._flag_get,
            life=self._cur['life'],
            score=self._cur['score'],
            stage=self._cur['stage0'] + 1,
            status=_STATUS_MAP[self._cur['player_status']],
            time=self._cur['time'],
            world=self._cur['world0'] + 1,
            x_pos=self._x_position,
            y_pos=self._y_position,
        )

        self._did_step(done)
        obs = em.get_screen() if self.want_obs else None

        return obs, reward, done, out_info

    def render(self, mode='rgb_array'):
        return self._retro.get_screen()

    def close(self):
        # Drop our cached emulator reference first, otherwise the C++ object
        # stays alive and stable-retro refuses to create another emulator in
        # this process.
        if hasattr(self, '_em'):
            del self._em
        self._retro.close()


class Wrapper:
    """Minimal old-gym-style wrapper base (no gym/gymnasium dependency)."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


class ObservationWrapper(Wrapper):
    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, observation):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The wrappers below are verbatim ports of mario_env.py (same math, same
# defaults), rebased on the local Wrapper class so they run without gym 0.25.
# ---------------------------------------------------------------------------


class EpisodicLifeMarioEnv(Wrapper):
    """Treat loss of life as end-of-episode for better value estimation.
    The real environment only resets when all lives are lost.
    """
    def __init__(self, env):
        Wrapper.__init__(self, env)
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


class MarioProgressWrapper(Wrapper):
    """Reward shaping for game completion (not just running forward).
    - Big bonus for completing a stage (flag_get)
    - Penalty for standing still (idle_penalty)
    - Tracks overall game progress as a metric
    """
    def __init__(self, env, stage_bonus=500.0, idle_penalty=0.5,
                 idle_threshold=10, progress_reward=0.001, progress_scale=1.0):
        Wrapper.__init__(self, env)
        self.stage_bonus = stage_bonus
        self.idle_penalty = idle_penalty
        self.idle_threshold = idle_threshold
        self.progress_reward = progress_reward
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

        # Growing progress reward: forward movement scaled by position
        # At x=0 bonus is ~0, at x=2000 bonus is +2 per step of forward movement
        # Cap x_delta to avoid spike on episode reset (episode_life resets
        # _prev_x_pos to 0 but env continues from death position)
        x_pos = info.get('x_pos', 0)
        x_delta = x_pos - self._prev_x_pos
        if x_delta > 0:
            x_delta = min(x_delta, 20)  # cap to normal per-step movement
            reward += x_delta * self.progress_reward * x_pos
            self._idle_steps = 0
        else:
            # Idle penalty: only after idle_threshold consecutive idle steps
            self._idle_steps += 1
            if self._idle_steps > self.idle_threshold:
                reward -= self.idle_penalty
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


class StickyActionWrapper(Wrapper):
    """Repeat the previous action with some probability for robustness."""
    def __init__(self, env, p=0.25):
        Wrapper.__init__(self, env)
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


class MaxAndSkipEnv(Wrapper):
    """Return only every skip-th frame, max-pool last 2 observations.

    Retro micro-optimization: only the last 2 frames of each skip group are
    consumed, so the base env is told to skip the framebuffer copy for the
    other frames (base step returns obs=None there, which is never used).
    """
    def __init__(self, env, skip=4):
        Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self._base = env.unwrapped

    def step(self, action):
        total_reward = 0.0
        done = None
        base = self._base
        try:
            for i in range(self._skip):
                base.want_obs = i >= self._skip - 2
                obs, reward, done, info = self.env.step(action)
                if i == self._skip - 2:
                    self._obs_buffer[0] = obs
                if i == self._skip - 1:
                    self._obs_buffer[1] = obs
                total_reward += reward
                if done:
                    break
        finally:
            base.want_obs = True
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(ObservationWrapper):
    """Resize frames to 84x84 grayscale."""
    def __init__(self, env, width=84, height=84, grayscale=True):
        ObservationWrapper.__init__(self, env)
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


class ScaledFloatFrame(ObservationWrapper):
    """Normalize pixel values to [0, 1]."""
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class FrameStack(Wrapper):
    """Stack k last frames along the last axis."""
    def __init__(self, env, k=4):
        Wrapper.__init__(self, env)
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


def _parse_name(name):
    """'SuperMarioBros-v0' -> None (full game); 'SuperMarioBros-1-2-v0' -> (1, 2)."""
    parts = name.split('-')
    if len(parts) == 4 and parts[0] == 'SuperMarioBros':
        return int(parts[1]), int(parts[2])
    return None


def create_mario_env(**kwargs):
    """Factory function for rl_games environment registration (retro backend).

    Same kwargs as mario_env.create_mario_env:
        name: environment id (default: SuperMarioBros-v0)
        action_type: 'simple' or 'complex' (default: 'complex')
        episode_life: treat each life as episode (default: True)
        stage_bonus: reward for completing a stage (default: 500)
        idle_penalty: per-step penalty after idle_threshold consecutive idle steps (default: 0.5)
        idle_threshold: steps of no progress before penalty kicks in (default: 10)
        progress_reward: growing bonus multiplier for forward movement (default: 0.001)
        skip: frame skip (default: 4)
        sticky_actions: probability of repeating previous action (default: 0)
        random_stages: list of stages to randomize, e.g. ['1-1','1-2','1-3','1-4']
                       if empty/None, uses the single env from 'name' (default: None)

    Note: unlike the nes_py backend, random_stages uses a single emulator that
    loads a random per-level savestate on each reset (stable-retro allows only
    one emulator per process).
    """
    name = kwargs.pop('name', 'SuperMarioBros-v0')
    action_type = kwargs.pop('action_type', 'complex')
    episode_life = kwargs.pop('episode_life', True)
    stage_bonus = kwargs.pop('stage_bonus', 500.0)
    idle_penalty = kwargs.pop('idle_penalty', 0.5)
    idle_threshold = kwargs.pop('idle_threshold', 10)
    progress_reward = kwargs.pop('progress_reward', 0.001)
    skip = kwargs.pop('skip', 4)
    sticky_prob = kwargs.pop('sticky_actions', 0.0)
    random_stages = kwargs.pop('random_stages', None)

    actions = COMPLEX_MOVEMENT if action_type == 'complex' else SIMPLE_MOVEMENT

    env = RetroMarioEnv(target=_parse_name(name),
                        random_stages=random_stages,
                        actions=actions)

    if sticky_prob > 0:
        env = StickyActionWrapper(env, p=sticky_prob)

    if episode_life:
        env = EpisodicLifeMarioEnv(env)

    env = MarioProgressWrapper(env, stage_bonus=stage_bonus, idle_penalty=idle_penalty,
                               idle_threshold=idle_threshold,
                               progress_reward=progress_reward)
    env = MaxAndSkipEnv(env, skip=skip)
    env = WarpFrame(env, width=84, height=84, grayscale=True)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, k=4)

    return env
