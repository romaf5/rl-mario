"""Vectorized Mario env on the native SMB core (native/libbatchenv.so).

The C++ side steps N emulators on a threadpool (4 frames + RAM hacks +
render/pool/resize per agent step) and returns 84x84 uint8 frames plus a
2KB RAM snapshot per env. ALL game semantics -- rewards (high-water x,
time, death), progress/warp bonus with debounce, loop/backtrack penalties,
novelty, episodic life, victory/done, stage sampling, self-restarts --
live here, vectorized in numpy. Reward constants match mario_env.py; the
granularity is per agent step (4 frames) instead of per frame, so numbers
are equivalent-in-expectation rather than bit-identical to the retro chain.
"""

import ctypes
import gzip
import os

import numpy as np
from gymnasium import spaces

from rl_games.common import vecenv
from rl_games.common.ivecenv import IVecEnv

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, 'native', 'libbatchenv.so')
ROM = os.path.join(HERE, 'retro_integration', 'SuperMarioBros-Nes-v0',
                   'rom.nes')
STATE_DIR = os.path.join(HERE, 'native', 'states')

FRAME_STACK = 4

# COMPLEX_MOVEMENT -> native button byte (bit0=A,1=B,2=Sel,3=Start,4=U,5=D,6=L,7=R)
_ACTION_BYTES = np.array([
    0x00,               # NOOP
    0x80,               # right
    0x81,               # right+A
    0x82,               # right+B
    0x83,               # right+A+B
    0x01,               # A
    0x40,               # left
    0x41,               # left+A
    0x42,               # left+B
    0x43,               # left+A+B
    0x20,               # down
    0x10,               # up
], dtype=np.int32)


def _load_state(name):
    with gzip.open(os.path.join(STATE_DIR, f'Level{name}.state'
                                if name != 'FullGame' else 'FullGame.state'),
                   'rb') as f:
        return f.read()


class _Lib:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            lib = ctypes.CDLL(LIB)
            lib.benv_create.restype = ctypes.c_void_p
            lib.benv_create.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int]
            lib.benv_destroy.argtypes = [ctypes.c_void_p]
            lib.benv_step.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3
            lib.benv_state_size.restype = ctypes.c_int
            lib.benv_save.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_char_p]
            lib.benv_load.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_char_p]
            lib.benv_frames.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int]
            lib.benv_obs.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_void_p, ctypes.c_void_p]
            lib.benv_render.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_void_p]
            cls._inst = lib
        return cls._inst


class MarioNativeVecEnv(IVecEnv):
    """rl_games IVecEnv over the native batched core."""

    def __init__(self, config_name, num_actors, name='SuperMarioBros-v0',
                 action_type='complex', episode_life=True, stage_bonus=500.0,
                 idle_penalty=0.5, idle_threshold=10, progress_reward=0.001,
                 skip=4, sticky_actions=0.0, random_stages=None,
                 full_game=False, reset_noops=0, x_reward='highwater',
                 loop_penalty=0.0, backtrack_penalty=0.0, novelty_bonus=0.0,
                 self_restart_prob=0.0, self_restart_cells=96,
                 n_threads=32, seed=None, dense_infos=False,
                 novelty_y_band=48, score_reward=0.0,
                 novelty_global=False, explore_eps=0.0,
                 archive_path=None, explore_episode_prob=0.0,
                 explore_episode_steps=150, **unknown):
        assert action_type == 'complex'
        n = self.num_actors = num_actors
        self.lib = _Lib()
        rom = open(ROM, 'rb').read()
        self.single_stage = (not full_game) and (random_stages is not None)
        self.env = self.lib.benv_create(rom, len(rom), n, n_threads,
                                        int(self.single_stage))
        self.state_size = self.lib.benv_state_size()

        self.stages = list(random_stages) if random_stages else ['FullGame']
        self.states = {s: _load_state(s) for s in self.stages}
        self.stage_weights = None
        self.episode_life = episode_life
        self.stage_bonus = stage_bonus
        self.idle_penalty = idle_penalty
        self.idle_threshold = idle_threshold
        self.progress_reward = progress_reward
        self.sticky = sticky_actions
        self.reset_noops = reset_noops
        self.x_reward = x_reward
        self.loop_penalty = loop_penalty
        self.backtrack_penalty = backtrack_penalty
        self.novelty_bonus = novelty_bonus
        self.sr_prob = self_restart_prob
        self.sr_cells = self_restart_cells
        self.dense_infos = dense_infos
        self.novelty_y_band = novelty_y_band
        self.score_reward = score_reward
        self.novelty_global = novelty_global
        self.explore_eps = explore_eps
        self.archive_path = archive_path
        self._archive_dirty = 0
        self.exp_ep_prob = explore_episode_prob
        self.exp_ep_steps = explore_episode_steps
        self.explorer = np.zeros(num_actors, dtype=np.int32)
        self.novelty_counts = {}    # cross-episode cell visit counts

        self.rng = np.random.RandomState(seed)
        self.obs_u8 = np.zeros((n, 84, 84), dtype=np.uint8)
        self.ram = np.zeros((n, 0x800), dtype=np.uint8)
        self.actions_buf = np.zeros(n, dtype=np.int32)

        self.observation_space = spaces.Box(0.0, 1.0, (84, 84, FRAME_STACK),
                                            np.float32)
        self.action_space = spaces.Discrete(12)
        self._ring = np.zeros((n, 84, 84, FRAME_STACK), dtype=np.float32)
        self._ptr = 0

        # per-env python-side state
        z = lambda dt=np.int32: np.zeros(n, dtype=dt)
        self.x_last = z(); self.time_last = z(); self.hw = z()
        self.lives = z(); self.prev_flag = z(bool)
        self.progress = z(); self.pending = np.full(n, -1, np.int32)
        self.start_progress = z(); self.cleared = z()
        self.warped = z(bool); self.looped = z(bool); self.vic_paid = z(bool)
        self.idle = z(); self.prev_x = z(); self.max_x = z()
        self.prev_area = z(); self.last_action = z()
        self.start_stage = [''] * n
        self.was_restart = z(bool)
        self.prev_score = np.zeros(n, dtype=np.int64)
        self.novelty_sets = [set() for _ in range(n)]
        # SHARED self-restart archive: all envs contribute and draw from one
        # pool (per-env archives dilute frontier discovery at large N)
        self.archive = {}                           # cell -> [state, uses]
        if archive_path and os.path.exists(archive_path):
            import pickle
            with open(archive_path, 'rb') as f:
                self.archive = pickle.load(f)
            print(f'[archive] loaded {len(self.archive)} cells '
                  f'from {archive_path}')
        self.ep_cells = [set() for _ in range(n)]
        self._sbuf = ctypes.create_string_buffer(self.state_size)

    # ------------------------------------------------------------- helpers
    def _field(self, addr):
        return self.ram[:, addr].astype(np.int32)

    def _x(self):
        return self._field(0x6D) * 256 + self._field(0x86)

    def _time(self):
        return (self._field(0x7F8) * 100 + self._field(0x7F9) * 10
                + self._field(0x7FA))

    def _score(self):
        d = self.ram[:, 0x7DD:0x7E3].astype(np.int64)
        return (d * np.array([100000, 10000, 1000, 100, 10, 1])).sum(axis=1)

    def _gp(self):
        return np.clip(self._field(0x75F) * 4 + self._field(0x75C), 0, 31)

    def _flag(self):
        et = self.ram[:, 0x16:0x1B].astype(np.int32)
        stage_over = (((et == 0x2D) | (et == 0x31)).any(axis=1)
                      & (self._field(0x1D) == 3))
        return (self._field(0x770) == 2) | stage_over

    def _reset_env(self, i, first=False):
        # self-restart from own archive?
        self.was_restart[i] = False
        if (self.sr_prob > 0 and self.archive
                and self.rng.random_sample() < self.sr_prob):
            # soft least-practiced: p(cell) ~ 1/(1+uses). Uniform-ish
            # coverage bridges the door->frontier gap; a hard frontier
            # bias starves the cells where the policy actually fails.
            cells = list(self.archive.keys())
            w = np.array([1.0 / (1 + self.archive[c][1]) for c in cells])
            cell = cells[self.rng.choice(len(cells), p=w / w.sum())]
            ent = self.archive[cell]
            ent[1] += 1
            self.lib.benv_load(self.env, i, ent[0])
            self.start_stage[i] = cell[0]
            self.was_restart[i] = True
            # Go-Explore phase 1: some restart episodes flail randomly to
            # EXPAND the archive past what the policy can reach
            if self.rng.random_sample() < self.exp_ep_prob:
                self.explorer[i] = self.exp_ep_steps
        else:
            if self.stage_weights is not None:
                s = self.stages[self.rng.choice(len(self.stages),
                                                p=self.stage_weights)]
            else:
                s = self.stages[self.rng.randint(len(self.stages))]
            self.lib.benv_load(self.env, i, self.states[s])
            self.start_stage[i] = s
            if self.reset_noops:
                self.lib.benv_frames(self.env, i,
                                     int(self.rng.randint(
                                         0, self.reset_noops + 1)), 0)
        self.ep_cells[i] = set()
        self.novelty_sets[i] = set()

    def _post_reset_init(self, idx, ram):
        """Re-init per-env python state for envs in idx from fresh RAM."""
        for i in idx:
            r = ram[i]
            x = int(r[0x6D]) * 256 + int(r[0x86])
            gp = min(max(int(r[0x75F]) * 4 + int(r[0x75C]), 0), 31)
            self.x_last[i] = x; self.prev_x[i] = 0; self.max_x[i] = 0
            self.hw[i] = x
            self.time_last[i] = (int(r[0x7F8]) * 100 + int(r[0x7F9]) * 10
                                 + int(r[0x7FA]))
            self.lives[i] = int(r[0x75A])
            d6 = r[0x7DD:0x7E3].astype(np.int64)
            self.prev_score[i] = int((d6 * np.array(
                [100000, 10000, 1000, 100, 10, 1])).sum())
            self.prev_flag[i] = False
            self.progress[i] = gp; self.start_progress[i] = gp
            self.pending[i] = -1; self.cleared[i] = 0
            self.warped[i] = False; self.looped[i] = False
            self.vic_paid[i] = False
            self.idle[i] = 0
            self.prev_area[i] = int(r[0x760])

    # ------------------------------------------------------------- IVecEnv
    def _fetch_obs(self, i):
        self.lib.benv_obs(self.env, int(i),
                          self.obs_u8[i].ctypes.data,
                          self.ram[i].ctypes.data)

    def reset(self):
        for i in range(self.num_actors):
            self._reset_env(i, first=True)
            self._fetch_obs(i)
        self._post_reset_init(range(self.num_actors), self.ram)
        f = self.obs_u8.astype(np.float32) / 255.0
        self._ring[:] = f[..., None]
        order = [(self._ptr + 1 + j) % FRAME_STACK for j in range(FRAME_STACK)]
        return self._ring[..., order]

    def step(self, actions):
        n = self.num_actors
        acts = np.asarray(actions).astype(np.int64).ravel()
        exp_mask = self.explorer > 0
        if exp_mask.any():
            acts = np.where(exp_mask, self.rng.randint(0, 12, size=n), acts)
            self.explorer = np.maximum(self.explorer - 1, 0)
        if self.explore_eps > 0:
            # permanent action-diversity floor: collapsed policy entropy
            # otherwise closes the discovery window for rare moves
            ex = self.rng.random_sample(n) < self.explore_eps
            acts = np.where(ex, self.rng.randint(0, 12, size=n), acts)
        if self.sticky > 0:
            rep = self.rng.random_sample(n) < self.sticky
            acts = np.where(rep, self.last_action, acts)
        self.last_action[:] = acts
        self.actions_buf[:] = _ACTION_BYTES[acts]
        self.lib.benv_step(self.env, self.actions_buf.ctypes.data,
                           self.obs_u8.ctypes.data, self.ram.ctypes.data)
        ram = self.ram

        x = self._x(); t = self._time(); gp = self._gp()
        # transition frames can leave garbage in the x page byte; no SMB
        # level exceeds ~3800px, so treat larger readings as glitches and
        # carry the previous x (self-corrects next block)
        x = np.where(x > 4000, self.x_last, x)
        life = self._field(0x75A); area = self._field(0x760)
        pstate = self._field(0x0E); yvp = self._field(0xB5)
        dying = (pstate == 0x0B) | (yvp > 1)
        dead = pstate == 0x06
        flag = self._flag()
        gmode = self._field(0x770)
        if self.single_stage:
            victory = np.zeros(n, dtype=bool)
        else:
            victory = (gmode == 2) & (gp == 31)

        # ---- base reward (block granularity) ----
        if self.x_reward == 'highwater':
            ctx_change = (life != self.lives) | (area != self.prev_area) | \
                         (gp != self.progress)
            self.hw = np.where(ctx_change, x, self.hw)
            r_x = np.clip(x - self.hw, 0, 20).astype(np.float32)
            self.hw = np.maximum(self.hw, x)
        else:
            dx = x - self.x_last
            r_x = np.where(np.abs(dx) > 24, 0, dx).astype(np.float32)
        r_t = np.minimum(t - self.time_last, 0).astype(np.float32)
        r_death = np.where(dying | dead, -15.0, 0.0).astype(np.float32)
        reward = np.clip(r_x + r_t + r_death, -15, 20)
        self.x_last = x
        self.time_last = t

        # ---- score reward: the game's own signal (coins, stomps, blocks;
        # notably the hidden-block reveal pays +200) ----
        if self.score_reward > 0:
            sc = self._score()
            ds = np.clip(sc - self.prev_score, 0, 2000)
            reward = reward + ds * self.score_reward
            self.prev_score = sc

        # ---- progress bonus (debounced, monotonic, jump-capped) ----
        inc = gp > self.progress
        confirm = inc & (gp == self.pending)
        delta = gp - self.progress
        ok = confirm & (delta <= 15)
        if ok.any():
            if not self.single_stage:
                reward = reward + np.where(ok, self.stage_bonus * delta, 0)
            self.cleared += ok.astype(np.int32)
            self.warped |= ok & (delta >= 2)
            self.progress = np.where(ok, gp, self.progress)
            # rebase within-stage x tracking on level change
            self.max_x = np.where(ok, 0, self.max_x)
            self.prev_x = np.where(ok, 0, self.prev_x)
        self.pending = np.where(inc, gp, -1)
        if self.single_stage:
            newflag = flag & ~self.prev_flag
            reward = reward + np.where(newflag, self.stage_bonus, 0)
        self.prev_flag = flag

        # victory terminal bonus
        vpay = victory & ~self.vic_paid
        reward = reward + np.where(vpay, self.stage_bonus, 0)
        self.vic_paid |= victory

        # ---- loop penalty / x shaping / idle ----
        xd = x - self.prev_x
        loop = (self.loop_penalty > 0) & (xd < -96) & \
               (area == self.prev_area) & ~flag & ~dying & ~dead
        reward = reward - np.where(loop, self.loop_penalty, 0)
        self.looped |= loop
        self.prev_area = area

        fwd = xd > 0
        new_ground = x > self.max_x
        grow = np.minimum(xd, 20) * self.progress_reward * x
        reward = reward + np.where(fwd & new_ground, grow, 0)
        reward = reward - np.where(fwd & ~new_ground,
                                   self.backtrack_penalty, 0)
        self.idle = np.where(fwd, 0, self.idle + 1)
        reward = reward - np.where(self.idle > self.idle_threshold,
                                   self.idle_penalty, 0)
        self.prev_x = x
        self.max_x = np.maximum(self.max_x, x)

        # ---- novelty (python sets; ~N ops/step) ----
        if self.novelty_bonus > 0:
            ypix = self._field(0x3B8)
            for i in range(n):
                cell = (int(area[i]), int(x[i]) // 64,
                        int(ypix[i]) // self.novelty_y_band)
                if self.novelty_global:
                    # cross-episode decaying counts: mundane cells deplete,
                    # never-reached cells keep a standing bonus
                    c = self.novelty_counts.get(cell, 0)
                    if cell not in self.novelty_sets[i]:
                        self.novelty_sets[i].add(cell)
                        self.novelty_counts[cell] = c + 1
                        reward[i] += self.novelty_bonus / (1 + c) ** 0.5
                elif cell not in self.novelty_sets[i]:
                    self.novelty_sets[i].add(cell)
                    reward[i] += self.novelty_bonus

        # ---- self-restart archiving ----
        if self.sr_prob > 0:
            fstate = self._field(0x1D)
            can = (fstate == 0) & ~dying & ~dead & (gmode == 1) & (t > 100)
            ypix_a = self._field(0x3B8)
            swim_a = self.ram[:, 0x704].astype(np.int32)
            for i in np.nonzero(can)[0]:
                # y-band in the key: standing ON a block/pipe is a different
                # rung than the floor below it; swim flag disambiguates the
                # water zone (same area byte + low x as the level start)
                cell = (self.start_stage[i], int(area[i]), int(x[i]) // 128,
                        int(ypix_a[i]) // 64, int(swim_a[i]))
                if cell in self.ep_cells[i]:
                    continue
                self.ep_cells[i].add(cell)
                if cell not in self.archive:
                    if len(self.archive) >= self.sr_cells:
                        worst = max(self.archive,
                                    key=lambda c: self.archive[c][1])
                        del self.archive[worst]
                    self.lib.benv_save(self.env, int(i), self._sbuf)
                    self.archive[cell] = [bytes(self._sbuf.raw), 0]
                    self._archive_dirty += 1
                elif self.rng.random_sample() < 0.02:
                    # refresh: stored states drift toward the current
                    # visitation distribution (e.g. post-reveal variants)
                    self.lib.benv_save(self.env, int(i), self._sbuf)
                    self.archive[cell][0] = bytes(self._sbuf.raw)
                    self._archive_dirty += 1
        if (self.archive_path and self._archive_dirty >= 10):
            self._archive_dirty = 0
            import pickle
            tmp = self.archive_path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(self.archive, f)
            os.replace(tmp, self.archive_path)

        # ---- dones ----
        game_over = life == 0xFF
        if self.single_stage:
            real_done = dying | dead | flag
        else:
            real_done = game_over | victory
        life_lost = (life < self.lives) & (life > 0)
        self.lives = life

        infos = []
        done_pre = real_done | (life_lost if self.episode_life else False)
        for i in range(n):
            if not done_pre[i] and not self.dense_infos:
                infos.append({})   # observer only reads infos of done envs
                continue
            infos.append({
                'x_pos': int(x[i]), 'max_x_pos': int(self.max_x[i]),
                'game_progress': int(self.progress[i]),
                'progress_gain': int(self.progress[i]
                                     - self.start_progress[i]),
                'stages_cleared': int(self.cleared[i]),
                'warped': bool(self.warped[i]),
                'victory': bool(victory[i]),
                'looped': bool(self.looped[i]),
                'flag_get': bool(flag[i]), 'life': int(life[i]),
                'world': int(ram[i, 0x75F]) + 1,
                'stage': int(ram[i, 0x75C]) + 1,
                'time': int(t[i]), 'coins': int(ram[i, 0x7ED]),
                'score': 0,
                'start_stage': self.start_stage[i],
                'self_restart': bool(self.was_restart[i]),
            })

        done = done_pre

        # frame stack (ring)
        f = self.obs_u8.astype(np.float32) / 255.0
        self._ptr = (self._ptr + 1) % FRAME_STACK
        self._ring[..., self._ptr] = f

        # resets for finished episodes
        done_idx = np.nonzero(done)[0]
        realdone_idx = np.nonzero(real_done)[0]
        if len(realdone_idx):
            for i in realdone_idx:
                self._reset_env(int(i))
                self._fetch_obs(i)
                self._ring[i] = (self.obs_u8[i].astype(np.float32)
                                 / 255.0)[..., None]
            self._post_reset_init(realdone_idx, self.ram)
        # life-loss boundaries: re-init episode trackers but keep playing
        soft_idx = [i for i in done_idx if i not in set(realdone_idx)]
        if soft_idx:
            self._post_reset_init(soft_idx, self.ram)

        order = [(self._ptr + 1 + j) % FRAME_STACK for j in range(FRAME_STACK)]
        obs = self._ring[..., order]
        return obs, reward.astype(np.float32), done, infos

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {'observation_space': self.observation_space,
                'action_space': self.action_space, 'agents': 1,
                'value_size': 1}

    def set_stage_weights(self, weights):
        w = np.array([max(float(weights.get(s, 0.0)), 0.0)
                      for s in self.stages])
        tot = w.sum()
        self.stage_weights = (w / tot) if tot > 0 else None

    def has_action_masks(self):
        return False

    def _save_archive(self):
        if self.archive_path and self.archive:
            import pickle
            tmp = self.archive_path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(self.archive, f)
            os.replace(tmp, self.archive_path)

    def close(self):
        self._save_archive()
        self.lib.benv_destroy(self.env)


class NativeEvalEnv:
    """Single-env adapter over MarioNativeVecEnv for the video/eval loop
    (old-gym API + .screen for frame capture)."""

    def __init__(self, **kwargs):
        kwargs.setdefault('n_threads', 1)
        kwargs.setdefault('episode_life', False)
        self.v = MarioNativeVecEnv('eval', 1, dense_infos=True, **kwargs)
        self._buf = ctypes.create_string_buffer(240 * 224)

    @property
    def unwrapped(self):
        return self

    @property
    def screen(self):
        self.v.lib.benv_render(self.v.env, 0, self._buf)
        g = np.frombuffer(self._buf, dtype=np.uint8).reshape(224, 240)
        return np.stack([g, g, g], axis=-1)

    def reset(self):
        return self.v.reset()[0]

    def step(self, action):
        obs, r, d, infos = self.v.step([int(action)])
        return obs[0], float(r[0]), bool(d[0]), infos[0]

    def close(self):
        self.v.close()


def register_mario_native_vecenv():
    vecenv.register(
        'MARIO_NATIVE',
        lambda config_name, num_actors, **kwargs: MarioNativeVecEnv(
            config_name, num_actors, **kwargs))
