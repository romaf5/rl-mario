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
            lib.benv_render_rgb.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_void_p]
            lib.benv_step_rgb4.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p]
            lib.benv_step_raw.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_void_p]
            lib.benv_set_ram.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_char_p]
            lib.benv_set_skip.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.benv_transit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
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
                 explore_episode_steps=150,
                 self_restart_frontier_prob=0.0,
                 self_restart_frontier_k=16, idle_timeout=150,
                 offroute_penalty=0.0, fail_penalty=15.0, **unknown):
        assert action_type == 'complex'
        n = self.num_actors = num_actors
        self.lib = _Lib()
        rom = open(ROM, 'rb').read()
        self.single_stage = (not full_game) and (random_stages is not None)
        self.env = self.lib.benv_create(rom, len(rom), n, n_threads,
                                        int(self.single_stage))
        self.skip = int(skip)
        if self.skip != 4:
            self.lib.benv_set_skip(self.env, self.skip)
        self.state_size = self.lib.benv_state_size()

        self.stages = list(random_stages) if random_stages else ['FullGame']
        self.states = {s: _load_state(s) for s in self.stages}
        # off-route guard (full-game training on a level SET): a confirmed
        # move into a level outside the set is a dead end -- penalised and
        # terminal. Without it the 1-2 flag paid +500 and then unlimited
        # x-reward in 1-3, so the flag beat the warp (which pays +5500 once).
        self.offroute_penalty = float(offroute_penalty)
        self.route_gps = None
        if full_game and random_stages and self.offroute_penalty > 0:
            self.route_gps = np.array(sorted(
                (int(s.split('-')[0]) - 1) * 4 + int(s.split('-')[1]) - 1
                for s in self.stages), dtype=np.int32)
        self.stage_weights = None
        self.episode_life = episode_life
        self.stage_bonus = stage_bonus
        self.idle_penalty = idle_penalty
        self.idle_threshold = idle_threshold
        self.idle_timeout = idle_timeout
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
        self.sr_frontier_prob = self_restart_frontier_prob
        self.sr_frontier_k = self_restart_frontier_k
        self.explorer = np.zeros(num_actors, dtype=np.int32)
        self.exp_action = np.zeros(num_actors, dtype=np.int64)
        self.ep_steps = np.zeros(num_actors, dtype=np.int32)
        self.start_cell = [None] * num_actors
        self.cell_early = {}
        self.cell_wins = {}
        self._novelty_tick = 0
        self.nongame = np.zeros(num_actors, dtype=np.int32)
        self.novelty_counts = {}    # cross-episode cell visit counts

        self.rng = np.random.RandomState(seed)
        self.obs_u8 = np.zeros((n, 84, 84), dtype=np.uint8)
        self.ram = np.zeros((n, 0x800), dtype=np.uint8)
        self.actions_buf = np.zeros(n, dtype=np.int32)
        # uniform cost of every attempt-ending failure (death, idle timeout;
        # loop/off-route use their own knobs): equal costs leave no reason
        # to prefer one failure over another (e.g. suicide over a loop)
        self.fail_penalty = float(fail_penalty)
        # per-life visited cells (level, area, swim, x//128) as ints: a jump
        # into a visited cell is a cycle whatever mechanism produced it
        self.visited = [set() for _ in range(n)]
        self.last_cid = np.full(n, -1, dtype=np.int64)
        self.prev_swim = np.zeros(n, dtype=np.int32)

        self.observation_space = spaces.Box(0.0, 1.0, (84, 84, FRAME_STACK),
                                            np.float32)
        self.action_space = spaces.Discrete(12)
        self._ring = np.zeros((n, 84, 84, FRAME_STACK), dtype=np.float32)
        self._ptr = 0

        # per-env python-side state
        z = lambda dt=np.int32: np.zeros(n, dtype=dt)
        self.x_last = z(); self.time_last = z(); self.hw = z()
        self.x_pending = z()
        self.lives = z(); self.prev_flag = z(bool)
        self.progress = z(); self.pending = np.full(n, -1, np.int32)
        self.start_progress = z(); self.cleared = z()
        self.warped = z(bool); self.looped = z(bool); self.vic_paid = z(bool)
        self.idle = z(); self.prev_x = z(); self.max_x = z()
        self.idle_paid = np.zeros(n, dtype=np.float32)
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
            # win counts persist as entry[3] (backward-chaining state)
            self.cell_wins = {c: e[3] for c, e in self.archive.items()
                              if len(e) > 3}
        self.ep_cells = [set() for _ in range(n)]
        self._sbuf = ctypes.create_string_buffer(self.state_size)
        self._rgb4 = None      # (4,224,240,3) capture buffer when recording
        self._raw_steps = False  # hack-free stepping (lockstep video eval)

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
        self.start_cell[i] = None      # door episodes credit no cell
        self.explorer[i] = 0           # no macro-noise leak across episodes
        if (self.sr_prob > 0 and self.archive
                and self.rng.random_sample() < self.sr_prob):
            # soft least-practiced: p(cell) ~ 1/(1+uses). Uniform-ish
            # coverage bridges the door->frontier gap; a hard frontier
            # bias starves the cells where the policy actually fails.
            # A recency slice (dict order = insertion order) concentrates
            # extra practice on the newest cells so a fresh frontier gets
            # enough episodes to keep expanding.
            cells = list(self.archive.keys())
            if (self.sr_frontier_prob > 0
                    and self.rng.random_sample() < self.sr_frontier_prob):
                # backward-chaining frontier: cells PROVEN to convert
                # (1+ wins), least-practiced first. Newly-winning
                # outer-ring cells have the fewest uses so they dominate
                # draws -- the band marches outward on its own; heavily
                # consolidated inner cells fade without any graduation
                # threshold (a cap here starved the pipeline: v24).
                cand = ([c for c in cells if self.cell_wins.get(c, 0) >= 1]
                        or cells)
                cand = sorted(cand,
                              key=lambda c: self.archive[c][1]
                              )[:self.sr_frontier_k]
                cell = cand[self.rng.randint(len(cand))]
            else:
                w = np.array([1.0 / (1 + self.archive[c][1]) for c in cells])
                cell = cells[self.rng.choice(len(cells), p=w / w.sum())]
            ent = self.archive[cell]
            ent[1] += 1
            # a cell may hold several state variants (different enemy/RNG
            # phases); sampling among them exposes the policy to the full
            # local distribution instead of one replayed setup
            states = ent[0] if isinstance(ent[0], list) else [ent[0]]
            self.lib.benv_load(self.env, i,
                               states[self.rng.randint(len(states))])
            self.start_stage[i] = cell[0]
            self.was_restart[i] = True
            self.start_cell[i] = cell
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
        self.ep_steps[i] = 0

    def _post_reset_init(self, idx, ram):
        """Re-init per-env python state for envs in idx from fresh RAM."""
        for i in idx:
            r = ram[i]
            x = int(r[0x6D]) * 256 + int(r[0x86])
            gp = min(max(int(r[0x75F]) * 4 + int(r[0x75C]), 0), 31)
            self.x_last[i] = x; self.prev_x[i] = 0; self.max_x[i] = 0
            self.hw[i] = x; self.x_pending[i] = x
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
            self.idle_paid[i] = 0.0
            self.nongame[i] = 0
            self.prev_area[i] = int(r[0x760])
            self.prev_swim[i] = int(r[0x704])
            self.visited[i] = set()
            self.last_cid[i] = -1

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
            # macro-action random walk: hold each random action ~8 steps.
            # Per-step uniform noise cannot produce directed multi-step
            # maneuvers (e.g. sustained sink or approach); persistence can.
            keep = self.rng.random_sample(n) >= 1.0 / 8
            macro = np.where(keep, self.exp_action,
                             self.rng.randint(0, 12, size=n))
            self.exp_action[:] = macro
            # 60% policy / 40% macro: keep the policy's behavioral prior
            # (pure uniform flail dies to hazards before it can discover)
            use_macro = self.rng.random_sample(n) < 0.4
            acts = np.where(exp_mask & use_macro, macro, acts)
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
        if self._raw_steps and n == 1:
            # pure frames, no hacks: a reference emulator fed the same
            # actions stays in bitwise lockstep (video replay)
            self.lib.benv_step_raw(self.env, 0, int(self.actions_buf[0]),
                                   self.obs_u8.ctypes.data,
                                   self.ram.ctypes.data)
        elif self._rgb4 is not None and n == 1:
            # eval recording: capture all 4 emulated frames (no aliasing)
            self.lib.benv_step_rgb4(self.env, 0, int(self.actions_buf[0]),
                                    self.obs_u8.ctypes.data,
                                    self.ram.ctypes.data,
                                    self._rgb4.ctypes.data)
        else:
            self.lib.benv_step(self.env, self.actions_buf.ctypes.data,
                               self.obs_u8.ctypes.data, self.ram.ctypes.data)
        ram = self.ram

        x = self._x(); t = self._time(); gp = self._gp()
        # transition frames can leave garbage in the x page byte. A hard
        # x>4000 cutoff is WRONG (8-4's post-water corridor runs to ~4830
        # in the same coordinate frame); instead debounce: accept a
        # teleport-scale jump only when it persists two consecutive steps.
        raw = x
        jump = np.abs(raw - self.x_last) > 600
        confirm = np.abs(raw - self.x_pending) <= 64
        held = jump & ~confirm          # x carried this step, decided next
        x = np.where(held, self.x_last, raw)
        self.x_pending = raw
        life = self._field(0x75A); area = self._field(0x760)
        # on a held step the area/context bookkeeping is deferred too, so
        # the rebase and the loop-penalty exemption land on the CONFIRMED
        # step with the true x (otherwise pipe exits paid -30 and hw
        # rebased to the stale pre-teleport x)
        area_b = np.where(held, self.prev_area, area)
        # death = life decrement (0 is the last playable life; 0xFF = game
        # over). The C++ kill-dying hack skips the dying frames inside the
        # step, so pstate can never be relied on for the death penalty.
        died = (life == 0xFF) | (life < self.lives)
        pstate = self._field(0x0E); yvp = self._field(0xB5)
        dying = (pstate == 0x0B) | (yvp > 1)
        dead = pstate == 0x06
        flag = self._flag()
        gmode = self._field(0x770)
        if self.single_stage:
            victory = np.zeros(n, dtype=bool)
        else:
            victory = (gmode == 2) & (gp == 31)

        # ---- backward x jumps: legit transition vs loop ----
        # A scripted transition (pipe/vine/entrance) seen by the C++ skip
        # loop explains a backward jump (8-4 pipe 3 -> water: same area
        # byte, x frame restarts at 0). Carried across a held step so it
        # lands on the CONFIRMED step with the jump. A backward jump with no
        # transition is a maze loop teleport: penalised and (in training)
        # terminal, so no state ever earns different rewards for the same
        # forward run depending on invisible history.
        # Cycle rule (frame = level, area byte, swim flag): SMB never scrolls
        # left, so a backward jump INSIDE one frame is never progress --
        # whether it came from a wrong pipe (scripted) or a maze teleport
        # (instant). A jump that changes frame is a transition (8-4 pipe 3
        # -> water: same area byte, swim 0->1) UNLESS it lands in a cell
        # this life already visited (re-entering a bonus room = cycle).
        swim = ram[:, 0x704].astype(np.int32)
        swim_b = np.where(held, self.prev_swim, swim)
        jump_c = (np.abs(x - self.x_last) > 96) & ~held & ~died & ~flag & \
                 (gp == self.progress)
        same_frame = (area_b == self.prev_area) & (swim_b == self.prev_swim)
        cid = ((gp.astype(np.int64) * 8 + area) * 2 + swim) * 64 + x // 128
        revisit = np.zeros(n, dtype=bool)
        for i in np.nonzero(jump_c & ~same_frame)[0]:
            revisit[i] = int(cid[i]) in self.visited[i]
        loop = jump_c & ((same_frame & (x < self.x_last)) | revisit) & \
               (self.loop_penalty > 0)
        legit = jump_c & ~loop

        # ---- base reward (block granularity) ----
        if self.x_reward == 'highwater':
            ctx_change = (life != self.lives) | (area_b != self.prev_area) | \
                         (swim_b != self.prev_swim) | (gp != self.progress) | \
                         legit
            self.hw = np.where(ctx_change, x, self.hw)
            self.max_x = np.where(legit, 0, self.max_x)
            r_x = np.clip(x - self.hw, 0, 20).astype(np.float32)
            self.hw = np.maximum(self.hw, x)
        else:
            dx = x - self.x_last
            r_x = np.where(np.abs(dx) > 24, 0, dx).astype(np.float32)
        r_t = np.minimum(t - self.time_last, 0).astype(np.float32)
        reward = np.clip(r_x + r_t, -15, 20) - \
            np.where(died, self.fail_penalty, 0.0).astype(np.float32)
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
        # off-route level entry: no bonus, a penalty, and terminal (below)
        if self.route_gps is not None:
            off = ok & ~np.isin(gp, self.route_gps)
        else:
            off = np.zeros(n, dtype=bool)
        if ok.any():
            good = ok & ~off
            if not self.single_stage:
                reward = reward + np.where(good, self.stage_bonus * delta, 0)
                reward = reward - np.where(off, self.offroute_penalty, 0)
            self.cleared += good.astype(np.int32)
            self.warped |= good & (delta >= 2)
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
        reward = reward - np.where(loop, self.loop_penalty, 0)
        self.looped |= loop
        area_changed = area_b != self.prev_area
        self.prev_area = np.where(held, self.prev_area, area)
        self.prev_swim = np.where(held, self.prev_swim, swim)
        # mark the (confirmed) current cell as visited this life
        for i in np.nonzero(~held & (cid != self.last_cid))[0]:
            self.visited[i].add(int(cid[i]))
            self.last_cid[i] = cid[i]

        fwd = xd > 0
        new_ground = x > self.max_x
        grow = np.minimum(xd, 20) * self.progress_reward * x
        reward = reward + np.where(fwd & new_ground, grow, 0)
        reward = reward - np.where(fwd & ~new_ground,
                                   self.backtrack_penalty, 0)
        self.idle = np.where(fwd, 0, self.idle + 1)
        # idle drip (config may zero it) capped at one death's worth
        idle_hit = (self.idle > self.idle_threshold) & (self.idle_paid < 15.0)
        reward = reward - np.where(idle_hit, self.idle_penalty, 0)
        self.idle_paid += np.where(idle_hit, self.idle_penalty, 0)
        # idle timeout: camping ends the episode at death cost. A capped
        # drip alone made stalling FREE once the cap was paid (policies
        # learned to park); unbounded drip made dying cheaper than trying.
        # Equal terminal cost removes both attractors.
        idle_to = self.idle >= self.idle_timeout
        reward = reward - np.where(idle_to, self.fail_penalty, 0)
        self.prev_x = x
        self.max_x = np.maximum(self.max_x, x)

        # ---- novelty (python sets; ~N ops/step) ----
        if self.novelty_bonus > 0:
            # annealing: counts decay so often-visited cells slowly regain
            # pull -- keeps a standing gradient toward the least-visited
            # corners of a stuck frontier instead of a fully depleted map
            self._novelty_tick += 1
            if self.novelty_global and self._novelty_tick % 3000 == 0:
                self.novelty_counts = {k: v * 0.5
                                       for k, v in self.novelty_counts.items()
                                       if v * 0.5 >= 0.1}
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
            ypix_a = self._field(0x3B8)
            swim_a = self.ram[:, 0x704].astype(np.int32)
            # grounded on land; swimming counts as controlled in water
            # (float_state never returns to 0 while afloat). Timer floor
            # stays low: chains reach deep cells with little time left,
            # and a state with ~25s is still a practiceable episode.
            can = (((fstate == 0) | (swim_a == 1)) & ~dying & ~dead
                   & (gmode == 1) & (t > 25)
                   & ~held & ~died & ~area_changed & ~jump_c)  # no phantom cells
            for i in np.nonzero(can)[0]:
                # y-band in the key: standing ON a block/pipe is a different
                # rung than the floor below it; swim flag disambiguates the
                # water zone (same area byte + low x as the level start)
                # keyed by the level Mario is IN (not the episode's start
                # stage): the same physical spot reached via 1-1 -> 1-2 or
                # from the 1-2 door is one cell. Start-stage keys held the
                # same states 3-4x over and crowded world 8 out at the cap.
                g = int(gp[i])
                cell = ('%d-%d' % (g // 4 + 1, g % 4 + 1), int(area[i]),
                        int(x[i]) // 128, int(ypix_a[i]) // 64, int(swim_a[i]))
                if cell in self.ep_cells[i]:
                    continue
                self.ep_cells[i].add(cell)
                if cell not in self.archive:
                    if len(self.archive) >= self.sr_cells:
                        # evict the OLDEST cell without wins (evicting the
                        # most-used one made practice impossible at cap)
                        losers = [c for c in self.archive
                                  if self.cell_wins.get(c, 0) == 0]
                        worst = losers[0] if losers else min(
                            self.archive, key=lambda c: self.cell_wins.get(c, 0))
                        del self.archive[worst]
                    self.lib.benv_save(self.env, int(i), self._sbuf)
                    self.archive[cell] = [[bytes(self._sbuf.raw)], 0,
                                          int(t[i])]
                    self._archive_dirty += 1
                else:
                    # refresh: grow a small reservoir of state variants
                    # per cell (different enemy/RNG phases), then rotate.
                    # Never rotate toward a more timer-doomed state unless
                    # the cell keeps killing its restarts (early deaths).
                    ent = self.archive[cell]
                    if not isinstance(ent[0], list):
                        ent[0] = [ent[0]]
                    early = self.cell_early.get(cell, 0)
                    p_ref = min(0.5, 0.05 + early / max(ent[1], 1))
                    if (self.rng.random_sample() < p_ref
                            and (len(ent[0]) < 4 or len(ent) < 3
                                 or int(t[i]) >= ent[2] or early >= 3)):
                        self.lib.benv_save(self.env, int(i), self._sbuf)
                        ent[0].append(bytes(self._sbuf.raw))
                        if len(ent[0]) > 4:
                            ent[0].pop(0)
                        if len(ent) > 2:
                            ent[2] = max(ent[2], int(t[i]))
                        self.cell_early.pop(cell, None)
                        self._archive_dirty += 1
        if (self.archive_path and self._archive_dirty >= 10):
            self._archive_dirty = 0
            self._save_archive()

        # ---- dones ----
        game_over = life == 0xFF
        # zombie guard: an env stuck outside normal gameplay (post-ending
        # screens, title/attract after a missed terminal) never comes back
        # on its own -- force a reset after 8 consecutive non-game steps
        self.nongame = np.where(gmode == 1, 0, self.nongame + 1)
        zombie = self.nongame >= 8
        # wrap guard: progress below the episode's start can only mean the
        # game rolled through the ending into a new quest -- terminal
        wrapped = gp < self.start_progress
        if self.single_stage:
            real_done = dying | dead | flag | zombie | wrapped | idle_to
        else:
            real_done = game_over | victory | zombie | wrapped | idle_to | off
        life_lost = life < self.lives
        self.lives = life

        infos = []
        # a loop ends the PPO episode like a lost life (the game itself
        # continues from the teleport point with rebased trackers)
        done_pre = real_done | ((life_lost | loop) if self.episode_life
                                else False)
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
                'idle_timeout': bool(idle_to[i]),
                'offroute': bool(off[i]),
            })

        done = done_pre

        # archive hygiene: a cell whose restarts mostly die within a few
        # steps was saved in a doomed spot (e.g. mid enemy contact) - prune
        self.ep_steps += 1
        for i in np.nonzero(done)[0]:
            cell = self.start_cell[i]
            if cell is None:
                continue
            if victory[i] or self.cleared[i] > 0:
                self.cell_wins[cell] = self.cell_wins.get(cell, 0) + 1
            if self.ep_steps[i] > 8:
                continue
            n_early = self.cell_early.get(cell, 0) + 1
            self.cell_early[cell] = n_early
            ent = self.archive.get(cell)
            if (ent is not None and n_early >= 12
                    and n_early > 0.5 * max(ent[1], 1)):
                del self.archive[cell]
                del self.cell_early[cell]
                self._archive_dirty += 1

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
            # new episode = fresh frame stack: the first observation of
            # the next life (or post-loop run) must not carry death /
            # pre-teleport frames from the episode that just ended
            for i in soft_idx:
                self._ring[i] = (self.obs_u8[i].astype(np.float32)
                                 / 255.0)[..., None]

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
            # fold win counts into entries (entry[3]) so backward-chaining
            # state survives restarts
            for c, w in self.cell_wins.items():
                e = self.archive.get(c)
                if e is not None:
                    while len(e) < 4:
                        e.append(0)
                    e[3] = w
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
        self._buf = ctypes.create_string_buffer(240 * 224 * 3)
        self.frames_per_step = self.v.skip
        self.v._rgb4 = np.zeros((self.v.skip, 224, 240, 3), dtype=np.uint8)

    @property
    def frames4(self):
        """All emulated RGB frames of the last step (60fps recording)."""
        return [self.v._rgb4[k].copy() for k in range(self.v.skip)]

    @property
    def unwrapped(self):
        return self

    @property
    def screen(self):
        self.v.lib.benv_render_rgb(self.v.env, 0, self._buf)
        return np.frombuffer(self._buf, dtype=np.uint8).reshape(224, 240, 3)

    def reset(self):
        return self.v.reset()[0]

    def step(self, action):
        obs, r, d, infos = self.v.step([int(action)])
        return obs[0], float(r[0]), bool(d[0]), infos[0]

    def close(self):
        self.v.close()


class LockstepVideoEnv:
    """Video-grade eval env with zero train/eval obs mismatch.

    The policy plays on the NATIVE core (hack-free stepping), so it sees
    exactly the training observation distribution. A stable-retro
    emulator, seeded from the same RAM, replays the identical actions in
    bitwise lockstep (property proven by native/deep_difftest.py) purely
    to render the video frames with the reference NES renderer.
    """

    def __init__(self, **kwargs):
        from mario_env import RetroMarioEnv
        kwargs.setdefault('n_threads', 1)
        kwargs.setdefault('episode_life', False)
        stages = kwargs.get('random_stages')
        self.v = MarioNativeVecEnv('lockstep', 1, dense_infos=True, **kwargs)
        self.v._raw_steps = True
        self.frames_per_step = self.v.skip
        self.r = RetroMarioEnv(random_stages=list(stages) if stages else None,
                               full_game=kwargs.get('full_game', False),
                               reset_noops=0)
        self.frames4 = []
        self._steps = 0
        self.desynced = False
        self._game = np.ones(0x800, dtype=bool)
        self._game[0x100:0x300] = False

    @property
    def unwrapped(self):
        return self

    @property
    def screen(self):
        if self.frames4:
            return self.frames4[-1]
        return self.r.screen

    def reset(self):
        self.v.reset()                    # native state: correct PPU/VRAM
        self.r.reset()                    # reference state: authoritative RAM
        ram = np.frombuffer(self.r._retro.get_ram(),
                            dtype=np.uint8)[:0x800].copy()
        self.v.lib.benv_set_ram(self.v.env, 0, ram.tobytes())
        self.v._post_reset_init([0], ram[None, :])
        self.v._fetch_obs(0)
        self.v._ring[0] = (self.v.obs_u8[0].astype(np.float32)
                           / 255.0)[..., None]
        self.frames4 = []
        self._steps = 0
        self.desynced = False
        order = [(self.v._ptr + 1 + j) % 4 for j in range(4)]
        return self.v._ring[..., order][0]

    def step(self, action):
        obs, rew, done, infos = self.v.step(np.array([int(action)]))
        m = self.r._masks[int(action)]
        fr = []
        for _ in range(self.v.skip):
            self.r._em.set_button_mask(m, 0)
            self.r._em.step()
            fr.append(self.r._retro.get_screen().copy())
        self.frames4 = fr
        self._steps += 1
        if self._steps % 32 == 0 and not self.desynced:
            rram = np.frombuffer(self.r._retro.get_ram(),
                                 dtype=np.uint8)[:0x800]
            if not np.array_equal(self.v.ram[0][self._game],
                                  rram[self._game]):
                self.desynced = True
                print(f'[lockstep] WARNING: replay desync at step '
                      f'{self._steps}')
        return obs[0], float(rew[0]), bool(done[0]), infos[0]

    def close(self):
        self.v.close()
        self.r.close()


def register_mario_native_vecenv():
    vecenv.register(
        'MARIO_NATIVE',
        lambda config_name, num_actors, **kwargs: MarioNativeVecEnv(
            config_name, num_actors, **kwargs))
