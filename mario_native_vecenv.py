"""Vectorized Mario env on the native SMB core (native/libbatchenv.so).

The C++ side steps N emulators on a threadpool (4 frames + RAM hacks +
render/pool/resize per agent step) and returns 84x84 uint8 frames plus a
2KB RAM snapshot per env. ALL game semantics -- rewards (high-water x,
positive-only rewards: first-visit progress + level clear (mario_rewards),
episodic life, victory/done, stage sampling, self-restarts --
live here, vectorized in numpy. Reward constants match mario_env.py; the
granularity is per agent step (4 frames) instead of per frame, so numbers
are equivalent-in-expectation rather than bit-identical to the retro chain.
"""

import ctypes
import zlib
import gzip
import atexit
import os
import time

import numpy as np
from gymnasium import spaces

from rl_games.common import vecenv
from rl_games.common.ivecenv import IVecEnv
from types import SimpleNamespace

from mario_rewards import Signals, RewardSet, FirstVisitProgress

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, 'native', 'libbatchenv.so')
ROM = os.path.join(HERE, 'retro_integration', 'SuperMarioBros-Nes-v0',
                   'rom.nes')
STATE_DIR = os.path.join(HERE, 'native', 'states')

FRAME_STACK = 4
# RAM feature layout: 12x13 tile grid (2 cols behind Mario, 9 ahead; all 13
# playfield rows) + 5 enemy slots x 4 + 12 Mario/game scalars

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
            if not os.path.exists(LIB):
                raise FileNotFoundError(
                    f'{LIB} is missing: build it with native/build.sh')
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
            lib.benv_step_raw_rgb4.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                               ctypes.c_int, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_void_p]
            lib.benv_step_raw.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_void_p]
            lib.benv_set_ram.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_char_p]
            lib.benv_set_skip.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.benv_transit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.benv_ram.restype = ctypes.POINTER(ctypes.c_uint8)
            lib.benv_ram.argtypes = [ctypes.c_void_p, ctypes.c_int]
            cls._inst = lib
        return cls._inst


class Infos(list):
    """Per-env info dicts (list) that also answers rl_games' vectorised
    `'time_outs' in infos` / `infos['time_outs']` (value bootstrap at our
    unpaid-steps cutoff, which is not a game terminal)."""
    time_outs = None

    def __contains__(self, k):
        if k == 'time_outs':
            return self.time_outs is not None
        return list.__contains__(self, k)

    def __getitem__(self, k):
        if k == 'time_outs':
            return self.time_outs
        return list.__getitem__(self, k)


class MarioNativeVecEnv(IVecEnv):
    """rl_games IVecEnv over the native batched core."""

    REMOVED_KWARGS = ('idle_penalty', 'idle_threshold', 'idle_timeout',
                      'progress_reward', 'x_reward', 'loop_penalty',
                      'loop_terminal', 'backtrack_penalty', 'novelty_bonus',
                      'novelty_y_band', 'novelty_global', 'score_reward',
                      'fail_penalty', 'offroute_penalty')

    def __init__(self, config_name, num_actors, name='SuperMarioBros-v0',
                 action_type='complex', episode_life=True, stage_bonus=500.0,
                 skip=4, sticky_actions=0.0, random_stages=None,
                 full_game=False, reset_noops=0,
                 self_restart_prob=0.0, self_restart_cells=96,
                 n_threads=32, seed=None, dense_infos=False,
                 explore_eps=0.0, archive_path=None, explore_episode_prob=0.0,
                 explore_episode_steps=150, self_restart_frontier_prob=0.0,
                 self_restart_frontier_k=16, unpaid_timeout=250,
                 page_reset_grace=60, page_reset_px=600,
                 reward=None, play_mode=False,
                 route_levels=None, cell_tiles=False,
                 frontier_predecessors=0, cell_y_band=64, explore_pure=False,
                 credit_vertical=False, cell_max_variants=0, cell_bonus=0.0, cell_bonus_relative=True, cell_x_bin=128,
                 frontier_per_level=False, explore_fresh_uses=0, cell_screen_bin=0,
                 explorer_envs=0, end_on_stage_exit=False, archive_save_secs=60.0,
                 cell_bonus_door_only=False, life_loss_reset=True, **unknown):
        assert action_type == 'complex'
        gone = [k for k in unknown if k in self.REMOVED_KWARGS]
        if gone:
            print('[env] ignoring removed reward knobs: %s (rewards are '
                  'positive-only since 2026-09-05, see mario_rewards.py)'
                  % ', '.join(gone))
        # invisible explorers (Go-Explore phase 1 outside the learner): extra
        # cores that random-walk from archive cells purely to grow the
        # archive. They step in the same batch, but their obs/rewards/dones
        # never reach the trainer. A walk inside a TRAINING env substitutes
        # random actions after rl_games stored the policy's own action and
        # log-prob, so PPO learned from actions that were never played.
        # Needs an archive (self_restart_prob > 0); never in play mode.
        self.n_train = int(num_actors)
        self.n_explorers = (max(0, int(explorer_envs))
                            if (self_restart_prob > 0 and not play_mode) else 0)
        n = self.num_actors = self.n_train + self.n_explorers
        self.is_explorer_env = np.arange(n) >= self.n_train
        self.explore_walks = {}       # cell -> explorer walks started there
        # wins of explorer walks started from a cell: existence proofs that
        # make it a frontier candidate, never mixed into the policy's win rate
        self.explore_wins = {}
        self.n_walks = 0
        self.lib = _Lib()
        rom = open(ROM, 'rb').read()
        self.single_stage = (not full_game) and (random_stages is not None)
        # the C++ pool with 0 threads returns stale obs without stepping;
        # more threads than cores only adds wakeups (configs say 24, a Mac
        # has 12); results are bitwise identical for any thread count
        n_threads = max(1, min(int(n_threads), os.cpu_count() or 1))
        self.env = self.lib.benv_create(rom, len(rom), n, n_threads,
                                        int(self.single_stage))
        # C clamps to >= 2; keep Python in step or the rgb4 buffer is short
        self.skip = max(2, int(skip))
        if self.skip != 4:
            self.lib.benv_set_skip(self.env, self.skip)
        self.state_size = self.lib.benv_state_size()

        self.stages = list(random_stages) if random_stages else ['FullGame']
        self.states = {s: _load_state(s) for s in self.stages}
        # on-route set (full-game training on a level SET): a confirmed move
        # into a level outside it is a wrong exit -- the episode ends, reward
        # 0. Explicit route_levels lets a tool start on ONE level without
        # turning the rest of the route into dead ends.
        route = route_levels or random_stages
        self._route_levels = list(route) if route else None
        self._full_game = bool(full_game)
        self.route_gps = None
        if full_game and route:
            self.route_gps = np.array(sorted(
                (int(s.split('-')[0]) - 1) * 4 + int(s.split('-')[1]) - 1
                for s in route), dtype=np.int32)
        # the TRAINED levels: the archive only saves states there (a 4-2-only
        # run with the full route archived and practised 8-x states after
        # the warp), and end_on_stage_exit ends an episode that leaves them
        # by a paid route exit instead of playing on into the next level
        self.train_gps = None
        if random_stages:
            self.train_gps = np.array(sorted(
                (int(s.split('-')[0]) - 1) * 4 + int(s.split('-')[1]) - 1
                for s in random_stages), dtype=np.int32)
        self.end_on_stage_exit = bool(end_on_stage_exit)
        self.stage_weights = None
        self.episode_life = episode_life
        self.stage_bonus = stage_bonus
        self.sticky = sticky_actions
        self.reset_noops = reset_noops
        self.sr_prob = self_restart_prob
        self.sr_cells = self_restart_cells
        self.dense_infos = dense_infos
        self.explore_eps = explore_eps
        self.archive_path = archive_path
        self._archive_dirty = 0
        self._archive_saved_at = 0.0
        self.archive_save_secs = float(archive_save_secs)
        self.exp_ep_prob = explore_episode_prob
        self.exp_ep_steps = explore_episode_steps
        self.sr_frontier_prob = self_restart_frontier_prob
        self.sr_frontier_k = self_restart_frontier_k
        # X contiguous steps without a positive reward end the episode
        # (reported to rl_games as a time-out so the critic bootstraps: it
        # is our cutoff, not part of the game)
        self.unpaid_timeout = int(unpaid_timeout)
        # the game's page-counter reset (x falls far below the frame's
        # highwater within the same frame): not terminal, highwater kept,
        # and only `page_reset_grace` unpaid steps remain to reach paid
        # ground (pipe 1: climb in, ~35 steps) before a dead-end cutoff
        self.page_reset_grace = int(page_reset_grace)
        self.page_reset_px = int(page_reset_px)
        # archive cells also keyed by the level tiles of the current x-bin
        self.cell_tiles = bool(cell_tiles)
        # frontier practice pool also includes never-won cells up to this
        # many x-bins before a winning cell (0 = winners only, legacy)
        self.frontier_pred = int(frontier_predecessors)
        # frontier restarts pick the LEVEL first (curriculum weights), then a
        # cell of that level: in one global pool the 16 hardest cells of the
        # biggest levels took every frontier draw and 1-2 / 4-2 got none
        self.frontier_per_level = bool(frontier_per_level)
        # Go-Explore's "explore from new cells": a restart from a cell used
        # fewer than this many times becomes a random-walk explorer episode
        # with probability 1 - uses/k (never below explore_episode_prob).
        # The 4-2 vine cell (on top of the revealed block) had ~1 restart per
        # 50 epochs and only 5% of those were walks: UP never met the vine.
        self.explore_fresh_uses = int(explore_fresh_uses)
        # the camera never scrolls back in SMB, so the screen's left edge
        # bounds what a state can still reach: with it in the cell key (px
        # bin; 0 = off), "next to the hidden block with the block on screen"
        # and "next to it with the block scrolled off" are different cells
        # (4-2: every archived state near the vine block had the camera past
        # it, so no restart could ever bump it)
        self.cell_screen_bin = int(cell_screen_bin)
        # vertical resolution of archive cells (px): 32 separates standing on
        # a block (ypix 112) from standing on the pipe top above it (64)
        self.cell_y_band = int(cell_y_band)
        # explore episodes: pure random with mixed persistence (True) or the
        # legacy 60% policy / 40% 8-step macro mix
        self.explore_pure = bool(explore_pure)
        # transitive credit also for climbing / revealing within one x-bin
        self.credit_vertical = bool(credit_vertical)
        # cap on tile-signature variants per spatial cell (0 = unlimited):
        # 3 covers a hidden block (hidden / bumped / revealed); without a cap
        # every broken-brick pattern in an underground level is a new cell
        # (1-2 grew 666 cells and swallowed the prompt pool)
        self.cell_max_variants = int(cell_max_variants)
        # archive x-bin in px (multiple of 16). 128 could not tell pipe 4 from
        # pipe 3 in the 1-2 warp zone (48 px apart): same cell, so neither the
        # novelty bonus nor the prompts could single the route exit out
        self.cell_x_bin = int(cell_x_bin); assert self.cell_x_bin % 16 == 0
        # novelty bonus: +cell_bonus the first time a life enters a grounded
        # archive cell (the block top and pipe top of a climb pay nothing
        # in x-progress terms; the same cell key the demos chain on)
        self.cell_bonus = float(cell_bonus)
        # relative novelty: pay only for cells the DOOR episodes do not reach
        # (a per-cell count of door-episode entries, decayed every 32 batch
        # steps); a bonus paid for every first visit rewarded the runners as
        # much as the climbers
        self.cell_bonus_relative = bool(cell_bonus_relative)
        # pay the novelty bonus to door episodes only: restarts inside an area
        # the door episodes never reach (4-2's warp area) collected +100 per new
        # cell every life (~856 per episode) and wandered instead of warping
        # (8% warp from warp-area winners, half the episodes still wandering
        # after 700 steps, Mario_PPO42i ep 3500)
        self.cell_bonus_door_only = bool(cell_bonus_door_only)
        # per-life training: every life loss ends the GAME too and the next
        # episode is a fresh draw (door or archive restart). The continued
        # life used to be labelled a door episode although it respawns at the
        # level's halfway point after a death past it: in 4-2 that is x 1576,
        # behind the vine and the coin-cache pipe (the camera never scrolls
        # back), so no such life can clear -- 43% of run j's training steps
        # were continued lives, 27% respawned at 1576 (clear rate 0). Lives
        # are invisible to the policy (HUD cropped) anyway. The multi-life
        # eval (episode_life False) and play mode keep playing on.
        self.life_loss_reset = bool(life_loss_reset)
        # a continued life (life_loss_reset off) is neither a door episode
        # nor a restart: it credits no cell and is not a door sample
        self.continuation = np.zeros(n, dtype=bool)
        self.door_seen = {}; self._seen_tick = 0
        # GRPO rollouts freeze the door counts (no decay, no growth) so the
        # bonus is the same function of the state for every rollout of a
        # group whenever it runs in the horizon; entries seen meanwhile
        # are merged when the trainer unfreezes
        self.door_seen_frozen = False; self._door_seen_pending = {}
        self.is_door = np.ones(n, dtype=bool)
        self.exp_persist = np.ones(n, dtype=np.int64)
        self.explorer = np.zeros(n, dtype=np.int32)
        self.exp_action = np.zeros(n, dtype=np.int64)
        self.ep_steps = np.zeros(n, dtype=np.int32)
        self.start_cell = [None] * n
        self.cell_early = {}
        self.cell_wins = {}
        self.cell_tries = {}      # policy restarts per cell (lifetime; persisted)
        self.nongame = np.zeros(n, dtype=np.int32)

        self.rng = np.random.RandomState(seed)
        self.obs_u8 = np.zeros((n, 84, 84), dtype=np.uint8)
        self.ram = np.zeros((n, 0x800), dtype=np.uint8)
        self.actions_buf = np.zeros(n, dtype=np.int32)

        self.observation_space = spaces.Box(0.0, 1.0, (84, 84, FRAME_STACK),
                                            np.float32)
        self.action_space = spaces.Discrete(12)
        self._ring = np.zeros((n, 84, 84, FRAME_STACK), dtype=np.float32)
        self._ptr = 0
        # optional uint8 twin of the ring for trainers that want raw bytes
        # (grpo/): stacked uint8 obs without any float conversion
        self.u8_obs = False
        self._ring_u8 = None

        # per-env python-side state
        z = lambda dt=np.int32: np.zeros(n, dtype=dt)
        self.x_last = z(np.int64); self.x_pending = z(np.int64)
        self.time_last = z()
        self.lives = z(); self.prev_flag = z(bool)
        self.progress = z(); self.pending = np.full(n, -1, np.int32)
        self.start_progress = z(); self.cleared = z()
        self.warped = z(bool); self.vic_paid = z(bool)
        self.max_x = z(np.int64); self.last_action = z()
        self.prev_frame = z(np.int64)
        self.unpaid = z(); self.max_gap = z(); self.page_resets = z()
        self.after_reset = z(bool)
        self.forced_timeup = z()      # eval: cutoffs turned into time-ups
        self.last_stuck = [None] * n   # eval: (level, x//16) of the last forced time-up
        self.hold_on_done = False     # video: never reset after done
        self.entered_cell = [None] * n  # cell first entered this step (grpo demos)
        self.prev_in_play = np.ones(n, dtype=bool)
        self.pending_life = z(bool); self.pending_life_at_resume = z(bool)
        self.start_stage = [''] * n
        self.was_restart = z(bool)
        self.prev_score = np.zeros(n, dtype=np.int64)
        # reward terms (mario_rewards): positive-only set from config, else
        # the default (first-visit progress + level clear at stage_bonus)
        specs = reward or [
            {'type': 'first_visit_progress', 'cap': 20},
            {'type': 'level_clear', 'base': stage_bonus, 'per_extra': 100}]
        self.reward_specs = specs
        self.rewards = RewardSet(n, specs)
        self._prog = self.rewards.get(FirstVisitProgress)
        self.last_terms = {}
        self.last_signals = None
        self.last_leaving = np.zeros(n, dtype=bool)
        # play_mode (tools/play.py): cutoffs and wrong exits are flagged but
        # never reset the game, so a human can inspect what follows; the
        # highwater is kept, exactly as in training minus the terminal
        self.play_mode = bool(play_mode)
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
            self.cell_tries = {c: e[4] for c, e in self.archive.items()
                               if len(e) > 4}
            self.explore_wins = {c: e[5] for c, e in self.archive.items()
                                 if len(e) > 5 and e[5]}
        self.ep_cells = [set() for _ in range(n)]
        self._sbuf = ctypes.create_string_buffer(self.state_size)
        if archive_path:
            # rl_games never closes its vec env: without this the archive
            # grown since the last throttled save was lost on every exit
            atexit.register(self._save_archive)
        self._rgb4 = None      # (4,224,240,3) capture buffer when recording
        self._raw_steps = False  # hack-free stepping (video clips, win searches)

    # ------------------------------------------------------------- helpers
    def load_state(self, i, state):
        """Load a savestate into env i. The C side memcpy's sizeof(Core)
        from the buffer unchecked, so a state of another layout / a
        truncated pickle would silently corrupt the core."""
        if len(state) != self.state_size:
            raise ValueError('savestate of %d bytes, core expects %d'
                             % (len(state), self.state_size))
        self.lib.benv_load(self.env, int(i), state)

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
        self.was_restart[i] = False; self.is_door[i] = True
        self.continuation[i] = False
        self.start_cell[i] = None      # door episodes credit no cell
        self.explorer[i] = 0           # no macro-noise leak across episodes
        self.forced_timeup[i] = 0; self.last_stuck[i] = None
        if self.is_explorer_env[i]:
            self._reset_explorer(i)
            self.ep_cells[i] = set()
            self.rewards.reset([i], None, hard=True)
            self.ep_steps[i] = 0
            return
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
                if self.frontier_per_level:
                    by = {}
                    for c in cells:
                        by.setdefault(c[0], []).append(c)
                    levels = [l for l in self.stages if l in by]
                    if levels:
                        if self.stage_weights is not None:
                            w = np.array([self.stage_weights[self.stages.index(l)]
                                          for l in levels])
                            w = w / w.sum() if w.sum() > 0 else None
                        else:
                            w = None
                        cells = by[levels[self.rng.choice(len(levels), p=w)]]
                # backward-chaining frontier: cells PROVEN to convert (1+
                # wins), plus -- with frontier_predecessors -- the never-won
                # cells right behind a winner (same frame, up to that many
                # x-bins before). A never-winning cell with thousands of
                # tries used to out-weigh every winner (weight ~1.05 vs
                # ~0.99), so the draw fed the dead ends: 4-2's end section
                # took 25-60k restarts, the 1-2 warp-zone winners ~50 each.
                winners = [c for c in cells if self._won(c)]
                pool = set(winners)
                if self.frontier_pred > 0:
                    for w in winners:
                        for c in cells:
                            if (c[0] == w[0] and c[1] == w[1] and c[4] == w[4]
                                    and c[5] == w[5]
                                    and w[2] - self.frontier_pred <= c[2] <= w[2]):
                                pool.add(c)
                if pool:
                    cells = list(pool)
                    # practice where it is learnable: weight by p (1 - p)
                    # (_frontier_weight), concentrated on the k best cells
                    w = np.array([self._frontier_weight(c) + 0.01 for c in cells])
                    k = self.sr_frontier_k
                    if k and 0 < k < len(cells):
                        top = np.argpartition(-w, k - 1)[:k]
                        cells = [cells[j] for j in top]; w = w[top]
                else:
                    # nothing proven here yet: least-practised coverage
                    w = np.array([1.0 / (1 + self.archive[c][1]) for c in cells])
                cell = cells[self.rng.choice(len(cells), p=w / w.sum())]
            else:
                w = np.array([1.0 / (1 + self.archive[c][1]) for c in cells])
                cell = cells[self.rng.choice(len(cells), p=w / w.sum())]
            ent = self.archive[cell]
            ent[1] += 1
            self.cell_tries[cell] = self.cell_tries.get(cell, 0) + 1
            # a cell may hold several state variants (different enemy/RNG
            # phases); sampling among them exposes the policy to the full
            # local distribution instead of one replayed setup
            states = ent[0] if isinstance(ent[0], list) else [ent[0]]
            self.load_state(i, states[self.rng.randint(len(states))])
            self.start_stage[i] = cell[0]
            self.was_restart[i] = True; self.is_door[i] = False
            self.start_cell[i] = cell
            # Go-Explore phase 1: some restart episodes flail randomly to
            # EXPAND the archive past what the policy can reach; a fresh
            # cell (few uses) is walked from almost every time it is drawn
            p_exp = self.exp_ep_prob
            if self.explore_fresh_uses > 0:
                p_exp = max(p_exp, 1.0 - (ent[1] - 1) / self.explore_fresh_uses)
            # with invisible explorers the walks happen there, never in a
            # training env (whose stored actions would not be the played ones)
            if self.n_explorers == 0 and self.rng.random_sample() < p_exp:
                self.explorer[i] = self.exp_ep_steps
        else:
            if self.stage_weights is not None:
                s = self.stages[self.rng.choice(len(self.stages),
                                                p=self.stage_weights)]
            else:
                s = self.stages[self.rng.randint(len(self.stages))]
            self.load_state(i, self.states[s])
            self.start_stage[i] = s
            if self.reset_noops:
                self.lib.benv_frames(self.env, i,
                                     int(self.rng.randint(
                                         0, self.reset_noops + 1)), 0)
        self.ep_cells[i] = set()
        self.rewards.reset([i], None, hard=True)
        self.ep_steps[i] = 0

    def _forget_cell(self, cell):
        """Remove a cell and all of its counters (a cell rediscovered later
        must start unproven; stale wins made it satisfy transitive credit)."""
        self.archive.pop(cell, None)
        self.cell_wins.pop(cell, None)
        self.cell_tries.pop(cell, None)
        self.cell_early.pop(cell, None)
        self.explore_walks.pop(cell, None)
        self.explore_wins.pop(cell, None)

    def _won(self, cell):
        """A cell proven to convert: a policy restart or an explorer walk from
        it reached a level advance (or a deeper winning cell)."""
        return self.cell_wins.get(cell, 0) > 0 or self.explore_wins.get(cell, 0) > 0

    def _frontier_weight(self, cell):
        """Practice weight of a winning cell: its learnability p (1 - p) for
        the policy, p = (wins + 1) / (tries + 2) over its policy restarts
        (lifetime counts). Highest where the policy converts about half the
        time; a winner it never tried (proven by explorer walks only) gets
        p = 1/2, the top weight. The failure rate 1 - p used before grew with
        every failed try, so the top-k frontier locked onto cells the policy
        never converts: in run j 16 cells took 59% of all restarts on
        winners at a 4.5% policy win rate (one had 884 tries, 0 policy wins)
        while the median winner got 3 tries."""
        t = self.cell_tries.get(cell, 0)
        pw = self.cell_wins.get(cell, 0)
        p = (pw + 1.0) / (max(t, pw) + 2.0)
        return p * (1.0 - p)

    def _reset_explorer(self, i):
        """Invisible explorer episode: a random walk from the least-walked
        archive cell (from a trained level's door while the archive is
        empty). Walks count neither as policy restarts (uses / tries, the
        practice weights) nor as door episodes (relative novelty counts);
        a walk that reaches a winning cell still credits its start cell."""
        self.is_door[i] = False
        self.explorer[i] = self.exp_ep_steps
        self.n_walks += 1
        if not self.archive:
            s = self.stages[self.rng.randint(len(self.stages))]
            self.load_state(i, self.states[s])
            self.start_stage[i] = s
            return
        cells = list(self.archive.keys())
        walks = np.array([self.explore_walks.get(c, 0) for c in cells])
        if self.explore_fresh_uses > 0 and (walks < self.explore_fresh_uses).any():
            # fresh cells first: every new cell gets its k walks (a link
            # found by 10% of walks is found with ~95% at k = 30) before
            # worn-out cells are walked again
            keep = np.nonzero(walks < self.explore_fresh_uses)[0]
            cells = [cells[j] for j in keep]; walks = walks[keep]
        w = 1.0 / (1.0 + walks)
        cell = cells[self.rng.choice(len(cells), p=w / w.sum())]
        self.explore_walks[cell] = self.explore_walks.get(cell, 0) + 1
        ent = self.archive[cell]
        states = ent[0] if isinstance(ent[0], list) else [ent[0]]
        self.load_state(i, states[self.rng.randint(len(states))])
        self.start_stage[i] = cell[0]
        self.was_restart[i] = True
        self.start_cell[i] = cell

    @staticmethod
    def _in_play_of(pstate, yvp):
        """Mario under player control: not a transition / intermission state
        ($0E 0-5, 7), not dying ($0B), dead ($06) or below the screen."""
        pstate = np.asarray(pstate); yvp = np.asarray(yvp)
        return ~((pstate <= 5) | (pstate == 7) | (pstate == 0x0B)
                 | (pstate == 0x06) | (yvp > 1))

    @staticmethod
    def _frame_of(gp, area, sub, atype, swim):
        """Frame id: the coordinate system x lives in. A level's sections
        that share it are monotone in x; anything else (pipe to a new
        section, vine to a bonus area, water) starts a new frame. `sub` is
        the sub-area ($074F, the area's offset in its type's table): $0760
        is one value for a whole level, so a pipe into a room of the same
        AreaType (4-2's coin room) used to be a same-frame teleport (paid
        +20 after a hold, and the room's ground raised the main area's
        highwater), and 4-2's flag area shared a frame with its warp area."""
        return ((((np.asarray(gp, dtype=np.int64) * 256 + area) * 256 + sub)
                 * 8 + atype) * 2 + swim)

    def _tile_sig(self, i, x):
        """Signature of the level geometry in Mario's current 128-px bin
        (8 metatile columns x 13 rows of the game's $0500 buffer). Changes
        exactly when a block is revealed / broken in that bin; ignores
        enemies and animation. Part of the archive cell key so a revealed
        hidden block is its own cell (Go-Explore: the cell representation
        must see the state that matters, or it can never be practised)."""
        return int(zlib.crc32(self._tile_grid(i, x).tobytes()) & 0xFFFF)

    def _tile_grid(self, i, x):
        """Metatiles of Mario's current x-bin: (bin columns x 13 rows) of the
        game's $0500 buffer."""
        ncol = self.cell_x_bin // 16
        col0 = (int(x) // self.cell_x_bin) * ncol
        cx = (col0 + np.arange(ncol)) * 16
        base = 0x500 + ((cx // 256) % 2) * 0xD0 + (cx % 256) // 16
        idx = base[:, None] + (np.arange(13) * 16)[None, :]
        return self.ram[i][idx]

    def cell_of(self, i):
        """Archive cell key of env i's CURRENT state (same composition as the
        archive save), regardless of whether it would be saved now."""
        r = self.ram[i]; x = int(r[0x6D]) * 256 + int(r[0x86])
        gp = min(max(int(r[0x75F]) * 4 + int(r[0x75C]), 0), 31)
        cell = ('%d-%d' % (gp // 4 + 1, gp % 4 + 1), self._area_key(r[0x760], r[0x74F]), x // self.cell_x_bin,
                int(r[0x3B8]) // self.cell_y_band, int(r[0x704]), int(r[0x74E]),
                self._tile_sig(i, x) if self.cell_tiles else 0)
        if self.cell_screen_bin > 0:
            cell = cell + ((int(r[0x71A]) * 256 + int(r[0x71C])) // self.cell_screen_bin,)
        return cell

    @staticmethod
    def _area_key(area, sub):
        """Archive key slot 1: area ($0760) and sub-area ($074F), so a room
        of the same AreaType is not a variant of the main area's spot."""
        return int(area) * 256 + int(sub)

    def _tile_variants(self, cell):
        """Archived tile-signature variants of `cell`'s spot: every key slot
        equal except the signature (slot 6); the camera bin (slot 7, with
        cell_screen_bin) belongs to the spot."""
        return sum(1 for c in self.archive
                   if c[:6] == cell[:6] and c[7:] == cell[7:])

    def _seed_cells(self, idx):
        """A new life's visited-cell set starts with the cell it stands in:
        the start cell is not a discovery (it used to pay the novelty bonus
        on the first grounded step, at an action-dependent time)."""
        for i in idx:
            self.ep_cells[i] = {self.cell_of(i)} if self.sr_prob > 0 else set()

    def _post_reset_init(self, idx, ram):
        self._post_reset_init_core(idx, ram); self._seed_cells(idx)

    def _post_reset_init_core(self, idx, ram):
        """Re-init per-env python state for envs in idx from fresh RAM
        (new episode or new life)."""
        x0 = np.zeros(self.num_actors, dtype=np.int64)
        f0 = np.zeros(self.num_actors, dtype=np.int64)
        y0 = np.zeros(self.num_actors, dtype=np.int64)
        for i in idx:
            r = ram[i]
            x = int(r[0x6D]) * 256 + int(r[0x86])
            gp = min(max(int(r[0x75F]) * 4 + int(r[0x75C]), 0), 31)
            x0[i] = x; y0[i] = int(r[0x3B8])
            f0[i] = self._frame_of(gp, int(r[0x760]), int(r[0x74F]),
                                   int(r[0x74E]), int(r[0x704]))
            self.x_last[i] = x; self.x_pending[i] = x; self.max_x[i] = x
            self.time_last[i] = (int(r[0x7F8]) * 100 + int(r[0x7F9]) * 10
                                 + int(r[0x7FA]))
            self.lives[i] = int(r[0x75A])
            d6 = r[0x7DD:0x7E3].astype(np.int64)
            self.prev_score[i] = int((d6 * np.array(
                [100000, 10000, 1000, 100, 10, 1])).sum())
            self.prev_flag[i] = False
            self.progress[i] = gp; self.start_progress[i] = gp
            self.pending[i] = -1; self.cleared[i] = 0
            self.warped[i] = False; self.vic_paid[i] = False
            self.nongame[i] = 0
            self.prev_frame[i] = f0[i]
            self.unpaid[i] = 0; self.max_gap[i] = 0; self.page_resets[i] = 0
            self.after_reset[i] = False
            self.pending_life[i] = False; self.pending_life_at_resume[i] = False
            self.prev_in_play[i] = bool(self._in_play_of(int(r[0x0E]), int(r[0xB5])))
        self.rewards.reset(list(idx), SimpleNamespace(x=x0, frame=f0, ypix=y0),
                           hard=False)

    @property
    def hw(self):
        """Highwater of each env's current frame (for tools/video)."""
        if self._prog is None:
            return np.zeros(self.num_actors, dtype=np.int64)
        return self._prog.hw_for(self.prev_frame, self.x_last)

    # ------------------------------------------------------------- IVecEnv
    def enable_u8_obs(self):
        self.u8_obs = True
        self._ring_u8 = np.zeros((self.num_actors, 84, 84, FRAME_STACK), dtype=np.uint8)
        self._ring_u8[:] = self.obs_u8[..., None]

    def obs_u8_stack(self):
        """(n, 84, 84, 4) uint8 frame stack, oldest first (same order as _obs)."""
        order = [(self._ptr + 1 + j) % FRAME_STACK for j in range(FRAME_STACK)]
        return self._ring_u8[..., order]

    def _obs(self):
        order = [(self._ptr + 1 + j) % FRAME_STACK for j in range(FRAME_STACK)]
        return self._ring[..., order]

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
        return self._obs()[:self.n_train]

    def step(self, actions):
        n = self.num_actors
        acts = np.asarray(actions).astype(np.int64).ravel()
        if self.n_explorers:
            # explorer cores take their random walk below; pad the batch
            acts = np.concatenate([acts, np.zeros(self.n_explorers, np.int64)])
        exp_mask = self.explorer > 0
        if exp_mask.any():
            # macro-action random walk: hold each random action for a
            # while. Per-step uniform noise cannot produce directed
            # multi-step maneuvers; persistence can.
            if self.explore_pure:
                # Go-Explore phase 1 proper: pure random actions with a MIX
                # of persistence (1..8 steps) -- short hops and nudges
                # exist in this distribution, the policy+8-step-macro mix
                # never produced the precision jump onto the 8-4 block in
                # ~4000 tries from the right cell
                new = self.rng.random_sample(n) >= 1.0 / self.exp_persist
                keep = new & (self.exp_persist > 1)
                self.exp_persist = np.where(keep, self.exp_persist,
                                            2 ** self.rng.randint(0, 4, size=n))
                macro = np.where(keep, self.exp_action,
                                 self.rng.randint(0, 12, size=n))
                self.exp_action[:] = macro
                acts = np.where(exp_mask, macro, acts)
            else:
                keep = self.rng.random_sample(n) >= 1.0 / 8
                macro = np.where(keep, self.exp_action,
                                 self.rng.randint(0, 12, size=n))
                self.exp_action[:] = macro
                # 60% policy / 40% macro: keep the policy's behavioral prior
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
        if self._raw_steps and self._rgb4 is not None and n == 1:
            # hack-free + RGB capture of all emulated frames (video clips)
            self.lib.benv_step_raw_rgb4(self.env, 0, int(self.actions_buf[0]),
                                        self.obs_u8.ctypes.data,
                                        self.ram.ctypes.data,
                                        self._rgb4.ctypes.data)
        elif self._raw_steps:
            # pure frames, no hacks (pipe travel, dying, inter-life screens
            # and the ending all play out): a reference emulator fed the
            # same actions stays in bitwise lockstep. One env per call in
            # the core; loop for batches (win searches that must replay).
            for i in range(n):      # the core offsets obs/ram by i itself
                self.lib.benv_step_raw(self.env, int(i), int(self.actions_buf[i]),
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
        return self._after_step()

    def _after_step(self):
        """Score the step from the freshly fetched RAM: rewards, dones,
        archive, infos, resets."""
        n = self.num_actors
        ram = self.ram

        x_raw = self._x(); t = self._time(); gp = self._gp()
        # world byte > 7 = a glitch world (the 1-2 warp-zone drops Mario into
        # world 36, an endless water level); _gp() clips it to 31
        bad_world = self._field(0x75F) > 7
        life = self._field(0x75A); area = self._field(0x760)
        sub = self._field(0x74F)
        atype = self._field(0x74E); swim = self._field(0x704)
        ypix = self._field(0x3B8)
        frame = self._frame_of(gp, area, sub, atype, swim)
        frame_change = frame != self.prev_frame
        pstate = self._field(0x0E)
        # player control ($0E not in 0-5,7, not dying / dead / below the
        # screen). The hacked training step always ends in control; the
        # hack-free eval path (videos) exposes the dying / intermission /
        # pipe frames step by step: those score nothing (a pit fall used to
        # pay progress on the way down, the death animation new cells), and
        # the first control step afterwards re-anchors x so a respawn or
        # pipe exit is never read as a page reset.
        yvp = self._field(0xB5)
        dying = (pstate == 0x0B) | (yvp > 1)
        dead = pstate == 0x06
        in_play = self._in_play_of(pstate, yvp)
        resume = in_play & ~self.prev_in_play
        self.prev_in_play = in_play.copy()
        self.x_last = np.where(resume, x_raw, self.x_last)
        self.x_pending = np.where(resume, x_raw, self.x_pending)
        # transition frames can leave garbage in the x page byte: a
        # teleport-scale jump INSIDE a frame only counts once it persists
        # two consecutive steps (the carried x pays nothing meanwhile)
        jump = (np.abs(x_raw - self.x_last) > 600) & ~frame_change
        confirm = np.abs(x_raw - self.x_pending) <= 64
        # the first control step of a new life is scored after its re-init
        # (hack-free path): it used to pay with the ended life's highwater and
        # cells, and the respawn cell was paid again a step later
        hold = (jump & ~confirm) | ~in_play | (resume & self.pending_life_at_resume)
        x = np.where(hold, self.x_last, x_raw)
        self.x_pending = np.where(in_play, x_raw, self.x_pending)
        # death = life decrement (0 is the last playable life; 0xFF = game
        # over). The C++ kill-dying hack skips the dying frames inside the
        # step, so pstate can never be relied on for it.
        died = (life == 0xFF) | (life < self.lives)
        flag = self._flag()
        gmode = self._field(0x770)
        if self.single_stage:
            victory = np.zeros(n, dtype=bool)
        else:
            victory = (gmode == 2) & (gp == 31)

        # ---- level progress (debounced, monotonic, jump-capped) ----
        inc = gp > self.progress
        confirm_gp = inc & (gp == self.pending)
        delta = gp - self.progress
        ok = confirm_gp & (delta <= 15)
        if self.route_gps is not None:
            wrong_exit = (ok & ~np.isin(gp, self.route_gps)) | bad_world
        else:
            wrong_exit = bad_world.copy()
        good = ok & ~wrong_exit
        if ok.any():
            self.cleared += good.astype(np.int32)
            self.warped |= good & (delta >= 2)
            self.progress = np.where(ok, gp, self.progress)
        self.pending = np.where(inc, gp, -1)
        level_delta = np.where(good, delta, 0).astype(np.int32)
        vpay = victory & ~self.vic_paid
        self.vic_paid |= victory
        self.prev_flag = flag

        # ---- page reset: x far below this frame's highwater, same frame ----
        # (the game's page-counter reset: pipe 1 when approached on the
        # ground, the corridor end). Not terminal: the highwater is kept so
        # the re-run pays nothing, and the agent has `page_reset_grace`
        # unpaid steps to reach paid ground (pipe 1: climb in and land in
        # section 2, ~35 steps) before the episode ends at 0.
        hw_now = self._prog.hw_for(frame, x) if self._prog is not None \
            else x
        # the reset is the STEP the drop happens (confirmed jump), not every
        # step spent below the highwater afterwards
        page_reset = (~hold & ~frame_change & ~died
                      & (self.x_last - x > self.page_reset_px)
                      & (x < hw_now - self.page_reset_px))
        self.page_resets += page_reset.astype(np.int32)

        # ---- self-restart archiving ----
        self.entered_cell = [None] * n
        entered = np.zeros(n, dtype=bool); paid_cell = np.zeros(n, dtype=bool)
        self._seen_tick += 1
        if self._seen_tick % 32 == 0 and self.door_seen and not self.door_seen_frozen:
            for k in self.door_seen:
                self.door_seen[k] *= 0.9
        # the bonus is decided against the counts as they stood BEFORE this
        # step for every env: the loop below used to bump the count as it
        # went, so of two envs entering the same new cell in one step only
        # the lower index was paid (a systematic bias for GRPO's clones)
        seen0 = self.door_seen
        seen_inc = self._door_seen_pending if self.door_seen_frozen else self.door_seen
        seen0 = dict(seen0)
        if self.sr_prob > 0:
            fstate = self._field(0x1D)
            # grounded on land; swimming counts as controlled in water
            # (float_state never returns to 0 while afloat). Timer floor
            # stays low: chains reach deep cells with little time left,
            # and a state with ~25s is still a practiceable episode.
            can = (((fstate == 0) | (swim == 1)) & ~dying & ~dead
                   & (gmode == 1) & (t > 25)
                   & ~hold & ~died & ~frame_change & ~wrong_exit
                   & ~bad_world & ~page_reset & ~self.after_reset)
            if self.route_gps is not None:
                can &= np.isin(gp, self.route_gps)
            if self.train_gps is not None:
                can &= np.isin(gp, self.train_gps)
            for i in np.nonzero(can)[0]:
                # keyed by the level Mario is IN, area, x-bin, y-band, swim
                # and AreaType: the same physical spot is one cell whatever
                # episode reached it
                g = int(gp[i])
                sig = 0
                if self.cell_tiles:
                    grid = self._tile_grid(i, x[i])
                    if (grid == 0x23).any():
                        # a block mid-bump ($23 stands in for it for a few
                        # frames): a transient, not a place. Saved, it was a
                        # tile variant of its own and filled the variant cap
                        continue
                    sig = int(zlib.crc32(grid.tobytes()) & 0xFFFF)
                cell = ('%d-%d' % (g // 4 + 1, g % 4 + 1), self._area_key(area[i], sub[i]),
                        int(x[i]) // self.cell_x_bin, int(ypix[i]) // self.cell_y_band,
                        int(swim[i]),
                        int(atype[i]),
                        sig)
                if self.cell_screen_bin > 0:
                    cell = cell + ((int(ram[i, 0x71A]) * 256 + int(ram[i, 0x71C])) // self.cell_screen_bin,)
                if cell in self.ep_cells[i]:
                    continue
                self.ep_cells[i].add(cell)
                self.entered_cell[i] = cell; entered[i] = True
                # the cap counts TILE variants of one spot; the camera bin
                # (key slot 7) is part of the spot. Counting camera positions
                # as variants let three screen bins of the 4-2 ledge fill the
                # cap, so its revealed-block state could never be archived.
                if cell not in self.archive and self.cell_max_variants > 0 and \
                        self._tile_variants(cell) >= self.cell_max_variants:
                    continue        # yet another tile variant of a known spot
                # novelty bonus only for cells the archive keeps (variants beyond
                # the cap are not new places) and, if relative, only for cells
                # the door episodes do not reach on their own (count taken
                # before this entry, so the first discoverer is paid)
                paid_cell[i] = (((not self.cell_bonus_relative) or seen0.get(cell, 0.0) < 1.0)
                                and (self.is_door[i] or not self.cell_bonus_door_only))
                if self.is_door[i]:
                    seen_inc[cell] = seen_inc.get(cell, 0.0) + 1.0
                if cell not in self.archive:
                    if len(self.archive) >= self.sr_cells:
                        # evict the OLDEST cell no episode is practising
                        in_use = {c for c in self.start_cell if c is not None}
                        cand = [c for c in self.archive if c not in in_use]
                        losers = [c for c in cand if not self._won(c)] or cand
                        if losers:
                            self._forget_cell(losers[0])
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
        # persistence only (restart recovery, probes), throttled by time: the
        # whole archive (thousands of cells x up to 4 savestates of 45 KB)
        # used to be rewritten after every 10 dirty entries
        if (self.archive_path and self._archive_dirty >= 10
                and time.time() - self._archive_saved_at >= self.archive_save_secs):
            self._archive_dirty = 0
            self._save_archive()

        # ---- rewards (positive-only terms, mario_rewards) ----
        x_last = self.x_last.copy(); t_last = self.time_last.copy()
        sig = Signals(n=n, x=x, x_last=x_last, frame=frame,
                      frame_change=frame_change, hold=hold, t=t, died=died,
                      game_over=(life == 0xFF), level_delta=level_delta,
                      wrong_exit=wrong_exit, victory_new=vpay,
                      page_reset=page_reset,
                      timeout=np.zeros(n, dtype=bool), gp=gp, area=area,
                      atype=atype, swim=swim, ypix=ypix,
                      single_stage=self.single_stage)
        reward = self.rewards(sig)
        if self.cell_bonus > 0:
            reward = reward + self.cell_bonus * paid_cell
        # every terminal pays 0: a wrong exit's pending step (already inside
        # the wrong level: a new frame, +2 cells) and its confirm step (x
        # progress there) used to pay a little. A real clear pays on its
        # confirm step (kept); its pending step pays nothing.
        leaving = (inc & ~good) | wrong_exit
        reward = np.where(leaving, 0.0, reward).astype(np.float32)
        paid = reward > 0
        # contiguous steps without a positive reward -> cutoff. After a page
        # reset the remaining budget shrinks to the grace window and the
        # cutoff is a TRUE terminal (the post-reset state looks like fresh
        # ground to the critic; bootstrapping there would reward looping).
        self.unpaid = np.where(paid, 0, self.unpaid + 1)
        self.max_gap = np.maximum(self.max_gap, self.unpaid)
        self.after_reset = (self.after_reset | page_reset) & ~paid
        self.unpaid = np.where(
            page_reset & ~paid,
            np.maximum(self.unpaid, self.unpaid_timeout - self.page_reset_grace),
            self.unpaid)
        timeout = self.unpaid >= self.unpaid_timeout
        sig.timeout = timeout
        # the per-term breakdown of what was PAID: the raw terms of a leaving
        # step used to stay in it (index.csv totals +2 per wrong exit, GRPO's
        # --outcome-progress re-paid the leaving steps)
        self.last_terms = {k: np.where(leaving, 0.0, v).astype(np.float32)
                           for k, v in self.rewards.last.items()}
        if self.cell_bonus > 0:
            self.last_terms['cell_bonus'] = np.where(
                leaving, 0.0, self.cell_bonus * paid_cell).astype(np.float32)
        self.last_leaving = leaving
        self.last_signals = sig

        # ---- trackers ----
        # copies: _post_reset_init writes these per env on a reset, which used
        # to rewrite last_signals (.frame / .t) of the step that just ended
        self.prev_frame = frame.copy()
        self.x_last = x.copy()
        self.time_last = t.copy()
        self.max_x = np.maximum(self.max_x, x)
        self.prev_score = self._score()

        # ---- dones (every terminal pays 0) ----
        game_over = life == 0xFF
        # zombie guard: an env stuck outside normal gameplay (post-ending
        # screens, title/attract after a missed terminal) never comes back
        # on its own -- force a reset after 8 consecutive non-game steps
        self.nongame = np.where(gmode == 1, 0, self.nongame + 1)
        zombie = self.nongame >= 8
        # wrap guard: progress below the episode's start can only mean the
        # game rolled through the ending into a new quest -- terminal
        wrapped = (gp < self.start_progress) | bad_world
        if self.single_stage:
            real_done = dying | dead | flag | zombie | wrapped | timeout
        else:
            real_done = (game_over | victory | zombie | wrapped | timeout
                         | wrong_exit)
            if (self.end_on_stage_exit and self.train_gps is not None
                    and not self.play_mode):
                # a single-level run ends at its paid route exit: the 4-2
                # warp used to play on into 8-1, paying 8-1 ground
                real_done = real_done | (good & ~np.isin(gp, self.train_gps))
        if self.episode_life and self.life_loss_reset and not self.play_mode:
            real_done = real_done | (life < self.lives)
        if self.play_mode:
            # inspection: flag, keep the highwater, keep playing
            real_done = real_done & ~(timeout | wrong_exit)
        elif not self.episode_life:
            # multi-life eval (videos): a stuck life must cost ONE life, as
            # in the real game, not the whole run -- zero the game timer so
            # the game itself runs the time-up death and the next life
            # starts. (Ending the run here made a loop fatal while a death
            # was not, which inverted the eval's incentives.)
            stuck = timeout & ~game_over & ~victory & ~zombie & ~wrapped
            repeat = np.zeros(n, dtype=bool)
            for i in np.nonzero(stuck)[0]:
                # a deterministic policy that got stuck at the same spot as
                # its previous life will repeat it forever: end the run
                # instead of spending the remaining lives on identical replays
                key = (int(gp[i]), int(x[i]) // 16)
                if self.last_stuck[i] == key:
                    repeat[i] = True; continue
                self.last_stuck[i] = key
                self.ram[i, 0x7F8:0x7FB] = 0
                self.lib.benv_set_ram(self.env, int(i), self.ram[i].tobytes())
                self.unpaid[i] = 0
                self.forced_timeup[i] += 1
            real_done = (real_done & ~timeout) | repeat
        if self.n_explorers:
            # an explorer walk ends at its step budget or its first death
            real_done = real_done | (self.is_explorer_env
                                     & ((self.explorer == 0) | (life < self.lives)))
        # rl_games value bootstrap: the plain cutoff is not part of the
        # game; the post-reset cutoff IS a dead end (no bootstrap)
        time_outs = timeout & ~self.after_reset & \
            ~(died | game_over | victory | zombie | wrapped)
        life_lost = life < self.lives
        self.lives = life.copy()
        # the new-life re-sync must see the RESPAWN position: in the
        # hack-free path the life counter drops during the intermission, so
        # defer it to the first control step (immediate in the hacked path)
        self.pending_life = (self.pending_life | life_lost) & ~in_play
        life_sync = (life_lost & in_play) | (resume & self.pending_life_at_resume)
        self.pending_life_at_resume = self.pending_life.copy()

        infos = Infos()
        infos.time_outs = time_outs
        n_front = sum(1 for c in self.archive if self._won(c)) \
            if self.archive else 0
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
                'flag_get': bool(flag[i]), 'life': int(life[i]),
                'world': int(ram[i, 0x75F]) + 1,
                'stage': int(ram[i, 0x75C]) + 1,
                'time': int(t[i]), 'coins': int(ram[i, 0x7ED]),
                'score': int(self.prev_score[i]),
                'start_stage': self.start_stage[i],
                'self_restart': bool(self.was_restart[i]),
                # started from the level's door state (not an archive cell,
                # not a continued life): what every door metric counts
                'door': bool(self.is_door[i]),
                'continuation': bool(self.continuation[i]),
                'timeout': bool(timeout[i]),
                'loop_timeout': bool(timeout[i] and self.after_reset[i]),
                'page_reset': bool(page_reset[i]),
                'page_resets': int(self.page_resets[i]),
                'wrong_exit': bool(wrong_exit[i]),
                'max_unpaid_gap': int(self.max_gap[i]),
                'forced_timeups': int(self.forced_timeup[i]),
                'frontier_cells': n_front,
                'archive_cells': len(self.archive),
                'explorer_walks': self.n_walks,
            })

        done = done_pre

        # archive hygiene: a cell whose restarts mostly die within a few
        # steps was saved in a doomed spot (e.g. mid enemy contact) - prune
        self.ep_steps += 1
        for i in np.nonzero(done)[0]:
            cell = self.start_cell[i]
            if cell is None:
                continue
            # transitive credit: reaching a DEEPER cell that already wins is
            # a win for this cell ("deeper" = another frame or >= 4 x-bins)
            # "deeper" = another frame, >= 4 x-bins further, or -- with
            # credit_vertical -- a winning cell in the SAME bin that is
            # higher up (smaller y-band) or has another tile signature
            # (the reveal). The flat next-door cell never counts, so the
            # trivial-credit failure stays closed, but climbing onto the
            # 8-4 block from the floor is finally a win for the floor cell
            # (without it the floor only won if the same episode also made
            # the pipe top and entered the pipe: 0 wins in ~5000 tries).
            reached = any(
                self._won(c) and (
                    c[0] != cell[0] or c[1] != cell[1] or c[4] != cell[4]
                    or c[5] != cell[5] or c[2] >= cell[2] + 4
                    or (self.credit_vertical and c[2] == cell[2]
                        and (c[3] < cell[3] or c[6] != cell[6]))
                    # ... or a HIGHER winning cell 1-3 bins ahead: the
                    # approach to the block (bins 16-17) could only win by
                    # reaching bin 20+, i.e. the whole reveal-climb-jump-
                    # enter sequence in one episode (0 wins in ~28k tries)
                    or (self.credit_vertical and cell[2] < c[2] < cell[2] + 4
                        and c[3] < cell[3]))
                for c in self.ep_cells[i] if c != cell)
            won = victory[i] or self.cleared[i] > 0 or reached
            if won:
                wins = self.explore_wins if self.is_explorer_env[i] else self.cell_wins
                wins[cell] = wins.get(cell, 0) + 1
            # a random walk dying early says nothing about the cell, and a cell
            # that has converted before is never pruned: only a win in the SAME
            # episode used to protect it, so a proven link saved on a ledge
            # edge could be deleted together with its win counts
            if (won or self.ep_steps[i] > 8 or self.is_explorer_env[i]
                    or self._won(cell)):
                continue
            n_early = self.cell_early.get(cell, 0) + 1
            self.cell_early[cell] = n_early
            ent = self.archive.get(cell)
            if (ent is not None and n_early >= 12
                    and n_early > 0.5 * max(ent[1], 1)):
                self._forget_cell(cell)
                self._archive_dirty += 1

        # frame stack (ring)
        f = self.obs_u8.astype(np.float32) / 255.0
        self._ptr = (self._ptr + 1) % FRAME_STACK
        self._ring[..., self._ptr] = f
        if self.u8_obs:
            self._ring_u8[..., self._ptr] = self.obs_u8

        # resets for finished episodes (recorders set hold_on_done so the
        # ending / game-over screen keeps playing instead of a fresh level)
        realdone_idx = np.nonzero(real_done)[0] if not self.hold_on_done \
            else np.zeros(0, dtype=np.int64)
        if len(realdone_idx):
            for i in realdone_idx:
                self._reset_env(int(i))
                self._fetch_obs(i)
                self._ring[i] = (self.obs_u8[i].astype(np.float32)
                                 / 255.0)[..., None]
                if self.u8_obs:
                    self._ring_u8[i] = self.obs_u8[i][..., None]
            self._post_reset_init(realdone_idx, self.ram)
        # life-loss boundaries: re-init episode trackers but keep playing
        soft_idx = list(np.nonzero(life_sync & ~real_done)[0])
        if soft_idx:
            self._post_reset_init(soft_idx, self.ram)
            for i in soft_idx:
                # the ended life's episode is over; the next life continues
                # the game (life_loss_reset off, or the multi-life eval): it
                # credits no cell and is not a door episode (it respawns at
                # the level start or its halfway point)
                self.start_cell[i] = None; self.is_door[i] = False
                self.was_restart[i] = False; self.continuation[i] = True
                self._seed_cells([i])
                self.explorer[i] = 0      # macro noise must not leak on
                # new life = fresh frame stack
                self._ring[i] = (self.obs_u8[i].astype(np.float32)
                                 / 255.0)[..., None]
                if self.u8_obs:
                    self._ring_u8[i] = self.obs_u8[i][..., None]

        obs = self._obs()
        if self.n_explorers:
            nt = self.n_train
            out = Infos(infos[:nt]); out.time_outs = time_outs[:nt]
            return obs[:nt], reward[:nt].astype(np.float32), done[:nt], out
        return obs, reward.astype(np.float32), done, infos

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {'observation_space': self.observation_space,
                'action_space': self.action_space, 'agents': 1,
                'value_size': 1}

    def freeze_door_seen(self, flag):
        """Freeze / unfreeze the door-episode cell counts of the relative
        novelty bonus. Unfreezing merges the entries seen meanwhile."""
        flag = bool(flag)
        if self.door_seen_frozen and not flag:
            for k, v in self._door_seen_pending.items():
                self.door_seen[k] = self.door_seen.get(k, 0.0) + v
            self._door_seen_pending = {}
        self.door_seen_frozen = flag

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
            # every archived cell, not just winners: iterating cell_wins
            # left never-winning cells as 3-slot entries, and the loader
            # then dropped their try counts -- the failure weight of the
            # hardest (never-winning) cells collapsed on every reload
            for c, e in self.archive.items():
                while len(e) < 6:
                    e.append(0)
                e[3] = self.cell_wins.get(c, 0)
                e[4] = self.cell_tries.get(c, 0)
                e[5] = self.explore_wins.get(c, 0)
            tmp = self.archive_path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(self.archive, f)
            os.replace(tmp, self.archive_path)
            self._archive_saved_at = time.time()

    def close(self):
        self._save_archive()
        self.lib.benv_destroy(self.env)


class NativeEvalEnv:
    """Single-env adapter over MarioNativeVecEnv for the video/eval loop
    (old-gym API + .screen for frame capture)."""

    def __init__(self, raw_steps=True, **kwargs):
        kwargs.setdefault('n_threads', 1)
        kwargs.setdefault('episode_life', False)
        self.v = MarioNativeVecEnv('eval', 1, dense_infos=True, **kwargs)
        # hack-free by default: deaths, pipe travel, the flag and the ending
        # are emulated and shown frame by frame, and the eval's timer hack
        # lands on the same core that renders the video (the lockstep
        # renderer replayed a different game after the first forced time-up)
        self.v._raw_steps = bool(raw_steps)
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


def register_mario_native_vecenv():
    vecenv.register(
        'MARIO_NATIVE',
        lambda config_name, num_actors, **kwargs: MarioNativeVecEnv(
            config_name, num_actors, **kwargs))
