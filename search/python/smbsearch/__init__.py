"""Python interface to the C++ route search (search/build/libsmbsearch.so).

    from smbsearch import Search, load_state, ROUTE
    s = Search(threads=40)
    ref = s.explore(load_state('4-2'), ROUTE, budget_s=300)       # discovery
    opt = s.optimize(load_state('4-2'), ROUTE, ref.actions)        # beam A*
    trace, end = s.replay(load_state('4-2'), opt.actions)

All loops run in C++ on a thread pool; ctypes releases the GIL for each call.
States are full native savestates (bytes); actions are COMPLEX_MOVEMENT indices,
one per 4 frames of the real game (no training hacks).
"""
import ctypes
import gzip
import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(SEARCH)
LIB = os.path.join(SEARCH, 'build', 'libsmbsearch.so')
ROM = os.path.join(REPO, 'retro_integration', 'SuperMarioBros-Nes-v0', 'rom.nes')
STATES = os.path.join(REPO, 'native', 'states')
ROUTE = ['1-1', '1-2', '4-1', '4-2', '8-1', '8-2', '8-3', '8-4']
ACTIONS = ['NOOP', 'right', 'right+A', 'right+B', 'right+A+B', 'A', 'left', 'left+A', 'left+B',
           'left+A+B', 'down', 'up']
ACTION_BUTTONS = [0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40, 0x41, 0x42, 0x43, 0x20, 0x10]
TRACE_FIELDS = ('x', 'y', 'level', 'area', 'sub', 'atype', 'engine', 'mode', 'camera', 'lives')
FRAME_SKIP = 4
OBS = 84          # the net's frame: 84x84 grayscale
FPS = 50.007      # the ROM is Super Mario Bros. (Europe): a PAL game, 50 frames per second


def gp(level):
    w, s = level.split('-')
    return (int(w) - 1) * 4 + int(s) - 1


def level_name(g):
    return '%d-%d' % (g // 4 + 1, g % 4 + 1)


def load_state(name='4-2'):
    """A native savestate: a level's door state ('4-2') or 'FullGame' (1-1 from boot)."""
    fn = 'FullGame.state' if name == 'FullGame' else 'Level%s.state' % name
    return gzip.open(os.path.join(STATES, fn)).read()


class _Stats(ctypes.Structure):
    _fields_ = [('frames', ctypes.c_int64), ('emu_frames', ctypes.c_int64), ('seconds', ctypes.c_double),
                ('cells', ctypes.c_int64), ('walks', ctypes.c_int64), ('found', ctypes.c_int32),
                ('n_actions', ctypes.c_int32)]


@dataclass
class Result:
    actions: np.ndarray
    found: bool
    frames: int
    stats: dict


class Search:
    MAX_ACTIONS = 200000

    def __init__(self, threads=None, rom=ROM, lib=LIB):
        if not os.path.exists(lib):
            raise FileNotFoundError('%s is missing: run search/build.sh' % lib)
        L = self._lib = ctypes.CDLL(lib)
        P, I, D, U64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_uint64
        L.ss_state_size.restype = I
        L.ss_create.restype = P; L.ss_create.argtypes = [ctypes.c_char_p, I, I]
        L.ss_destroy.argtypes = [P]
        L.ss_threads.restype = I; L.ss_threads.argtypes = [P]
        L.ss_ram.restype = I; L.ss_ram.argtypes = [P, ctypes.c_char_p, P]
        L.ss_replay.restype = I; L.ss_replay.argtypes = [P, ctypes.c_char_p, P, I, P, P]
        L.ss_settle.restype = I; L.ss_settle.argtypes = [P, ctypes.c_char_p, I, P]
        L.ss_bench.restype = D; L.ss_bench.argtypes = [P, ctypes.c_char_p, ctypes.c_int64]
        L.ss_selftest.restype = I; L.ss_selftest.argtypes = [P, ctypes.c_char_p, I, U64]
        L.ss_explore.restype = I
        L.ss_explore.argtypes = [P, ctypes.c_char_p, P, I, D, D, I, U64, I, P, I, ctypes.POINTER(_Stats)]
        L.ss_optimize.restype = I
        L.ss_optimize.argtypes = [P, ctypes.c_char_p, P, I, P, I, I, I, I, I, P, I, ctypes.POINTER(_Stats),
                                  ctypes.c_char_p]
        L.ss_frames.restype = I; L.ss_frames.argtypes = [P, ctypes.c_char_p, I, I, P]
        L.ss_obs.restype = I; L.ss_obs.argtypes = [P, ctypes.c_char_p, P]
        L.ss_replay_obs.restype = I; L.ss_replay_obs.argtypes = [P, ctypes.c_char_p, P, I, P, P, P]
        L.ss_forced_along.restype = I; L.ss_forced_along.argtypes = [P, ctypes.c_char_p, P, I, P]
        L.ss_classify_along.restype = I; L.ss_classify_along.argtypes = [P, ctypes.c_char_p, P, I, P, I, P]
        L.ss_progress_along.restype = I
        L.ss_progress_along.argtypes = [P, ctypes.c_char_p, P, I, P, I, ctypes.c_char_p, P, I, P]
        L.ss_lookahead.restype = I
        L.ss_lookahead.argtypes = [P, ctypes.c_char_p, P, I, P, I, ctypes.c_char_p, I, I, I, P, I,
                                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int32)]
        rom_bytes = open(rom, 'rb').read()
        self._ctx = L.ss_create(rom_bytes, len(rom_bytes), int(threads or os.cpu_count()))
        if not self._ctx:
            raise RuntimeError('ss_create failed (bad ROM?)')
        self.threads = L.ss_threads(self._ctx)
        self.state_size = L.ss_state_size()

    def close(self):
        if getattr(self, '_ctx', None):
            self._lib.ss_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def _state(self, s):
        if len(s) != self.state_size:
            raise ValueError('state of %d bytes, the core uses %d' % (len(s), self.state_size))
        return bytes(s)

    @staticmethod
    def _route(route):
        arr = np.array([gp(l) for l in route], dtype=np.int32)
        return arr, arr.ctypes.data, len(arr)

    def ram(self, state):
        """The 2 KB work RAM of a state."""
        out = np.zeros(0x800, dtype=np.uint8)
        self._lib.ss_ram(self._ctx, self._state(state), out.ctypes.data)
        return out

    def level(self, state):
        r = self.ram(state)
        return level_name(int(r[0x75F]) * 4 + int(r[0x75C]))

    def replay(self, state, actions):
        """Step the actions from state. Returns (trace (n, 10) int32 per step, end state)."""
        a = np.ascontiguousarray(actions, dtype=np.uint8)
        tr = np.zeros((len(a), len(TRACE_FIELDS)), dtype=np.int32)
        end = ctypes.create_string_buffer(self.state_size)
        n = self._lib.ss_replay(self._ctx, self._state(state), a.ctypes.data, len(a), tr.ctypes.data, end)
        if n < 0:
            raise ValueError('bad action index')
        return tr, end.raw

    def settle(self, state, max_steps=3000):
        """NOOP steps until Mario is in control (after a level transition): (steps, state)."""
        end = ctypes.create_string_buffer(self.state_size)
        n = self._lib.ss_settle(self._ctx, self._state(state), max_steps, end)
        return n, end.raw

    def bench(self, state, frames=4_000_000):
        return self._lib.ss_bench(self._ctx, self._state(state), int(frames))

    def selftest(self, state, steps=400, seed=1):
        return self._lib.ss_selftest(self._ctx, self._state(state), steps, seed) == 0

    def _result(self, out, n, st):
        if n == -2:
            raise RuntimeError('action buffer too small')
        stats = {k: getattr(st, k) for k, _ in _Stats._fields_}
        return Result(actions=out[:max(n, 0)].copy(), found=bool(st.found), frames=int(st.frames), stats=stats)

    def explore(self, state, route, budget_s=120.0, settle_s=30.0, max_walk=300, seed=0, verbose=0):
        """Go-Explore discovery of the segment starting at state (to the next route level)."""
        r, rp, rn = self._route(route)
        out = np.zeros(self.MAX_ACTIONS, dtype=np.uint8)
        st = _Stats()
        n = self._lib.ss_explore(self._ctx, self._state(state), rp, rn, float(budget_s), float(settle_s),
                                 int(max_walk), int(seed), int(verbose), out.ctypes.data, len(out), ctypes.byref(st))
        return self._result(out, n, st)

    def optimize(self, state, route, reference, beam=20000, per_cell=16, max_depth=6000, verbose=0, ref_start=None):
        """Beam A* over time along the reference's waypoints (never slower than the reference).
        ref_start: the state the reference was found from, when it is not `state` (e.g. no start delay)."""
        r, rp, rn = self._route(route)
        ref = np.ascontiguousarray(reference, dtype=np.uint8)
        out = np.zeros(self.MAX_ACTIONS, dtype=np.uint8)
        st = _Stats()
        n = self._lib.ss_optimize(self._ctx, self._state(state), rp, rn, ref.ctypes.data, len(ref), int(beam),
                                  int(per_cell), int(max_depth), int(verbose), out.ctypes.data, len(out),
                                  ctypes.byref(st), None if ref_start is None else self._state(ref_start))
        return self._result(out, n, st)

    def lookahead(self, state, route, reference, ref_start=None, beam=200, per_cell=16, horizon=50):
        """Local teacher: the beam for at most `horizon` steps along the reference.
        Returns (actions of the best path, frames to the goal (exact if found), found)."""
        r, rp, rn = self._route(route)
        ref = np.ascontiguousarray(reference, dtype=np.uint8)
        out = np.zeros(max(horizon, 1), dtype=np.uint8)
        est, found = ctypes.c_double(), ctypes.c_int32()
        n = self._lib.ss_lookahead(self._ctx, self._state(state), rp, rn, ref.ctypes.data, len(ref),
                                   None if ref_start is None else self._state(ref_start), int(beam), int(per_cell),
                                   int(horizon), out.ctypes.data, len(out), ctypes.byref(est), ctypes.byref(found))
        return out[:max(n, 0)].copy(), est.value, bool(found.value)

    def frames(self, state, n, buttons=0):
        """n raw frames holding buttons (a start delay: NOOP frames). Returns the end state."""
        end = ctypes.create_string_buffer(self.state_size)
        self._lib.ss_frames(self._ctx, self._state(state), int(n), int(buttons), end)
        return end.raw

    def obs(self, state):
        """The net's view of the current frame: (84, 84) uint8."""
        out = np.zeros((OBS, OBS), dtype=np.uint8)
        self._lib.ss_obs(self._ctx, self._state(state), out.ctypes.data)
        return out

    def progress_along(self, state, route, reference, actions, ref_start=None):
        """The route's frames to go at the start and after each action (n + 1 values).
        The waste of step k is max(0, R[k+1] - R[k] + 4)."""
        r, rp, rn = self._route(route)
        ref = np.ascontiguousarray(reference, dtype=np.uint8)
        a = np.ascontiguousarray(actions, dtype=np.uint8)
        out = np.zeros(len(a) + 1, np.float32)
        n = self._lib.ss_progress_along(self._ctx, self._state(state), rp, rn, ref.ctypes.data, len(ref),
                                        None if ref_start is None else self._state(ref_start), a.ctypes.data,
                                        len(a), out.ctypes.data)
        if n < 0:
            raise ValueError('bad action index or state off the route')
        return out

    def classify_along(self, state, route, actions):
        """Per step: 0 running, 1 goal, 2 dead (the search's rule); stops at the first end.
        Returns (outcomes[:n_played], n_played)."""
        r, rp, rn = self._route(route)
        a = np.ascontiguousarray(actions, dtype=np.uint8)
        out = np.zeros(len(a), np.uint8)
        n = self._lib.ss_classify_along(self._ctx, self._state(state), rp, rn, a.ctypes.data, len(a), out.ctypes.data)
        if n < 0:
            raise ValueError('bad action index or state off the route')
        return out[:n], n

    def forced_along(self, state, actions):
        """Per step of the replay: True where the input did nothing (all actions reach one state)."""
        a = np.ascontiguousarray(actions, dtype=np.uint8)
        out = np.zeros(len(a), np.uint8)
        if self._lib.ss_forced_along(self._ctx, self._state(state), a.ctypes.data, len(a), out.ctypes.data) < 0:
            raise ValueError('bad action index')
        return out.astype(bool)

    def replay_obs(self, state, actions):
        """replay() plus each step's 84x84 frame: (obs (n, 84, 84) uint8, trace, end state)."""
        a = np.ascontiguousarray(actions, dtype=np.uint8)
        obs = np.zeros((len(a), OBS, OBS), dtype=np.uint8)
        tr = np.zeros((len(a), len(TRACE_FIELDS)), dtype=np.int32)
        end = ctypes.create_string_buffer(self.state_size)
        n = self._lib.ss_replay_obs(self._ctx, self._state(state), a.ctypes.data, len(a), obs.ctypes.data,
                                    tr.ctypes.data, end)
        if n < 0:
            raise ValueError('bad action index')
        return obs, tr, end.raw


class _MctsParams(ctypes.Structure):
    _fields_ = [('c_puct', ctypes.c_float), ('fpu', ctypes.c_float), ('scale', ctypes.c_float),
                ('v_death', ctypes.c_float), ('max_nodes', ctypes.c_int32), ('value_mix', ctypes.c_float),
                ('min_backup', ctypes.c_int32), ('relative', ctypes.c_int32)]


class Forest:
    """MCTS trees on the real game (search/src/mcts), one game per tree; the net lives in the caller.

        f = Forest(search, n_trees=1)
        f.reset(0, state, ROUTE)
        n = f.select([0], per_tree=256)        # leaves f.leaves[:n], net inputs f.stacks[:n] (n, 4, 84, 84)
        f.backup(n, priors, values)            # priors (n, 12) probabilities, values (n,) frames to go
        visits, best, root_b, root_n = f.root(0)
        term = f.commit(0, action)             # 0 running, 1 goal (next route level), 2 dead

    stacks: optional (max_leaves, 4, 84, 84) uint8 buffer to fill (e.g. pinned memory for the GPU).
    """
    RUNNING, GOAL, DEAD = 0, 1, 2

    def __init__(self, search, n_trees, c_puct=1.5, fpu=0.5, scale=32.0, v_death=4096.0, max_nodes=1 << 16,
                 max_leaves=2048, stacks=None, value_mix=1.0, min_backup=False, relative=False):
        self.s = search
        L = self._lib = search._lib
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        L.ss_mcts_create.restype = P; L.ss_mcts_create.argtypes = [P, I, ctypes.POINTER(_MctsParams)]
        L.ss_mcts_destroy.argtypes = [P]
        L.ss_mcts_reset.restype = I; L.ss_mcts_reset.argtypes = [P, I, ctypes.c_char_p, P, I]
        L.ss_mcts_select.restype = I; L.ss_mcts_select.argtypes = [P, P, I, I, I, P, P]
        L.ss_mcts_backup.argtypes = [P, I, P, P, P]
        L.ss_mcts_root.restype = I; L.ss_mcts_root.argtypes = [P, I, P, P, P]
        L.ss_mcts_noise.argtypes = [P, I, P, F]
        L.ss_mcts_commit.restype = I; L.ss_mcts_commit.argtypes = [P, I, I]
        L.ss_mcts_forced.argtypes = [P, P, I, P]
        L.ss_mcts_state.argtypes = [P, I, P, P, P]
        L.ss_mcts_nodes.restype = I; L.ss_mcts_nodes.argtypes = [P, I]
        L.ss_mcts_set_route.argtypes = [P, I, ctypes.c_char_p, P, I]
        L.ss_mcts_set_value_mix.argtypes = [P, F]
        L.ss_mcts_safe.argtypes = [P, I, P, I, I, P]
        L.ss_mcts_leaf_info.argtypes = [P, I, P, P, P]
        L.ss_mcts_dump.restype = I; L.ss_mcts_dump.argtypes = [P, I, I, ctypes.c_uint64, P, P, P, P]
        L.ss_mcts_root_value.restype = F; L.ss_mcts_root_value.argtypes = [P, I]
        self.params = _MctsParams(c_puct, fpu, scale, v_death, max_nodes, value_mix, int(min_backup),
                                  int(relative))
        self.v_death = v_death
        self.n_trees = n_trees
        self._m = L.ss_mcts_create(search._ctx, n_trees, ctypes.byref(self.params))
        self.max_leaves = max_leaves
        self.stacks = np.zeros((max_leaves, 4, OBS, OBS), np.uint8) if stacks is None else stacks
        assert self.stacks.shape == (max_leaves, 4, OBS, OBS) and self.stacks.dtype == np.uint8
        assert self.stacks.flags['C_CONTIGUOUS']
        self.leaves = np.zeros((max_leaves, 2), np.int32)

    def close(self):
        if getattr(self, '_m', None):
            self._lib.ss_mcts_destroy(self._m)
            self._m = None

    def __del__(self):
        self.close()

    def reset(self, tree, state, route):
        r = np.array([gp(l) for l in route], dtype=np.int32)
        return self._lib.ss_mcts_reset(self._m, tree, self.s._state(state), r.ctypes.data, len(r)) == 0

    def select(self, trees, per_tree):
        t = np.ascontiguousarray(trees, dtype=np.int32)
        return self._lib.ss_mcts_select(self._m, t.ctypes.data, len(t), int(per_tree), self.max_leaves,
                                        self.leaves.ctypes.data, self.stacks.ctypes.data)

    def backup(self, n, priors, values):
        p = np.ascontiguousarray(priors, dtype=np.float32)
        v = np.ascontiguousarray(values, dtype=np.float32)
        assert p.shape == (n, 12) and v.shape == (n,)
        self._lib.ss_mcts_backup(self._m, n, self.leaves.ctypes.data, p.ctypes.data, v.ctypes.data)

    def root(self, tree):
        visits = np.zeros(12, np.int32)
        best = np.zeros(12, np.float32)
        rb = ctypes.c_float()
        n = self._lib.ss_mcts_root(self._m, tree, visits.ctypes.data, best.ctypes.data, ctypes.byref(rb))
        return visits, best, rb.value, n

    def noise(self, tree, noise, frac):
        z = np.ascontiguousarray(noise, dtype=np.float32)
        self._lib.ss_mcts_noise(self._m, tree, z.ctypes.data, float(frac))

    def commit(self, tree, action):
        return self._lib.ss_mcts_commit(self._m, tree, int(action))

    def forced(self, trees):
        t = np.ascontiguousarray(trees, dtype=np.int32)
        out = np.zeros(len(t), np.int32)
        self._lib.ss_mcts_forced(self._m, t.ctypes.data, len(t), out.ctypes.data)
        return out.astype(bool)

    def state(self, tree):
        """(full state bytes, ram (2048,), root stack (4, 84, 84))"""
        full = ctypes.create_string_buffer(self.s.state_size)
        ram = np.zeros(0x800, np.uint8)
        stack = np.zeros((4, OBS, OBS), np.uint8)
        self._lib.ss_mcts_state(self._m, tree, full, ram.ctypes.data, stack.ctypes.data)
        return full.raw, ram, stack

    def nodes(self, tree):
        return self._lib.ss_mcts_nodes(self._m, tree)

    def set_route(self, level, start, actions):
        """Leaf values for the level also use frames to go along this route (its actions from start)."""
        a = np.ascontiguousarray(actions, dtype=np.uint8)
        self._lib.ss_mcts_set_route(self._m, gp(level), self.s._state(start), a.ctypes.data, len(a))

    def safe(self, tree, actions, horizon=24):
        """Per candidate root action: does some button held for `horizon` steps after it survive?"""
        a = np.ascontiguousarray(actions, dtype=np.int32)
        out = np.zeros(len(a), np.int32)
        self._lib.ss_mcts_safe(self._m, tree, a.ctypes.data, len(a), int(horizon), out.ctypes.data)
        return out.astype(bool)

    def leaf_info(self, n):
        """The last select's leaves: (route frames to go (n,), depth from the root (n,))."""
        v = np.zeros(n, np.float32); d = np.zeros(n, np.int32)
        self._lib.ss_mcts_leaf_info(self._m, n, self.leaves.ctypes.data, v.ctypes.data, d.ctypes.data)
        return v, d

    def dump(self, tree, max_n=64, seed=0):
        """A sample of the tree's visited nodes: (b relative to the root, depth, visits,
        stacks (n, 4, 84, 84)) -- the search's own value targets, no route needed."""
        b = np.zeros(max_n, np.float32); d = np.zeros(max_n, np.int32); v = np.zeros(max_n, np.int32)
        st = np.zeros((max_n, 4, OBS, OBS), np.uint8)
        n = self._lib.ss_mcts_dump(self._m, tree, max_n, seed, b.ctypes.data, d.ctypes.data,
                                   v.ctypes.data, st.ctypes.data)
        return b[:n], d[:n], v[:n], st[:n]

    def root_value(self, tree):
        """The route's frames to go at the tree's root."""
        return self._lib.ss_mcts_root_value(self._m, tree)

    def set_value_mix(self, mix):
        """Leaf value = mix x net + (1 - mix) x route (where a route is set)."""
        self._lib.ss_mcts_set_value_mix(self._m, float(mix))
