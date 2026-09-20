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
        L.ss_optimize.argtypes = [P, ctypes.c_char_p, P, I, P, I, I, I, I, I, P, I, ctypes.POINTER(_Stats)]
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

    def optimize(self, state, route, reference, beam=20000, per_cell=16, max_depth=6000, verbose=0):
        """Beam A* over time along the reference's waypoints (never slower than the reference)."""
        r, rp, rn = self._route(route)
        ref = np.ascontiguousarray(reference, dtype=np.uint8)
        out = np.zeros(self.MAX_ACTIONS, dtype=np.uint8)
        st = _Stats()
        n = self._lib.ss_optimize(self._ctx, self._state(state), rp, rn, ref.ctypes.data, len(ref), int(beam),
                                  int(per_cell), int(max_depth), int(verbose), out.ctypes.data, len(out),
                                  ctypes.byref(st))
        return self._result(out, n, st)
