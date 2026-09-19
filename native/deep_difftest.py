"""Deep lockstep differential test: native smbcore vs stable-retro.

Part 1 -- hack-free core (libsmbcore), per level: load retro's savestate,
sync non-stack RAM into a booted native core, then drive both with
identical seeded runner-like button sequences (run/jump right, A released
between jumps, short back-offs, DOWN/UP holds for pipes and vines, waits),
comparing game-state RAM every frame. Lives are topped up in both
emulators (the same RAM write) so the play stays inside the level instead
of ending in game over and the attract demo.

Part 2 -- training path (libbatchenv benv_step: 4 frames + the RAM hacks)
vs RetroMarioEnv (the reference port of the same hacks), per agent step,
from the native door states training starts from (stable-retro adopts the
native RAM and PPU memory):
  * the recorded 4-2 routes (tests/data): vine + bonus area + warp pipe
    into 8-1, and the flag into 4-3 -- vines, pipes, flagpole, level
    changes and the transition skips;
  * seeded runner play in a few levels (deaths -> the kill / inter-life
    skip hacks), lives topped up as in part 1.

Usage: deep_difftest.py [FRAMES] [Level1-1,Level4-2,...]   (~35 s)
"""
import ctypes
import gzip
import os
import struct
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.join(ROOT, 'native')
ROM = os.path.join(ROOT, 'retro_integration', 'SuperMarioBros-Nes-v0', 'rom.nes')

lib = ctypes.CDLL(os.path.join(HERE, 'libsmbcore.so'))
lib.smb_create.restype = ctypes.c_void_p
lib.smb_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.smb_destroy = getattr(lib, 'smb_destroy')
lib.smb_destroy.argtypes = [ctypes.c_void_p]
lib.smb_frame.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
lib.smb_ram.restype = ctypes.POINTER(ctypes.c_uint8)
lib.smb_ram.argtypes = [ctypes.c_void_p]
lib.smb_set_ram.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

blib = ctypes.CDLL(os.path.join(HERE, 'libbatchenv.so'))
_V, _I = ctypes.c_void_p, ctypes.c_int
for _n, _r, _a in [('benv_create', _V, [ctypes.c_char_p, _I, _I, _I, _I]),
                   ('benv_destroy', None, [_V]),
                   ('benv_load', None, [_V, _I, ctypes.c_char_p]),
                   ('benv_step', None, [_V, _V, _V, _V]),
                   ('benv_get_ppu', None, [_V, _I, _V, _V, _V]),
                   ('benv_ram', ctypes.POINTER(ctypes.c_uint8), [_V, _I])]:
    _f = getattr(blib, _n); _f.restype = _r; _f.argtypes = _a

rom = open(ROM, 'rb').read()

from mario_env import RetroMarioEnv, _register_integration  # noqa: E402
from mario_native_vecenv import _ACTION_BYTES  # noqa: E402
import stable_retro as retro  # noqa: E402
_register_integration()

GAME = np.ones(0x800, dtype=bool)
GAME[0x100:0x300] = False          # stack page (own per CPU) + OAM shadow
LIVES, MODE = 0x75A, 0x770

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
LEVELS = sys.argv[2].split(',') if len(sys.argv) > 2 else [
    'Level1-1', 'Level1-2', 'Level1-4', 'Level2-2', 'Level3-1', 'Level4-1',
    'Level4-2', 'Level6-2', 'Level7-2', 'Level8-1', 'Level8-2', 'Level8-3',
    'Level8-4']
TRAIN_LEVELS = ['1-1', '1-2', '8-1']
TRAIN_STEPS = 700


def seed_of(level):
    # hash() of a str is salted per process (PYTHONHASHSEED): not reproducible
    return zlib.crc32(level.encode()) & 0xFFFF


def action_seq(seed, n):
    """runner-like pad bytes (bit0..7 = A,B,Sel,Start,U,D,L,R)"""
    rng = np.random.RandomState(seed)
    seq = []
    while len(seq) < n:
        r = rng.random_sample()
        if r < 0.55:
            seq += [0x82] * rng.randint(2, 20)      # run right (B)
            seq += [0x83] * rng.randint(4, 30)      # running jump, then A up
        elif r < 0.70:
            seq += [0x81] * rng.randint(4, 30)      # walking jump
            seq += [0x80] * rng.randint(1, 6)
        elif r < 0.80:
            seq += [0x42] * rng.randint(2, 16)      # back off (left+B)
        elif r < 0.88:
            seq += [0x20] * rng.randint(4, 40)      # down: pipes
        elif r < 0.93:
            seq += [0x10] * rng.randint(4, 40)      # up: vines
        else:
            seq += [0x00] * rng.randint(4, 40)      # wait: enemy timing
    return seq[:n]


def step_seq(seed, n):
    """runner-like COMPLEX_MOVEMENT indices, held 1-8 agent steps"""
    rng = np.random.RandomState(seed)
    w = np.array([2, 6, 8, 14, 14, 3, 2, 1, 2, 1, 3, 2], float)
    seq = []
    while len(seq) < n:
        a = int(rng.choice(12, p=w / w.sum()))
        seq += [a] * rng.randint(1, 9)
        if a in (2, 4, 5, 7, 9):                    # release A after a jump
            seq += [3]
    return seq[:n]


def pad_mask(b):
    m = np.zeros(9, dtype=np.uint8)
    m[0] = (b >> 1) & 1; m[2] = (b >> 2) & 1; m[3] = (b >> 3) & 1
    m[4] = (b >> 4) & 1; m[5] = (b >> 5) & 1; m[6] = (b >> 6) & 1
    m[7] = (b >> 7) & 1; m[8] = b & 1
    return m


def chunk(st, tag, size):
    """payload offset of a chunk in an FCEUmm savestate (tag + u32 size)"""
    key = tag + struct.pack('<I', size)
    i = st.find(key)
    assert i >= 0 and st.find(key, i + 1) < 0, tag
    return i + 8


def adopt_native(em, benv):
    """stable-retro takes the native core's RAM (all but the stack page) and
    its nametables / palette / OAM (savestate patch)"""
    st = bytearray(em.get_state())
    ram = np.ctypeslib.as_array(blib.benv_ram(benv, 0), shape=(0x800,))
    r = chunk(st, b'RAM\x00', 0x800)
    st[r:r + 0x100] = ram[:0x100].tobytes()
    st[r + 0x200:r + 0x800] = ram[0x200:].tobytes()
    bufs = [ctypes.create_string_buffer(n) for n in (0x800, 32, 256)]
    blib.benv_get_ppu(benv, 0, *bufs)
    vram, pal, oam = (bytearray(b.raw) for b in bufs)
    pal[4] = pal[8] = pal[12] = pal[0]     # FCEUmm's backdrop copies
    for tag, b in zip((b'NTAR', b'PRAM', b'SPRA'), (vram, pal, oam)):
        o = chunk(st, tag, len(b)); st[o:o + len(b)] = b
    em.set_state(bytes(st))


class Check:
    """game-state RAM equal every tick; full RAM (minus the stack page)
    may differ only in OAM-shadow transients healing within 3 ticks"""

    def __init__(self, name):
        self.name, self.fail, self.transients, self.pending = name, None, 0, 0

    def __call__(self, t, nr, rr):
        if not np.array_equal(nr[GAME], rr[GAME]):
            d = np.nonzero((nr != rr) & GAME)[0]
            print(f'{self.name}: GAME-STATE MISMATCH at {t}: {len(d)} bytes '
                  + ' '.join(f'${a:04X}:{nr[a]:02X}!={rr[a]:02X}' for a in d[:6]))
            self.fail = t
            return False
        if not (np.array_equal(nr[:0x100], rr[:0x100])
                and np.array_equal(nr[0x200:], rr[0x200:])):
            self.pending += 1
            if self.pending > 3:
                print(f'{self.name}: unhealed OAM-shadow divergence at {t}')
                self.fail = t
                return False
        else:
            self.transients += self.pending > 0
            self.pending = 0
        return True


class Cover:
    """what the run exercised: time inside the start level, progress, deaths,
    distinct (world, stage, area, area type) places"""

    def __init__(self, start):
        self.start, self.maxx, self.deaths, self.lives = start, 0, 0, None
        self.inlevel, self.n, self.places = 0, 0, set()

    def __call__(self, r):
        self.n += 1
        if r[MODE] == 1 and int(r[0x75F]) * 4 + int(r[0x75C]) == self.start:
            self.inlevel += 1
            self.maxx = max(self.maxx, int(r[0x6D]) * 256 + int(r[0x86]))
        # a life lost (the top-up raises it again; on the training path the
        # hacks skip the dying animation, so $0E never shows it)
        self.deaths += self.lives is not None and int(r[LIVES]) < self.lives
        self.lives = int(r[LIVES])
        self.places.add((int(r[0x75F]), int(r[0x75C]), int(r[0x760]), int(r[0x74E])))

    def __str__(self):
        return (f'in-level {100 * self.inlevel // max(self.n, 1)}%, max x {self.maxx}, '
                f'{self.deaths} deaths, {len(self.places)} areas')


def part1(level):
    renv = retro.make('SuperMarioBros-Nes-v0', state=level,
                      inttype=retro.data.Integrations.CUSTOM_ONLY,
                      use_restricted_actions=retro.Actions.ALL,
                      render_mode='rgb_array')
    renv.reset()
    renv.data.update_ram()

    def retro_frame(b):
        renv.em.set_button_mask(pad_mask(b), 0)
        renv.em.step()
        renv.data.update_ram()

    def retro_ram():
        return np.frombuffer(renv.get_ram(), dtype=np.uint8)[:0x800]

    core = lib.smb_create(rom, len(rom))
    for _ in range(30):
        lib.smb_frame(core, 0)
    nv = np.ctypeslib.as_array(lib.smb_ram(core), shape=(0x800,))
    rv = retro_ram()
    for a in np.nonzero(nv != rv)[0]:
        if a < 0x100 or a >= 0x200:
            nv[a] = rv[a]

    # settle: one aligned frame lets OAM DMA/render state converge
    lib.smb_frame(core, 0); retro_frame(0)

    chk = Check(level)
    cov = Cover(int(nv[0x75F]) * 4 + int(nv[0x75C]))
    for f, b in enumerate(action_seq(seed_of(level), FRAMES)):
        lib.smb_frame(core, b)
        retro_frame(b)
        nr = np.ctypeslib.as_array(lib.smb_ram(core), shape=(0x800,))
        if not chk(f, nr, retro_ram()):
            break
        cov(nr)
        if nr[LIVES] < 2 and nr[MODE] == 1:         # stay in the level
            nr[LIVES] = 2; cov.lives = 2
            renv.data.memory.assign(LIVES, '|u1', 2)
    if chk.fail is None:
        nr = np.ctypeslib.as_array(lib.smb_ram(core), shape=(0x800,))
        print(f'{level}: PASS {FRAMES} frames ({chk.transients} healed transients; {cov}) '
              f'end w={nr[0x75F]+1}-{nr[0x75C]+1} x={int(nr[0x6D])*256+int(nr[0x86])}')
    lib.smb_destroy(core)
    renv.close()
    return chk.fail is None


def part2(name, level, acts, keep_lives, want=None):
    env = RetroMarioEnv(random_stages=[level], full_game=True)   # as the 4-2 config
    env.reset()
    be = blib.benv_create(rom, len(rom), 1, 1, 0)                # full game too
    with gzip.open(os.path.join(HERE, 'states', f'Level{level}.state'), 'rb') as fh:
        blib.benv_load(be, 0, fh.read())
    adopt_native(env._em, be)
    env._data.update_ram(); env._cur = env._data.lookup_all()
    nram = np.ctypeslib.as_array(blib.benv_ram(be, 0), shape=(0x800,))
    obs = np.zeros((1, 84, 84), np.uint8); ram = np.zeros((1, 0x800), np.uint8)
    act = np.zeros(1, np.int32)
    chk = Check(name)
    cov = Cover(int(nram[0x75F]) * 4 + int(nram[0x75C]))
    for t, a in enumerate(acts):
        act[0] = _ACTION_BYTES[a]
        blib.benv_step(be, act.ctypes.data, obs.ctypes.data, ram.ctypes.data)
        for _ in range(4):
            env.step(int(a))
        if not chk(t, ram[0], np.frombuffer(env._retro.get_ram(), np.uint8)[:0x800]):
            break
        cov(ram[0])
        if keep_lives and nram[LIVES] < 2 and nram[MODE] == 1:
            nram[LIVES] = 2; cov.lives = 2
            env._assign(LIVES, 2)
    end = f'{nram[0x75F]+1}-{nram[0x75C]+1}'
    ok = chk.fail is None and (want is None or end == want)
    if ok:
        print(f'{name}: PASS {len(acts)} steps ({chk.transients} healed transients; {cov}) '
              f'end {end} x={int(nram[0x6D])*256+int(nram[0x86])}')
    elif chk.fail is None:
        print(f'{name}: FAIL route ended in {end}, expected {want}')
    blib.benv_destroy(be)
    env.close()
    return ok


overall_fail = 0
print('-- part 1: hack-free core, per frame')
for level in LEVELS:
    overall_fail += not part1(level)
print('-- part 2: training path (benv_step vs RetroMarioEnv), per step')
for fname, want in (('vine_route_4-2.npy', '8-1'), ('exit_flag_4-2.npy', '4-3')):
    acts = list(np.load(os.path.join(ROOT, 'tests', 'data', fname)).astype(int))
    overall_fail += not part2('route ' + fname, '4-2', acts, False, want)
for level in TRAIN_LEVELS:
    overall_fail += not part2('train ' + level, level,
                              step_seq(seed_of('train' + level), TRAIN_STEPS), True)

print('OVERALL:', 'PASS' if overall_fail == 0 else f'{overall_fail} RUNS FAILED')
sys.exit(1 if overall_fail else 0)
