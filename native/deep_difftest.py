"""Deep lockstep differential test: native smbcore vs stable-retro.

Part 1 -- hack-free core (libsmbcore), per level: load retro's savestate,
sync non-stack RAM into a booted native core, then drive both with
identical seeded runner-like button sequences (run/jump right, A released
between jumps, short back-offs, DOWN/UP holds for pipes and vines, waits),
comparing game-state RAM every frame. Lives are topped up in both
emulators (the same RAM write) so the play stays inside the level instead
of ending in game over and the attract demo.

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

rom = open(ROM, 'rb').read()

import stable_retro as retro  # noqa: E402
retro.data.Integrations.add_custom_path(os.path.join(ROOT, 'retro_integration'))

GAME = np.ones(0x800, dtype=bool)
GAME[0x100:0x300] = False          # stack page (own per CPU) + OAM shadow
LIVES, MODE = 0x75A, 0x770

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
LEVELS = sys.argv[2].split(',') if len(sys.argv) > 2 else [
    'Level1-1', 'Level1-2', 'Level1-4', 'Level2-2', 'Level3-1', 'Level4-1',
    'Level4-2', 'Level6-2', 'Level7-2', 'Level8-1', 'Level8-2', 'Level8-3',
    'Level8-4']


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


overall_fail = 0
print('-- part 1: hack-free core, per frame')
for level in LEVELS:
    overall_fail += not part1(level)
print('OVERALL:', 'PASS' if overall_fail == 0 else f'{overall_fail} RUNS FAILED')
sys.exit(1 if overall_fail else 0)
