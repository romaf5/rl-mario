"""Deep lockstep differential test: native smbcore vs stable-retro.

Per level: load retro's savestate, sync non-stack RAM into a booted
native core, then drive both with identical seeded gameplay-like button
sequences for N frames, comparing game-state RAM every frame.
"""
import ctypes
import os
import sys

import numpy as np

ROOT = '/home/mario/workdir/rl-mario'
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

from mario_env import _register_integration  # noqa: E402
import stable_retro as retro  # noqa: E402
_register_integration()

GAME = np.ones(0x800, dtype=bool)
GAME[0x100:0x300] = False

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
LEVELS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['Level1-1', 'Level1-2', 'Level2-2', 'Level4-1', 'Level4-2', 'Level8-1', 'Level8-2', 'Level8-3', 'Level8-4']

def action_seq(seed, n):
    rng = np.random.RandomState(seed)
    seq = []
    cur = 0x82
    for _ in range(n):
        r = rng.random_sample()
        if r < 0.55:
            pass                      # hold current
        elif r < 0.75:
            cur = 0x82                # right+B
        elif r < 0.85:
            cur = 0x83                # right+B+A
        elif r < 0.90:
            cur = 0x01                # A (stroke/jump)
        elif r < 0.94:
            cur = 0x40                # left
        elif r < 0.97:
            cur = 0x20                # down
        else:
            cur = 0x00
        seq.append(cur)
    return seq

overall_fail = 0
for level in LEVELS:
    renv = retro.make('SuperMarioBros-Nes-v0', state=level,
                      inttype=retro.data.Integrations.CUSTOM_ONLY,
                      use_restricted_actions=retro.Actions.ALL,
                      render_mode='rgb_array')
    renv.reset()
    renv.data.update_ram()

    def retro_frame(b):
        m = np.zeros(9, dtype=np.uint8)
        m[0] = (b >> 1) & 1; m[2] = (b >> 2) & 1; m[3] = (b >> 3) & 1
        m[4] = (b >> 4) & 1; m[5] = (b >> 5) & 1; m[6] = (b >> 6) & 1
        m[7] = (b >> 7) & 1; m[8] = b & 1
        renv.em.set_button_mask(m, 0)
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

    seq = action_seq(hash(level) & 0xFFFF, FRAMES)
    fail = None
    transients = 0
    pending = 0
    for f, b in enumerate(seq):
        lib.smb_frame(core, b)
        retro_frame(b)
        nr = np.ctypeslib.as_array(lib.smb_ram(core), shape=(0x800,))
        rr = retro_ram()
        if not np.array_equal(nr[GAME], rr[GAME]):
            d = np.nonzero((nr != rr) & GAME)[0]
            print(f'{level}: GAME-STATE MISMATCH frame {f}: {len(d)} bytes '
                  + ' '.join(f'${a:04X}:{nr[a]:02X}!={rr[a]:02X}' for a in d[:6]))
            fail = f
            overall_fail += 1
            break
        full = np.concatenate([nr[:0x100], nr[0x200:]])
        rull = np.concatenate([rr[:0x100], rr[0x200:]])
        if not np.array_equal(full, rull):
            pending += 1
            if pending > 3:
                print(f'{level}: unhealed OAM-shadow divergence frame {f}')
                fail = f
                overall_fail += 1
                break
        else:
            if pending:
                transients += 1
            pending = 0
    if fail is None:
        nr = np.ctypeslib.as_array(lib.smb_ram(core), shape=(0x800,))
        print(f'{level}: PASS {FRAMES} frames ({transients} healed transients) '
              f'end w={nr[0x75F]+1}-{nr[0x75C]+1} x={int(nr[0x6D])*256+int(nr[0x86])} '
              f'lives={nr[0x75A]}')
    lib.smb_destroy(core)
    renv.close()

print('OVERALL:', 'PASS' if overall_fail == 0 else f'{overall_fail} LEVELS FAILED')
