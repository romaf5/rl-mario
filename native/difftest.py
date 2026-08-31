"""Lockstep differential test: native smbcore vs stable-retro.

Boots both from power-on (native adopts retro's initial RAM), feeds
identical button sequences, compares all 2KB of work RAM every frame.
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
ROM = os.path.join(HERE, '..', 'retro_integration',
                   'SuperMarioBros-Nes-v0', 'rom.nes')

lib = ctypes.CDLL(os.path.join(HERE, 'libsmbcore.so'))
lib.smb_create.restype = ctypes.c_void_p
lib.smb_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.smb_frame.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
lib.smb_ram.restype = ctypes.POINTER(ctypes.c_uint8)
lib.smb_ram.argtypes = [ctypes.c_void_p]
lib.smb_set_ram.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

rom = open(ROM, 'rb').read()
core = lib.smb_create(rom, len(rom))
assert core, 'smb_create failed'


def native_ram():
    return np.ctypeslib.as_array(lib.smb_ram(core), shape=(0x800,))


# retro reference
from mario_env import RetroMarioEnv, _register_integration  # noqa: E402
import stable_retro as retro  # noqa: E402
_register_integration()
renv = retro.make('SuperMarioBros-Nes-v0', state=retro.State.NONE,
                  inttype=retro.data.Integrations.CUSTOM_ONLY,
                  use_restricted_actions=retro.Actions.ALL,
                  render_mode='rgb_array')
renv.reset()

# native pad bits: A,B,Sel,Start,U,D,L,R (bit0..7)
# retro mask order: ['B',None,'SELECT','START','UP','DOWN','LEFT','RIGHT','A']
def to_retro_mask(b):
    m = np.zeros(9, dtype=np.uint8)
    m[0] = (b >> 1) & 1   # B
    m[2] = (b >> 2) & 1   # Select
    m[3] = (b >> 3) & 1   # Start
    m[4] = (b >> 4) & 1   # Up
    m[5] = (b >> 5) & 1   # Down
    m[6] = (b >> 6) & 1   # Left
    m[7] = (b >> 7) & 1   # Right
    m[8] = b & 1          # A
    return m


def retro_frame(b):
    renv.em.set_button_mask(to_retro_mask(b), 0)
    renv.em.step()
    renv.data.update_ram()


def retro_ram():
    return np.frombuffer(renv.get_ram(), dtype=np.uint8)[:0x800]


# boot both cores independently past the boot transient, then sync only
# the divergent NON-STACK bytes (each CPU keeps its own stack/registers;
# at the frame boundary both idle in the same NMI-wait loop, so other
# state is quiescent). After this, any divergence is a real emu bug.
for _ in range(30):
    lib.smb_frame(core, 0)
    retro_frame(0)
nv = np.ctypeslib.as_array(lib.smb_ram(core), shape=(0x800,))
rv = retro_ram()
synced = 0
for a in np.nonzero(nv != rv)[0]:
    if a < 0x100 or a >= 0x200:
        nv[a] = rv[a]
        synced += 1
print(f'post-boot sync: {synced} bytes copied (stack page untouched)')

# scripted sequence: boot idle, START presses, then plausible gameplay
rng = np.random.RandomState(0)
seq = []
seq += [0] * 60
seq += ([8] * 2 + [0] * 8) * 12          # tap START
for i in range(4000):                     # gameplay-ish inputs
    r = rng.random_sample()
    if r < 0.5:
        seq.append(0x82)                  # Right+B
    elif r < 0.8:
        seq.append(0x83)                  # Right+B+A
    elif r < 0.9:
        seq.append(0x40)                  # Left
    else:
        seq.append(0)

# two-tier check:
#  - game-state bytes (all RAM except stack $01xx and OAM shadow $02xx)
#    must match EVERY frame
#  - full-RAM transients (OAM shadow during screen transitions) must heal
#    within 3 frames
game = np.ones(0x800, dtype=bool)
game[0x100:0x300] = False
pending = 0
transients = 0
fail = None
for f, b in enumerate(seq):
    lib.smb_frame(core, b)
    retro_frame(b)
    nr = native_ram()
    rr = retro_ram()
    if not np.array_equal(nr[game], rr[game]):
        d = np.nonzero((nr != rr) & game)[0]
        print(f'GAME-STATE MISMATCH at frame {f}: {len(d)} bytes')
        for a in d[:10]:
            print(f'  ${a:04X}: native={nr[a]:02X} retro={rr[a]:02X}')
        fail = f
        break
    if not np.array_equal(nr, rr):
        pending += 1
        if pending > 3:
            print(f'UNHEALED render-buffer divergence at frame {f}')
            fail = f
            break
    else:
        if pending:
            transients += 1
        pending = 0
if fail is None:
    nr = native_ram()
    print(f'PASS: {len(seq)} frames, game-state RAM identical every frame; '
          f'{transients} self-healing render transients')
    print(f"end state: world={nr[0x75F]+1}-{nr[0x75C]+1} "
          f"x={int(nr[0x6D])*256+int(nr[0x86])} lives={nr[0x75A]}")
renv.close()
