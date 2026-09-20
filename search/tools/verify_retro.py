"""Replay a route in stable-retro (the reference emulator) in lockstep with the
native core: stable-retro adopts the native start state (RAM, nametables,
palette, OAM -- the deep_difftest method), both get the same buttons every
frame, and game-state RAM must match every frame.

  venv_retro/bin/python search/tools/verify_retro.py search/out/4-2/route.npz
"""
import ctypes, os, struct, sys
import numpy as np
SEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SEARCH, 'python'))
from smbsearch import ACTION_BUTTONS, REPO, ROM, load_state, level_name

GAME = np.ones(0x800, dtype=bool)
GAME[0x100:0x300] = False            # stack page + OAM shadow (as native/deep_difftest.py)


def _pad_mask(b):
    m = np.zeros(9, dtype=np.uint8)
    m[0] = (b >> 1) & 1; m[2] = (b >> 2) & 1; m[3] = (b >> 3) & 1
    m[4] = (b >> 4) & 1; m[5] = (b >> 5) & 1; m[6] = (b >> 6) & 1
    m[7] = (b >> 7) & 1; m[8] = b & 1
    return m


def _chunk(st, tag, size):
    key = tag + struct.pack('<I', size)
    i = st.find(key)
    assert i >= 0 and st.find(key, i + 1) < 0, tag
    return i + 8


class _Native:
    """the native core on the hack-free path, frame by frame (native/libbatchenv.so)"""
    def __init__(self, state):
        L = self.lib = ctypes.CDLL(os.path.join(REPO, 'native', 'libbatchenv.so'))
        P, I = ctypes.c_void_p, ctypes.c_int
        L.benv_create.restype = P; L.benv_create.argtypes = [ctypes.c_char_p, I, I, I, I]
        L.benv_destroy.argtypes = [P]
        L.benv_load.argtypes = [P, I, ctypes.c_char_p]
        L.benv_ram.restype = ctypes.POINTER(ctypes.c_uint8); L.benv_ram.argtypes = [P, I]
        L.benv_get_ppu.argtypes = [P, I] + [ctypes.c_char_p] * 3
        L.benv_frames.argtypes = [P, I, I, I]
        rom = open(ROM, 'rb').read()
        self.env = L.benv_create(rom, len(rom), 1, 1, 0)
        L.benv_load(self.env, 0, state)

    def ram(self):
        return np.ctypeslib.as_array(self.lib.benv_ram(self.env, 0), shape=(0x800,))

    def ppu(self):
        bufs = [ctypes.create_string_buffer(n) for n in (0x800, 32, 256)]
        self.lib.benv_get_ppu(self.env, 0, *bufs)
        return [bytearray(b.raw) for b in bufs]

    def frame(self, buttons):
        self.lib.benv_frames(self.env, 0, 1, int(buttons))

    def close(self):
        self.lib.benv_destroy(self.env)


def retro_state_of(start):
    """stable-retro savestate that supplies CPU/mapper registers for a native start"""
    return 'FullGame' if start == 'FullGame' else 'Level%s' % start


def verify(start_state, actions, on_frame=None, retro_state='Level1-1'):
    import stable_retro as retro
    retro.data.Integrations.add_custom_path(os.path.join(REPO, 'retro_integration'))
    nat = _Native(start_state)
    renv = retro.make('SuperMarioBros-Nes-v0', state=retro_state, inttype=retro.data.Integrations.CUSTOM_ONLY,
                      use_restricted_actions=retro.Actions.ALL, render_mode='rgb_array')
    renv.reset()
    st = bytearray(renv.em.get_state())
    ram = nat.ram()
    r = _chunk(st, b'RAM\x00', 0x800)
    st[r:r + 0x100] = ram[:0x100].tobytes(); st[r + 0x200:r + 0x800] = ram[0x200:].tobytes()
    vram, pal, oam = nat.ppu()
    pal[4] = pal[8] = pal[12] = pal[0]
    for tag, b in zip((b'NTAR', b'PRAM', b'SPRA'), (vram, pal, oam)):
        o = _chunk(st, tag, len(b)); st[o:o + len(b)] = b
    renv.em.set_state(bytes(st))
    f = 0
    rr = np.frombuffer(renv.get_ram(), dtype=np.uint8)[:0x800]
    for a in np.asarray(actions, dtype=np.uint8):
        b = ACTION_BUTTONS[int(a)]
        for _ in range(4):
            nat.frame(b)
            renv.em.set_button_mask(_pad_mask(b), 0)
            renv.em.step()
            rr = np.frombuffer(renv.get_ram(), dtype=np.uint8)[:0x800]
            nr = nat.ram()
            if not np.array_equal(nr[GAME], rr[GAME]):
                d = np.nonzero((nr != rr) & GAME)[0]
                renv.close()
                return dict(ok=False, frames=f, mismatch=['$%04X:%02X!=%02X' % (x, nr[x], rr[x]) for x in d[:6]])
            if on_frame is not None:
                on_frame(f, renv.em.get_screen(), rr)
            f += 1
    renv.close()
    return dict(ok=True, frames=f, mismatch=[], end_level=level_name(int(rr[0x75F]) * 4 + int(rr[0x75C])))


if __name__ == '__main__':
    z = np.load(sys.argv[1])
    res = verify(load_state(str(z['start'])), z['actions'], retro_state=retro_state_of(str(z['start'])))
    print('[verify] %s: %d frames, %s%s' % ('PASS' if res['ok'] else 'FAIL', res['frames'],
                                            res.get('end_level', ''), ' ' + ' '.join(res['mismatch'])))
    sys.exit(0 if res['ok'] else 1)
