"""Render/obs lockstep bench: native core vs stable-retro on the recorded 4-2 routes.
Run: venv_retro/bin/python tests/render_bench.py        (~15 s, CPU)

Routes (COMPLEX_MOVEMENT indices, recorded on the training path from the
native Level4-2 door): tests/data/vine_route_4-2.npy (vine -> bonus area ->
world-8 pipe) and tests/data/exit_flag_4-2.npy (flag into 4-3).

Start: the native Level4-2 door, exactly where training starts. stable-retro
(its own Level4-2 state: same frame-boundary point, other frame counter/RNG)
adopts the native RAM (all but the stack page, whose contents belong to each
core's own CPU) and the native nametables, palette and OAM by patching its
savestate, so both emulators start from the same game state AND the same
picture.

Two paths per route:
  raw   -- hack-free. Native core A steps with benv_step_raw (the video/win
           search path), core B one frame at a time; stable-retro one frame
           at a time. B vs retro every frame: game-state RAM and the rendered
           frame. A vs retro every 4 frames: RAM and the 84x84 obs.
  train -- the training path: native benv_step (4 frames + the RAM hacks of
           frame_hacks) vs RetroMarioEnv.step x4 (the reference port of the
           same hacks, mario_env.py). Every step: RAM, the last frame, obs.
The table's 'checked' column counts the frames compared (raw: every
emulated frame; train: the last frame of every step).
Checks:
  * game-state RAM identical ($0000-$00FF, $0300-$07FF). Excluded, as in
    native/difftest.py and deep_difftest.py: the stack page $0100-$01FF
    (each core keeps its own; never synced) and the OAM shadow $0200-$02FF
    (a sprite buffer the game rebuilds every frame; it may differ during
    screen transitions and must heal within 3 frames -- checked, and the
    rendered frames compare what is actually displayed).
  * frames pixel-identical: native palette index -> stable-retro RGB with
    stable-retro's own palette, measured here by patching the backdrop
    colour into its savestate (the native RGB palette differs from it).
  * obs identical: the obs benv_step(_raw) returned == the native obs
    pipeline (benv_obs_from_idx: GRAY_LUT, max-pool, HUD crop, 84x84 resize)
    run on stable-retro's frames 3 and 4 of the step.
  * route outcome on the train path: the vine route ends in 8-1, the flag
    route in 4-3 (the replay really followed the recorded trajectory).
"""
import ctypes
import gzip
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mario_env import RetroMarioEnv, _register_integration  # noqa: E402
from mario_native_vecenv import _ACTION_BYTES  # noqa: E402
import stable_retro as retro  # noqa: E402

_register_integration()
LIB = ctypes.CDLL(os.path.join(ROOT, 'native', 'libbatchenv.so'))
V, I = ctypes.c_void_p, ctypes.c_int
for name, res, args in [
        ('benv_create', V, [ctypes.c_char_p, I, I, I, I]),
        ('benv_destroy', None, [V]),
        ('benv_load', None, [V, I, ctypes.c_char_p]),
        ('benv_step', None, [V, V, V, V]),
        ('benv_step_raw', None, [V, I, I, V, V]),
        ('benv_frames', None, [V, I, I, I]),
        ('benv_render_idx', None, [V, I, V]),
        ('benv_obs_from_idx', None, [V, V, V]),
        ('benv_get_ppu', None, [V, I, V, V, V]),
        ('benv_ram', ctypes.POINTER(ctypes.c_uint8), [V, I])]:
    f = getattr(LIB, name); f.restype = res; f.argtypes = args
ROM = open(os.path.join(ROOT, 'retro_integration', 'SuperMarioBros-Nes-v0', 'rom.nes'), 'rb').read()
DOOR = gzip.decompress(open(os.path.join(ROOT, 'native', 'states', 'Level4-2.state'), 'rb').read())
GAME = np.ones(0x800, bool); GAME[0x100:0x300] = False
OAM_SHADOW = slice(0x200, 0x300)
W, H = 240, 224


def pad_mask(b):
    """native pad byte (bit0..7 = A,B,Sel,Start,U,D,L,R) -> stable-retro mask"""
    m = np.zeros(9, np.uint8)
    m[0] = (b >> 1) & 1; m[2] = (b >> 2) & 1; m[3] = (b >> 3) & 1; m[4] = (b >> 4) & 1
    m[5] = (b >> 5) & 1; m[6] = (b >> 6) & 1; m[7] = (b >> 7) & 1; m[8] = b & 1
    return m


def chunk(st, tag, size):
    """offset of a chunk's payload in an FCEUmm savestate (tag + u32 size)"""
    key = tag + struct.pack('<I', size)
    i = st.find(key)
    assert i >= 0 and st.find(key, i + 1) < 0, tag
    return i + 8


def patch_retro(em, native_env):
    """stable-retro adopts the native core's RAM (not the stack page) and PPU memory"""
    st = bytearray(em.get_state())
    ram = np.ctypeslib.as_array(LIB.benv_ram(native_env, 0), shape=(0x800,))
    r = chunk(st, b'RAM\x00', 0x800)
    st[r:r + 0x100] = ram[:0x100].tobytes(); st[r + 0x200:r + 0x800] = ram[0x200:].tobytes()
    vram, pal, oam = (ctypes.create_string_buffer(n) for n in (0x800, 32, 256))
    LIB.benv_get_ppu(native_env, 0, vram, pal, oam)
    pal = bytearray(pal.raw)
    # FCEUmm keeps the backdrop in $3F00/04/08/0C (a $3F00/$3F10 write sets
    # all four, $3F04/08/0C writes are dropped) and draws colour 0 of every
    # palette from them; the native core stores what the game wrote there
    # (never displayed: colour 0 is the backdrop)
    pal[4] = pal[8] = pal[12] = pal[0]
    for tag, buf in ((b'NTAR', vram.raw), (b'PRAM', bytes(pal)), (b'SPRA', oam.raw)):
        o = chunk(st, tag, len(buf)); st[o:o + len(buf)] = buf
    em.set_state(bytes(st))


def retro_palette():
    """stable-retro's RGB for each NES colour index: patch the backdrop
    colour into a savestate and read an open-sky pixel of 1-1"""
    env = retro.make('SuperMarioBros-Nes-v0', state='Level1-1',
                     inttype=retro.data.Integrations.CUSTOM_ONLY,
                     use_restricted_actions=retro.Actions.ALL, render_mode='rgb_array')
    env.reset()
    base = env.em.get_state(); o = chunk(base, b'PRAM', 32)
    pal = np.zeros(64, np.int64)
    for c in range(64):
        st = bytearray(base); st[o] = c
        env.em.set_state(bytes(st)); env.em.set_button_mask(np.zeros(9, np.uint8), 0); env.em.step()
        p = env.em.get_screen()[100, 20].astype(np.int64)
        pal[c] = (p[0] << 16) | (p[1] << 8) | p[2]
    env.close()
    return pal


RPAL = retro_palette()
# stable-retro RGB -> a native palette index (first of the duplicates; the
# duplicate colours -- the blacks, 0x20/0x30 white -- share one GRAY_LUT value)
INV = {}
for c in range(63, -1, -1):
    INV[int(RPAL[c])] = c
INV_KEYS = np.array(sorted(INV), np.int64)
INV_VALS = np.array([INV[k] for k in INV_KEYS], np.uint8)


def pack(rgb):
    rgb = rgb.reshape(-1, 3).astype(np.int64)
    return (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]


def to_idx(rgb):
    k = pack(rgb); j = np.searchsorted(INV_KEYS, k)
    assert (INV_KEYS[np.minimum(j, len(INV_KEYS) - 1)] == k).all(), 'colour outside the retro palette'
    return INV_VALS[j]


def native_idx(env, i):
    out = np.zeros(W * H, np.uint8)
    LIB.benv_render_idx(env, i, out.ctypes.data)
    return out


def obs_of(idx_a, idx_b):
    out = np.zeros((84, 84), np.uint8)
    LIB.benv_obs_from_idx(np.ascontiguousarray(idx_a).ctypes.data,
                          np.ascontiguousarray(idx_b).ctypes.data, out.ctypes.data)
    return out


class Tally:
    def __init__(self, route, path):
        self.route, self.path = route, path
        self.frames = self.steps = 0
        self.ram_bad = []; self.oam_run = 0; self.oam_trans = 0; self.oam_unhealed = []
        self.px_frames = []; self.obs_steps = []; self.outcome = None; self.ok_outcome = True

    def ram(self, t, nr, rr):
        if not np.array_equal(nr[GAME], rr[GAME]) and len(self.ram_bad) < 5:
            d = np.nonzero((nr != rr) & GAME)[0]
            self.ram_bad.append((t, ' '.join('$%04X:%02X!=%02X' % (a, nr[a], rr[a]) for a in d[:4])))
        if np.array_equal(nr[OAM_SHADOW], rr[OAM_SHADOW]):
            self.oam_trans += self.oam_run > 0; self.oam_run = 0
        else:
            self.oam_run += 1
            if self.oam_run > 3 and len(self.oam_unhealed) < 5:
                self.oam_unhealed.append(t)

    def pixels(self, t, idx, rgb):
        n = int(np.count_nonzero(RPAL[idx.astype(np.int64)] != pack(rgb)))
        if n:
            self.px_frames.append((t, n))

    def obs(self, t, got, want):
        if not np.array_equal(got, want):
            self.obs_steps.append((t, int(np.count_nonzero(got != want))))

    def passed(self):
        return not (self.ram_bad or self.oam_unhealed or self.px_frames or self.obs_steps) and self.ok_outcome


def info(ram):
    r = ram.astype(int)
    return '%d-%d x=%d' % (r[0x75F] + 1, r[0x75C] + 1, r[0x6D] * 256 + r[0x86])


def raw_pass(route, acts):
    T = Tally(route, 'raw')
    env = retro.make('SuperMarioBros-Nes-v0', state='Level4-2',
                     inttype=retro.data.Integrations.CUSTOM_ONLY,
                     use_restricted_actions=retro.Actions.ALL, render_mode='rgb_array')
    env.reset()
    ne = LIB.benv_create(ROM, len(ROM), 2, 1, 0)      # 0: benv_step_raw, 1: per frame
    LIB.benv_load(ne, 0, DOOR); LIB.benv_load(ne, 1, DOOR)
    patch_retro(env.em, ne)
    ram_a = np.ctypeslib.as_array(LIB.benv_ram(ne, 0), shape=(0x800,))
    ram_b = np.ctypeslib.as_array(LIB.benv_ram(ne, 1), shape=(0x800,))
    obs = np.zeros((2, 84, 84), np.uint8); ram_out = np.zeros((2, 0x800), np.uint8)
    for t, a in enumerate(acts):
        bt = int(_ACTION_BYTES[a])
        LIB.benv_step_raw(ne, 0, bt, obs.ctypes.data, ram_out.ctypes.data)
        rf = []
        for k in range(4):
            LIB.benv_frames(ne, 1, 1, bt)
            env.em.set_button_mask(pad_mask(bt), 0); env.em.step(); env.data.update_ram()
            rr = np.frombuffer(env.get_ram(), np.uint8)[:0x800]
            scr = env.em.get_screen()
            T.frames += 1
            T.ram(T.frames, ram_b, rr)
            T.pixels(T.frames, native_idx(ne, 1), scr)
            rf.append(to_idx(scr))
        T.steps += 1
        if not np.array_equal(ram_a[GAME], rr[GAME]) and len(T.ram_bad) < 5:
            T.ram_bad.append(('step %d (benv_step_raw core)' % t, ''))
        T.obs(t, obs[0], obs_of(rf[2], rf[3]))
    T.outcome = info(ram_b)
    LIB.benv_destroy(ne); env.close()
    return T


def train_pass(route, acts, want):
    T = Tally(route, 'train')
    env = RetroMarioEnv(random_stages=['4-2'], full_game=True)
    env.reset()
    ne = LIB.benv_create(ROM, len(ROM), 1, 1, 0)      # full game, as the 4-2 config
    LIB.benv_load(ne, 0, DOOR)
    patch_retro(env._em, ne)
    env._data.update_ram(); env._cur = env._data.lookup_all()
    obs = np.zeros((1, 84, 84), np.uint8); ram_out = np.zeros((1, 0x800), np.uint8)
    act = np.zeros(1, np.int32)
    for t, a in enumerate(acts):
        act[0] = _ACTION_BYTES[a]
        LIB.benv_step(ne, act.ctypes.data, obs.ctypes.data, ram_out.ctypes.data)
        rf = [to_idx(env.step(int(a))[0]) for _ in range(4)]
        rr = np.frombuffer(env._retro.get_ram(), np.uint8)[:0x800]
        T.steps += 1; T.frames += 1
        T.ram(t, ram_out[0], rr)
        T.pixels(t, native_idx(ne, 0), env._em.get_screen())
        T.obs(t, obs[0], obs_of(rf[2], rf[3]))
    T.outcome = info(ram_out[0])
    T.ok_outcome = T.outcome.startswith(want)
    LIB.benv_destroy(ne); env.close()
    return T


def main():
    # the two action tables agree (native pad bytes vs the retro chain's masks)
    probe = RetroMarioEnv(target=(1, 1))
    same = all(np.array_equal(probe._masks[a], pad_mask(int(_ACTION_BYTES[a]))) for a in range(12))
    probe.close()
    rows = []
    for route, fname, want in (('vine', 'vine_route_4-2.npy', '8-1'), ('flag', 'exit_flag_4-2.npy', '4-3')):
        acts = np.load(os.path.join(ROOT, 'tests', 'data', fname)).astype(int)
        rows.append(raw_pass(route, acts))
        rows.append(train_pass(route, acts, want))
    print('action tables (native bytes vs RetroMarioEnv masks) agree:', same)
    print('%-5s %-6s %6s %7s | %-8s %-9s %-10s %-9s | %-14s %s' % (
        'route', 'path', 'steps', 'checked', 'RAM', 'OAM-heal', 'px-frames', 'obs', 'end', 'result'))
    for T in rows:
        print('%-5s %-6s %6d %7d | %-8s %-9s %-10s %-9s | %-14s %s' % (
            T.route, T.path, T.steps, T.frames, len(T.ram_bad) or 'ok',
            ('%d ok' % T.oam_trans) if not T.oam_unhealed else 'FAIL', len(T.px_frames) or 'ok',
            len(T.obs_steps) or 'ok', T.outcome + ('' if T.ok_outcome else ' (!)'), 'PASS' if T.passed() else 'FAIL'))
        for what, lst in (('RAM', T.ram_bad), ('OAM unhealed', T.oam_unhealed), ('pixels', T.px_frames[:5]),
                          ('obs', T.obs_steps[:5])):
            if lst:
                print('      first %s mismatches: %s' % (what, lst))
    ok = same and all(T.passed() for T in rows)
    print('\nrender bench:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
