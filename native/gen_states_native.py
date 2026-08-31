"""Generate native-core savestates for every SMB level (mirrors
gen_states.py: boot, press START, poke world/stage/area RAM during the
pre-level timer). States go to native/states/Level<W>-<S>.state (+FullGame).
"""
import ctypes
import gzip
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROM = os.path.join(HERE, '..', 'retro_integration',
                   'SuperMarioBros-Nes-v0', 'rom.nes')
OUT = os.path.join(HERE, 'states')

lib = ctypes.CDLL(os.path.join(HERE, 'libsmbcore.so'))
lib.smb_create.restype = ctypes.c_void_p
lib.smb_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.smb_frame.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
lib.smb_ram.restype = ctypes.POINTER(ctypes.c_uint8)
lib.smb_ram.argtypes = [ctypes.c_void_p]
lib.smb_state_size.restype = ctypes.c_int
lib.smb_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.smb_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

START = 0x08


def decode_area(world, stage):
    area = stage
    if world in {1, 2, 4, 7} and stage >= 2:
        area += 1
    return area


def make_state(core, ram, poweron, target):
    def T():
        return int(ram[0x7F8]) * 100 + int(ram[0x7F9]) * 10 + int(ram[0x7FA])

    lib.smb_load(core, poweron)
    lib.smb_frame(core, 0)
    lib.smb_frame(core, START)
    lib.smb_frame(core, 0)
    # phase 1: press START until gameplay begins; poke the target level
    # into RAM while the pre-level timer runs
    for _ in range(2000):
        if ram[0x770] == 1 and ram[0x75A] == 2 and 0 < T() <= 401:
            break
        lib.smb_frame(core, START)
        if target is not None:
            world, stage = target
            ram[0x75F] = world - 1
            ram[0x75C] = stage - 1
            ram[0x760] = decode_area(world, stage) - 1
        lib.smb_frame(core, 0)
        ram[0x7A0] = 0
    else:
        raise RuntimeError(f'start-screen skip failed for {target}')

    # phase 2: wait for the SECOND timer decrement -- the first may be the
    # level's preset write (401 -> 400); the second is a genuine tick
    decs, tl = 0, T()
    for _ in range(6000):
        lib.smb_frame(core, 0)
        t = T()
        if t == tl - 1 and 0 < t <= 400:
            decs += 1
            if decs == 2:
                break
        tl = t
    else:
        raise RuntimeError(f'timer never ticked twice for {target}')

    if target is not None:
        world, stage = target
        assert ram[0x75F] == world - 1 and ram[0x75C] == stage - 1, \
            f'wanted {world}-{stage}, got {ram[0x75F]+1}-{ram[0x75C]+1}'
    buf = ctypes.create_string_buffer(lib.smb_state_size())
    lib.smb_save(core, buf)
    return buf.raw


def main():
    os.makedirs(OUT, exist_ok=True)
    rom = open(ROM, 'rb').read()
    core = lib.smb_create(rom, len(rom))
    ram = np.ctypeslib.as_array(lib.smb_ram(core), shape=(0x800,))
    poweron = ctypes.create_string_buffer(lib.smb_state_size())
    lib.smb_save(core, poweron)

    st = make_state(core, ram, poweron, None)
    with gzip.open(os.path.join(OUT, 'FullGame.state'), 'wb') as f:
        f.write(st)
    t = int(ram[0x7F8]) * 100 + int(ram[0x7F9]) * 10 + int(ram[0x7FA])
    print(f'FullGame: {ram[0x75F]+1}-{ram[0x75C]+1} time={t} '
          f'lives={ram[0x75A]}')

    for world in range(1, 9):
        for stage in range(1, 5):
            st = make_state(core, ram, poweron, (world, stage))
            with gzip.open(os.path.join(OUT, f'Level{world}-{stage}.state'),
                           'wb') as f:
                f.write(st)
            x = int(ram[0x6D]) * 256 + int(ram[0x86])
            t = int(ram[0x7F8]) * 100 + int(ram[0x7F9]) * 10 + int(ram[0x7FA])
            print(f'Level{world}-{stage}: time={t} x={x} lives={ram[0x75A]}')
    print('done')


if __name__ == '__main__':
    main()
