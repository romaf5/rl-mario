#!/usr/bin/env python3
"""Generate stable-retro .state files for every SMB level (worlds 1-8, stages 1-4).

Replicates gym_super_mario_bros' level selection: boot the ROM from power-on,
skip the start screen by pressing START, and while the pre-level timer runs,
write the target (world, stage, area) into RAM (0x075F/0x075C/0x0760) so the
game loads that level instead of 1-1 (see smb_env.py::_write_stage /
_skip_start_screen in gym_super_mario_bros).

States are written to retro_integration/SuperMarioBros-Nes-v0/Level<W>-<S>.state
(plus FullGame.state = power-on start of a normal 3-life game at 1-1).

Run from venv_retro:
    ./venv_retro/bin/python gen_states.py
"""

import gzip
import os

import numpy as np
import stable_retro as retro

HERE = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(HERE, 'retro_integration')
GAME = 'SuperMarioBros-Nes-v0'
STATE_DIR = os.path.join(INTEGRATION_PATH, GAME)

# NES button order in stable-retro:
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
START = 3

# RAM addresses (same as gym_super_mario_bros)
ADDR_WORLD = 0x075F   # current world, 0-based
ADDR_STAGE = 0x075C   # current stage, 0-based
ADDR_AREA = 0x0760    # current area, 0-based
ADDR_PRELEVEL_TIMER = 0x07A0


def decode_area(world, stage):
    """Area for (world, stage), from gym_super_mario_bros decode_target."""
    area = stage
    if world in {1, 2, 4, 7} and stage >= 2:
        area += 1
    return area


class StateGenerator:
    def __init__(self):
        retro.data.Integrations.add_custom_path(INTEGRATION_PATH)
        self.env = retro.make(
            GAME, state=retro.State.NONE,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            use_restricted_actions=retro.Actions.ALL,
            render_mode='rgb_array')
        self.env.reset()
        # Snapshot power-on so every level generation starts identically
        self._poweron = self.env.em.get_state()

    def _frame(self, *buttons):
        mask = np.zeros(9, dtype=np.uint8)
        for b in buttons:
            mask[b] = 1
        self.env.em.set_button_mask(mask, 0)
        self.env.em.step()
        self.env.data.update_ram()

    def _vals(self):
        return self.env.data.lookup_all()

    def _write(self, addr, value):
        self.env.data.memory.assign(addr, '|u1', value)

    def _time(self):
        return self._vals()['time']

    def make_state(self, target=None):
        """Boot from power-on and advance to the start of the target level.

        target: (world, stage) 1-based, or None for a normal full game start.
        Returns the raw emulator state bytes.
        """
        self.env.em.set_state(self._poweron)
        self._frame()

        # press and release START once (skip title)
        self._frame(START)
        self._frame()

        # Press START until the game starts; write the target stage into RAM
        # while the pre-level timer runs (exactly like smb_env.py).
        for _ in range(2000):
            v = self._vals()
            if v['life'] == 2 and 0 < v['time'] <= 401:
                break
            self._frame(START)
            if target is not None:
                world, stage = target
                self._write(ADDR_WORLD, world - 1)
                self._write(ADDR_STAGE, stage - 1)
                self._write(ADDR_AREA, decode_area(world, stage) - 1)
            self._frame()
            self._write(ADDR_PRELEVEL_TIMER, 0)
        else:
            raise RuntimeError(f'start screen skip failed for {target}')

        # idle a few frames until the in-game timer starts ticking
        time_last = self._time()
        for _ in range(2000):
            if self._time() < time_last:
                break
            time_last = self._time()
            self._frame(START)
            self._frame()
        else:
            raise RuntimeError(f'timer never started for {target}')

        v = self._vals()
        if target is not None:
            world, stage = target
            if v['world0'] != world - 1 or v['stage0'] != stage - 1:
                raise RuntimeError(
                    f'wanted {world}-{stage}, got '
                    f"{v['world0'] + 1}-{v['stage0'] + 1}")
        return self.env.em.get_state(), v

    def close(self):
        self.env.close()


def main():
    gen = StateGenerator()
    os.makedirs(STATE_DIR, exist_ok=True)

    state, v = gen.make_state(None)
    path = os.path.join(STATE_DIR, 'FullGame.state')
    with gzip.open(path, 'wb') as f:
        f.write(state)
    print(f"FullGame.state: world={v['world0']+1}-{v['stage0']+1} "
          f"time={v['time']} life={v['life']}")

    failed = []
    for world in range(1, 9):
        for stage in range(1, 5):
            try:
                state, v = gen.make_state((world, stage))
            except RuntimeError as e:
                print(f'Level{world}-{stage}: FAILED ({e})')
                failed.append((world, stage))
                continue
            path = os.path.join(STATE_DIR, f'Level{world}-{stage}.state')
            with gzip.open(path, 'wb') as f:
                f.write(state)
            print(f"Level{world}-{stage}.state: time={v['time']} "
                  f"life={v['life']} x={v['xpos_hi']*256 + v['xpos_lo']}")

    gen.close()
    if failed:
        print(f'\nFailed levels: {failed}')
    else:
        print('\nAll 32 level states + FullGame.state generated.')


if __name__ == '__main__':
    main()
