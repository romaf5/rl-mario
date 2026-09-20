# Super Mario Bros, solved by search

The whole game, 1-1 to Bowser's axe, found by a C++ search on the real game in 45 minutes on
one machine: **5:26.4** from first control to the axe (16,324 frames; the ROM is the European
version, a PAL game at 50 fps). [Full video: 4x size, real speed](docs/media/full_game.mp4).

| level | route found → optimised (decisions) | search | PAL TAS | gap |
|---|---|---|---|---|
| 1-1 | 522 → 396 | 31.7 s | 28.5 s | +3.2 s |
| 1-2 (warp zone → world 4) | 736 → 289 | 32.4 s | 30.2 s | +2.2 s |
| 4-1 | 495 → 455 | 40.0 s | 37.2 s | +2.9 s |
| 4-2 (vine → warp zone → world 8) | 800 → 359 | 38.1 s | 28.0 s | +10.1 s |
| 8-1 | 873 → 625 | 53.8 s | 50.6 s | +3.2 s |
| 8-2 | 549 → 450 | 38.9 s | 35.7 s | +3.2 s |
| 8-3 | 459 → 441 | 38.2 s | 33.1 s | +5.0 s |
| 8-4 (to the axe) | 981 → 632 | 53.4 s | 48.5 s | +4.9 s |
| **total** | | **5:26.4** | **4:51.7** | **+34.7 s** |

One decision every 4 frames, from the 12 actions of COMPLEX_MOVEMENT. Times include each
level's intro screen and end sequence. PAL TAS: HappyLee's
[Super Mario Bros. (Europe) "warps"](https://tasvideos.org/6622M), replayed on this ROM in
stable-retro and timed the same way. The famous 4:54 records are for the NTSC version
(60 fps, different timers and physics), so they don't compare.

Where the 34.7 s go:

| | search | TAS | gap |
|---|---|---|---|
| level ends (flag slide, castle walk, score count, fireworks) | 59.7 s | 43.0 s | +16.7 s |
| running | 204.4 s | 193.1 s | +11.3 s |
| pipes, the vine, level entrances | 38.1 s | 31.3 s | +6.8 s |

**4-2: two hidden blocks, the vine, the warp zone** (real speed)

![4-2](docs/media/4-2_vine_warp.gif)

**8-4: the maze, the water, Bowser** (real speed)

![8-4](docs/media/8-4_finish.gif)

## How it works

```
level start ─► EXPLORE ─────────────► a route out ─► OPTIMISE ─────────────► fast route ─► next level
               Go-Explore in C++:                    beam A* over time:
               cells of (area, x, y, camera,         nodes ranked by how far along the
               tiles); walk from new cells first;    found route they are; bumped blocks
               keep the fastest state per cell       and grown vines are checkpoints
```

- **Real game.** No emulator shortcuts: pipes, flag sequences and castles play out in full.
- **Verified.** Every route is replayed in [stable-retro](https://github.com/Farama-Foundation/stable-retro),
  the reference emulator: game RAM matches our core on every one of the 16,328 frames.
- **Fast.** A single-purpose NES core (`native/`), 4.6 KB compact savestates, 64 threads:
  ~375k emulated frames per second.
- **No route hints.** It is given only the level order of the warp route (1-1, 1-2, 4-1, 4-2,
  8-1 to 8-4); where the warp zones, the vine and the way through the 8-4 maze are, it found itself.

## Run it

```bash
./setup.sh                      # venv, dependencies, ROM, builds
search/full_game.sh             # search 1-1 -> 8-4, verify, render search/out/e2e/demo.mp4
```

One level: `python search/tools/solve.py --start 4-2 --segments 1 --out search/out/4-2`.
Python API and engine layout: [search/README.md](search/README.md).

## Repository

| path | |
|---|---|
| `search/` | the search engine (C++), its Python interface, tools and checks |
| `native/` | the NES core, a batched renderer, level states, lockstep tests vs stable-retro |
| `retro_integration/` | stable-retro integration for Super Mario Bros (used for verification) |

## Next: SMBZero

A policy network trained from the search (expert iteration: the search teaches the network,
the network guides the search), meant to play on its own even when the game's timing differs
from run to run (random delays at the start shift enemies and the RNG).

The earlier reinforcement-learning attempts (PPO, GRPO, curricula) are in git tag `pre-cleanup`.
