# smbsearch: C++ route search for SMB (discover -> optimise -> demo)

Status: approved 2026-09-19. Phase 1: 4-2. Phase 2: end to end (1-1 -> 8-4 axe, warp route).

## Why

RL runs find the 4-2 route (explorers, minutes) but never turn it into a reliable policy; run l's route is
1478 steps with ~500 wasted. With a deterministic emulator, savestates and 12k frames/s per thread, search
solves this class of problem directly and produces the demonstrations for later distillation.

## Pipeline

```
start state ─► EXPLORE (Go-Explore in C++) ──► reference path ─► OPTIMISE (beam A* over time) ──► route
  (door /                cells, random walks,        (any exit into          dedupe + dominance,         (actions,
   arrival)              fastest state per cell)      the next route level)   waypoint progress rank      frames)
                                                                                       │
e2e: route's arrival state in level k+1 = next segment's start ◄──────────────────────┘
                                                                                       ▼
                        verify in stable-retro (lockstep RAM) ─► demo mp4 / GIF with HUD + splits
```

## Folder (all new code in `search/`, nothing else in the repo changes)

| path | responsibility |
|---|---|
| `search/include/smbsearch.h` | the C API (only thing Python and the CLI see) |
| `search/src/emu/` | compact state (~5.5 KB: Core minus PRG/CHR), hack-free frame stepping, RAM decode; the ONLY file that includes `native/smbcore.cpp` |
| `search/src/core/` | thread pool, AVX2 hashing, sharded hash tables |
| `search/src/route/` | segments (start, route levels, goal), waypoints from a reference path, action files |
| `search/src/explore/` | Go-Explore discovery |
| `search/src/optimize/` | beam A* over time |
| `search/src/api.cpp`, `search/src/cli.cpp` | C API; `smbsearch` CLI (`bench`, `explore`, `optimize`, `e2e`) |
| `search/python/smbsearch/` | ctypes wrapper (`Search`: `explore`, `optimize`, `replay`, `bench`); long calls release the GIL |
| `search/tools/` | `verify_retro.py` (stable-retro lockstep + frames), `render_demo.py` (mp4/GIF, HUD, splits) |
| `search/tests/search_bench.py` | checks (below) |
| `search/build.sh` | `-O3 -march=native -flto`, PGO (profile run on 4-2) -> `libsmbsearch.so` + `smbsearch` |
| `search/out/` | results (gitignored): routes, stats JSON, videos |

## Semantics

- Stepping: the real game (no training hacks); one decision per 4 frames, the policy's 12 actions (COMPLEX_MOVEMENT).
- Segment goal: the first frame whose level is the next route level (8-4: `$0770 == 2`, the axe); cost = frames.
  The next segment starts at the first in-control frame of that level (NOOPs through the transition, counted).
- Dead ends: a death (`$0E` 0x0B / life lost / y page > 1), an off-route level, or the timer < 10 end a branch.

## Explore

- Cell = (level, area `$0760`, sub-area `$074F`, AreaType `$074E`, x/32, y/16, camera x/64, tile signature of
  Mario's 128-px bin) -- the key that found the 4-2 route in the Python env.
- Each cell keeps its fastest state (fewest frames) and the actions to it. Pick weight 1/sqrt(1 + picks);
  walks of <= 300 steps, random actions held 1/2/4/8 steps.
- Stops `settle` seconds after the first goal (keeps the fastest goal path found) or at the budget.

## Optimise (beam A* over time)

- Waypoints: replay the reference path; each maximal run in one frame (level, area, sub, AreaType) gives
  (frame, exit x); `rank = |exit_x[k] - x| + sum_{j>k} |exit_x[j] - entry_x[j]|` px, k = the node's current
  waypoint (advances when the node enters frame k+1; other frames are dropped). Ties: higher x speed.
- Depth d = all nodes at 4d frames. Children of the B best nodes x 12 actions are stepped in parallel;
  dedupe by a 64-bit key hash (Mario x incl. subpixel, y, speeds, float state, power-up, camera x, frame,
  enemy slots, tile buffer hash); a global closed set drops keys reached at an earlier depth.
- The first depth with a goal child ends the search; parents give the actions.

## Verification and demo

- `verify_retro.py`: stable-retro adopts the native start state (the deep_difftest method), replays the route
  frame by frame; game-state RAM must match every frame and the goal must be reached. Its frames are the demo.
- `render_demo.py`: 60 fps mp4 + GIF with HUD (level, frame counter, split per level, delta to a baseline).

## Efficiency

Compact states (8x less copying than the 45 KB savestate), no rendering in search, 40 threads alongside the
running RL job (all 64 when alone), lock-sharded tables, AVX2 for hashing/compare, LTO + PGO. The 6502
interpreter is branchy (SIMD does not apply there). Baseline: 11.8k frames/s/thread, 325k frames/s at 40 threads.

## Tests (`search/tests/search_bench.py`)

1. compact save/load round-trips: stepping from a restored compact state equals stepping the original.
2. engine stepping equals `benv_step_raw` / `smb_frame` frame by frame (RAM identical).
3. explore finds the 1-1 flag; optimise beats a run-right-and-jump baseline on 1-1; both replay to the goal.
4. the 4-2 route reaches 8-1 on replay and in `verify_retro.py`.

## Deliverables

Phase 1: `search/out/4-2/` route, stats, `demo.mp4` / `demo.gif` (+ side-by-side vs run l's route).
Phase 2: `search/out/e2e/` one continuous 1-1 -> 8-4 demo with per-level splits.
