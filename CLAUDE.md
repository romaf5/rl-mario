# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## Project

Super Mario Bros solved by search: a C++ engine explores each level of the warp route
(1-1, 1-2, 4-1, 4-2, 8-1, 8-2, 8-3, 8-4) until it finds an exit (Go-Explore), then a beam
A* search makes that route as fast as it can. Every route is replayed in stable-retro, the
reference emulator, frame by frame, and rendered as a video. Next: SMBZero, a neural
policy trained from the search (expert iteration) that also plays with random start delays.

The PPO / GRPO / retro-chain training code that preceded this lives in git tag `pre-cleanup`.

## Commands

```bash
./setup.sh                          # fresh machine: venv_retro, deps, ROM (SHA-1 checked), builds
source venv_retro/bin/activate
native/build.sh                     # the NES core libraries (after editing native/*.cpp)
search/build.sh                     # libsmbsearch.so + smbsearch CLI (-O3 -march=native -flto)

search/full_game.sh                 # 1-1 -> 8-4: search, verify in stable-retro, render demo.mp4
python search/tools/solve.py --start 4-2 --segments 1 --out search/out/4-2   # one level
python search/tools/verify_retro.py search/out/4-2/route.npz                 # lockstep check
python search/tools/render_demo.py search/out/4-2/route.npz --compare        # mp4 / gif

python search/tests/search_bench.py # engine checks (~5 min on 64 threads)
python native/difftest.py           # native core vs stable-retro from power-on
python native/deep_difftest.py      # per-level lockstep, hack-free core
```

Long jobs: launch detached (`setsid nohup ... &`), with `PYTHONUNBUFFERED=1` and a log file.

## Layout

- `native/`: the NES core (`smbcore.cpp`, 6502 in `cpu6502.h`), a batched renderer
  (`batchenv.cpp`: 84x84 observations, used by the verifier and SMBZero), level door states
  (`states/`, `gen_states_native.py`), lockstep tests.
- `search/`: see `search/README.md`. C++: `src/emu` (the only file that includes
  `native/smbcore.cpp`; compact states), `src/core` (thread pool, AVX2 hash, state keys),
  `src/route` (segments, outcomes), `src/explore` (Go-Explore), `src/optimize` (beam A*,
  `progress.h` = its rank), `src/api.cpp` (C API, `include/smbsearch.h`), `src/cli.cpp`.
  Python: `python/smbsearch` (ctypes wrapper), `tools/` (solve, verify, render), `tests/`.
- `retro_integration/`: stable-retro integration (data.json, per-level states, the ROM once
  restored; `gen_states.py` regenerates the states).

## Semantics that matter

- The search plays the real game: no training hacks, 4 frames per decision, the 12
  COMPLEX_MOVEMENT actions. States across the API are full native savestates (45,568 B);
  inside, compact states (4.6 KB: the core minus its ROM copies).
- A segment runs from a level's first in-control frame to the first frame of the next route
  level (8-4: `$0770 == 2`). Dead: dying/dead `$0E`, a life lost, below the screen in
  control, an off-route level, timer < 10.
- The beam's rank (`progress.h`) is progress along the explore route: the latest reference
  step whose situation a node matches (area, x/16, y/16, camera/64, tiles without coins),
  so bumped blocks and grown vines are checkpoints.
- The ROM (`retro_integration/SuperMarioBros-Nes-v0/rom.nes`) is gitignored; `setup.sh`
  fetches it.

## Working rules

- Commit and push at verified checkpoints without asking; messages end with the
  `Co-Authored-By` line only.
- Keep READMEs short and visual.
