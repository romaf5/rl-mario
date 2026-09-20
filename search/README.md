# smbsearch

C++ route search for Super Mario Bros on the native NES core, with a Python interface.

```
start state ──► EXPLORE ──────────► reference route ──► OPTIMISE ──────────► route
                Go-Explore: cells,   (any exit into      beam A* over time:   (actions,
                random walks, the    the next route      rank = progress      frames)
                fastest state per    level)              along the reference
                cell                                            │
                          verify in stable-retro (every frame) ◄┘ ──► demo.mp4 / gif
```

| | |
|---|---|
| game | the real one: no emulator shortcuts, one decision per 4 frames, 12 actions |
| state | 4.6 KB compact savestate (the 45 KB core minus its ROM copies) |
| speed | ~375k frames/s on 64 threads (Threadripper 3970X), 11.8k per thread |
| proof | each route replays in stable-retro with identical game RAM on every frame |

## Use

```bash
search/build.sh                                   # g++ -O3 -march=native -flto, AVX2 hashing
search/full_game.sh                               # 1-1 -> 8-4, verified, rendered
python search/tools/solve.py --start 4-2 --segments 1 --out search/out/4-2
```

```python
import sys; sys.path.insert(0, 'search/python')
from smbsearch import Search, load_state, ROUTE
s = Search(threads=64)
ref = s.explore(load_state('4-2'), ROUTE, budget_s=300)      # find any exit
opt = s.optimize(load_state('4-2'), ROUTE, ref.actions)       # make it fast
trace, end_state = s.replay(load_state('4-2'), opt.actions)
```

## Code

| path | what |
|---|---|
| `include/smbsearch.h` | the C API |
| `src/emu/` | `Emu`: compact states, frame stepping, RAM decode (the only includer of `native/smbcore.cpp`) |
| `src/core/` | thread pool, AVX2 hash, cell / exact / coarse state keys |
| `src/route/` | segments (level -> next route level) and step outcomes |
| `src/explore/` | Go-Explore: fresh cells first (60 walks each), 8 tile variants per spot |
| `src/optimize/` | beam A* (`beam.cpp`) and its rank (`progress.h`) |
| `python/smbsearch/` | `Search`: `explore`, `optimize`, `replay`, `settle`, `bench` |
| `tools/` | `solve.py` (chains levels), `verify_retro.py`, `render_demo.py` |
| `tests/search_bench.py` | stepping = native core, compact round trip, 1-1 explore + optimise |
