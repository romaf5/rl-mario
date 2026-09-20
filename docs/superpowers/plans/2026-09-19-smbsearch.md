# smbsearch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A C++ route search (Go-Explore discovery + beam A* over time) with a Python interface that finds and optimises SMB routes on the real game, verifies them in stable-retro and renders a demo — 4-2 first, then 1-1 -> 8-4.

**Architecture:** `search/` holds everything. One TU (`src/emu/emu.cpp`) includes `native/smbcore.cpp` and exposes an `Emu` class with compact states; `core/` has the thread pool, AVX2 hashing and state keys; `route/` segments, outcomes and waypoints; `explore/` and `optimize/` the two algorithms; `api.cpp` a C API used by `python/smbsearch` (ctypes) and `cli.cpp`. Python only orchestrates, verifies and renders.

**Tech Stack:** C++17 (g++ 13, `-O3 -march=native -flto`, PGO, AVX2), ctypes, numpy, stable-retro 1.0, imageio-ffmpeg, Pillow.

**Spec:** `docs/superpowers/specs/2026-09-19-smbsearch-design.md`

## Global Constraints

- All new code in `search/`; nothing else in the repo changes (except a pointer line in CLAUDE.md / EXPERIMENTS.md).
- Stepping: the real game (no training hacks); one decision per 4 frames; the 12 COMPLEX_MOVEMENT actions.
- States crossing the API are full native savestates (45568 bytes); compact states (Core minus PRG/CHR) inside.
- Goal: first frame in the next route level (8-4: `$0770 == 2`); dead: `$0E` 0x0B/0x06, a life lost, below the screen in control, off-route level, timer < 10.
- Route: `1-1, 1-2, 4-1, 4-2, 8-1, 8-2, 8-3, 8-4`.
- Efficiency: no rendering in search, compact states, 64-thread pool (40 while run l trains), AVX2 hashing, LTO + PGO.
- Python: `venv_retro/bin/python`; checks are standalone scripts (`check()` style).
- Commits end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`; commit + push at every green task.

---

## File Structure

```
search/
  README.md                     diagram, build, usage, numbers
  build.sh                      objects -> libsmbsearch.so + smbsearch; --pgo
  .gitignore                    out/
  include/smbsearch.h           C API
  src/emu/emu.h, emu.cpp        Emu, compact states, RAM decode (only includer of native/smbcore.cpp)
  src/core/pool.h               thread pool
  src/core/hash.h               AVX2 hash
  src/core/keys.h               tile grid, cell / exact / coarse keys
  src/route/route.h, route.cpp  Segment, Outcome, Waypoints
  src/explore/explore.h, .cpp   Go-Explore discovery
  src/optimize/beam.h, .cpp     beam A* over time
  src/api.cpp                   C API
  src/cli.cpp                   smbsearch CLI (bench / explore / optimize)
  python/smbsearch/__init__.py  Search wrapper, load_state, ROUTE
  tools/solve.py                explore -> optimize -> settle per segment; route.npz + stats.json
  tools/verify_retro.py         stable-retro lockstep replay
  tools/render_demo.py          mp4 / gif with HUD, side-by-side
  tests/search_bench.py         checks
```

Extraction convention for execution: every file below is given as a bold path followed by one fenced code block.

---

### Task 1: Emulator layer, pool, hashing, C API core, Python wrapper, build

**Files:** create `search/include/smbsearch.h`, `search/src/emu/emu.h`, `search/src/emu/emu.cpp`, `search/src/core/pool.h`, `search/src/core/hash.h`, `search/src/core/keys.h`, `search/src/route/route.h`, `search/src/route/route.cpp`, `search/src/explore/explore.h`, `search/src/explore/explore.cpp`, `search/src/optimize/beam.h`, `search/src/optimize/beam.cpp`, `search/src/api.cpp`, `search/src/cli.cpp`, `search/build.sh`, `search/.gitignore`, `search/python/smbsearch/__init__.py`, `search/tests/search_bench.py`.
(All sources are created in this task so the library links; Tasks 2-3 are verified by their own checks.)

**Interfaces (Produces):** C API in `smbsearch.h`; Python `Search(threads)` with `replay(state, actions) -> (trace (n,10) int32, end_state bytes)`, `settle(state, max_steps=3000) -> (steps, state)`, `bench(state, frames) -> frames/s`, `selftest(state, steps) -> bool`, `explore(state, route, budget_s, settle_s, max_walk=300, seed=0) -> Result`, `optimize(state, route, reference, beam, per_cell, max_depth, verbose) -> Result`; `Result(actions: np.uint8 array, found: bool, frames: int, stats: dict)`; `load_state(name)`, `gp(level)`, `ROUTE`, `ACTION_BUTTONS`, `TRACE_FIELDS`.

- [ ] **Step 1: the check script** (fails until the library exists)

**`search/tests/search_bench.py`**
```python
"""smbsearch checks. Run: venv_retro/bin/python search/tests/search_bench.py [--level42]

  * compact states: stepping through a compact save/load equals stepping straight through
  * the engine's stepping equals the native core's smb_frame (RAM identical after every action)
  * the button table equals the training env's COMPLEX_MOVEMENT bytes
  * explore finds the 1-1 flag; optimise is no slower than its reference and replays to 1-2
  * (--level42) 4-2: explore + optimise reach 8-1
"""
import ctypes, os, sys, time
import numpy as np
SEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SEARCH, 'python'))
from smbsearch import Search, load_state, ROUTE, gp, ROM, REPO, ACTION_BUTTONS

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)), flush=True)


s = Search(threads=int(os.environ.get('SS_THREADS', '32')))
st42, st11 = load_state('4-2'), load_state('1-1')
rs = np.random.RandomState(0)

# ---------------------------------------------------------------- engine
check('compact states: save/load mid-run changes nothing (400 random steps)', s.selftest(st42, 400))
acts = rs.randint(0, 12, 120).astype(np.uint8)
tr, end = s.replay(st42, acts)
lib = ctypes.CDLL(os.path.join(REPO, 'native', 'libsmbcore.so'))
lib.smb_create.restype = ctypes.c_void_p; lib.smb_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.smb_frame.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
lib.smb_ram.restype = ctypes.POINTER(ctypes.c_uint8); lib.smb_ram.argtypes = [ctypes.c_void_p]
lib.smb_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
rom = open(ROM, 'rb').read()
ca, cb = lib.smb_create(rom, len(rom)), lib.smb_create(rom, len(rom))
lib.smb_load(ca, st42)
for a in acts:
    for _ in range(4):
        lib.smb_frame(ca, int(ACTION_BUTTONS[a]))
lib.smb_load(cb, end)
ra = np.ctypeslib.as_array(lib.smb_ram(ca), shape=(0x800,)).copy()
rb = np.ctypeslib.as_array(lib.smb_ram(cb), shape=(0x800,)).copy()
check('stepping: 120 actions equal smb_frame x 4 (RAM identical)', np.array_equal(ra, rb), np.nonzero(ra != rb)[0][:8])
sys.path.insert(0, REPO)
from mario_native_vecenv import _ACTION_BYTES
check('actions: button table equals the training env', list(_ACTION_BYTES) == list(ACTION_BUTTONS))
fps = s.bench(st42, 2_000_000)
check('bench: %.0f frames/s on %d threads' % (fps, s.threads), fps > 1e5, fps)

# ---------------------------------------------------------------- explore / optimise 1-1
t = time.time()
ref = s.explore(st11, ROUTE, budget_s=90, settle_s=15, seed=1)
check('explore 1-1: flag found (%d actions, %d cells, %d walks, %.0f s)'
      % (len(ref.actions), ref.stats['cells'], ref.stats['walks'], time.time() - t), ref.found)
tr, _ = s.replay(st11, ref.actions)
check('explore 1-1: its actions replay into 1-2 at the last step', ref.found and tr[-1, 2] == gp('1-2') and (tr[:-1, 2] == gp('1-1')).all())
t = time.time()
opt = s.optimize(st11, ROUTE, ref.actions, beam=4000, per_cell=16)
check('optimise 1-1: %d actions vs reference %d (%.0f s)' % (len(opt.actions), len(ref.actions), time.time() - t),
      opt.found and len(opt.actions) <= len(ref.actions))
tr, _ = s.replay(st11, opt.actions)
check('optimise 1-1: replays into 1-2 at the last step', opt.found and tr[-1, 2] == gp('1-2') and (tr[:-1, 2] == gp('1-1')).all())

if '--level42' in sys.argv:
    ref = s.explore(st42, ROUTE, budget_s=600, settle_s=120, seed=1)
    check('explore 4-2: route into 8-1 found (%d actions)' % len(ref.actions), ref.found)
    opt = s.optimize(st42, ROUTE, ref.actions, beam=20000, per_cell=16, verbose=1)
    tr, _ = s.replay(st42, opt.actions)
    check('optimise 4-2: %d actions vs %d, replays into 8-1' % (len(opt.actions), len(ref.actions)),
          opt.found and tr[-1, 2] == gp('8-1'))

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
```

- [ ] **Step 2: sources**

**`search/include/smbsearch.h`**
```c
/* smbsearch C API: route search for Super Mario Bros on the native core.
 * States crossing the API are full native savestates (ss_state_size() bytes:
 * native/states and MarioNativeVecEnv use the same format). Actions are
 * COMPLEX_MOVEMENT indices (0-11), one per 4 frames of the real game. A route
 * is a list of level indices (world-1)*4 + (stage-1) in play order; a segment
 * runs from its start level to the next route level (the last: the axe). */
#ifndef SMBSEARCH_H
#define SMBSEARCH_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define SS_API __attribute__((visibility("default")))
typedef struct ss_ctx ss_ctx;
typedef struct {
    int64_t frames;      /* frames of the returned actions (4 per action) */
    int64_t emu_frames;  /* frames emulated in total */
    double seconds;
    int64_t cells;       /* explore: archive cells; optimize: nodes kept */
    int64_t walks;       /* explore: random walks; optimize: depths searched */
    int32_t found;       /* 1 if the segment goal was reached */
    int32_t n_actions;
} ss_stats;
/* replay trace per step: x, y, level, area, sub-area, AreaType, $0E, $0770, camera x, lives */
#define SS_TRACE 10
SS_API int ss_state_size(void);
SS_API ss_ctx* ss_create(const uint8_t* rom, int rom_len, int threads);
SS_API void ss_destroy(ss_ctx* ctx);
SS_API int ss_threads(ss_ctx* ctx);
SS_API int ss_ram(ss_ctx* ctx, const uint8_t* state, uint8_t* ram_out /* 0x800 */);
SS_API int ss_replay(ss_ctx* ctx, const uint8_t* start, const uint8_t* actions, int n,
                     int32_t* trace, uint8_t* end_state);
SS_API int ss_settle(ss_ctx* ctx, const uint8_t* state, int max_steps, uint8_t* end_state);
SS_API double ss_bench(ss_ctx* ctx, const uint8_t* start, int64_t frames);
SS_API int ss_selftest(ss_ctx* ctx, const uint8_t* start, int steps, uint64_t seed);
SS_API int ss_explore(ss_ctx* ctx, const uint8_t* start, const int32_t* route, int n_route,
                      double budget_s, double settle_s, int max_walk, uint64_t seed,
                      uint8_t* out, int max_out, ss_stats* st);
SS_API int ss_optimize(ss_ctx* ctx, const uint8_t* start, const int32_t* route, int n_route,
                       const uint8_t* ref, int n_ref, int beam, int per_cell, int max_depth,
                       int verbose, uint8_t* out, int max_out, ss_stats* st);
#ifdef __cplusplus
}
#endif
#endif
```

**`search/src/emu/emu.h`**
```cpp
// Emulator access for the search: one Emu per worker thread over the native SMB
// core (the real game: no training hacks), compact states and RAM decoding.
#pragma once
#include <cstddef>
#include <cstdint>

namespace ss {

constexpr int kFrameSkip = 4;       // frames per decision (the policy's step)
constexpr int kNumActions = 12;     // COMPLEX_MOVEMENT
extern const uint8_t kActionButtons[kNumActions];

size_t full_state_size();           // the native savestate: sizeof(Core)
size_t compact_state_size();        // what changes: sizeof(Core) - PRG - CHR copies

class Emu {
public:
    Emu(const uint8_t* rom, int rom_len);
    ~Emu();
    Emu(const Emu&) = delete;
    Emu& operator=(const Emu&) = delete;
    bool ok() const { return core_ != nullptr; }
    void load_full(const uint8_t* full);
    void save_full(uint8_t* full) const;
    void load(const uint8_t* compact);
    void save(uint8_t* compact) const;
    void frame(uint8_t buttons);
    void step(int action);          // kFrameSkip frames holding the action
    const uint8_t* ram() const;
    bool jammed() const;
private:
    void* core_;
};

// ---- SMB RAM decode ----
inline int level_gp(const uint8_t* r) { return r[0x75F] > 7 ? -1 : r[0x75F] * 4 + r[0x75C]; }
inline int mario_x(const uint8_t* r) { return r[0x6D] * 256 + r[0x86]; }
inline int mario_y(const uint8_t* r) { return r[0xB5] * 256 + r[0xCE]; }
inline int camera_x(const uint8_t* r) { return r[0x71A] * 256 + r[0x71C]; }
inline int lives(const uint8_t* r) { return r[0x75A]; }
inline int game_timer(const uint8_t* r) { return r[0x7F8] * 100 + r[0x7F9] * 10 + r[0x7FA]; }
inline bool in_control(const uint8_t* r) { return r[0x0E] == 0x08 && r[0x770] == 1; }
// dying animation / dead, or fallen below the screen while in control
inline bool dying(const uint8_t* r) {
    return r[0x0E] == 0x0B || r[0x0E] == 0x06 || (in_control(r) && r[0xB5] > 1);
}
// the coordinate system x lives in: level, area, sub-area, AreaType, swimming
inline uint32_t frame_id(const uint8_t* r) {
    int g = level_gp(r);
    if (g < 0) g = 63;
    return (uint32_t)((((g * 256 + r[0x760]) * 256 + r[0x74F]) * 8 + (r[0x74E] & 7)) * 2 + (r[0x704] & 1));
}

}  // namespace ss
```

**`search/src/emu/emu.cpp`**
```cpp
// The only translation unit that includes the native core (single-TU core:
// its extern "C" symbols exist once; LTO inlines across the Emu boundary).
#include "emu.h"
#include <cstring>
#include "../../../native/smbcore.cpp"

namespace ss {

const uint8_t kActionButtons[kNumActions] = {
    0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40, 0x41, 0x42, 0x43, 0x20, 0x10};

namespace {
// compact state = Core minus its CHR (inside Ppu) and PRG ROM copies
const size_t kChrOff = offsetof(Core, ppu) + offsetof(Ppu, chr);
const size_t kChrLen = sizeof(Ppu::chr);
const size_t kPrgOff = offsetof(Core, prg);
const size_t kPrgLen = sizeof(Core::prg);
const size_t kSegA = kChrOff;
const size_t kSegB = kPrgOff - (kChrOff + kChrLen);
const size_t kSegC = sizeof(Core) - (kPrgOff + kPrgLen);
static_assert(offsetof(Core, ppu) + offsetof(Ppu, chr) + sizeof(Ppu::chr) <= offsetof(Core, prg),
              "compact layout: CHR before PRG");
}  // namespace

size_t full_state_size() { return sizeof(Core); }
size_t compact_state_size() { return kSegA + kSegB + kSegC; }

Emu::Emu(const uint8_t* rom, int rom_len) : core_(smb_create(rom, rom_len)) {}
Emu::~Emu() { if (core_) smb_destroy(static_cast<Core*>(core_)); }
void Emu::load_full(const uint8_t* full) { smb_load(static_cast<Core*>(core_), full); }
void Emu::save_full(uint8_t* full) const { smb_save(static_cast<Core*>(core_), full); }

void Emu::load(const uint8_t* s) {
    uint8_t* c = static_cast<uint8_t*>(core_);
    memcpy(c, s, kSegA);
    memcpy(c + kChrOff + kChrLen, s + kSegA, kSegB);
    memcpy(c + kPrgOff + kPrgLen, s + kSegA + kSegB, kSegC);
}

void Emu::save(uint8_t* s) const {
    const uint8_t* c = static_cast<const uint8_t*>(core_);
    memcpy(s, c, kSegA);
    memcpy(s + kSegA, c + kChrOff + kChrLen, kSegB);
    memcpy(s + kSegA + kSegB, c + kPrgOff + kPrgLen, kSegC);
}

void Emu::frame(uint8_t buttons) { smb_frame(static_cast<Core*>(core_), buttons); }

void Emu::step(int action) {
    Core* c = static_cast<Core*>(core_);
    const uint8_t b = kActionButtons[action];
    for (int k = 0; k < kFrameSkip; k++) smb_frame(c, b);
}

const uint8_t* Emu::ram() const { return static_cast<const Core*>(core_)->ram; }
bool Emu::jammed() const { return static_cast<const Core*>(core_)->cpu.jammed; }

}  // namespace ss
```

**`search/src/core/pool.h`**
```cpp
// Persistent worker pool: parallel_for(n, f, chunk) runs f(i, worker) for i in
// [0, n) with dynamic chunks; the calling thread is worker 0.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ss {

class Pool {
public:
    explicit Pool(int threads) : n_(threads < 1 ? 1 : threads) {
        for (int w = 1; w < n_; w++) th_.emplace_back([this, w] { work(w); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return n_; }
    void parallel_for(int64_t n, const std::function<void(int64_t, int)>& f, int64_t chunk = 1) {
        if (n <= 0) return;
        {
            std::lock_guard<std::mutex> l(m_);
            f_ = &f; items_ = n; chunk_ = chunk < 1 ? 1 : chunk; next_ = 0; busy_ = n_ - 1; gen_++;
        }
        cv_.notify_all();
        run(0);
        std::unique_lock<std::mutex> l(m_);
        done_.wait(l, [&] { return busy_ == 0; });
    }

private:
    void run(int w) {
        for (;;) {
            int64_t i = next_.fetch_add(chunk_);
            if (i >= items_) return;
            const int64_t e = std::min(i + chunk_, items_);
            for (; i < e; i++) (*f_)(i, w);
        }
    }
    void work(int w) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
            }
            run(w);
            std::lock_guard<std::mutex> l(m_);
            if (--busy_ == 0) done_.notify_one();
        }
    }
    int n_;
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    uint64_t gen_ = 0;
    int busy_ = 0;
    bool stop_ = false;
    const std::function<void(int64_t, int)>* f_ = nullptr;
    std::atomic<int64_t> next_{0};
    int64_t items_ = 0, chunk_ = 1;
};

}  // namespace ss
```

**`search/src/core/hash.h`**
```cpp
// 64-bit hashing of byte ranges: AVX2 multiply-accumulate over 32-byte blocks
// (xxh3-style lanes), 8-byte scalar tail. Used for state keys (2 KB RAM, tile grids).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ss {

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33);
}

inline uint64_t hash_bytes(const uint8_t* p, size_t n, uint64_t seed = 0) {
    uint64_t h = seed ^ (n * 0x9E3779B97F4A7C15ULL);
    size_t i = 0;
#if defined(__AVX2__)
    if (n >= 32) {
        __m256i acc = _mm256_set1_epi64x((long long)h);
        const __m256i k = _mm256_set_epi64x(0x165667B19E3779F9LL, 0x27D4EB2F165667C5LL,
                                            (long long)0x85EBCA77C2B2AE63ULL, (long long)0x9E3779B185EBCA87ULL);
        for (; i + 32 <= n; i += 32) {
            const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            const __m256i dk = _mm256_xor_si256(d, k);
            const __m256i prod = _mm256_mul_epu32(dk, _mm256_srli_epi64(dk, 32));
            acc = _mm256_add_epi64(acc, _mm256_add_epi64(prod, d));
            acc = _mm256_xor_si256(acc, _mm256_srli_epi64(acc, 29));
        }
        alignas(32) uint64_t lane[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lane), acc);
        h = mix64(lane[0] ^ mix64(lane[1] ^ mix64(lane[2] ^ mix64(lane[3]))));
    }
#endif
    for (; i + 8 <= n; i += 8) { uint64_t v; memcpy(&v, p + i, 8); h = mix64(h ^ v); }
    if (i < n) { uint64_t v = 0; memcpy(&v, p + i, n - i); h = mix64(h ^ v ^ 0xA5); }
    return mix64(h);
}

}  // namespace ss
```

**`search/src/core/keys.h`**
```cpp
// State keys. cell_key: the Go-Explore cell (the key that found the 4-2 route in
// the Python env). exact_key: the RAM that is game state (beam dedupe within a
// depth). coarse_key: the beam's diversity cell.
#pragma once
#include <cstdint>
#include <cstring>
#include "../emu/emu.h"
#include "hash.h"

namespace ss {

// metatiles of Mario's 128-px bin: 8 columns x 13 rows of the $0500 buffer;
// false for a block mid-bump ($23): a transient, not a place
inline bool tile_grid(const uint8_t* r, uint8_t g[104]) {
    const int col0 = (mario_x(r) / 128) * 8;
    bool bump = false;
    for (int j = 0; j < 8; j++) {
        const int cx = (col0 + j) * 16;
        const int base = 0x500 + ((cx / 256) % 2) * 0xD0 + (cx % 256) / 16;
        for (int row = 0; row < 13; row++) {
            const uint8_t t = r[base + row * 16];
            g[j * 13 + row] = t;
            bump |= t == 0x23;
        }
    }
    return !bump;
}

// frame, x/32, y band/16 ($03B8), camera/64, tile signature; 0 = not a cell
inline uint64_t cell_key(const uint8_t* r) {
    uint8_t g[104];
    if (!tile_grid(r, g)) return 0;
    uint64_t k = frame_id(r);
    k = k * 1024 + (uint64_t)(mario_x(r) / 32);
    k = k * 32 + (uint64_t)(r[0x3B8] / 16);
    k = k * 1024 + (uint64_t)(camera_x(r) / 64);
    return mix64(k ^ (hash_bytes(g, sizeof g) * 0x9E3779B97F4A7C15ULL)) | 1;
}

// all game-state RAM: temps, frame counter, stack, OAM buffer, score, coins and
// timer digits excluded
inline uint64_t exact_key(const uint8_t* r) {
    alignas(32) uint8_t m[0x800];
    memcpy(m, r, sizeof m);
    memset(m, 0, 8); m[0x09] = 0;
    memset(m + 0x100, 0, 0x200);
    memset(m + 0x7DD, 0, 6); m[0x7ED] = m[0x7EE] = 0; memset(m + 0x7F8, 0, 3);
    return hash_bytes(m, sizeof m) | 1;
}

// frame, x/8, y/8, camera/64, power-up, area pointer $0750, float state, tiles
inline uint64_t coarse_key(const uint8_t* r) {
    uint64_t k = frame_id(r);
    k = k * 1024 + (uint64_t)(mario_x(r) / 8);
    k = k * 128 + (uint64_t)(mario_y(r) / 8);
    k = k * 1024 + (uint64_t)(camera_x(r) / 64);
    k = k * 4 + (r[0x756] & 3);
    k = k * 256 + r[0x750];
    k = k * 4 + (r[0x1D] & 3);
    uint8_t g[104];
    tile_grid(r, g);
    return mix64(k) ^ hash_bytes(g, sizeof g);
}

}  // namespace ss
```

**`search/src/route/route.h`**
```cpp
// Segments of a route (start level -> next route level or the axe), step
// outcomes, and waypoints: progress along a reference path.
#pragma once
#include <cstdint>
#include <vector>
#include "../emu/emu.h"

namespace ss {

enum class Outcome { Running, Goal, Dead };

struct Segment {
    std::vector<int> route;          // level indices in play order
    int start_gp = -1, goal_gp = -1;
    bool goal_victory = false;       // last route level: the axe ($0770 == 2)
    int start_lives = 0;
    static Segment make(const std::vector<int>& route, const uint8_t* start_ram);
    Outcome classify(const uint8_t* r) const;
};

// the reference's frames (frame_id) in visit order and where each was entered /
// left; rank = px still to cover: |exit - x| in the current frame + later frames
struct Waypoints {
    std::vector<uint32_t> frame;
    std::vector<int> entry_x, exit_x;
    std::vector<int64_t> tail;
    bool empty() const { return frame.empty(); }
    // node in waypoint k -> (k', px left); false if an in-control frame is off the path
    bool rank(uint32_t f, int x, bool control, int k, int64_t parent_px, int* k_out, int64_t* px_out) const;
};

Waypoints waypoints_from_trace(const std::vector<uint32_t>& frames, const std::vector<int>& xs,
                               const std::vector<uint8_t>& control);

}  // namespace ss
```

**`search/src/route/route.cpp`**
```cpp
#include "route.h"
#include <cstdlib>

namespace ss {

Segment Segment::make(const std::vector<int>& route, const uint8_t* r) {
    Segment s;
    s.route = route;
    s.start_lives = lives(r);
    const int g = level_gp(r);
    for (size_t i = 0; i < route.size(); i++) {
        if (route[i] != g) continue;
        s.start_gp = g;
        if (i + 1 < route.size()) s.goal_gp = route[i + 1];
        else s.goal_victory = true;
        break;
    }
    return s;
}

Outcome Segment::classify(const uint8_t* r) const {
    const int g = level_gp(r);
    if (goal_victory) {
        if (g == start_gp && r[0x770] == 2) return Outcome::Goal;
    } else if (g == goal_gp) {
        return Outcome::Goal;
    }
    if (g != start_gp) return Outcome::Dead;              // off-route level, glitch world
    if (dying(r) || lives(r) < start_lives) return Outcome::Dead;
    if (in_control(r) && game_timer(r) < 10) return Outcome::Dead;
    return Outcome::Running;
}

Waypoints waypoints_from_trace(const std::vector<uint32_t>& f, const std::vector<int>& x,
                               const std::vector<uint8_t>& c) {
    Waypoints w;
    for (size_t i = 0; i < f.size(); i++) {
        if (!c[i]) continue;
        if (w.frame.empty() || w.frame.back() != f[i]) {
            w.frame.push_back(f[i]); w.entry_x.push_back(x[i]); w.exit_x.push_back(x[i]);
        } else {
            w.exit_x.back() = x[i];
        }
    }
    const int n = (int)w.frame.size();
    w.tail.assign(n, 0);
    for (int k = n - 2; k >= 0; k--)
        w.tail[k] = w.tail[k + 1] + std::abs(w.exit_x[k + 1] - w.entry_x[k + 1]);
    return w;
}

bool Waypoints::rank(uint32_t f, int x, bool control, int k, int64_t parent_px, int* k_out,
                     int64_t* px_out) const {
    const int n = (int)frame.size();
    int m = -1;
    if (frame[k] == f) m = k;
    else
        for (int j = k + 1; j < n && j <= k + 3; j++)
            if (frame[j] == f) { m = j; break; }       // later frames: shortcuts allowed
    if (m < 0) {
        if (control) return false;                      // off the reference's topology
        *k_out = k; *px_out = parent_px;                // transition frames keep the rank
        return true;
    }
    *k_out = m;
    *px_out = std::abs(exit_x[m] - x) + tail[m];
    return true;
}

}  // namespace ss
```

**`search/src/explore/explore.h`**
```cpp
// Go-Explore phase 1 in C++: an archive of cells (cell_key), each holding its
// fastest state and the actions to it; workers pick cells (1/sqrt(1 + picks)),
// random-walk from them and merge new or faster cells. Returns the fastest path
// into the segment goal found within the budget.
#pragma once
#include <cstdint>
#include <vector>
#include "../core/pool.h"
#include "../emu/emu.h"
#include "../route/route.h"

namespace ss {

struct ExploreParams {
    double budget_s = 120, settle_s = 30;   // stop settle_s after the first goal
    int max_walk = 300;                     // steps per random walk
    uint64_t seed = 0;
    int64_t max_cells = 2000000;
};

struct ExploreResult {
    bool found = false;
    std::vector<uint8_t> actions;
    int64_t emu_frames = 0, cells = 0, walks = 0;
    double seconds = 0;
};

ExploreResult explore(Pool& pool, std::vector<Emu*>& emus, const uint8_t* start_full,
                      const std::vector<int>& route, const ExploreParams& p);

}  // namespace ss
```

**`search/src/explore/explore.cpp`**
```cpp
#include "explore.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include "../core/keys.h"

namespace ss {
namespace {

struct Cell {
    std::vector<uint8_t> state, actions;
    std::atomic<uint32_t> picks{0};
    Cell(std::vector<uint8_t> s, std::vector<uint8_t> a) : state(std::move(s)), actions(std::move(a)) {}
};

// grounded (or swimming), in control, with time left: a state worth restarting from
bool storable(const uint8_t* r) {
    return in_control(r) && (r[0x1D] == 0 || (r[0x704] & 1)) && game_timer(r) > 25;
}

}  // namespace

ExploreResult explore(Pool& pool, std::vector<Emu*>& emus, const uint8_t* start_full,
                      const std::vector<int>& route, const ExploreParams& p) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto elapsed = [&] { return std::chrono::duration<double>(clk::now() - t0).count(); };
    const size_t CS = compact_state_size();
    ExploreResult res;
    Emu& e0 = *emus[0];
    e0.load_full(start_full);
    const Segment seg = Segment::make(route, e0.ram());
    if (seg.start_gp < 0) return res;

    std::shared_mutex mu;                               // guards cells + index
    std::deque<Cell> cells;
    std::unordered_map<uint64_t, uint32_t> index;
    {
        std::vector<uint8_t> s0(CS);
        e0.save(s0.data());
        cells.emplace_back(std::move(s0), std::vector<uint8_t>());
        const uint64_t k0 = cell_key(e0.ram());
        if (k0) index.emplace(k0, 0u);
    }
    std::mutex gmu;                                     // guards the goal path
    std::vector<uint8_t> best;
    bool found = false;
    double found_at = 0;
    std::atomic<int64_t> frames{0}, walks{0};

    pool.parallel_for(pool.size(), [&](int64_t wi, int w) {
        Emu& e = *emus[w];
        std::mt19937_64 rng(p.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)wi + 1);
        std::vector<uint8_t> buf(CS), acts;
        struct Cand { std::vector<uint8_t> state, actions; };
        std::unordered_map<uint64_t, Cand> local;
        int64_t fr = 0, nw = 0;
        for (;;) {
            const double t = elapsed();
            {
                std::lock_guard<std::mutex> g(gmu);
                if (t > p.budget_s || (found && t > found_at + p.settle_s)) break;
            }
            {
                std::shared_lock<std::shared_mutex> l(mu);
                const size_t n = cells.size();
                size_t ci;
                for (;;) {
                    ci = (size_t)(rng() % n);
                    const double a = 1.0 / std::sqrt(1.0 + cells[ci].picks.load(std::memory_order_relaxed));
                    if ((double)(rng() >> 11) * 0x1.0p-53 < a) break;
                }
                cells[ci].picks.fetch_add(1, std::memory_order_relaxed);
                memcpy(buf.data(), cells[ci].state.data(), CS);
                acts = cells[ci].actions;
            }
            e.load(buf.data());
            local.clear();
            int hold = 0, a = 0;
            for (int s = 0; s < p.max_walk; s++) {
                if (hold == 0) { a = (int)(rng() % kNumActions); hold = 1 << (rng() % 4); }
                hold--;
                e.step(a);
                fr += kFrameSkip;
                acts.push_back((uint8_t)a);
                const uint8_t* r = e.ram();
                const Outcome o = seg.classify(r);
                if (o == Outcome::Dead) break;
                if (o == Outcome::Goal) {
                    std::lock_guard<std::mutex> g(gmu);
                    if (!found || acts.size() < best.size()) {
                        best = acts;
                        if (!found) found_at = elapsed();
                        found = true;
                    }
                    break;
                }
                if (!storable(r)) continue;
                const uint64_t k = cell_key(r);
                if (!k || local.count(k)) continue;         // first visit of a walk is its fastest
                bool better;
                {
                    std::shared_lock<std::shared_mutex> l(mu);
                    auto g = index.find(k);
                    better = g == index.end() || cells[g->second].actions.size() > acts.size();
                }
                if (better) {
                    Cand c;
                    c.state.resize(CS);
                    e.save(c.state.data());
                    c.actions = acts;
                    local.emplace(k, std::move(c));
                }
            }
            if (!local.empty()) {
                std::unique_lock<std::shared_mutex> l(mu);
                for (auto& kv : local) {
                    auto g = index.find(kv.first);
                    if (g == index.end()) {
                        if ((int64_t)cells.size() < p.max_cells) {
                            index.emplace(kv.first, (uint32_t)cells.size());
                            cells.emplace_back(std::move(kv.second.state), std::move(kv.second.actions));
                        }
                    } else if (cells[g->second].actions.size() > kv.second.actions.size()) {
                        cells[g->second].state = std::move(kv.second.state);
                        cells[g->second].actions = std::move(kv.second.actions);
                    }
                }
            }
            nw++;
        }
        frames += fr;
        walks += nw;
    });
    res.found = found;
    res.actions = best;
    res.emu_frames = frames;
    res.walks = walks;
    res.cells = (int64_t)cells.size();
    res.seconds = elapsed();
    return res;
}

}  // namespace ss
```

**`search/src/optimize/beam.h`**
```cpp
// Beam A* over time: depth d holds nodes after 4d frames; the B best (px left
// along the reference's waypoints, then x speed) are expanded by all 12 actions,
// deduplicated by exact state, spread by at most per_cell per coarse cell. The
// reference's own node is always kept, so the result is never slower than it.
#pragma once
#include <cstdint>
#include <vector>
#include "../core/pool.h"
#include "../emu/emu.h"
#include "../route/route.h"

namespace ss {

struct OptimizeParams {
    int beam = 20000, per_cell = 16, max_depth = 6000;
    bool verbose = false;
};

struct OptimizeResult {
    bool found = false;
    std::vector<uint8_t> actions;
    int64_t emu_frames = 0, nodes = 0;
    int depth = 0;
    double seconds = 0;
};

OptimizeResult optimize(Pool& pool, std::vector<Emu*>& emus, const uint8_t* start_full,
                        const std::vector<int>& route, const std::vector<uint8_t>& ref,
                        const OptimizeParams& p);

}  // namespace ss
```

**`search/src/optimize/beam.cpp`**
```cpp
#include "beam.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include "../core/keys.h"

namespace ss {

OptimizeResult optimize(Pool& pool, std::vector<Emu*>& emus, const uint8_t* start_full,
                        const std::vector<int>& route, const std::vector<uint8_t>& ref,
                        const OptimizeParams& p) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto elapsed = [&] { return std::chrono::duration<double>(clk::now() - t0).count(); };
    const size_t CS = compact_state_size();
    OptimizeResult res;
    Emu& e0 = *emus[0];
    e0.load_full(start_full);
    const Segment seg = Segment::make(route, e0.ram());
    if (seg.start_gp < 0) return res;
    std::vector<uint8_t> s0(CS);
    e0.save(s0.data());

    // the reference path -> waypoints; ref_len = its steps before the goal step
    std::vector<uint32_t> ff;
    std::vector<int> xs;
    std::vector<uint8_t> ctl;
    auto rec = [&](const uint8_t* r) { ff.push_back(frame_id(r)); xs.push_back(mario_x(r)); ctl.push_back(in_control(r)); };
    rec(e0.ram());
    int ref_len = 0;
    for (size_t d = 0; d < ref.size(); d++) {
        e0.step(ref[d]);
        if (seg.classify(e0.ram()) != Outcome::Running) break;
        rec(e0.ram());
        ref_len = (int)d + 1;
    }
    const Waypoints wp = waypoints_from_trace(ff, xs, ctl);
    auto rank_of = [&](uint32_t f, int x, bool c, int k, int64_t pd, int* ko, int64_t* d) {
        if (wp.empty()) { *ko = 0; *d = 1000000 - x; return true; }     // no reference: go right
        return wp.rank(f, x, c, k, pd, ko, d);
    };
    int k0 = 0;
    int64_t d0 = 0;
    if (!rank_of(ff[0], xs[0], false, 0, 1 << 30, &k0, &d0)) d0 = 1 << 30;

    const int B = std::max(1, p.beam);
    std::vector<uint8_t> cur(s0), nxt;
    std::vector<uint16_t> cur_k{(uint16_t)k0};
    std::vector<int64_t> cur_d{d0};
    int ref_node = ref_len > 0 ? 0 : -1;                 // the reference's node in cur
    std::vector<std::vector<uint32_t>> par;              // par[d-1][j]: parent of node j at depth d
    std::vector<std::vector<uint8_t>> act;
    std::vector<uint8_t> chs, ok;
    std::vector<uint16_t> ch_k;
    std::vector<int64_t> ch_d;
    std::vector<int8_t> spd;
    std::vector<uint64_t> ek, cek;
    int64_t emu = 0, kept = 0;

    for (int depth = 1; depth <= p.max_depth; depth++) {
        const int np = (int)cur_k.size();
        const size_t nc = (size_t)np * kNumActions;
        if (chs.size() < nc * CS) chs.resize(nc * CS);
        ok.assign(nc, 0);
        ch_k.resize(nc); ch_d.resize(nc); spd.resize(nc); ek.resize(nc); cek.resize(nc);
        pool.parallel_for(np, [&](int64_t i, int w) {
            Emu& e = *emus[w];
            for (int a = 0; a < kNumActions; a++) {
                const size_t c = (size_t)i * kNumActions + a;
                e.load(cur.data() + (size_t)i * CS);
                e.step(a);
                const uint8_t* r = e.ram();
                const Outcome o = seg.classify(r);
                if (o == Outcome::Dead) continue;
                if (o == Outcome::Goal) { ok[c] = 2; continue; }
                int k;
                int64_t d;
                if (!rank_of(frame_id(r), mario_x(r), in_control(r), cur_k[i], cur_d[i], &k, &d)) continue;
                ok[c] = 1; ch_k[c] = (uint16_t)k; ch_d[c] = d; spd[c] = (int8_t)r[0x57];
                ek[c] = exact_key(r); cek[c] = coarse_key(r);
                e.save(chs.data() + c * CS);
            }
        }, 1);
        emu += (int64_t)nc * kFrameSkip;

        size_t goal = nc;
        for (size_t c = 0; c < nc; c++)
            if (ok[c] == 2) { goal = c; break; }
        if (goal < nc) {                                  // parents give the actions
            std::vector<uint8_t> path{(uint8_t)(goal % kNumActions)};
            uint32_t node = (uint32_t)(goal / kNumActions);
            for (int d = depth - 1; d >= 1; d--) { path.push_back(act[d - 1][node]); node = par[d - 1][node]; }
            std::reverse(path.begin(), path.end());
            res.found = true; res.actions = std::move(path); res.depth = depth;
            break;
        }

        std::vector<uint32_t> order;
        order.reserve(nc);
        for (size_t c = 0; c < nc; c++)
            if (ok[c] == 1) order.push_back((uint32_t)c);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (ch_d[a] != ch_d[b]) return ch_d[a] < ch_d[b];
            if (spd[a] != spd[b]) return spd[a] > spd[b];
            return a < b;
        });
        std::vector<uint32_t> sel;
        sel.reserve((size_t)B + 1);
        std::unordered_set<uint64_t> seen;
        seen.reserve(order.size() * 2 + 16);
        std::unordered_map<uint64_t, int> per;
        per.reserve(order.size() + 16);
        int new_ref = -1;
        if (ref_node >= 0 && depth <= ref_len) {           // the reference always stays
            const uint32_t c = (uint32_t)ref_node * kNumActions + ref[depth - 1];
            if (ok[c] == 1) { sel.push_back(c); seen.insert(ek[c]); per[cek[c]]++; new_ref = 0; }
        }
        for (uint32_t c : order) {
            if ((int)sel.size() >= B) break;
            if (!seen.insert(ek[c]).second) continue;
            int& n = per[cek[c]];
            if (n >= p.per_cell) continue;
            n++;
            sel.push_back(c);
        }
        if (sel.empty()) break;

        const int ns = (int)sel.size();
        nxt.resize((size_t)ns * CS);
        std::vector<uint16_t> nk(ns);
        std::vector<int64_t> nd(ns);
        std::vector<uint32_t> pp(ns);
        std::vector<uint8_t> pa(ns);
        pool.parallel_for(ns, [&](int64_t j, int) {
            const uint32_t c = sel[j];
            memcpy(nxt.data() + (size_t)j * CS, chs.data() + (size_t)c * CS, CS);
            nk[j] = ch_k[c]; nd[j] = ch_d[c]; pp[j] = c / kNumActions; pa[j] = (uint8_t)(c % kNumActions);
        }, 256);
        cur.swap(nxt); cur_k.swap(nk); cur_d.swap(nd);
        par.push_back(std::move(pp)); act.push_back(std::move(pa));
        ref_node = new_ref;
        kept += ns;
        if (p.verbose && depth % 25 == 0)
            fprintf(stderr, "[beam] depth %d (%d frames): %d nodes of %zu, best %lld px, waypoint %d/%zu, %.0f frames/s\n",
                    depth, depth * kFrameSkip, ns, order.size(), (long long)*std::min_element(cur_d.begin(), cur_d.end()),
                    (int)*std::max_element(cur_k.begin(), cur_k.end()), wp.frame.size(), emu / elapsed());
    }
    res.emu_frames = emu;
    res.nodes = kept;
    res.seconds = elapsed();
    return res;
}

}  // namespace ss
```

**`search/src/api.cpp`**
```cpp
#include "../include/smbsearch.h"
#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include "core/pool.h"
#include "emu/emu.h"
#include "explore/explore.h"
#include "optimize/beam.h"

using namespace ss;

struct ss_ctx {
    std::vector<uint8_t> rom;
    std::unique_ptr<Pool> pool;
    std::vector<std::unique_ptr<Emu>> owned;
    std::vector<Emu*> emus;
};

static int copy_out(const std::vector<uint8_t>& a, uint8_t* out, int max_out) {
    if ((int)a.size() > max_out) return -2;
    if (!a.empty()) memcpy(out, a.data(), a.size());
    return (int)a.size();
}

extern "C" {

int ss_state_size(void) { return (int)full_state_size(); }

ss_ctx* ss_create(const uint8_t* rom, int rom_len, int threads) {
    if (threads < 1) threads = (int)std::thread::hardware_concurrency();
    auto* c = new ss_ctx;
    c->rom.assign(rom, rom + rom_len);
    for (int i = 0; i < threads; i++) {
        c->owned.emplace_back(new Emu(c->rom.data(), rom_len));
        if (!c->owned.back()->ok()) { delete c; return nullptr; }
        c->emus.push_back(c->owned.back().get());
    }
    c->pool.reset(new Pool(threads));
    return c;
}

void ss_destroy(ss_ctx* c) { delete c; }
int ss_threads(ss_ctx* c) { return c->pool->size(); }

int ss_ram(ss_ctx* c, const uint8_t* state, uint8_t* ram_out) {
    c->emus[0]->load_full(state);
    memcpy(ram_out, c->emus[0]->ram(), 0x800);
    return 0x800;
}

int ss_replay(ss_ctx* c, const uint8_t* start, const uint8_t* actions, int n, int32_t* trace,
              uint8_t* end_state) {
    Emu& e = *c->emus[0];
    e.load_full(start);
    for (int i = 0; i < n; i++) {
        if (actions[i] >= kNumActions) return -1;
        e.step(actions[i]);
        if (trace) {
            const uint8_t* r = e.ram();
            int32_t* t = trace + (size_t)i * SS_TRACE;
            t[0] = mario_x(r); t[1] = mario_y(r); t[2] = level_gp(r); t[3] = r[0x760]; t[4] = r[0x74F];
            t[5] = r[0x74E]; t[6] = r[0x0E]; t[7] = r[0x770]; t[8] = camera_x(r); t[9] = lives(r);
        }
    }
    if (end_state) e.save_full(end_state);
    return n;
}

int ss_settle(ss_ctx* c, const uint8_t* state, int max_steps, uint8_t* end_state) {
    Emu& e = *c->emus[0];
    e.load_full(state);
    int n = 0;
    while (n < max_steps && !in_control(e.ram())) { e.step(0); n++; }
    if (end_state) e.save_full(end_state);
    return in_control(e.ram()) ? n : -1;
}

double ss_bench(ss_ctx* c, const uint8_t* start, int64_t frames) {
    const size_t CS = compact_state_size();
    std::vector<uint8_t> s(CS);
    c->emus[0]->load_full(start);
    c->emus[0]->save(s.data());
    const int T = c->pool->size();
    const int64_t per = frames / T / kFrameSkip + 1;
    const auto t0 = std::chrono::steady_clock::now();
    c->pool->parallel_for(T, [&](int64_t i, int w) {
        Emu& e = *c->emus[w];
        std::mt19937 rng((uint32_t)i + 1);
        for (int64_t k = 0; k < per; k++) {
            if (k % 400 == 0) e.load(s.data());
            e.step((int)(rng() % kNumActions));
        }
    });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)per * T * kFrameSkip / sec;
}

int ss_selftest(ss_ctx* c, const uint8_t* start, int steps, uint64_t seed) {
    if (c->emus.size() < 2) return -1;
    Emu& a = *c->emus[0];
    Emu& b = *c->emus[1];
    std::mt19937_64 rng(seed);
    std::vector<int> acts(steps);
    for (auto& x : acts) x = (int)(rng() % kNumActions);
    a.load_full(start);
    for (int x : acts) a.step(x);
    b.load_full(start);
    std::vector<uint8_t> s(compact_state_size());
    for (int i = 0; i < steps; i++) {
        b.step(acts[i]);
        if (i % 7 == 3) { b.save(s.data()); b.load(s.data()); }   // through a compact state
    }
    std::vector<uint8_t> fa(full_state_size()), fb(full_state_size());
    a.save_full(fa.data());
    b.save_full(fb.data());
    return memcmp(fa.data(), fb.data(), fa.size()) == 0 ? 0 : 1;
}

int ss_explore(ss_ctx* c, const uint8_t* start, const int32_t* route, int n_route, double budget_s,
               double settle_s, int max_walk, uint64_t seed, uint8_t* out, int max_out, ss_stats* st) {
    ExploreParams p;
    p.budget_s = budget_s; p.settle_s = settle_s; p.max_walk = max_walk; p.seed = seed;
    const ExploreResult r = explore(*c->pool, c->emus, start, std::vector<int>(route, route + n_route), p);
    const int n = copy_out(r.actions, out, max_out);
    if (st) {
        st->frames = (int64_t)r.actions.size() * kFrameSkip; st->emu_frames = r.emu_frames;
        st->seconds = r.seconds; st->cells = r.cells; st->walks = r.walks;
        st->found = r.found; st->n_actions = (int32_t)r.actions.size();
    }
    return n;
}

int ss_optimize(ss_ctx* c, const uint8_t* start, const int32_t* route, int n_route, const uint8_t* ref,
                int n_ref, int beam, int per_cell, int max_depth, int verbose, uint8_t* out, int max_out,
                ss_stats* st) {
    OptimizeParams p;
    p.beam = beam; p.per_cell = per_cell; p.max_depth = max_depth; p.verbose = verbose != 0;
    const OptimizeResult r = optimize(*c->pool, c->emus, start, std::vector<int>(route, route + n_route),
                                      std::vector<uint8_t>(ref, ref + n_ref), p);
    const int n = copy_out(r.actions, out, max_out);
    if (st) {
        st->frames = (int64_t)r.actions.size() * kFrameSkip; st->emu_frames = r.emu_frames;
        st->seconds = r.seconds; st->cells = r.nodes; st->walks = r.depth;
        st->found = r.found; st->n_actions = (int32_t)r.actions.size();
    }
    return n;
}

}  // extern "C"
```

**`search/src/cli.cpp`**
```cpp
// smbsearch CLI over the C API: benchmarks and the PGO profile workload.
//   smbsearch bench    ROM STATE [threads] [frames]
//   smbsearch explore  ROM STATE ROUTE BUDGET_S OUT [threads]
//   smbsearch optimize ROM STATE ROUTE REF OUT [beam] [threads]
// STATE: a raw (not gzipped) native savestate; ROUTE like 1-1,1-2,4-1,4-2,8-1,8-2,8-3,8-4;
// REF / OUT: action files (one byte per action).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "../include/smbsearch.h"

static std::vector<uint8_t> slurp(const char* path) {
    std::vector<uint8_t> d;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    uint8_t b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) d.insert(d.end(), b, b + n);
    fclose(f);
    return d;
}

static std::vector<int32_t> parse_route(const char* s) {
    std::vector<int32_t> r;
    std::string t(s);
    size_t i = 0;
    while (i < t.size()) {
        const int w = t[i] - '0', l = t[i + 2] - '0';
        r.push_back((w - 1) * 4 + (l - 1));
        i += 4;
    }
    return r;
}

static void dump(const char* path, const std::vector<uint8_t>& a, int n) {
    FILE* f = fopen(path, "wb");
    fwrite(a.data(), 1, (size_t)n, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: see cli.cpp\n"); return 2; }
    const std::string cmd = argv[1];
    std::vector<uint8_t> rom = slurp(argv[2]), state = slurp(argv[3]);
    if ((int)state.size() != ss_state_size()) { fprintf(stderr, "state size %zu != %d\n", state.size(), ss_state_size()); return 2; }
    if (cmd == "bench") {
        ss_ctx* c = ss_create(rom.data(), (int)rom.size(), argc > 4 ? atoi(argv[4]) : 0);
        const long long frames = argc > 5 ? atoll(argv[5]) : 4000000LL;
        printf("%.0f frames/s on %d threads\n", ss_bench(c, state.data(), frames), ss_threads(c));
        ss_destroy(c);
        return 0;
    }
    std::vector<int32_t> route = parse_route(argv[4]);
    std::vector<uint8_t> out(200000);
    ss_stats st{};
    if (cmd == "explore" && argc >= 7) {
        ss_ctx* c = ss_create(rom.data(), (int)rom.size(), argc > 7 ? atoi(argv[7]) : 0);
        const int n = ss_explore(c, state.data(), route.data(), (int)route.size(), atof(argv[5]), 10.0, 300, 1,
                                 out.data(), (int)out.size(), &st);
        printf("explore: found %d, %d actions, %lld cells, %lld walks, %.1f s\n", st.found, n,
               (long long)st.cells, (long long)st.walks, st.seconds);
        if (n > 0) dump(argv[6], out, n);
        ss_destroy(c);
        return st.found ? 0 : 1;
    }
    if (cmd == "optimize" && argc >= 7) {
        ss_ctx* c = ss_create(rom.data(), (int)rom.size(), argc > 8 ? atoi(argv[8]) : 0);
        std::vector<uint8_t> ref = slurp(argv[5]);
        const int n = ss_optimize(c, state.data(), route.data(), (int)route.size(), ref.data(), (int)ref.size(),
                                  argc > 7 ? atoi(argv[7]) : 20000, 16, 6000, 1, out.data(), (int)out.size(), &st);
        printf("optimize: found %d, %d actions (reference %zu), %.1f s\n", st.found, n, ref.size(), st.seconds);
        if (n > 0) dump(argv[6], out, n);
        ss_destroy(c);
        return st.found ? 0 : 1;
    }
    fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
```

**`search/build.sh`**
```bash
#!/usr/bin/env bash
# Build search/build/libsmbsearch.so (Python) and search/build/smbsearch (CLI).
#   search/build.sh          -O3 -march=native -flto
#   search/build.sh --pgo    + profile-guided optimisation (profiles bench + explore on 1-1 / 4-2)
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}
PY=${PYTHON:-../venv_retro/bin/python}
SRCS="emu/emu route/route explore/explore optimize/beam api"
FLAGS="-std=c++17 -O3 -march=native -mtune=native -flto=auto -fno-plt -fomit-frame-pointer \
  -fvisibility=hidden -fPIC -DNDEBUG -Wall -Wno-invalid-offsetof -pthread"
OBJ=build/obj

compile() {  # $1: extra flags
  mkdir -p $OBJ
  for s in $SRCS cli; do
    mkdir -p "$OBJ/$(dirname $s)"
    $CXX $FLAGS $1 -c src/$s.cpp -o $OBJ/$s.o
  done
  local objs=""
  for s in $SRCS; do objs="$objs $OBJ/$s.o"; done
  $CXX $FLAGS $1 -shared -o build/libsmbsearch.so $objs
  $CXX $FLAGS $1 -o build/smbsearch $objs $OBJ/cli.o
}

if [ "${1:-}" = "--pgo" ]; then
  rm -rf build/pgo $OBJ && mkdir -p build/pgo
  compile "-fprofile-generate -fprofile-update=atomic -fprofile-dir=$(pwd)/build/pgo"
  ROM=../retro_integration/SuperMarioBros-Nes-v0/rom.nes
  for L in 1-1 4-2; do
    $PY -c "import gzip,sys; open('build/L$L.state','wb').write(gzip.open('../native/states/Level$L.state').read())"
  done
  ./build/smbsearch bench $ROM build/L4-2.state 0 3000000
  ./build/smbsearch explore $ROM build/L1-1.state 1-1,1-2,4-1,4-2,8-1,8-2,8-3,8-4 20 build/pgo_ref.bin || true
  ./build/smbsearch optimize $ROM build/L1-1.state 1-1,1-2,4-1,4-2,8-1,8-2,8-3,8-4 build/pgo_ref.bin build/pgo_opt.bin 2000 || true
  rm -rf $OBJ
  compile "-fprofile-use -fprofile-partial-training -fprofile-dir=$(pwd)/build/pgo -Wno-missing-profile"
else
  rm -rf $OBJ
  compile ""
fi
echo "[build] ok: $(pwd)/build/libsmbsearch.so $(pwd)/build/smbsearch"
```

**`search/.gitignore`**
```
out/
```

**`search/python/smbsearch/__init__.py`**
```python
"""Python interface to the C++ route search (search/build/libsmbsearch.so).

    from smbsearch import Search, load_state, ROUTE
    s = Search(threads=40)
    ref = s.explore(load_state('4-2'), ROUTE, budget_s=300)       # discovery
    opt = s.optimize(load_state('4-2'), ROUTE, ref.actions)        # beam A*
    trace, end = s.replay(load_state('4-2'), opt.actions)

All loops run in C++ on a thread pool; ctypes releases the GIL for each call.
States are full native savestates (bytes); actions are COMPLEX_MOVEMENT indices,
one per 4 frames of the real game (no training hacks).
"""
import ctypes
import gzip
import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(SEARCH)
LIB = os.path.join(SEARCH, 'build', 'libsmbsearch.so')
ROM = os.path.join(REPO, 'retro_integration', 'SuperMarioBros-Nes-v0', 'rom.nes')
STATES = os.path.join(REPO, 'native', 'states')
ROUTE = ['1-1', '1-2', '4-1', '4-2', '8-1', '8-2', '8-3', '8-4']
ACTIONS = ['NOOP', 'right', 'right+A', 'right+B', 'right+A+B', 'A', 'left', 'left+A', 'left+B',
           'left+A+B', 'down', 'up']
ACTION_BUTTONS = [0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40, 0x41, 0x42, 0x43, 0x20, 0x10]
TRACE_FIELDS = ('x', 'y', 'level', 'area', 'sub', 'atype', 'engine', 'mode', 'camera', 'lives')
FRAME_SKIP = 4


def gp(level):
    w, s = level.split('-')
    return (int(w) - 1) * 4 + int(s) - 1


def level_name(g):
    return '%d-%d' % (g // 4 + 1, g % 4 + 1)


def load_state(name='4-2'):
    """A native savestate: a level's door state ('4-2') or 'FullGame' (1-1 from boot)."""
    fn = 'FullGame.state' if name == 'FullGame' else 'Level%s.state' % name
    return gzip.open(os.path.join(STATES, fn)).read()


class _Stats(ctypes.Structure):
    _fields_ = [('frames', ctypes.c_int64), ('emu_frames', ctypes.c_int64), ('seconds', ctypes.c_double),
                ('cells', ctypes.c_int64), ('walks', ctypes.c_int64), ('found', ctypes.c_int32),
                ('n_actions', ctypes.c_int32)]


@dataclass
class Result:
    actions: np.ndarray
    found: bool
    frames: int
    stats: dict


class Search:
    MAX_ACTIONS = 200000

    def __init__(self, threads=None, rom=ROM, lib=LIB):
        if not os.path.exists(lib):
            raise FileNotFoundError('%s is missing: run search/build.sh' % lib)
        L = self._lib = ctypes.CDLL(lib)
        P, I, D, U64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_uint64
        L.ss_state_size.restype = I
        L.ss_create.restype = P; L.ss_create.argtypes = [ctypes.c_char_p, I, I]
        L.ss_destroy.argtypes = [P]
        L.ss_threads.restype = I; L.ss_threads.argtypes = [P]
        L.ss_ram.restype = I; L.ss_ram.argtypes = [P, ctypes.c_char_p, P]
        L.ss_replay.restype = I; L.ss_replay.argtypes = [P, ctypes.c_char_p, P, I, P, P]
        L.ss_settle.restype = I; L.ss_settle.argtypes = [P, ctypes.c_char_p, I, P]
        L.ss_bench.restype = D; L.ss_bench.argtypes = [P, ctypes.c_char_p, ctypes.c_int64]
        L.ss_selftest.restype = I; L.ss_selftest.argtypes = [P, ctypes.c_char_p, I, U64]
        L.ss_explore.restype = I
        L.ss_explore.argtypes = [P, ctypes.c_char_p, P, I, D, D, I, U64, P, I, ctypes.POINTER(_Stats)]
        L.ss_optimize.restype = I
        L.ss_optimize.argtypes = [P, ctypes.c_char_p, P, I, P, I, I, I, I, I, P, I, ctypes.POINTER(_Stats)]
        rom_bytes = open(rom, 'rb').read()
        self._ctx = L.ss_create(rom_bytes, len(rom_bytes), int(threads or os.cpu_count()))
        if not self._ctx:
            raise RuntimeError('ss_create failed (bad ROM?)')
        self.threads = L.ss_threads(self._ctx)
        self.state_size = L.ss_state_size()

    def close(self):
        if getattr(self, '_ctx', None):
            self._lib.ss_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def _state(self, s):
        if len(s) != self.state_size:
            raise ValueError('state of %d bytes, the core uses %d' % (len(s), self.state_size))
        return bytes(s)

    @staticmethod
    def _route(route):
        arr = np.array([gp(l) for l in route], dtype=np.int32)
        return arr, arr.ctypes.data, len(arr)

    def ram(self, state):
        """The 2 KB work RAM of a state."""
        out = np.zeros(0x800, dtype=np.uint8)
        self._lib.ss_ram(self._ctx, self._state(state), out.ctypes.data)
        return out

    def level(self, state):
        r = self.ram(state)
        return level_name(int(r[0x75F]) * 4 + int(r[0x75C]))

    def replay(self, state, actions):
        """Step the actions from state. Returns (trace (n, 10) int32 per step, end state)."""
        a = np.ascontiguousarray(actions, dtype=np.uint8)
        tr = np.zeros((len(a), len(TRACE_FIELDS)), dtype=np.int32)
        end = ctypes.create_string_buffer(self.state_size)
        n = self._lib.ss_replay(self._ctx, self._state(state), a.ctypes.data, len(a), tr.ctypes.data, end)
        if n < 0:
            raise ValueError('bad action index')
        return tr, end.raw

    def settle(self, state, max_steps=3000):
        """NOOP steps until Mario is in control (after a level transition): (steps, state)."""
        end = ctypes.create_string_buffer(self.state_size)
        n = self._lib.ss_settle(self._ctx, self._state(state), max_steps, end)
        return n, end.raw

    def bench(self, state, frames=4_000_000):
        return self._lib.ss_bench(self._ctx, self._state(state), int(frames))

    def selftest(self, state, steps=400, seed=1):
        return self._lib.ss_selftest(self._ctx, self._state(state), steps, seed) == 0

    def _result(self, out, n, st):
        if n == -2:
            raise RuntimeError('action buffer too small')
        stats = {k: getattr(st, k) for k, _ in _Stats._fields_}
        return Result(actions=out[:max(n, 0)].copy(), found=bool(st.found), frames=int(st.frames), stats=stats)

    def explore(self, state, route, budget_s=120.0, settle_s=30.0, max_walk=300, seed=0):
        """Go-Explore discovery of the segment starting at state (to the next route level)."""
        r, rp, rn = self._route(route)
        out = np.zeros(self.MAX_ACTIONS, dtype=np.uint8)
        st = _Stats()
        n = self._lib.ss_explore(self._ctx, self._state(state), rp, rn, float(budget_s), float(settle_s),
                                 int(max_walk), int(seed), out.ctypes.data, len(out), ctypes.byref(st))
        return self._result(out, n, st)

    def optimize(self, state, route, reference, beam=20000, per_cell=16, max_depth=6000, verbose=0):
        """Beam A* over time along the reference's waypoints (never slower than the reference)."""
        r, rp, rn = self._route(route)
        ref = np.ascontiguousarray(reference, dtype=np.uint8)
        out = np.zeros(self.MAX_ACTIONS, dtype=np.uint8)
        st = _Stats()
        n = self._lib.ss_optimize(self._ctx, self._state(state), rp, rn, ref.ctypes.data, len(ref), int(beam),
                                  int(per_cell), int(max_depth), int(verbose), out.ctypes.data, len(out),
                                  ctypes.byref(st))
        return self._result(out, n, st)
```

- [ ] **Step 3: build** — `search/build.sh` → `[build] ok`.
- [ ] **Step 4: checks** — `SS_THREADS=32 venv_retro/bin/python search/tests/search_bench.py` → all pass (engine, bench > 1e5 frames/s, 1-1 explore + optimise).
- [ ] **Step 5: commit** — "search/: C++ route search engine (compact-state emulator layer, Go-Explore discovery, beam A* over time), C API, Python interface, checks"; push.

---

### Task 2: PGO build and measured throughput

- [ ] **Step 1:** `venv_retro/bin/python -c "import sys; sys.path.insert(0,'search/python'); from smbsearch import *; s=Search(40); print(s.bench(load_state('4-2'), 8_000_000))"` with the plain build → note frames/s.
- [ ] **Step 2:** `search/build.sh --pgo` → ok; repeat the bench → note frames/s (expect +10-30%).
- [ ] **Step 3:** rerun `search/tests/search_bench.py` → all pass on the PGO build.
- [ ] **Step 4:** commit "search/build.sh: PGO" if build.sh changed; numbers go into the README in Task 5.

---

### Task 3: Verification in stable-retro, demo renderer, solve driver; phase 1 (4-2)

**Files:** create `search/tools/solve.py`, `search/tools/verify_retro.py`, `search/tools/render_demo.py`.

**Interfaces:** `solve.py --start 4-2 --segments 1 --out search/out/4-2` writes `route.npz` (`start`, `actions` uint8 [all segments + settles], `levels` (str per segment), `seg_actions`, `seg_settle`, `ref_actions_<i>`) and `stats.json`; `verify_retro.verify(start_state, actions, on_frame=None) -> dict(ok, frames, mismatch)`; `render_demo.py route.npz [--compare] --out DIR` writes `demo.mp4`, `demo.gif` (and `compare.mp4`).

**`search/tools/solve.py`**
```python
"""Solve route segments: explore -> optimise -> settle, chained.

  venv_retro/bin/python search/tools/solve.py --start 4-2 --segments 1 --out search/out/4-2
  venv_retro/bin/python search/tools/solve.py --start FullGame --segments 8 --out search/out/e2e

Each segment starts where the previous one's optimised route settled (first
in-control frame of the next level). Writes route.npz and stats.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
from smbsearch import Search, load_state, ROUTE, gp, level_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='4-2')
    ap.add_argument('--segments', type=int, default=1)
    ap.add_argument('--route', default=','.join(ROUTE))
    ap.add_argument('--threads', type=int, default=40)
    ap.add_argument('--explore-budget', type=float, default=600)
    ap.add_argument('--explore-settle', type=float, default=120)
    ap.add_argument('--beam', type=int, default=20000)
    ap.add_argument('--per-cell', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    route = a.route.split(',')
    s = Search(threads=a.threads)
    state = load_state(a.start)
    n0, state = s.settle(state)
    assert n0 >= 0, 'start state never reaches player control'
    all_actions = [np.zeros(n0, np.uint8)]
    rec = dict(start=a.start, levels=[], seg_actions=[], seg_settle=[], lead_in=n0)
    stats = dict(segments=[], threads=s.threads, lead_in_steps=n0)
    for k in range(a.segments):
        lvl = s.level(state)
        t0 = time.time()
        ref = s.explore(state, route, budget_s=a.explore_budget, settle_s=a.explore_settle, seed=a.seed + k)
        t1 = time.time()
        if not ref.found:
            print('[solve] %s: explore found no exit in %.0f s' % (lvl, t1 - t0)); break
        print('[solve] %s: reference %d steps (%d cells, %d walks, %.0f s)'
              % (lvl, len(ref.actions), ref.stats['cells'], ref.stats['walks'], t1 - t0), flush=True)
        opt = s.optimize(state, route, ref.actions, beam=a.beam, per_cell=a.per_cell, verbose=1)
        t2 = time.time()
        best = opt.actions if opt.found and len(opt.actions) <= len(ref.actions) else ref.actions
        print('[solve] %s: optimised %d steps (reference %d, %.0f s)' % (lvl, len(best), len(ref.actions), t2 - t1), flush=True)
        tr, end = s.replay(state, best)
        nxt = level_name(int(tr[-1, 2])) if int(tr[-1, 2]) >= 0 else '?'
        settle, nstate = (0, end) if int(tr[-1, 7]) == 2 else s.settle(end)
        rec['levels'].append(lvl); rec['seg_actions'].append(len(best)); rec['seg_settle'].append(max(settle, 0))
        rec['ref_actions_%d' % k] = ref.actions
        all_actions += [best, np.zeros(max(settle, 0), np.uint8)]
        stats['segments'].append(dict(level=lvl, next=nxt, reference_steps=len(ref.actions),
                                      optimised_steps=len(best), settle_steps=settle,
                                      explore_s=round(t1 - t0, 1), optimise_s=round(t2 - t1, 1),
                                      explore=ref.stats, optimise=opt.stats))
        state = nstate
        if int(tr[-1, 7]) == 2:
            break
    rec['actions'] = np.concatenate(all_actions)
    np.savez(os.path.join(a.out, 'route.npz'), **{k: np.asarray(v) for k, v in rec.items()})
    stats['total_steps'] = int(len(rec['actions'])); stats['total_frames'] = int(len(rec['actions'])) * 4
    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=1, default=float)
    print('[solve] total %d steps = %d frames (%.2f s)' % (stats['total_steps'], stats['total_frames'],
                                                         stats['total_frames'] / 60.0988))


if __name__ == '__main__':
    main()
```

**`search/tools/verify_retro.py`**
```python
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
sys.path.insert(0, REPO)

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
        from mario_native_vecenv import _Lib
        self.lib = _Lib()
        rom = open(ROM, 'rb').read()
        self.env = self.lib.benv_create(rom, len(rom), 1, 1, 0)
        self.lib.benv_load(self.env, 0, state)
        self.lib.benv_get_ppu.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_char_p] * 3
        self.lib.benv_frames.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    def ram(self):
        return np.ctypeslib.as_array(self.lib.benv_ram(self.env, 0), shape=(0x800,))

    def ppu(self):
        bufs = [ctypes.create_string_buffer(n) for n in (0x800, 32, 256)]
        self.lib.benv_get_ppu(self.env, 0, *bufs)
        return [bytearray(b.raw) for b in bufs]

    def frame(self, buttons):
        self.lib.benv_frames(self.env, 0, 1, int(buttons))


def verify(start_state, actions, on_frame=None, retro_state='Level1-1'):
    import stable_retro as retro
    from mario_env import _register_integration
    _register_integration()
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
    res = verify(load_state(str(z['start'])), z['actions'])
    print('[verify] %s: %d frames, %s%s' % ('PASS' if res['ok'] else 'FAIL', res['frames'],
                                            res.get('end_level', ''), ' ' + ' '.join(res['mismatch'])))
    sys.exit(0 if res['ok'] else 1)
```
(`benv_frames(env, i, nframes, buttons)` steps raw frames, verified in `native/batchenv.cpp:364`. stable-retro adopts all RAM but its stack page from the native start state; the retro state only supplies CPU/mapper registers at a frame boundary. The lead-in NOOPs of solve.py are part of `actions`.)

**`search/tools/render_demo.py`**
```python
"""Render a verified route as a demo: 60 fps mp4 (3x, HUD with level, frame
counter and per-level splits) and a smaller GIF; --compare adds a side-by-side
of the explore reference vs the optimised route for single-segment routes.

  venv_retro/bin/python search/tools/render_demo.py search/out/4-2/route.npz --compare
"""
import argparse, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_retro import verify
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
from smbsearch import load_state, level_name

FPS = 60.0988


def _font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def frames_of(start, actions, title):
    """verified frames with the HUD; returns (list of HxWx3, splits)"""
    out, splits, cur = [], [], [None]
    font, small = _font(20), _font(16)

    def on_frame(f, screen, ram):
        lvl = level_name(int(ram[0x75F]) * 4 + int(ram[0x75C]))
        if lvl != cur[0]:
            splits.append((lvl, f)); cur[0] = lvl
        im = Image.fromarray(screen).resize((screen.shape[1] * 3, screen.shape[0] * 3), Image.NEAREST)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
        d.text((8, 4), '%s   %s   frame %d   %.2f s' % (title, lvl, f, f / FPS), fill=(255, 255, 255), font=font)
        for j, (l, f0) in enumerate(splits[-6:]):
            d.text((im.width - 180, 40 + 20 * j), '%s @ %.2f s' % (l, f0 / FPS), fill=(255, 255, 0), font=small)
        out.append(np.asarray(im))

    res = verify(start, actions, on_frame=on_frame)
    if not res['ok']:
        raise SystemExit('[render] verification FAILED at frame %d: %s' % (res['frames'], res['mismatch']))
    return out, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('route')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    z = np.load(a.route)
    out = a.out or os.path.dirname(os.path.abspath(a.route))
    start = load_state(str(z['start']))
    fr, splits = frames_of(start, z['actions'], 'optimised')
    imageio.mimwrite(os.path.join(out, 'demo.mp4'), fr, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    gif = [np.asarray(Image.fromarray(x).resize((x.shape[1] // 3, x.shape[0] // 3))) for x in fr[::3]]
    imageio.mimwrite(os.path.join(out, 'demo.gif'), gif, duration=3 / FPS, loop=0)
    print('[render] demo.mp4: %d frames (%.2f s); splits %s' % (len(fr), len(fr) / FPS,
                                                              ', '.join('%s %.2f' % (l, f / FPS) for l, f in splits)))
    if a.compare and 'ref_actions_0' in z.files:
        lead = np.zeros(int(z['lead_in']), np.uint8)
        rf, _ = frames_of(start, np.concatenate([lead, z['ref_actions_0']]), 'discovered')
        n = max(len(rf), len(fr))
        pad = lambda L: L + [L[-1]] * (n - len(L))
        both = [np.concatenate([x, y], axis=1) for x, y in zip(pad(rf), pad(fr))]
        imageio.mimwrite(os.path.join(out, 'compare.mp4'), both, fps=FPS, codec='libx264', quality=7, macro_block_size=1)
        print('[render] compare.mp4: discovered %d vs optimised %d frames' % (len(rf), len(fr)))


if __name__ == '__main__':
    main()
```

- [ ] **Step 1:** verify stable-retro lockstep on a known-good route: `venv_retro/bin/python -c` replaying the 1-1 route from Task 1 through `verify()` → ok.
- [ ] **Step 2:** phase 1: `venv_retro/bin/python search/tools/solve.py --start 4-2 --segments 1 --out search/out/4-2` (explore budget 600 s, settle 120 s, beam 20000) → reference and optimised step counts, into 8-1.
- [ ] **Step 3:** `search/tools/verify_retro.py search/out/4-2/route.npz` → PASS, end level 8-1.
- [ ] **Step 4:** `search/tools/render_demo.py search/out/4-2/route.npz --compare` → demo.mp4, demo.gif, compare.mp4; inspect frames (first / middle / last) visually.
- [ ] **Step 5:** commit tools + route.npz/stats.json summary numbers (videos stay in out/, gitignored); push.

---

### Task 4: Phase 2 — end to end (1-1 -> 8-4 axe)

- [ ] **Step 1:** `venv_retro/bin/python search/tools/solve.py --start FullGame --segments 8 --out search/out/e2e` → 8 segments, last one ends with `$0770 == 2`.
- [ ] **Step 2:** `verify_retro.py search/out/e2e/route.npz` → PASS.
- [ ] **Step 3:** `render_demo.py search/out/e2e/route.npz` → demo.mp4 with splits; inspect frames at each level start.
- [ ] **Step 4:** if a segment fails to explore within budget: rerun that segment alone with a larger `--explore-budget` (solve.py `--start` accepts the saved state of that segment, add `--start-file`); record which in stats.

---

### Task 5: README, docs, EXPERIMENTS

**`search/README.md`**: a pipeline diagram (as the spec), build/usage commands, a table of results (per-level frames, discovered vs optimised), throughput numbers (plain vs PGO, threads), and where the videos are.
- CLAUDE.md: one line under Commands (`search/build.sh`, `search/tools/solve.py`), one under Architecture pointing to `search/README.md`.
- EXPERIMENTS.md: entry with the 4-2 and e2e numbers.
- Commit + push.
