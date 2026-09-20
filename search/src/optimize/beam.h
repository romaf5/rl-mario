// Beam A* over time: depth d holds nodes after 4d frames; the B best (estimated
// time to the goal along the reference, progress.h; then x speed) are expanded by all 12 actions,
// deduplicated by exact state, spread by at most per_cell per coarse cell. The
// reference's own node is always kept, so the result is never slower than it.
// ref_start (optional): the reference was found from another state of this level
// (e.g. without the start delay); its checkpoints are replayed from there, and no
// reference node is kept.
#pragma once
#include <cstdint>
#include <vector>
#include "../core/pool.h"
#include "../emu/emu.h"
#include "../route/route.h"

namespace ss {

struct OptimizeParams {
    int beam = 20000, per_cell = 16, max_depth = 6000;
    int verbose = 0;               // 1: progress every 25 depths, 2: + the best node's state
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
                        const OptimizeParams& p, const uint8_t* ref_start_full = nullptr);

}  // namespace ss
