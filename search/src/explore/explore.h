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
