// Progress along the reference path, the beam's rank. A node is credited with
// the latest reference step whose situation it matches (match_key: area, x/16,
// y/16, camera/64 and the tiles around Mario, coins ignored) and ranked by the
// time the reference still needed from a little past that point, plus the
// node's distance to where the reference was then. Reaching a reference
// situation sooner than the reference did is a shortcut; bumped blocks and a
// grown vine are part of the match, so puzzle steps count as progress (a rank
// by distance to the exit point rewarded 4-2 nodes that jumped in place under
// the vine instead of revealing the blocks that lead up to it).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include "../core/keys.h"

namespace ss {

constexpr int64_t kStepUnits = 4 * 16;                       // rank units: 1/16 frame; one decision
inline int64_t px_units(int px) { return (int64_t)px * 32 / 5; }   // at top running speed, 2.5 px/frame

inline uint64_t match_key(const uint8_t* r) {
    uint8_t g[104];
    tile_grid(r, g);
    for (auto& t : g)
        if (t == 0xC2 || t == 0xC3) t = 0;                   // coins, collected or not
    uint64_t k = frame_id(r);
    k = k * 4096 + (uint64_t)(mario_x(r) / 16);
    k = k * 64 + (uint64_t)(mario_y(r) / 16);
    k = k * 1024 + (uint64_t)(camera_x(r) / 64);
    return mix64(k ^ (hash_bytes(g, sizeof g) * 0x9E3779B97F4A7C15ULL)) | 1;
}

class RefProgress {
public:
    static constexpr int kLook = 8;                          // steps past the matched point

    // the reference's state after tau steps, tau = 0, 1, ... in order
    void add(const uint8_t* r, int tau) {
        frame_.push_back(frame_id(r));
        x_.push_back(mario_x(r));
        y_.push_back(mario_y(r));
        if (in_control(r)) {
            int& v = best_[match_key(r)];
            v = std::max(v, tau);
        }
    }
    int len() const { return (int)x_.size() - 1; }

    // the reference step nearest to a state in its area (a start that matches no
    // situation exactly: mid-level, reached another way); 0 if the area never occurs
    int nearest(const uint8_t* r) const {
        const uint32_t f = frame_id(r);
        int best = 0, bd = 1 << 30;
        for (int t = 0; t <= len(); t++) {
            if (frame_[t] != f) continue;
            const int d = std::max(std::abs(x_[t] - mario_x(r)), std::abs(y_[t] - mario_y(r)));
            if (d <= bd) { bd = d; best = t; }                   // ties: the later step
        }
        return best;
    }

    // a child's (tau, rank) from its RAM and its parent's (tau, rank)
    void rank(const uint8_t* r, int ptau, int64_t prank, int* tau, int64_t* rank) const {
        if (len() <= 0) { *tau = 0; *rank = px_units(100000 - mario_x(r)); return; }   // no reference: go right
        if (!in_control(r)) { *tau = ptau; *rank = prank - kStepUnits; return; }       // transitions: time runs
        int t = ptau;
        auto it = best_.find(match_key(r));
        if (it != best_.end() && it->second > t) t = it->second;
        *tau = t;
        const int t2 = std::min(t + kLook, len());
        int64_t v = (int64_t)(len() - t2) * kStepUnits;
        if (frame_[t2] == frame_id(r)) {
            const int dx = std::abs(x_[t2] - mario_x(r)), dy = std::abs(y_[t2] - mario_y(r));
            v += px_units(std::max(dx, dy));
        } else {
            v += (int64_t)(t2 - t) * kStepUnits;             // the look-ahead point is in another area
        }
        *rank = v;
    }

private:
    std::vector<uint32_t> frame_;
    std::vector<int> x_, y_;
    std::unordered_map<uint64_t, int> best_;                 // match_key -> latest reference step
};

}  // namespace ss
