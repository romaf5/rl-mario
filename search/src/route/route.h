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

// the reference's frames (frame_id) in visit order, where each was entered /
// left and how long its exit transition took. rank = estimated time to the goal
// in 1/16 frames: px at top running speed (2.5 px/frame) plus the reference's
// transition times (pipes, vine, flag sequence).
constexpr int64_t kStepUnits = 4 * 16;                       // one decision
inline int64_t px_units(int px) { return (int64_t)px * 32 / 5; }

struct Waypoints {
    std::vector<uint32_t> frame;
    std::vector<int> entry_x, exit_x;
    std::vector<int64_t> tail;       // time after reaching waypoint k's exit point
    bool empty() const { return frame.empty(); }
    // node in waypoint k -> (k', rank). exiting: control lost to an exit routine
    // ($0E 1-5: vine, pipes, flagpole, castle) -- further along ranks better.
    // false if an in-control frame is off the path.
    bool rank(uint32_t f, int x, bool control, bool exiting, int k, int64_t parent, int* k_out,
              int64_t* rank_out) const;
};

Waypoints waypoints_from_trace(const std::vector<uint32_t>& frames, const std::vector<int>& xs,
                               const std::vector<uint8_t>& control);

}  // namespace ss
