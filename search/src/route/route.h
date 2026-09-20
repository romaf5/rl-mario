// Segments of a route (start level -> next route level or the axe) and step outcomes.
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

}  // namespace ss
