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

}  // namespace ss
