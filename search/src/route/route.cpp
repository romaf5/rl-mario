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
    std::vector<int> first, last;                    // in-control trace indices per waypoint
    for (size_t i = 0; i < f.size(); i++) {
        if (!c[i]) continue;
        if (w.frame.empty() || w.frame.back() != f[i]) {
            w.frame.push_back(f[i]); w.entry_x.push_back(x[i]); w.exit_x.push_back(x[i]);
            first.push_back((int)i); last.push_back((int)i);
        } else {
            w.exit_x.back() = x[i]; last.back() = (int)i;
        }
    }
    const int n = (int)w.frame.size();
    w.tail.assign(n, 0);
    if (n == 0) return w;
    // the trace ends one step before the goal step
    w.tail[n - 1] = ((int64_t)f.size() - last[n - 1]) * kStepUnits;
    for (int k = n - 2; k >= 0; k--)
        w.tail[k] = (int64_t)(first[k + 1] - last[k]) * kStepUnits
                    + px_units(std::abs(w.exit_x[k + 1] - w.entry_x[k + 1])) + w.tail[k + 1];
    return w;
}

bool Waypoints::rank(uint32_t f, int x, bool control, bool exiting, int k, int64_t parent, int* k_out,
                     int64_t* rank_out) const {
    const int n = (int)frame.size();
    if (!control) {
        if (frame[k] == f && exiting) {              // an exit under way: time runs down
            *k_out = k; *rank_out = std::min(parent, tail[k]) - kStepUnits;
            return true;
        }
        // arriving in a later waypoint: x is not Mario's yet (0 during the area change),
        // so the countdown goes on from the reference's entry estimate. Ranking by x here
        // put every node leaving 1-1's bonus room ~1260 frames from the goal: the beam
        // pruned exactly the nodes that were ahead.
        for (int j = k + 1; j < n && j <= k + 3; j++)
            if (frame[j] == f) {
                *k_out = j;
                *rank_out = std::min(parent, px_units(std::abs(exit_x[j] - entry_x[j])) + tail[j]) - kStepUnits;
                return true;
            }
        *k_out = k; *rank_out = parent;              // power-up / arrival animations: neutral
        return true;
    }
    // in control: the best matching waypoint among this one and the next three (a frame
    // the reference revisits -- 1-1's main area after the bonus room -- may be skipped to)
    int m = -1;
    int64_t best = 0;
    for (int j = k; j < n && j <= k + 3; j++) {
        if (frame[j] != f) continue;
        const int64_t v = px_units(std::abs(exit_x[j] - x)) + tail[j];
        if (m < 0 || v < best) { m = j; best = v; }
    }
    if (m < 0) return false;                         // off the reference's topology
    *k_out = m; *rank_out = best;
    return true;
}

}  // namespace ss
