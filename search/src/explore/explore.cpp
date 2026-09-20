#include "explore.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
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
    std::unordered_map<uint64_t, int> spots;            // tile variants per spot
    {
        std::vector<uint8_t> s0(CS);
        e0.save(s0.data());
        cells.emplace_back(std::move(s0), std::vector<uint8_t>());
        const uint64_t k0 = cell_key(e0.ram());
        if (k0) { index.emplace(k0, 0u); spots[spot_key(e0.ram())] = 1; }
    }
    std::mutex pick_mu;                                 // guards fresh; taken before mu
    std::vector<uint32_t> fresh{0};                     // cells with < fresh_uses walks
    std::mutex gmu;                                     // guards the goal path
    std::vector<uint8_t> best;
    bool found = false;
    double found_at = 0;
    std::atomic<int64_t> frames{0}, walks{0};

    pool.parallel_for(pool.size(), [&](int64_t wi, int w) {
        Emu& e = *emus[w];
        std::mt19937_64 rng(p.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)wi + 1);
        std::vector<uint8_t> buf(CS), acts;
        struct Cand { std::vector<uint8_t> state, actions; uint64_t spot; };
        std::unordered_map<uint64_t, Cand> local;
        int64_t fr = 0, nw = 0;
        double next_log = 15;
        for (;;) {
            const double t = elapsed();
            {
                std::lock_guard<std::mutex> g(gmu);
                if (t > p.budget_s || (found && t > found_at + p.settle_s)) break;
            }
            {   // fresh cells first: a link found by 10% of walks is found with ~99% at 60
                std::lock_guard<std::mutex> pl(pick_mu);
                std::shared_lock<std::shared_mutex> l(mu);
                size_t ci;
                if (!fresh.empty()) {
                    const size_t j = (size_t)(rng() % fresh.size());
                    ci = fresh[j];
                    if ((int)cells[ci].picks.fetch_add(1) + 1 >= p.fresh_uses) { fresh[j] = fresh.back(); fresh.pop_back(); }
                } else {
                    const size_t n = cells.size();
                    for (;;) {
                        ci = (size_t)(rng() % n);
                        if ((double)(rng() >> 11) * 0x1.0p-53 < 1.0 / (1.0 + cells[ci].picks.load())) break;
                    }
                    cells[ci].picks.fetch_add(1);
                }
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
                    c.spot = spot_key(r);
                    local.emplace(k, std::move(c));
                }
            }
            if (!local.empty()) {
                std::lock_guard<std::mutex> pl(pick_mu);
                std::unique_lock<std::shared_mutex> l(mu);
                for (auto& kv : local) {
                    auto g = index.find(kv.first);
                    if (g == index.end()) {
                        int& nv = spots[kv.second.spot];
                        if (nv >= p.max_variants) continue;
                        nv++;
                        if ((int64_t)cells.size() < p.max_cells) {
                            index.emplace(kv.first, (uint32_t)cells.size());
                            fresh.push_back((uint32_t)cells.size());
                            cells.emplace_back(std::move(kv.second.state), std::move(kv.second.actions));
                        }
                    } else if (cells[g->second].actions.size() > kv.second.actions.size()) {
                        cells[g->second].state = std::move(kv.second.state);
                        cells[g->second].actions = std::move(kv.second.actions);
                    }
                }
            }
            nw++;
            if (p.verbose && w == 0 && t >= next_log) {
                next_log = t + 15;
                std::lock_guard<std::mutex> pl(pick_mu);
                std::shared_lock<std::shared_mutex> l(mu);
                fprintf(stderr, "[explore] %.0f s: %zu cells in %zu spots (%zu fresh), ~%lld walks, goal %s\n", t,
                        cells.size(), spots.size(), fresh.size(), (long long)(nw * pool.size()), found ? "found" : "not yet");
            }
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
