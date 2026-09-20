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
