#include "beam.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include "../core/keys.h"
#include "progress.h"

namespace ss {

OptimizeResult optimize(Pool& pool, std::vector<Emu*>& emus, const uint8_t* start_full,
                        const std::vector<int>& route, const std::vector<uint8_t>& ref,
                        const OptimizeParams& p, const uint8_t* ref_start_full) {
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

    // the reference path: progress checkpoints; ref_len = its steps before the goal step
    if (ref_start_full) e0.load_full(ref_start_full);
    RefProgress rp;
    rp.add(e0.ram(), 0);
    int ref_len = 0;
    for (size_t d = 0; d < ref.size(); d++) {
        e0.step(ref[d]);
        if (seg.classify(e0.ram()) != Outcome::Running) break;
        rp.add(e0.ram(), (int)d + 1);
        ref_len = (int)d + 1;
    }
    e0.load(s0.data());
    int t0v = 0;
    int64_t d0 = 0;
    // a start away from the reference's own (mid-level): anchor at its nearest reference step
    const int tau0 = ref_start_full ? rp.nearest(e0.ram()) : 0;
    rp.rank(e0.ram(), tau0, (int64_t)1 << 40, &t0v, &d0);
    if (!in_control(e0.ram())) d0 = (int64_t)(rp.len() - tau0) * kStepUnits;   // in a transition: the reference's time left

    const int B = std::max(1, p.beam);
    std::vector<uint8_t> cur(s0), nxt;
    std::vector<int32_t> cur_k{t0v};
    std::vector<int64_t> cur_d{d0};
    int ref_node = ref_len > 0 && !ref_start_full ? 0 : -1;   // the reference's node in cur
    std::vector<std::vector<uint32_t>> par;              // par[d-1][j]: parent of node j at depth d
    std::vector<std::vector<uint8_t>> act;
    std::vector<uint8_t> chs, ok;
    std::vector<int32_t> ch_k;
    std::vector<int64_t> ch_d;
    std::vector<int8_t> spd;
    std::vector<uint64_t> ek, cek;
    int64_t emu = 0, kept = 0;

    int last = 0;
    for (int depth = 1; depth <= p.max_depth; depth++) {
        last = depth;
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
                rp.rank(r, cur_k[i], cur_d[i], &k, &d);
                ok[c] = 1; ch_k[c] = k; ch_d[c] = d; spd[c] = (int8_t)r[0x57];
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
            res.est_frames = (double)depth * kFrameSkip;
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
        if (sel.empty()) { last = depth - 1; break; }

        const int ns = (int)sel.size();
        nxt.resize((size_t)ns * CS);
        std::vector<int32_t> nk(ns);
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
        if (p.verbose > 1 && depth % 25 == 0) {            // diagnostics: the best node's state
            const int b = (int)(std::min_element(cur_d.begin(), cur_d.end()) - cur_d.begin());
            Emu& e = *emus[0];
            e.load(cur.data() + (size_t)b * CS);
            const uint8_t* r = e.ram();
            int nneg = 0;
            for (auto v : cur_d) nneg += v < 0;
            fprintf(stderr, "[beam]   best: $0E %02X x %d y %d level %d area %d/%d mode %d timer %d; %d nodes ranked < 0\n",
                    r[0x0E], mario_x(r), mario_y(r), level_gp(r), r[0x760], r[0x74F], r[0x770], game_timer(r), nneg);
        }
        if (p.verbose && depth % 25 == 0)
            fprintf(stderr, "[beam] depth %d (%d frames): %d nodes of %zu, best %.0f frames left, reference step %d/%d, %.0f frames/s\n",
                    depth, depth * kFrameSkip, ns, order.size(), *std::min_element(cur_d.begin(), cur_d.end()) / 16.0,
                    (int)*std::max_element(cur_k.begin(), cur_k.end()), ref_len, emu / elapsed());
    }
    if (!res.found && p.partial && last > 0 && !cur_d.empty()) {   // the best node's path
        uint32_t node = (uint32_t)(std::min_element(cur_d.begin(), cur_d.end()) - cur_d.begin());
        res.est_frames = (double)last * kFrameSkip + cur_d[node] / 16.0;
        std::vector<uint8_t> path;
        for (int d = last; d >= 1; d--) { path.push_back(act[d - 1][node]); node = par[d - 1][node]; }
        std::reverse(path.begin(), path.end());
        res.actions = std::move(path); res.depth = last;
    }
    res.emu_frames = emu;
    res.nodes = kept;
    res.seconds = elapsed();
    return res;
}

}  // namespace ss
