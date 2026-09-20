#include "../include/smbsearch.h"
#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include "core/pool.h"
#include "emu/emu.h"
#include "explore/explore.h"
#include "optimize/beam.h"

using namespace ss;

struct ss_ctx {
    std::vector<uint8_t> rom;
    std::unique_ptr<Pool> pool;
    std::vector<std::unique_ptr<Emu>> owned;
    std::vector<Emu*> emus;
};

static int copy_out(const std::vector<uint8_t>& a, uint8_t* out, int max_out) {
    if ((int)a.size() > max_out) return -2;
    if (!a.empty()) memcpy(out, a.data(), a.size());
    return (int)a.size();
}

extern "C" {

int ss_state_size(void) { return (int)full_state_size(); }

ss_ctx* ss_create(const uint8_t* rom, int rom_len, int threads) {
    if (threads < 1) threads = (int)std::thread::hardware_concurrency();
    auto* c = new ss_ctx;
    c->rom.assign(rom, rom + rom_len);
    for (int i = 0; i < threads; i++) {
        c->owned.emplace_back(new Emu(c->rom.data(), rom_len));
        if (!c->owned.back()->ok()) { delete c; return nullptr; }
        c->emus.push_back(c->owned.back().get());
    }
    c->pool.reset(new Pool(threads));
    return c;
}

void ss_destroy(ss_ctx* c) { delete c; }
int ss_threads(ss_ctx* c) { return c->pool->size(); }

int ss_ram(ss_ctx* c, const uint8_t* state, uint8_t* ram_out) {
    c->emus[0]->load_full(state);
    memcpy(ram_out, c->emus[0]->ram(), 0x800);
    return 0x800;
}

int ss_replay(ss_ctx* c, const uint8_t* start, const uint8_t* actions, int n, int32_t* trace,
              uint8_t* end_state) {
    Emu& e = *c->emus[0];
    e.load_full(start);
    for (int i = 0; i < n; i++) {
        if (actions[i] >= kNumActions) return -1;
        e.step(actions[i]);
        if (trace) {
            const uint8_t* r = e.ram();
            int32_t* t = trace + (size_t)i * SS_TRACE;
            t[0] = mario_x(r); t[1] = mario_y(r); t[2] = level_gp(r); t[3] = r[0x760]; t[4] = r[0x74F];
            t[5] = r[0x74E]; t[6] = r[0x0E]; t[7] = r[0x770]; t[8] = camera_x(r); t[9] = lives(r);
        }
    }
    if (end_state) e.save_full(end_state);
    return n;
}

int ss_settle(ss_ctx* c, const uint8_t* state, int max_steps, uint8_t* end_state) {
    Emu& e = *c->emus[0];
    e.load_full(state);
    int n = 0;
    while (n < max_steps && !in_control(e.ram())) { e.step(0); n++; }
    if (end_state) e.save_full(end_state);
    return in_control(e.ram()) ? n : -1;
}

double ss_bench(ss_ctx* c, const uint8_t* start, int64_t frames) {
    const size_t CS = compact_state_size();
    std::vector<uint8_t> s(CS);
    c->emus[0]->load_full(start);
    c->emus[0]->save(s.data());
    const int T = c->pool->size();
    const int64_t per = frames / T / kFrameSkip + 1;
    const auto t0 = std::chrono::steady_clock::now();
    c->pool->parallel_for(T, [&](int64_t i, int w) {
        Emu& e = *c->emus[w];
        std::mt19937 rng((uint32_t)i + 1);
        for (int64_t k = 0; k < per; k++) {
            if (k % 400 == 0) e.load(s.data());
            e.step((int)(rng() % kNumActions));
        }
    });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)per * T * kFrameSkip / sec;
}

int ss_selftest(ss_ctx* c, const uint8_t* start, int steps, uint64_t seed) {
    if (c->emus.size() < 2) return -1;
    Emu& a = *c->emus[0];
    Emu& b = *c->emus[1];
    std::mt19937_64 rng(seed);
    std::vector<int> acts(steps);
    for (auto& x : acts) x = (int)(rng() % kNumActions);
    a.load_full(start);
    for (int x : acts) a.step(x);
    b.load_full(start);
    std::vector<uint8_t> s(compact_state_size());
    for (int i = 0; i < steps; i++) {
        b.step(acts[i]);
        if (i % 7 == 3) { b.save(s.data()); b.load(s.data()); }   // through a compact state
    }
    std::vector<uint8_t> fa(full_state_size()), fb(full_state_size());
    a.save_full(fa.data());
    b.save_full(fb.data());
    return memcmp(fa.data(), fb.data(), fa.size()) == 0 ? 0 : 1;
}

int ss_explore(ss_ctx* c, const uint8_t* start, const int32_t* route, int n_route, double budget_s,
               double settle_s, int max_walk, uint64_t seed, int verbose, uint8_t* out, int max_out, ss_stats* st) {
    ExploreParams p;
    p.budget_s = budget_s; p.settle_s = settle_s; p.max_walk = max_walk; p.seed = seed; p.verbose = verbose;
    const ExploreResult r = explore(*c->pool, c->emus, start, std::vector<int>(route, route + n_route), p);
    const int n = copy_out(r.actions, out, max_out);
    if (st) {
        st->frames = (int64_t)r.actions.size() * kFrameSkip; st->emu_frames = r.emu_frames;
        st->seconds = r.seconds; st->cells = r.cells; st->walks = r.walks;
        st->found = r.found; st->n_actions = (int32_t)r.actions.size();
    }
    return n;
}

int ss_optimize(ss_ctx* c, const uint8_t* start, const int32_t* route, int n_route, const uint8_t* ref,
                int n_ref, int beam, int per_cell, int max_depth, int verbose, uint8_t* out, int max_out,
                ss_stats* st) {
    OptimizeParams p;
    p.beam = beam; p.per_cell = per_cell; p.max_depth = max_depth; p.verbose = verbose;
    const OptimizeResult r = optimize(*c->pool, c->emus, start, std::vector<int>(route, route + n_route),
                                      std::vector<uint8_t>(ref, ref + n_ref), p);
    const int n = copy_out(r.actions, out, max_out);
    if (st) {
        st->frames = (int64_t)r.actions.size() * kFrameSkip; st->emu_frames = r.emu_frames;
        st->seconds = r.seconds; st->cells = r.nodes; st->walks = r.depth;
        st->found = r.found; st->n_actions = (int32_t)r.actions.size();
    }
    return n;
}

}  // extern "C"
