#include "../include/smbsearch.h"
#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include "core/keys.h"
#include "core/pool.h"
#include "emu/emu.h"
#include "emu/obs.h"
#include "explore/explore.h"
#include "optimize/beam.h"
#include "mcts/mcts.h"

using namespace ss;

struct ss_ctx {
    std::vector<uint8_t> rom;
    std::unique_ptr<Pool> pool;
    std::vector<std::unique_ptr<Emu>> owned;
    std::vector<Emu*> emus;
};

struct ss_mcts {
    Forest forest;
    ss_mcts(ss_ctx* c, int n, const MctsParams& p) : forest(*c->pool, c->emus, n, p) {}
};

static int copy_out(const std::vector<uint8_t>& a, uint8_t* out, int max_out) {
    if ((int)a.size() > max_out) return -2;
    if (!a.empty()) memcpy(out, a.data(), a.size());
    return (int)a.size();
}

static void trace_row(const uint8_t* r, int32_t* t) {
    t[0] = mario_x(r); t[1] = mario_y(r); t[2] = level_gp(r); t[3] = r[0x760]; t[4] = r[0x74F];
    t[5] = r[0x74E]; t[6] = r[0x0E]; t[7] = r[0x770]; t[8] = camera_x(r); t[9] = lives(r);
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
        if (trace) trace_row(e.ram(), trace + (size_t)i * SS_TRACE);
    }
    if (end_state) e.save_full(end_state);
    return n;
}

int ss_lookahead(ss_ctx* c, const uint8_t* start, const int32_t* route, int n_route, const uint8_t* ref, int n_ref,
                 const uint8_t* ref_start, int beam, int per_cell, int horizon, uint8_t* out, int max_out,
                 double* est_frames, int32_t* found) {
    OptimizeParams p;
    p.beam = beam; p.per_cell = per_cell; p.max_depth = horizon; p.partial = true;
    const OptimizeResult r = optimize(*c->pool, c->emus, start, std::vector<int>(route, route + n_route),
                                      std::vector<uint8_t>(ref, ref + n_ref), p, ref_start);
    if (est_frames) *est_frames = r.est_frames;
    if (found) *found = r.found;
    return copy_out(r.actions, out, max_out);
}

int ss_frames(ss_ctx* c, const uint8_t* state, int n, int buttons, uint8_t* end_state) {
    Emu& e = *c->emus[0];
    e.load_full(state);
    for (int i = 0; i < n; i++) e.frame((uint8_t)buttons);
    if (end_state) e.save_full(end_state);
    return n;
}

int ss_obs(ss_ctx* c, const uint8_t* state, uint8_t* obs_out) {
    c->emus[0]->load_full(state);
    c->emus[0]->obs_now(obs_out);
    return kObsSize;
}

int ss_replay_obs(ss_ctx* c, const uint8_t* start, const uint8_t* actions, int n, uint8_t* obs_out,
                  int32_t* trace, uint8_t* end_state) {
    Emu& e = *c->emus[0];
    e.load_full(start);
    for (int i = 0; i < n; i++) {
        if (actions[i] >= kNumActions) return -1;
        e.step_obs(actions[i], obs_out + (size_t)i * kObsSize);
        if (trace) trace_row(e.ram(), trace + (size_t)i * SS_TRACE);
    }
    if (end_state) e.save_full(end_state);
    return n;
}

int ss_classify_along(ss_ctx* c, const uint8_t* start, const int32_t* route, int n_route,
                      const uint8_t* actions, int n, uint8_t* out) {
    Emu& e = *c->emus[0];
    e.load_full(start);
    const Segment seg = Segment::make(std::vector<int>(route, route + n_route), e.ram());
    if (seg.start_gp < 0) return -1;
    for (int i = 0; i < n; i++) {
        if (actions[i] >= kNumActions) return -1;
        e.step(actions[i]);
        const Outcome o = seg.classify(e.ram());
        out[i] = o == Outcome::Goal ? 1 : o == Outcome::Dead ? 2 : 0;
        if (out[i]) return i + 1;                  // the segment ended here
    }
    return n;
}

int ss_forced_along(ss_ctx* c, const uint8_t* start, const uint8_t* actions, int n, uint8_t* out) {
    const size_t CS = compact_state_size();
    std::vector<uint8_t> st((size_t)n * CS);
    Emu& e0 = *c->emus[0];
    e0.load_full(start);
    for (int i = 0; i < n; i++) {
        if (actions[i] >= kNumActions) return -1;
        e0.save(st.data() + (size_t)i * CS);
        e0.step(actions[i]);
    }
    std::vector<uint64_t> key((size_t)n * kNumActions);
    c->pool->parallel_for((int64_t)n * kNumActions, [&](int64_t j, int w) {
        Emu& e = *c->emus[w];
        e.load(st.data() + (size_t)(j / kNumActions) * CS);
        e.step((int)(j % kNumActions));
        e.step(0);
        key[j] = exact_key(e.ram());
    }, 4);
    for (int i = 0; i < n; i++) {
        out[i] = 1;
        for (int a = 1; a < kNumActions; a++)
            if (key[(size_t)i * kNumActions + a] != key[(size_t)i * kNumActions]) { out[i] = 0; break; }
    }
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
                ss_stats* st, const uint8_t* ref_start) {
    OptimizeParams p;
    p.beam = beam; p.per_cell = per_cell; p.max_depth = max_depth; p.verbose = verbose;
    const OptimizeResult r = optimize(*c->pool, c->emus, start, std::vector<int>(route, route + n_route),
                                      std::vector<uint8_t>(ref, ref + n_ref), p, ref_start);
    const int n = copy_out(r.actions, out, max_out);
    if (st) {
        st->frames = (int64_t)r.actions.size() * kFrameSkip; st->emu_frames = r.emu_frames;
        st->seconds = r.seconds; st->cells = r.nodes; st->walks = r.depth;
        st->found = r.found; st->n_actions = (int32_t)r.actions.size();
    }
    return n;
}

ss_mcts* ss_mcts_create(ss_ctx* c, int n_trees, const ss_mcts_params* p) {
    MctsParams q;
    if (p) {
        q.c_puct = p->c_puct; q.fpu = p->fpu; q.scale = p->scale; q.v_death = p->v_death;
        q.max_nodes = p->max_nodes; q.value_mix = p->value_mix; q.min_backup = p->min_backup;
    }
    return n_trees > 0 ? new ss_mcts(c, n_trees, q) : nullptr;
}

void ss_mcts_destroy(ss_mcts* m) { delete m; }

int ss_mcts_reset(ss_mcts* m, int t, const uint8_t* state, const int32_t* route, int n_route) {
    if (t < 0 || t >= m->forest.size()) return -1;
    return m->forest.reset(t, state, std::vector<int>(route, route + n_route)) ? 0 : -1;
}

int ss_mcts_select(ss_mcts* m, const int32_t* trees, int n_trees, int per_tree, int max_leaves, int32_t* leaves,
                   uint8_t* stacks) {
    return m->forest.select(trees, n_trees, per_tree, max_leaves, reinterpret_cast<MctsLeaf*>(leaves), stacks);
}

void ss_mcts_backup(ss_mcts* m, int n, const int32_t* leaves, const float* priors, const float* values) {
    m->forest.backup(n, reinterpret_cast<const MctsLeaf*>(leaves), priors, values);
}

int ss_mcts_root(ss_mcts* m, int t, int32_t* visits, float* best, float* root_b) {
    return m->forest.root_stats(t, visits, best, root_b);
}

void ss_mcts_noise(ss_mcts* m, int t, const float* noise, float frac) { m->forest.root_noise(t, noise, frac); }
int ss_mcts_commit(ss_mcts* m, int t, int action) {
    return action < 0 || action >= kNumActions ? -1 : m->forest.commit(t, action);
}
void ss_mcts_forced(ss_mcts* m, const int32_t* trees, int n, int32_t* out) { m->forest.forced(trees, n, out); }
void ss_mcts_state(ss_mcts* m, int t, uint8_t* full, uint8_t* ram, uint8_t* stack) {
    m->forest.root_state(t, full, ram, stack);
}
int ss_mcts_nodes(ss_mcts* m, int t) { return m->forest.nodes(t); }
void ss_mcts_set_route(ss_mcts* m, int level_gp, const uint8_t* start, const uint8_t* actions, int n) {
    m->forest.set_route(level_gp, start, actions, n);
}
void ss_mcts_set_value_mix(ss_mcts* m, float mix) { m->forest.set_value_mix(mix); }
void ss_mcts_leaf_info(ss_mcts* m, int n, const int32_t* leaves, float* values, int32_t* depths) {
    m->forest.leaf_info(n, reinterpret_cast<const MctsLeaf*>(leaves), values, depths);
}

float ss_mcts_root_value(ss_mcts* m, int t) { return m->forest.root_value(t); }

void ss_mcts_safe(ss_mcts* m, int t, const int32_t* actions, int n, int horizon, int32_t* out) {
    m->forest.safe(t, actions, n, horizon, out);
}

}  // extern "C"
