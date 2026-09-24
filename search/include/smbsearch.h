/* smbsearch C API: route search for Super Mario Bros on the native core.
 * States crossing the API are full native savestates (ss_state_size() bytes:
 * native/states and MarioNativeVecEnv use the same format). Actions are
 * COMPLEX_MOVEMENT indices (0-11), one per 4 frames of the real game. A route
 * is a list of level indices (world-1)*4 + (stage-1) in play order; a segment
 * runs from its start level to the next route level (the last: the axe). */
#ifndef SMBSEARCH_H
#define SMBSEARCH_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define SS_API __attribute__((visibility("default")))
typedef struct ss_ctx ss_ctx;
typedef struct {
    int64_t frames;      /* frames of the returned actions (4 per action) */
    int64_t emu_frames;  /* frames emulated in total */
    double seconds;
    int64_t cells;       /* explore: archive cells; optimize: nodes kept */
    int64_t walks;       /* explore: random walks; optimize: depths searched */
    int32_t found;       /* 1 if the segment goal was reached */
    int32_t n_actions;
} ss_stats;
/* replay trace per step: x, y, level, area, sub-area, AreaType, $0E, $0770, camera x, lives */
#define SS_TRACE 10
SS_API int ss_state_size(void);
SS_API ss_ctx* ss_create(const uint8_t* rom, int rom_len, int threads);
SS_API void ss_destroy(ss_ctx* ctx);
SS_API int ss_threads(ss_ctx* ctx);
SS_API int ss_ram(ss_ctx* ctx, const uint8_t* state, uint8_t* ram_out /* 0x800 */);
SS_API int ss_replay(ss_ctx* ctx, const uint8_t* start, const uint8_t* actions, int n,
                     int32_t* trace, uint8_t* end_state);
SS_API int ss_settle(ss_ctx* ctx, const uint8_t* state, int max_steps, uint8_t* end_state);
SS_API double ss_bench(ss_ctx* ctx, const uint8_t* start, int64_t frames);
SS_API int ss_selftest(ss_ctx* ctx, const uint8_t* start, int steps, uint64_t seed);
SS_API int ss_explore(ss_ctx* ctx, const uint8_t* start, const int32_t* route, int n_route,
                      double budget_s, double settle_s, int max_walk, uint64_t seed, int verbose,
                      uint8_t* out, int max_out, ss_stats* st);
/* ref_start: NULL, or the state the reference was found from (another start of this level) */
SS_API int ss_optimize(ss_ctx* ctx, const uint8_t* start, const int32_t* route, int n_route,
                       const uint8_t* ref, int n_ref, int beam, int per_cell, int max_depth,
                       int verbose, uint8_t* out, int max_out, ss_stats* st, const uint8_t* ref_start);
/* local teacher: the beam for at most `horizon` steps from start along the reference;
 * returns the best path (to the goal if within the horizon, else to the best-ranked
 * node) and its frames to the goal in *est_frames (exact if the goal was reached) */
SS_API int ss_lookahead(ss_ctx* ctx, const uint8_t* start, const int32_t* route, int n_route,
                        const uint8_t* ref, int n_ref, const uint8_t* ref_start, int beam, int per_cell,
                        int horizon, uint8_t* out, int max_out, double* est_frames, int32_t* found);
/* the net's frames: 84x84 grayscale, status bar cropped, max of each step's last 2 frames */
#define SS_OBS 84
SS_API int ss_frames(ss_ctx* ctx, const uint8_t* state, int n, int buttons, uint8_t* end_state);
SS_API int ss_obs(ss_ctx* ctx, const uint8_t* state, uint8_t* obs_out /* 84*84: the current frame */);
/* per step of a replay: the segment outcome (0 running, 1 goal, 2 dead) -- the search's own
 * rule, as the world model's event labels */
SS_API int ss_classify_along(ss_ctx* ctx, const uint8_t* start, const int32_t* route, int n_route,
                             const uint8_t* actions, int n, uint8_t* out);
/* per step of a replay: 1 if the input did nothing there (every (action, NOOP) pair from
 * that state reaches one exact state: transitions, flag, pipes) */
SS_API int ss_forced_along(ss_ctx* ctx, const uint8_t* start, const uint8_t* actions, int n, uint8_t* out);
SS_API int ss_replay_obs(ss_ctx* ctx, const uint8_t* start, const uint8_t* actions, int n,
                         uint8_t* obs_out /* n*84*84 */, int32_t* trace, uint8_t* end_state);

/* MCTS forest (search/src/mcts): n trees, each one game. A wave selects new leaves in
 * the listed trees, emulates and renders them on the pool; the caller's net scores
 * them (priors as probabilities, values in frames to the segment goal) and backs up.
 * Leaves are (tree, node) int32 pairs; stacks are 4 x 84 x 84 uint8, oldest first. */
typedef struct ss_mcts ss_mcts;
typedef struct { float c_puct, fpu, scale, v_death; int32_t max_nodes; float value_mix;
                 int32_t min_backup, relative; } ss_mcts_params;
SS_API ss_mcts* ss_mcts_create(ss_ctx* ctx, int n_trees, const ss_mcts_params* p);
SS_API void ss_mcts_destroy(ss_mcts* m);
SS_API int ss_mcts_reset(ss_mcts* m, int tree, const uint8_t* state, const int32_t* route, int n_route);
SS_API int ss_mcts_select(ss_mcts* m, const int32_t* trees, int n_trees, int per_tree, int max_leaves,
                          int32_t* leaves, uint8_t* stacks);
SS_API void ss_mcts_backup(ss_mcts* m, int n, const int32_t* leaves, const float* priors, const float* values);
SS_API int ss_mcts_root(ss_mcts* m, int tree, int32_t* visits, float* best, float* root_b);
SS_API void ss_mcts_noise(ss_mcts* m, int tree, const float* noise, float frac);
SS_API int ss_mcts_commit(ss_mcts* m, int tree, int action);   /* new root: 0 running, 1 goal, 2 dead */
SS_API void ss_mcts_forced(ss_mcts* m, const int32_t* trees, int n, int32_t* out);
SS_API void ss_mcts_state(ss_mcts* m, int tree, uint8_t* full, uint8_t* ram, uint8_t* stack);
SS_API int ss_mcts_nodes(ss_mcts* m, int tree);
/* leaf values = value_mix x net + (1 - value_mix) x frames to go along the level's route
 * (the search's progress rank) where a route is set: its actions from its start state */
SS_API void ss_mcts_set_route(ss_mcts* m, int level_gp, const uint8_t* start, const uint8_t* actions, int n);
SS_API void ss_mcts_set_value_mix(ss_mcts* m, float mix);
/* survival check of candidate root actions: out[i] = 1 if some button held for `horizon`
 * steps after actions[i] does not die */
/* per leaf of the last select: the route's frames to go and the depth from the root
 * (training data for a relative value: D = value(leaf) - value(root)) */
SS_API void ss_mcts_leaf_info(ss_mcts* m, int n, const int32_t* leaves, float* values, int32_t* depths);
SS_API float ss_mcts_root_value(ss_mcts* m, int tree);
/* after a decision: up to max_n visited nodes of the tree with the value the search backed up
 * into them (relative to the root), depth, visits and 4-frame stacks -- value targets from the
 * agent's own search, needing no route */
SS_API int ss_mcts_dump(ss_mcts* m, int tree, int max_n, uint64_t seed, float* b, int32_t* depth,
                        int32_t* visits, uint8_t* stacks);
SS_API void ss_mcts_safe(ss_mcts* m, int tree, const int32_t* actions, int n, int horizon, int32_t* out);
#ifdef __cplusplus
}
#endif
#endif
