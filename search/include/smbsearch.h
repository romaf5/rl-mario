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
SS_API int ss_optimize(ss_ctx* ctx, const uint8_t* start, const int32_t* route, int n_route,
                       const uint8_t* ref, int n_ref, int beam, int per_cell, int max_depth,
                       int verbose, uint8_t* out, int max_out, ss_stats* st);
#ifdef __cplusplus
}
#endif
#endif
