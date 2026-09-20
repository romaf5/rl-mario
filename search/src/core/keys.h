// State keys. cell_key: the Go-Explore cell (the key that found the 4-2 route in
// the Python env). exact_key: the RAM that is game state (beam dedupe within a
// depth). coarse_key: the beam's diversity cell.
#pragma once
#include <cstdint>
#include <cstring>
#include "../emu/emu.h"
#include "hash.h"

namespace ss {

// metatiles of Mario's 128-px bin: 8 columns x 13 rows of the $0500 buffer;
// false for a block mid-bump ($23): a transient, not a place
inline bool tile_grid(const uint8_t* r, uint8_t g[104]) {
    const int col0 = (mario_x(r) / 128) * 8;
    bool bump = false;
    for (int j = 0; j < 8; j++) {
        const int cx = (col0 + j) * 16;
        const int base = 0x500 + ((cx / 256) % 2) * 0xD0 + (cx % 256) / 16;
        for (int row = 0; row < 13; row++) {
            const uint8_t t = r[base + row * 16];
            g[j * 13 + row] = t;
            bump |= t == 0x23;
        }
    }
    return !bump;
}

// frame, x/32, y band/16 ($03B8), camera/64, tile signature; 0 = not a cell
inline uint64_t cell_key(const uint8_t* r) {
    uint8_t g[104];
    if (!tile_grid(r, g)) return 0;
    uint64_t k = frame_id(r);
    k = k * 1024 + (uint64_t)(mario_x(r) / 32);
    k = k * 32 + (uint64_t)(r[0x3B8] / 16);
    k = k * 1024 + (uint64_t)(camera_x(r) / 64);
    return mix64(k ^ (hash_bytes(g, sizeof g) * 0x9E3779B97F4A7C15ULL)) | 1;
}

// all game-state RAM: temps, frame counter, stack, OAM buffer, score, coins and
// timer digits excluded
inline uint64_t exact_key(const uint8_t* r) {
    alignas(32) uint8_t m[0x800];
    memcpy(m, r, sizeof m);
    memset(m, 0, 8); m[0x09] = 0;
    memset(m + 0x100, 0, 0x200);
    memset(m + 0x7DD, 0, 6); m[0x7ED] = m[0x7EE] = 0; memset(m + 0x7F8, 0, 3);
    return hash_bytes(m, sizeof m) | 1;
}

// frame, x/8, y/8, camera/64, power-up, area pointer $0750, float state, tiles
inline uint64_t coarse_key(const uint8_t* r) {
    uint64_t k = frame_id(r);
    k = k * 1024 + (uint64_t)(mario_x(r) / 8);
    k = k * 128 + (uint64_t)(mario_y(r) / 8);
    k = k * 1024 + (uint64_t)(camera_x(r) / 64);
    k = k * 4 + (r[0x756] & 3);
    k = k * 256 + r[0x750];
    k = k * 4 + (r[0x1D] & 3);
    uint8_t g[104];
    tile_grid(r, g);
    return mix64(k) ^ hash_bytes(g, sizeof g);
}

}  // namespace ss
