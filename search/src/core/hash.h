// 64-bit hashing of byte ranges: AVX2 multiply-accumulate over 32-byte blocks
// (xxh3-style lanes), 8-byte scalar tail. Used for state keys (2 KB RAM, tile grids).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ss {

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33);
}

inline uint64_t hash_bytes(const uint8_t* p, size_t n, uint64_t seed = 0) {
    uint64_t h = seed ^ (n * 0x9E3779B97F4A7C15ULL);
    size_t i = 0;
#if defined(__AVX2__)
    if (n >= 32) {
        __m256i acc = _mm256_set1_epi64x((long long)h);
        const __m256i k = _mm256_set_epi64x(0x165667B19E3779F9LL, 0x27D4EB2F165667C5LL,
                                            (long long)0x85EBCA77C2B2AE63ULL, (long long)0x9E3779B185EBCA87ULL);
        for (; i + 32 <= n; i += 32) {
            const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            const __m256i dk = _mm256_xor_si256(d, k);
            const __m256i prod = _mm256_mul_epu32(dk, _mm256_srli_epi64(dk, 32));
            acc = _mm256_add_epi64(acc, _mm256_add_epi64(prod, d));
            acc = _mm256_xor_si256(acc, _mm256_srli_epi64(acc, 29));
        }
        alignas(32) uint64_t lane[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lane), acc);
        h = mix64(lane[0] ^ mix64(lane[1] ^ mix64(lane[2] ^ mix64(lane[3]))));
    }
#endif
    for (; i + 8 <= n; i += 8) { uint64_t v; memcpy(&v, p + i, 8); h = mix64(h ^ v); }
    if (i < n) { uint64_t v = 0; memcpy(&v, p + i, n - i); h = mix64(h ^ v ^ 0xA5); }
    return mix64(h);
}

}  // namespace ss
