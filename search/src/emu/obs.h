// The network's frame: 240x224 grayscale -> status bar (top 24 rows) cropped ->
// 84x84 area resize, fixed point. Same pixels as native/batchenv.cpp's resize_area.
#pragma once
#include <cstdint>

namespace ss {

constexpr int kScrW = 240, kScrH = 224, kHud = 24, kObsW = 84, kObsH = 84;
constexpr int kObsSize = kObsW * kObsH;

struct ObsResizer {
    int cs[kObsW], cn[kObsW]; uint16_t cw[kObsW][4];
    int rs[kObsH], rn[kObsH]; uint16_t rw[kObsH][4];
    ObsResizer() {
        build(kScrW, kObsW, cs, cn, &cw[0][0]);
        build(kScrH - kHud, kObsH, rs, rn, &rw[0][0]);
    }
    static void build(int in, int out, int* st, int* n, uint16_t* wts) {
        const double scale = (double)in / out;
        for (int o = 0; o < out; o++) {
            const double a = o * scale, b = a + scale;
            const int ia = (int)a;
            int ib = (int)(b - 1e-9);
            if (ib >= in) ib = in - 1;
            int cnt = ib - ia + 1;
            if (cnt > 4) cnt = 4;
            st[o] = ia; n[o] = cnt;
            double w[4], tot = 0;
            for (int k = 0; k < cnt; k++) {
                double lo = ia + k, hi = lo + 1;
                if (lo < a) lo = a;
                if (hi > b) hi = b;
                w[k] = hi - lo; tot += w[k];
            }
            for (int k = 0; k < cnt; k++) wts[o * 4 + k] = (uint16_t)(w[k] / tot * 4096 + 0.5);
        }
    }
    // in: 240x224 grayscale; out: 84x84
    void operator()(const uint8_t* in, uint8_t* out) const {
        uint16_t tmp[kObsW * (kScrH - kHud)];
        in += kHud * kScrW;
        for (int y = 0; y < kScrH - kHud; y++) {
            const uint8_t* row = in + y * kScrW;
            for (int o = 0; o < kObsW; o++) {
                uint32_t acc = 0;
                for (int k = 0; k < cn[o]; k++) acc += (uint32_t)row[cs[o] + k] * cw[o][k];
                tmp[y * kObsW + o] = (uint16_t)(acc >> 6);
            }
        }
        for (int o = 0; o < kObsH; o++)
            for (int x = 0; x < kObsW; x++) {
                uint32_t acc = 0;
                for (int k = 0; k < rn[o]; k++) acc += (uint32_t)tmp[(rs[o] + k) * kObsW + x] * rw[o][k];
                out[o * kObsW + x] = (uint8_t)(acc >> 18);
            }
    }
};

}  // namespace ss
