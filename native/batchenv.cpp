// Batched threadpool wrapper around smbcore: N envs, T worker threads.
// Per step and per env: 4 emulated frames (with the SMB RAM hacks ported
// from RetroMarioEnv._did_step), renders of the last 2 frames, max-pool,
// area-resize to 84x84, plus a 2KB RAM snapshot for Python-side semantics.
//
// Build: g++ -O3 -march=native -fPIC -shared -o libbatchenv.so batchenv.cpp -lpthread
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

// pull in the whole core (single translation unit build)
#include "smbcore.cpp"

namespace {

constexpr int W = 240, H = 224, OW = 84, OH = 84;
// the status bar (top 24 view rows) is cropped out of the agent's obs:
// score/timer digits are UI, not world -- the policy shouldn't read them
constexpr int HUD = 24, HP = H - HUD;

// separable box-filter resize 240x224 -> 84x84, fixed-point weights
struct Resizer {
    // per output col/row: source start index, tap count, weights (<=4 taps)
    int cs[OW], cn[OW]; uint16_t cw[OW][4];
    int rs[OH], rn[OH]; uint16_t rw[OH][4];
    Resizer() {
        build(W, OW, cs, cn, (uint16_t*)cw);
        build(HP, OH, rs, rn, (uint16_t*)rw);
    }
    static void build(int in, int out, int* st, int* n, uint16_t* wts) {
        double scale = (double)in / out;
        for (int o = 0; o < out; o++) {
            double a = o * scale, b = a + scale;
            int ia = (int)a, ib = (int)(b - 1e-9);
            if (ib >= in) ib = in - 1;
            int cnt = ib - ia + 1;
            if (cnt > 4) cnt = 4;
            st[o] = ia; n[o] = cnt;
            double tot = 0;
            double w[4];
            for (int k = 0; k < cnt; k++) {
                double lo = ia + k, hi = lo + 1;
                if (lo < a) lo = a;
                if (hi > b) hi = b;
                w[k] = hi - lo;
                tot += w[k];
            }
            for (int k = 0; k < cnt; k++)
                wts[o * 4 + k] = (uint16_t)(w[k] / tot * 4096 + 0.5);
        }
    }
};
const Resizer RZ;

void resize_area(const uint8_t* in, uint8_t* out) {
    static thread_local uint16_t tmp[OW * HP];
    in += HUD * W;                      // skip the status bar
    for (int y = 0; y < HP; y++) {
        const uint8_t* row = in + y * W;
        uint16_t* trow = tmp + y * OW;
        for (int o = 0; o < OW; o++) {
            int s = RZ.cs[o];
            uint32_t acc = 0;
            for (int k = 0; k < RZ.cn[o]; k++)
                acc += (uint32_t)row[s + k] * RZ.cw[o][k];
            trow[o] = (uint16_t)(acc >> 6);      // keep 6 extra bits
        }
    }
    for (int o = 0; o < OH; o++) {
        int s = RZ.rs[o];
        for (int x = 0; x < OW; x++) {
            uint32_t acc = 0;
            for (int k = 0; k < RZ.rn[o]; k++)
                acc += (uint32_t)tmp[(s + k) * OW + x] * RZ.rw[o][k];
            out[o * OW + x] = (uint8_t)(acc >> 18);   // 12 + 6 fractional
        }
    }
}

// ---- SMB RAM-hack helpers (port of RetroMarioEnv._did_step) -------------
inline bool is_dying(Core& c) {
    return c.ram[0x0E] == 0x0B || c.ram[0xB5] > 1;
}
inline bool is_dead(Core& c) { return c.ram[0x0E] == 0x06; }
inline bool is_busy(Core& c) { return c.ram[0x0E] <= 0x05 || c.ram[0x0E] == 0x07; }
inline bool is_world_over(Core& c) { return c.ram[0x770] == 2; }
inline int game_time(Core& c) {
    return c.ram[0x7F8] * 100 + c.ram[0x7F9] * 10 + c.ram[0x7FA];
}

// returns true if a skip loop exhausted its budget without reaching
// normal gameplay -- the game is in a terminal screen (e.g. the ending
// after the final axe) and further per-step skipping would burn 5000
// frames forever; the caller latches post_game and stops hacking.
bool frame_hacks(Core& c, bool single_stage, uint8_t* transit = nullptr) {
    // kill mario during the dying animation
    if (is_dying(c)) {
        c.ram[0x0E] = 0x06;
        smb_frame(&c, 0);
    }
    // scripted transition (pipe/vine/entrance engine states, $0E 0-5,7):
    // reported to the caller so a same-area backward x jump can be told
    // apart from a maze loop teleport, which never leaves player control
    if (transit && is_busy(c)) *transit = 1;
    if (!single_stage && is_world_over(c)) {   // skip end-of-world cutscene
        int t = game_time(c);
        for (int i = 0; i < 2000 && game_time(c) == t; i++)
            smb_frame(&c, 0);
        if (game_time(c) == t && is_world_over(c)) return true;
    }
    uint8_t timer = c.ram[0x6DE];              // skip area-change animation
    if (timer > 1 && timer < 255) c.ram[0x6DE] = 1;
    if (is_busy(c) || is_world_over(c)) {      // skip black inter-life screens
        // a flag sequence needs ~1000 busy frames; 600 latched post_game
        // after every stage clear and disabled all hacks for the rest of
        // the game. Latch only on a persistent world-over (the ending).
        for (int i = 0; i < 2000 && (is_busy(c) || is_world_over(c)); i++) {
            c.ram[0x7A0] = 0;
            smb_frame(&c, 0);
        }
        if (is_world_over(c)) return true;
    }
    return false;
}

struct BatchEnv {
    int n = 0;
    int skip = 4;
    bool single_stage = false;
    std::vector<Core*> cores;
    std::vector<uint8_t> post_game;   // per-env: terminal screen latched
    std::vector<uint8_t> transit;     // per-env: scripted transition this step
    // threadpool
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv_go, cv_done;
    uint64_t gen = 0;
    std::atomic<int> next_env{0};
    std::atomic<int> done_count{0};
    bool quit = false;
    // step io
    const int32_t* actions = nullptr;
    uint8_t* obs = nullptr;      // N*84*84
    uint8_t* ram_out = nullptr;  // N*2048

    void worker() {
        uint64_t my_gen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_go.wait(lk, [&]{ return gen != my_gen || quit; });
                if (quit) return;
                my_gen = gen;
            }
            for (;;) {
                int i = next_env.fetch_add(1);
                if (i >= n) break;
                step_env(i);
            }
            if (done_count.fetch_add(1) + 1 == (int)threads.size()) {
                std::lock_guard<std::mutex> lk(mu);
                cv_done.notify_one();
            }
        }
    }

    void step_env(int i) {
        Core* c = cores[i];
        uint8_t b = (uint8_t)actions[i];
        static thread_local uint8_t fa[W * H], fb2[W * H], mx[W * H];
        transit[i] = 0;
        for (int k = 0; k < skip; k++) {
            smb_frame(c, b);
            if (!post_game[i] && frame_hacks(*c, single_stage, &transit[i]))
                post_game[i] = 1;
            if (k == skip - 2) render_gray(*c, fa);
            if (k == skip - 1) render_gray(*c, fb2);
        }
        for (int p = 0; p < W * H; p++)
            mx[p] = fa[p] > fb2[p] ? fa[p] : fb2[p];
        resize_area(mx, obs + (size_t)i * OW * OH);
        memcpy(ram_out + (size_t)i * 0x800, c->ram, 0x800);
    }
};

}  // namespace

extern "C" {

BatchEnv* benv_create(const uint8_t* rom, int rom_len, int n, int n_threads,
                      int single_stage) {
    BatchEnv* e = new BatchEnv();
    e->n = n;
    e->single_stage = single_stage;
    e->post_game.assign(n, 0);
    e->transit.assign(n, 0);
    for (int i = 0; i < n; i++) e->cores.push_back(smb_create(rom, rom_len));
    for (int t = 0; t < n_threads; t++)
        e->threads.emplace_back([e]{ e->worker(); });
    return e;
}

void benv_destroy(BatchEnv* e) {
    {
        std::lock_guard<std::mutex> lk(e->mu);
        e->quit = true;
        e->cv_go.notify_all();
    }
    for (auto& t : e->threads) t.join();
    for (auto* c : e->cores) smb_destroy(c);
    delete e;
}

void benv_step(BatchEnv* e, const int32_t* actions, uint8_t* obs,
               uint8_t* ram_out) {
    e->actions = actions;
    e->obs = obs;
    e->ram_out = ram_out;
    e->next_env.store(0);
    e->done_count.store(0);
    {
        std::lock_guard<std::mutex> lk(e->mu);
        e->gen++;
        e->cv_go.notify_all();
    }
    std::unique_lock<std::mutex> lk(e->mu);
    e->cv_done.wait(lk, [&]{
        return e->done_count.load() == (int)e->threads.size(); });
}

int benv_state_size(void) { return smb_state_size(); }
void benv_set_skip(BatchEnv* e, int skip) { e->skip = skip < 2 ? 2 : skip; }
// per-env flag: a scripted transition (pipe/vine/entrance) ran inside the
// last benv_step -- distinguishes legit x-frame changes from loop teleports
void benv_transit(BatchEnv* e, uint8_t* out) {
    memcpy(out, e->transit.data(), e->n);
}

// standard NES palette (2C02), RGB triplets per color index
static const uint8_t NES_PAL[64][3] = {
    {84,84,84},{0,30,116},{8,16,144},{48,0,136},{68,0,100},{92,0,48},{84,4,0},{60,24,0},
    {32,42,0},{8,58,0},{0,64,0},{0,60,0},{0,50,60},{0,0,0},{0,0,0},{0,0,0},
    {152,150,152},{8,76,196},{48,50,236},{92,30,228},{136,20,176},{160,20,100},{152,34,32},{120,60,0},
    {84,90,0},{40,114,0},{8,124,0},{0,118,40},{0,102,120},{0,0,0},{0,0,0},{0,0,0},
    {236,238,236},{76,154,236},{120,124,236},{176,98,236},{228,84,236},{236,88,180},{236,106,100},{212,136,32},
    {160,170,0},{116,196,0},{76,208,32},{56,204,108},{56,180,204},{60,60,60},{0,0,0},{0,0,0},
    {236,238,236},{168,204,236},{188,188,236},{212,178,236},{236,174,236},{236,174,212},{236,180,176},{228,196,144},
    {204,210,120},{180,222,120},{168,226,144},{152,226,180},{160,214,228},{160,162,160},{0,0,0},{0,0,0},
};

// adopt reference-emulator RAM (retro->native seed for lockstep eval:
// both emulators then replay identically, bitwise -- see deep_difftest)
void benv_set_ram(BatchEnv* e, int i, const uint8_t* ram) {
    smb_set_ram(e->cores[i], ram);
    e->post_game[i] = 0;
}

// hack-free single-env step: pure emulated frames so a reference
// emulator fed the same actions stays in bitwise lockstep (video replay)
void benv_step_raw(BatchEnv* e, int i, int action, uint8_t* obs,
                   uint8_t* ram_out) {
    Core* c = e->cores[i];
    static thread_local uint8_t fa[W * H], fb2[W * H], mx[W * H];
    for (int k = 0; k < e->skip; k++) {
        smb_frame(c, (uint8_t)action);
        if (k == e->skip - 2) render_gray(*c, fa);
        if (k == e->skip - 1) render_gray(*c, fb2);
    }
    for (int p = 0; p < W * H; p++)
        mx[p] = fa[p] > fb2[p] ? fa[p] : fb2[p];
    resize_area(mx, obs + (size_t)i * OW * OH);
    memcpy(ram_out + (size_t)i * 0x800, c->ram, 0x800);
}

// single-env step that also captures all 4 emulated frames as RGB --
// eval video recording without temporal aliasing (blinking sprites
// strobe when only 1 of 4 frames is sampled). Semantics identical to
// step_env; obs/ram written the same way.
void benv_step_rgb4(BatchEnv* e, int i, int action, uint8_t* obs,
                    uint8_t* ram_out, uint8_t* rgb4 /*4*224*240*3*/) {
    Core* c = e->cores[i];
    static thread_local uint8_t fa[W * H], fb2[W * H], mx[W * H],
        idx[W * H];
    for (int k = 0; k < e->skip; k++) {
        smb_frame(c, (uint8_t)action);
        if (!e->post_game[i] && frame_hacks(*c, e->single_stage))
            e->post_game[i] = 1;
        render_idx(*c, idx);
        uint8_t* out = rgb4 + (size_t)k * W * H * 3;
        for (int p = 0; p < W * H; p++) {
            const uint8_t* cc = NES_PAL[idx[p] & 0x3F];
            out[p * 3 + 0] = cc[0];
            out[p * 3 + 1] = cc[1];
            out[p * 3 + 2] = cc[2];
        }
        // gray obs frames derive from the same palette indices
        if (k == e->skip - 2) for (int p = 0; p < W * H; p++) fa[p] = GRAY_LUT[idx[p] & 0x3F];
        if (k == e->skip - 1) for (int p = 0; p < W * H; p++) fb2[p] = GRAY_LUT[idx[p] & 0x3F];
    }
    for (int p = 0; p < W * H; p++)
        mx[p] = fa[p] > fb2[p] ? fa[p] : fb2[p];
    resize_area(mx, obs + (size_t)i * OW * OH);
    memcpy(ram_out + (size_t)i * 0x800, c->ram, 0x800);
}

void benv_render_rgb(BatchEnv* e, int i, uint8_t* out /*224*240*3*/) {
    static thread_local uint8_t idx[240 * 224];
    render_idx(*e->cores[i], idx);
    for (int p = 0; p < 240 * 224; p++) {
        const uint8_t* c = NES_PAL[idx[p] & 0x3F];
        out[p * 3 + 0] = c[0];
        out[p * 3 + 1] = c[1];
        out[p * 3 + 2] = c[2];
    }
}
void benv_save(BatchEnv* e, int i, uint8_t* out) { smb_save(e->cores[i], out); }
void benv_load(BatchEnv* e, int i, const uint8_t* in) {
    smb_load(e->cores[i], in);
    e->post_game[i] = 0;
}
void benv_frames(BatchEnv* e, int i, int nframes, int buttons) {
    for (int k = 0; k < nframes; k++)
        smb_frame(e->cores[i], (uint8_t)buttons);
}
void benv_render(BatchEnv* e, int i, uint8_t* out240x224) {
    render_gray(*e->cores[i], out240x224);
}

// render current state to 84x84 + RAM snapshot WITHOUT stepping any frames
// (used after per-env resets)
void benv_obs(BatchEnv* e, int i, uint8_t* out84, uint8_t* ram_out) {
    static uint8_t fb[W * H];
    render_gray(*e->cores[i], fb);
    resize_area(fb, out84);
    memcpy(ram_out, e->cores[i]->ram, 0x800);
}
uint8_t* benv_ram(BatchEnv* e, int i) { return e->cores[i]->ram; }

}  // extern "C"
