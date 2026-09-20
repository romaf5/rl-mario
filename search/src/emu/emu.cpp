// The only translation unit that includes the native core (single-TU core:
// its extern "C" symbols exist once; LTO inlines across the Emu boundary).
#include "emu.h"
#include <cstring>
#include "obs.h"
#include "../../../native/smbcore.cpp"

namespace ss {

const uint8_t kActionButtons[kNumActions] = {
    0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40, 0x41, 0x42, 0x43, 0x20, 0x10};

namespace {
// compact state = Core minus its CHR (inside Ppu) and PRG ROM copies
const size_t kChrOff = offsetof(Core, ppu) + offsetof(Ppu, chr);
const size_t kChrLen = sizeof(Ppu::chr);
const size_t kPrgOff = offsetof(Core, prg);
const size_t kPrgLen = sizeof(Core::prg);
const size_t kSegA = kChrOff;
const size_t kSegB = kPrgOff - (kChrOff + kChrLen);
const size_t kSegC = sizeof(Core) - (kPrgOff + kPrgLen);
static_assert(offsetof(Core, ppu) + offsetof(Ppu, chr) + sizeof(Ppu::chr) <= offsetof(Core, prg),
              "compact layout: CHR before PRG");
}  // namespace

size_t full_state_size() { return sizeof(Core); }
size_t compact_state_size() { return kSegA + kSegB + kSegC; }

Emu::Emu(const uint8_t* rom, int rom_len) : core_(smb_create(rom, rom_len)) {}
Emu::~Emu() { if (core_) smb_destroy(static_cast<Core*>(core_)); }
void Emu::load_full(const uint8_t* full) { smb_load(static_cast<Core*>(core_), full); }
void Emu::save_full(uint8_t* full) const { smb_save(static_cast<Core*>(core_), full); }

void Emu::load(const uint8_t* s) {
    uint8_t* c = static_cast<uint8_t*>(core_);
    memcpy(c, s, kSegA);
    memcpy(c + kChrOff + kChrLen, s + kSegA, kSegB);
    memcpy(c + kPrgOff + kPrgLen, s + kSegA + kSegB, kSegC);
}

void Emu::save(uint8_t* s) const {
    const uint8_t* c = static_cast<const uint8_t*>(core_);
    memcpy(s, c, kSegA);
    memcpy(s + kSegA, c + kChrOff + kChrLen, kSegB);
    memcpy(s + kSegA + kSegB, c + kPrgOff + kPrgLen, kSegC);
}

void Emu::frame(uint8_t buttons) { smb_frame(static_cast<Core*>(core_), buttons); }

void Emu::step(int action) {
    Core* c = static_cast<Core*>(core_);
    const uint8_t b = kActionButtons[action];
    for (int k = 0; k < kFrameSkip; k++) smb_frame(c, b);
}

namespace {
const ObsResizer kResize;
}

void Emu::step_obs(int action, uint8_t* obs84) {
    Core* c = static_cast<Core*>(core_);
    const uint8_t b = kActionButtons[action];
    thread_local uint8_t fa[kScrW * kScrH], fb[kScrW * kScrH];
    for (int k = 0; k < kFrameSkip; k++) {
        smb_frame(c, b);
        if (k == kFrameSkip - 2) render_gray(*c, fa);
        if (k == kFrameSkip - 1) render_gray(*c, fb);
    }
    for (int p = 0; p < kScrW * kScrH; p++) fa[p] = fa[p] > fb[p] ? fa[p] : fb[p];
    kResize(fa, obs84);
}

void Emu::obs_now(uint8_t* obs84) const {
    thread_local uint8_t f[kScrW * kScrH];
    render_gray(*static_cast<Core*>(core_), f);
    kResize(f, obs84);
}

const uint8_t* Emu::ram() const { return static_cast<const Core*>(core_)->ram; }
bool Emu::jammed() const { return static_cast<const Core*>(core_)->cpu.jammed; }

}  // namespace ss
