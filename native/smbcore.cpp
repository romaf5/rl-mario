// Minimal ultra-fast NES core specialized for Super Mario Bros (NROM-0).
// Emulates: 6502 (documented ops), frame-level PPU timing (vblank/NMI,
// sprite-0 hit), controller, OAM DMA. No APU, no rendering yet (phase 2).
// Verified by lockstep RAM-differential testing against stable-retro.
//
// Build: g++ -O3 -fPIC -shared -o libsmbcore.so smbcore.cpp
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "cpu6502.h"

namespace {

struct Ppu {
    // registers / latches
    uint8_t ctrl = 0, mask = 0, status = 0, oam_addr = 0;
    uint8_t oam[256] = {0};
    uint8_t vram[0x800] = {0};      // two nametables
    uint8_t palette[32] = {0};
    uint8_t chr[0x2000] = {0};      // CHR ROM copy (reads only)
    uint16_t v = 0, t = 0;          // loopy regs (addr behavior for $2007)
    uint8_t fine_x = 0;
    bool latch = false;             // $2005/$2006 write toggle
    uint8_t read_buf = 0;
    // timing
    int scanline = 261;             // pre-render
    int dot = 0;
    uint64_t frame = 0;
    bool nmi_out = false;
    int pending = 0;                // dots not yet applied (lazy catch-up)
    int next_event = 0;             // dots until nearest scheduled event
    // scroll snapshots for frame rendering (SMB status-bar raster split):
    // (scanline, t, fine_x, ctrl) captured at each $2000/$2005/$2006 write
    struct Scroll { int line; uint16_t t; uint8_t fx, ctrl; };
    Scroll slog[16];
    int slog_n = 0;
};


// sprite-0 hit line: hardware triggers on the sprite's first OPAQUE row
// overlapping background, not its top line. SMB's HUD split depends on
// this being right (a few lines early = the HUD's bottom rows scroll).
inline int s0_line(Ppu& u) {
    uint16_t base = (u.ctrl & 0x08) ? 0x1000 : 0x0000;
    const uint8_t* pat = u.chr + base + u.oam[1] * 16;
    int r0 = 0;
    for (int r = 0; r < 8; r++) {
        int rr = (u.oam[2] & 0x80) ? 7 - r : r;
        if (pat[rr] | pat[rr + 8]) { r0 = r; break; }
    }
    return u.oam[0] + 1 + r0;
}

struct Core {
    Cpu6502 cpu;
    Ppu ppu;
    uint8_t ram[0x800] = {0};
    uint8_t prg[0x8000] = {0};      // 32KB PRG (SMB)
    // controller
    uint8_t pad_state = 0, pad_shift = 0, pad_pending = 0;
    bool pad_strobe = false;
    uint64_t frames_done = 0;
};

// ---------------------------------------------------------------- PPU mem
inline uint16_t nt_mirror(uint16_t addr) {
    // SMB: vertical mirroring (horizontal scroll): $2000/$2800 -> NT0,
    // $2400/$2C00 -> NT1
    addr &= 0x0FFF;
    return (addr & 0x400) ? (0x400 | (addr & 0x3FF)) : (addr & 0x3FF);
}

inline uint8_t ppu_read(Core& c, uint16_t addr) {
    addr &= 0x3FFF;
    if (addr < 0x2000) return c.ppu.chr[addr];
    if (addr < 0x3F00) return c.ppu.vram[nt_mirror(addr)];
    uint16_t p = addr & 0x1F;
    if ((p & 0x13) == 0x10) p &= ~0x10;
    return c.ppu.palette[p];
}

inline void ppu_write(Core& c, uint16_t addr, uint8_t val) {
    addr &= 0x3FFF;
    if (addr < 0x2000) return;                       // CHR ROM
    if (addr < 0x3F00) { c.ppu.vram[nt_mirror(addr)] = val; return; }
    uint16_t p = addr & 0x1F;
    if ((p & 0x13) == 0x10) p &= ~0x10;
    c.ppu.palette[p] = val;
}

// ---------------------------------------------------------------- CPU bus
void ppu_sync(Core& c);

uint8_t read8(Core& c, uint16_t addr) {
    if (addr < 0x2000) return c.ram[addr & 0x7FF];
    if (addr < 0x4000) {
        ppu_sync(c);
        switch (addr & 7) {
            case 2: {   // PPUSTATUS
                uint8_t r = c.ppu.status;
                c.ppu.status &= 0x7F;               // clear vblank
                c.ppu.latch = false;
                return r;
            }
            case 4: return c.ppu.oam[c.ppu.oam_addr];
            case 7: {   // PPUDATA buffered read
                uint8_t r;
                uint16_t a = c.ppu.v & 0x3FFF;
                if (a >= 0x3F00) {
                    r = ppu_read(c, a);
                    c.ppu.read_buf = c.ppu.vram[nt_mirror(a)];
                } else {
                    r = c.ppu.read_buf;
                    c.ppu.read_buf = ppu_read(c, a);
                }
                c.ppu.v += (c.ppu.ctrl & 0x04) ? 32 : 1;
                return r;
            }
            default: return 0;
        }
    }
    if (addr == 0x4016) {
        uint8_t r;
        if (c.pad_strobe) r = c.pad_state & 1;
        else { r = c.pad_shift & 1; c.pad_shift = 0x80 | (c.pad_shift >> 1); }
        return r | 0x40;                             // open bus high bits
    }
    if (addr == 0x4017) return 0x40;                 // pad 2 not connected
    if (addr == 0x4015) return 0;                    // APU status stub
    if (addr >= 0x8000) return c.prg[addr - 0x8000];
    return 0;
}

void oam_dma(Core& c, uint8_t page);

void write8(Core& c, uint16_t addr, uint8_t val) {
    if (addr < 0x2000) { c.ram[addr & 0x7FF] = val; return; }
    if (addr < 0x4000) {
        ppu_sync(c);
        switch (addr & 7) {
            case 0: {
                bool was = c.ppu.ctrl & 0x80;
                c.ppu.ctrl = val;
                // enabling NMI while vblank flag is set fires NMI at once
                if (!was && (val & 0x80) && (c.ppu.status & 0x80))
                    c.cpu.nmi_pending = true;
                c.ppu.t = (uint16_t)((c.ppu.t & 0xF3FF) | ((val & 3) << 10));
                if (c.ppu.slog_n < 16)
                    c.ppu.slog[c.ppu.slog_n++] = {c.ppu.scanline, c.ppu.t,
                                                  c.ppu.fine_x, c.ppu.ctrl};
                return;
            }
            case 1: c.ppu.mask = val; return;
            case 3: c.ppu.oam_addr = val; return;
            case 4: c.ppu.oam[c.ppu.oam_addr++] = val; return;
            case 5:
                if (!c.ppu.latch) {
                    c.ppu.fine_x = val & 7;
                    c.ppu.t = (uint16_t)((c.ppu.t & 0xFFE0) | (val >> 3));
                } else {
                    c.ppu.t = (uint16_t)((c.ppu.t & 0x8C1F)
                              | ((val & 0xF8) << 2) | ((val & 7) << 12));
                }
                c.ppu.latch = !c.ppu.latch;
                if (!c.ppu.latch && c.ppu.slog_n < 16)
                    c.ppu.slog[c.ppu.slog_n++] = {c.ppu.scanline, c.ppu.t,
                                                  c.ppu.fine_x, c.ppu.ctrl};
                return;
            case 6:
                if (!c.ppu.latch)
                    c.ppu.t = (uint16_t)((c.ppu.t & 0x00FF)
                              | ((val & 0x3F) << 8));
                else {
                    c.ppu.t = (uint16_t)((c.ppu.t & 0xFF00) | val);
                    c.ppu.v = c.ppu.t;
                    if (c.ppu.slog_n < 16)
                        c.ppu.slog[c.ppu.slog_n++] = {c.ppu.scanline, c.ppu.t,
                                                      c.ppu.fine_x, c.ppu.ctrl};
                }
                c.ppu.latch = !c.ppu.latch;
                return;
            case 7:
                ppu_write(c, c.ppu.v & 0x3FFF, val);
                c.ppu.v += (c.ppu.ctrl & 0x04) ? 32 : 1;
                return;
            default: return;
        }
    }
    if (addr == 0x4014) { oam_dma(c, val); return; }
    if (addr == 0x4016) {
        bool was = c.pad_strobe;
        c.pad_strobe = val & 1;
        if (was && !c.pad_strobe) c.pad_shift = c.pad_state;
        return;
    }
    // APU $4000-$4017: ignored
}

// ---------------------------------------------------------------- 6502
// addressing helpers (return effective address; xpage = page crossed)
inline uint8_t fetch8(Core& c) {
    uint16_t pc = c.cpu.pc++;
    return (pc >= 0x8000) ? c.prg[pc - 0x8000] : read8(c, pc);
}
inline uint16_t fetch16(Core& c) {
    uint16_t lo = fetch8(c);
    return lo | ((uint16_t)fetch8(c) << 8);
}
inline void push(Core& c, uint8_t v) { c.ram[0x100 + c.cpu.sp--] = v; }
inline uint8_t pop(Core& c) { return c.ram[0x100 + ++c.cpu.sp]; }
inline uint8_t flags_byte(Cpu6502& p, bool brk) {
    return (uint8_t)((p.n << 7) | (p.v << 6) | 0x20 | (brk << 4)
                     | (p.d << 3) | (p.i << 2) | (p.z << 1) | (uint8_t)p.c);
}
inline void set_flags(Cpu6502& p, uint8_t f) {
    p.n = f & 0x80; p.v = f & 0x40; p.d = f & 0x08;
    p.i = f & 0x04; p.z = f & 0x02; p.c = f & 0x01;
}
inline void setnz(Cpu6502& p, uint8_t v) { p.n = v & 0x80; p.z = v == 0; }

int cpu_step(Core& c) {
    Cpu6502& p = c.cpu;
    if (p.nmi_pending) {
        p.nmi_pending = false;
        push(c, (uint8_t)(p.pc >> 8)); push(c, (uint8_t)p.pc);
        push(c, flags_byte(p, false));
        p.i = true;
        p.pc = read8(c, 0xFFFA) | ((uint16_t)read8(c, 0xFFFB) << 8);
        return 7;
    }
    uint8_t op = fetch8(c);
    int cyc = 2;
    bool xp = false;                 // page crossed (adds a cycle on loads)
    uint16_t ea = 0;

    auto zp  = [&]{ ea = fetch8(c); };
    auto zpx = [&]{ ea = (uint8_t)(fetch8(c) + p.x); };
    auto zpy = [&]{ ea = (uint8_t)(fetch8(c) + p.y); };
    auto abs_ = [&]{ ea = fetch16(c); };
    auto abx = [&]{ uint16_t b = fetch16(c); ea = b + p.x;
                    xp = (b & 0xFF00) != (ea & 0xFF00); };
    auto aby = [&]{ uint16_t b = fetch16(c); ea = b + p.y;
                    xp = (b & 0xFF00) != (ea & 0xFF00); };
    auto izx = [&]{ uint8_t z = (uint8_t)(fetch8(c) + p.x);
                    ea = c.ram[z] | ((uint16_t)c.ram[(uint8_t)(z + 1)] << 8); };
    auto izy = [&]{ uint8_t z = fetch8(c);
                    uint16_t b = c.ram[z]
                        | ((uint16_t)c.ram[(uint8_t)(z + 1)] << 8);
                    ea = b + p.y;
                    xp = (b & 0xFF00) != (ea & 0xFF00); };

    auto adc = [&](uint8_t m){
        unsigned s = p.a + m + (unsigned)p.c;
        p.v = (~(p.a ^ m) & (p.a ^ s) & 0x80) != 0;
        p.c = s > 0xFF; p.a = (uint8_t)s; setnz(p, p.a); };
    auto sbc = [&](uint8_t m){ adc((uint8_t)~m); };
    auto cmp = [&](uint8_t r, uint8_t m){
        p.c = r >= m; setnz(p, (uint8_t)(r - m)); };
    auto branch = [&](bool cond){
        int8_t off = (int8_t)fetch8(c);
        if (cond) {
            uint16_t old = p.pc;
            p.pc = (uint16_t)(p.pc + off);
            cyc += ((old & 0xFF00) != (p.pc & 0xFF00)) ? 2 : 1;
        } };
    auto asl = [&](uint8_t v){ p.c = v & 0x80; v <<= 1; setnz(p, v); return v; };
    auto lsr = [&](uint8_t v){ p.c = v & 1; v >>= 1; setnz(p, v); return v; };
    auto rol = [&](uint8_t v){ bool oc = p.c; p.c = v & 0x80;
        v = (uint8_t)((v << 1) | oc); setnz(p, v); return v; };
    auto ror = [&](uint8_t v){ bool oc = p.c; p.c = v & 1;
        v = (uint8_t)((v >> 1) | (oc << 7)); setnz(p, v); return v; };
    auto rmw = [&](uint8_t (*fn)(Cpu6502&, uint8_t), int base){
        (void)fn; (void)base; };
    (void)rmw;

    switch (op) {
    // loads
    case 0xA9: p.a = fetch8(c); setnz(p, p.a); cyc = 2; break;
    case 0xA5: zp();  p.a = read8(c, ea); setnz(p, p.a); cyc = 3; break;
    case 0xB5: zpx(); p.a = read8(c, ea); setnz(p, p.a); cyc = 4; break;
    case 0xAD: abs_(); p.a = read8(c, ea); setnz(p, p.a); cyc = 4; break;
    case 0xBD: abx(); p.a = read8(c, ea); setnz(p, p.a); cyc = 4 + xp; break;
    case 0xB9: aby(); p.a = read8(c, ea); setnz(p, p.a); cyc = 4 + xp; break;
    case 0xA1: izx(); p.a = read8(c, ea); setnz(p, p.a); cyc = 6; break;
    case 0xB1: izy(); p.a = read8(c, ea); setnz(p, p.a); cyc = 5 + xp; break;
    case 0xA2: p.x = fetch8(c); setnz(p, p.x); cyc = 2; break;
    case 0xA6: zp();  p.x = read8(c, ea); setnz(p, p.x); cyc = 3; break;
    case 0xB6: zpy(); p.x = read8(c, ea); setnz(p, p.x); cyc = 4; break;
    case 0xAE: abs_(); p.x = read8(c, ea); setnz(p, p.x); cyc = 4; break;
    case 0xBE: aby(); p.x = read8(c, ea); setnz(p, p.x); cyc = 4 + xp; break;
    case 0xA0: p.y = fetch8(c); setnz(p, p.y); cyc = 2; break;
    case 0xA4: zp();  p.y = read8(c, ea); setnz(p, p.y); cyc = 3; break;
    case 0xB4: zpx(); p.y = read8(c, ea); setnz(p, p.y); cyc = 4; break;
    case 0xAC: abs_(); p.y = read8(c, ea); setnz(p, p.y); cyc = 4; break;
    case 0xBC: abx(); p.y = read8(c, ea); setnz(p, p.y); cyc = 4 + xp; break;
    // stores
    case 0x85: zp();  write8(c, ea, p.a); cyc = 3; break;
    case 0x95: zpx(); write8(c, ea, p.a); cyc = 4; break;
    case 0x8D: abs_(); write8(c, ea, p.a); cyc = 4; break;
    case 0x9D: abx(); write8(c, ea, p.a); cyc = 5; break;
    case 0x99: aby(); write8(c, ea, p.a); cyc = 5; break;
    case 0x81: izx(); write8(c, ea, p.a); cyc = 6; break;
    case 0x91: izy(); write8(c, ea, p.a); cyc = 6; break;
    case 0x86: zp();  write8(c, ea, p.x); cyc = 3; break;
    case 0x96: zpy(); write8(c, ea, p.x); cyc = 4; break;
    case 0x8E: abs_(); write8(c, ea, p.x); cyc = 4; break;
    case 0x84: zp();  write8(c, ea, p.y); cyc = 3; break;
    case 0x94: zpx(); write8(c, ea, p.y); cyc = 4; break;
    case 0x8C: abs_(); write8(c, ea, p.y); cyc = 4; break;
    // transfers
    case 0xAA: p.x = p.a; setnz(p, p.x); break;
    case 0xA8: p.y = p.a; setnz(p, p.y); break;
    case 0x8A: p.a = p.x; setnz(p, p.a); break;
    case 0x98: p.a = p.y; setnz(p, p.a); break;
    case 0xBA: p.x = p.sp; setnz(p, p.x); break;
    case 0x9A: p.sp = p.x; break;
    // stack
    case 0x48: push(c, p.a); cyc = 3; break;
    case 0x68: p.a = pop(c); setnz(p, p.a); cyc = 4; break;
    case 0x08: push(c, flags_byte(p, true)); cyc = 3; break;
    case 0x28: set_flags(p, pop(c)); cyc = 4; break;
    // logic
    case 0x29: p.a &= fetch8(c); setnz(p, p.a); break;
    case 0x25: zp();  p.a &= read8(c, ea); setnz(p, p.a); cyc = 3; break;
    case 0x35: zpx(); p.a &= read8(c, ea); setnz(p, p.a); cyc = 4; break;
    case 0x2D: abs_(); p.a &= read8(c, ea); setnz(p, p.a); cyc = 4; break;
    case 0x3D: abx(); p.a &= read8(c, ea); setnz(p, p.a); cyc = 4 + xp; break;
    case 0x39: aby(); p.a &= read8(c, ea); setnz(p, p.a); cyc = 4 + xp; break;
    case 0x21: izx(); p.a &= read8(c, ea); setnz(p, p.a); cyc = 6; break;
    case 0x31: izy(); p.a &= read8(c, ea); setnz(p, p.a); cyc = 5 + xp; break;
    case 0x09: p.a |= fetch8(c); setnz(p, p.a); break;
    case 0x05: zp();  p.a |= read8(c, ea); setnz(p, p.a); cyc = 3; break;
    case 0x15: zpx(); p.a |= read8(c, ea); setnz(p, p.a); cyc = 4; break;
    case 0x0D: abs_(); p.a |= read8(c, ea); setnz(p, p.a); cyc = 4; break;
    case 0x1D: abx(); p.a |= read8(c, ea); setnz(p, p.a); cyc = 4 + xp; break;
    case 0x19: aby(); p.a |= read8(c, ea); setnz(p, p.a); cyc = 4 + xp; break;
    case 0x01: izx(); p.a |= read8(c, ea); setnz(p, p.a); cyc = 6; break;
    case 0x11: izy(); p.a |= read8(c, ea); setnz(p, p.a); cyc = 5 + xp; break;
    case 0x49: p.a ^= fetch8(c); setnz(p, p.a); break;
    case 0x45: zp();  p.a ^= read8(c, ea); setnz(p, p.a); cyc = 3; break;
    case 0x55: zpx(); p.a ^= read8(c, ea); setnz(p, p.a); cyc = 4; break;
    case 0x4D: abs_(); p.a ^= read8(c, ea); setnz(p, p.a); cyc = 4; break;
    case 0x5D: abx(); p.a ^= read8(c, ea); setnz(p, p.a); cyc = 4 + xp; break;
    case 0x59: aby(); p.a ^= read8(c, ea); setnz(p, p.a); cyc = 4 + xp; break;
    case 0x41: izx(); p.a ^= read8(c, ea); setnz(p, p.a); cyc = 6; break;
    case 0x51: izy(); p.a ^= read8(c, ea); setnz(p, p.a); cyc = 5 + xp; break;
    case 0x24: zp();  { uint8_t m = read8(c, ea);
        p.z = (p.a & m) == 0; p.n = m & 0x80; p.v = m & 0x40; } cyc = 3; break;
    case 0x2C: abs_(); { uint8_t m = read8(c, ea);
        p.z = (p.a & m) == 0; p.n = m & 0x80; p.v = m & 0x40; } cyc = 4; break;
    // arithmetic
    case 0x69: adc(fetch8(c)); break;
    case 0x65: zp();  adc(read8(c, ea)); cyc = 3; break;
    case 0x75: zpx(); adc(read8(c, ea)); cyc = 4; break;
    case 0x6D: abs_(); adc(read8(c, ea)); cyc = 4; break;
    case 0x7D: abx(); adc(read8(c, ea)); cyc = 4 + xp; break;
    case 0x79: aby(); adc(read8(c, ea)); cyc = 4 + xp; break;
    case 0x61: izx(); adc(read8(c, ea)); cyc = 6; break;
    case 0x71: izy(); adc(read8(c, ea)); cyc = 5 + xp; break;
    case 0xE9: sbc(fetch8(c)); break;
    case 0xE5: zp();  sbc(read8(c, ea)); cyc = 3; break;
    case 0xF5: zpx(); sbc(read8(c, ea)); cyc = 4; break;
    case 0xED: abs_(); sbc(read8(c, ea)); cyc = 4; break;
    case 0xFD: abx(); sbc(read8(c, ea)); cyc = 4 + xp; break;
    case 0xF9: aby(); sbc(read8(c, ea)); cyc = 4 + xp; break;
    case 0xE1: izx(); sbc(read8(c, ea)); cyc = 6; break;
    case 0xF1: izy(); sbc(read8(c, ea)); cyc = 5 + xp; break;
    case 0xC9: cmp(p.a, fetch8(c)); break;
    case 0xC5: zp();  cmp(p.a, read8(c, ea)); cyc = 3; break;
    case 0xD5: zpx(); cmp(p.a, read8(c, ea)); cyc = 4; break;
    case 0xCD: abs_(); cmp(p.a, read8(c, ea)); cyc = 4; break;
    case 0xDD: abx(); cmp(p.a, read8(c, ea)); cyc = 4 + xp; break;
    case 0xD9: aby(); cmp(p.a, read8(c, ea)); cyc = 4 + xp; break;
    case 0xC1: izx(); cmp(p.a, read8(c, ea)); cyc = 6; break;
    case 0xD1: izy(); cmp(p.a, read8(c, ea)); cyc = 5 + xp; break;
    case 0xE0: cmp(p.x, fetch8(c)); break;
    case 0xE4: zp();  cmp(p.x, read8(c, ea)); cyc = 3; break;
    case 0xEC: abs_(); cmp(p.x, read8(c, ea)); cyc = 4; break;
    case 0xC0: cmp(p.y, fetch8(c)); break;
    case 0xC4: zp();  cmp(p.y, read8(c, ea)); cyc = 3; break;
    case 0xCC: abs_(); cmp(p.y, read8(c, ea)); cyc = 4; break;
    // inc/dec
    case 0xE6: zp();  { uint8_t v2 = read8(c, ea) + 1; write8(c, ea, v2);
        setnz(p, v2); } cyc = 5; break;
    case 0xF6: zpx(); { uint8_t v2 = read8(c, ea) + 1; write8(c, ea, v2);
        setnz(p, v2); } cyc = 6; break;
    case 0xEE: abs_(); { uint8_t v2 = read8(c, ea) + 1; write8(c, ea, v2);
        setnz(p, v2); } cyc = 6; break;
    case 0xFE: abx(); { uint8_t v2 = read8(c, ea) + 1; write8(c, ea, v2);
        setnz(p, v2); } cyc = 7; break;
    case 0xC6: zp();  { uint8_t v2 = read8(c, ea) - 1; write8(c, ea, v2);
        setnz(p, v2); } cyc = 5; break;
    case 0xD6: zpx(); { uint8_t v2 = read8(c, ea) - 1; write8(c, ea, v2);
        setnz(p, v2); } cyc = 6; break;
    case 0xCE: abs_(); { uint8_t v2 = read8(c, ea) - 1; write8(c, ea, v2);
        setnz(p, v2); } cyc = 6; break;
    case 0xDE: abx(); { uint8_t v2 = read8(c, ea) - 1; write8(c, ea, v2);
        setnz(p, v2); } cyc = 7; break;
    case 0xE8: p.x++; setnz(p, p.x); break;
    case 0xC8: p.y++; setnz(p, p.y); break;
    case 0xCA: p.x--; setnz(p, p.x); break;
    case 0x88: p.y--; setnz(p, p.y); break;
    // shifts
    case 0x0A: p.a = asl(p.a); break;
    case 0x06: zp();  write8(c, ea, asl(read8(c, ea))); cyc = 5; break;
    case 0x16: zpx(); write8(c, ea, asl(read8(c, ea))); cyc = 6; break;
    case 0x0E: abs_(); write8(c, ea, asl(read8(c, ea))); cyc = 6; break;
    case 0x1E: abx(); write8(c, ea, asl(read8(c, ea))); cyc = 7; break;
    case 0x4A: p.a = lsr(p.a); break;
    case 0x46: zp();  write8(c, ea, lsr(read8(c, ea))); cyc = 5; break;
    case 0x56: zpx(); write8(c, ea, lsr(read8(c, ea))); cyc = 6; break;
    case 0x4E: abs_(); write8(c, ea, lsr(read8(c, ea))); cyc = 6; break;
    case 0x5E: abx(); write8(c, ea, lsr(read8(c, ea))); cyc = 7; break;
    case 0x2A: p.a = rol(p.a); break;
    case 0x26: zp();  write8(c, ea, rol(read8(c, ea))); cyc = 5; break;
    case 0x36: zpx(); write8(c, ea, rol(read8(c, ea))); cyc = 6; break;
    case 0x2E: abs_(); write8(c, ea, rol(read8(c, ea))); cyc = 6; break;
    case 0x3E: abx(); write8(c, ea, rol(read8(c, ea))); cyc = 7; break;
    case 0x6A: p.a = ror(p.a); break;
    case 0x66: zp();  write8(c, ea, ror(read8(c, ea))); cyc = 5; break;
    case 0x76: zpx(); write8(c, ea, ror(read8(c, ea))); cyc = 6; break;
    case 0x6E: abs_(); write8(c, ea, ror(read8(c, ea))); cyc = 6; break;
    case 0x7E: abx(); write8(c, ea, ror(read8(c, ea))); cyc = 7; break;
    // jumps
    case 0x4C: p.pc = fetch16(c); cyc = 3; break;
    case 0x6C: { uint16_t a = fetch16(c);        // JMP indirect + page bug
        uint16_t hi = (uint16_t)((a & 0xFF00) | ((a + 1) & 0xFF));
        p.pc = read8(c, a) | ((uint16_t)read8(c, hi) << 8); } cyc = 5; break;
    case 0x20: { uint16_t a = fetch16(c);
        push(c, (uint8_t)((p.pc - 1) >> 8)); push(c, (uint8_t)(p.pc - 1));
        p.pc = a; } cyc = 6; break;
    case 0x60: { uint16_t lo = pop(c);
        p.pc = (uint16_t)((lo | ((uint16_t)pop(c) << 8)) + 1); } cyc = 6; break;
    case 0x40: set_flags(p, pop(c)); { uint16_t lo = pop(c);
        p.pc = lo | ((uint16_t)pop(c) << 8); } cyc = 6; break;
    case 0x00: { p.pc++;                             // BRK
        push(c, (uint8_t)(p.pc >> 8)); push(c, (uint8_t)p.pc);
        push(c, flags_byte(p, true)); p.i = true;
        p.pc = read8(c, 0xFFFE) | ((uint16_t)read8(c, 0xFFFF) << 8); }
        cyc = 7; break;
    // branches
    case 0x10: branch(!p.n); break;
    case 0x30: branch(p.n); break;
    case 0x50: branch(!p.v); break;
    case 0x70: branch(p.v); break;
    case 0x90: branch(!p.c); break;
    case 0xB0: branch(p.c); break;
    case 0xD0: branch(!p.z); break;
    case 0xF0: branch(p.z); break;
    // flags
    case 0x18: p.c = false; break;
    case 0x38: p.c = true; break;
    case 0x58: p.i = false; break;
    case 0x78: p.i = true; break;
    case 0xB8: p.v = false; break;
    case 0xD8: p.d = false; break;
    case 0xF8: p.d = true; break;
    case 0xEA: break;                                // NOP
    default:
        fprintf(stderr, "smbcore: illegal opcode %02X at %04X\n",
                op, (unsigned)(p.pc - 1));
        abort();
    }
    p.cycles += (unsigned)cyc;
    return cyc;
}

// ---------------------------------------------------------------- DMA
void oam_dma(Core& c, uint8_t page) {
    uint16_t base = (uint16_t)(page << 8);
    for (int i = 0; i < 256; i++)
        c.ppu.oam[(uint8_t)(c.ppu.oam_addr + i)] = read8(c, base + i);
    c.cpu.cycles += 513;
}

// ---------------------------------------------------------------- PPU tick
// Advance PPU by CPU-cycle count * 3 dots via event scheduling: only a
// handful of dot-positions per frame matter (vblank set/clear, frame mark,
// sprite-0 hit), so we jump between them instead of stepping every dot.
constexpr int FRAME_DOTS = 262 * 341;
constexpr int VBL_SET   = 241 * 341 + 1;
constexpr int VBL_CLEAR = 261 * 341 + 1;
constexpr int FRAME_MARK = 240 * 341;

void ppu_apply(Core& c, int dots) {
    Ppu& u = c.ppu;
    int idx = u.scanline * 341 + u.dot;
    int remaining = dots;
    while (remaining > 0) {
        int s0 = -1;
        if ((u.mask & 0x18) == 0x18 && !(u.status & 0x40)) {
            int line = s0_line(u);
            if (line < 240) s0 = line * 341 + u.oam[3] + 4;
        }
        int next = FRAME_DOTS;
        int ev[4] = {VBL_SET, VBL_CLEAR, FRAME_MARK, s0};
        for (int k = 0; k < 4; k++) {
            if (ev[k] < 0) continue;
            int d = ev[k] - idx;
            if (d <= 0) d += FRAME_DOTS;
            if (d < next) next = d;
        }
        int step = remaining < next ? remaining : next;
        idx += step;
        if (idx >= FRAME_DOTS) idx -= FRAME_DOTS;
        remaining -= step;
        if (step == next) {
            if (idx == VBL_SET) {
                u.status |= 0x80;
                if (u.ctrl & 0x80) c.cpu.nmi_pending = true;
            } else if (idx == VBL_CLEAR) {
                u.status &= 0x1F;
                // start the new frame's scroll log: writes made during
                // vblank (NMI status-bar setup) become the line-0 baseline;
                // visible-frame entries of the finished frame are dropped.
                // (Resetting at FRAME_MARK erased the log at the exact
                // moment the caller renders -- HUD scrolled with Mario.)
                {
                    int m = 0;
                    for (int i = 0; i < u.slog_n; i++)
                        if (u.slog[i].line >= 240) {
                            u.slog[m] = u.slog[i];
                            u.slog[m].line = 0;
                            m++;
                        }
                    u.slog_n = m;
                }
            } else if (idx == FRAME_MARK) {
                u.frame++;
            }
            if (s0 >= 0 && idx == s0) u.status |= 0x40;
        }
    }
    u.scanline = idx / 341;
    u.dot = idx % 341;
    // recompute distance to the nearest upcoming event
    int s0 = -1;
    if ((u.mask & 0x18) == 0x18 && !(u.status & 0x40)) {
        int line = s0_line(u);
        if (line < 240) s0 = line * 341 + u.oam[3] + 4;
    }
    int next = FRAME_DOTS;
    int ev[4] = {VBL_SET, VBL_CLEAR, FRAME_MARK, s0};
    for (int k = 0; k < 4; k++) {
        if (ev[k] < 0) continue;
        int d = ev[k] - idx;
        if (d <= 0) d += FRAME_DOTS;
        if (d < next) next = d;
    }
    u.next_event = next;
}

void ppu_sync(Core& c) {
    if (c.ppu.pending > 0) {
        int p = c.ppu.pending;
        c.ppu.pending = 0;
        ppu_apply(c, p);
    }
}


// ---------------------------------------------------------------- renderer
// NES 2C02 palette -> grayscale (BT.601 luma of the canonical palette)
static const uint8_t GRAY_LUT[64] = {
     96,  25,  16,  49,  36,  38,  33,  23,  27,  32,  38,  32,  27,   0,   0,   0,
    158,  77,  74, 110,  92, 100,  99,  81,  81,  86, 101,  87,  79,   0,   0,   0,
    239, 148, 141, 168, 154, 166, 172, 160, 152, 148, 164, 155, 150,  60,   0,   0,
    239, 200, 195, 202, 195, 200, 204, 203, 199, 194, 201, 199, 197, 154,   0,   0,
};

inline uint8_t bg_color_gray(Core& c) {
    return GRAY_LUT[c.ppu.palette[0] & 0x3F];
}

// render one full frame to 256x240 through a 64-entry LUT (GRAY_LUT for
// training obs -- bit-identical; identity LUT yields palette indices for
// RGB mapping), then crop to the reference view (240x224)
void render_lut(Core& c, uint8_t* out /*240*224*/, const uint8_t* lut) {
    Ppu& u = c.ppu;
    static thread_local uint8_t fb[256 * 240];
    uint8_t ubg = lut[c.ppu.palette[0] & 0x3F];
    if (!(u.mask & 0x08)) {
        memset(fb, ubg, sizeof(fb));
    } else {
        int si = 0;
        // state at frame start: earliest snapshot from pre-render/NMI writes
        uint16_t st = u.slog_n ? u.slog[0].t : u.t;
        uint8_t sfx = u.slog_n ? u.slog[0].fx : u.fine_x;
        uint8_t sctrl = u.slog_n ? u.slog[0].ctrl : u.ctrl;
        for (int y = 0; y < 240; y++) {
            while (si < u.slog_n && u.slog[si].line <= y) {
                st = u.slog[si].t; sfx = u.slog[si].fx;
                sctrl = u.slog[si].ctrl; si++;
            }
            int coarse_x = st & 0x1F;
            int nt_x = (st >> 10) & 1;
            int scroll_x = ((nt_x * 256) + (coarse_x * 8) + sfx) & 0x1FF;
            int coarse_y = (st >> 5) & 0x1F;
            int fine_y = (st >> 12) & 7;
            int nt_y = (st >> 11) & 1;
            int scroll_y = ((nt_y * 240) + (coarse_y * 8) + fine_y) % 480;
            int sy = (scroll_y + y) % 480;
            int row_nt = (sy >= 240);
            int ry = sy % 240;
            int tile_row = ry >> 3, py = ry & 7;
            uint16_t bg_base = (sctrl & 0x10) ? 0x1000 : 0x0000;
            uint8_t* line = fb + y * 256;
            for (int x = 0; x < 256; ) {
                int sx = (scroll_x + x) & 0x1FF;
                int col_nt = (sx >= 256);
                int rx = sx & 0xFF;
                int tile_col = rx >> 3, px = rx & 7;
                int nt = (col_nt ^ row_nt) ? 0 : 0;  // placeholder
                // nametable index: horizontal from col_nt, vertical mirroring
                nt = col_nt;
                uint16_t nt_addr = (uint16_t)(nt * 0x400
                                   + tile_row * 32 + tile_col);
                uint8_t tid = u.vram[nt_addr];
                uint8_t at = u.vram[nt * 0x400 + 0x3C0
                                    + (tile_row >> 2) * 8 + (tile_col >> 2)];
                int shift = ((tile_row & 2) << 1) | (tile_col & 2);
                uint8_t pal = (at >> shift) & 3;
                const uint8_t* pat = u.chr + bg_base + tid * 16 + py;
                uint8_t lo = pat[0], hi = pat[8];
                int n = 8 - px;
                if (n > 256 - x) n = 256 - x;
                for (int k = 0; k < n; k++) {
                    int bit = 7 - (px + k);
                    uint8_t ci = (uint8_t)(((lo >> bit) & 1)
                                 | (((hi >> bit) & 1) << 1));
                    line[x + k] = ci ? lut[u.palette[pal * 4 + ci] & 0x3F]
                                     : ubg;
                }
                x += n;
            }
        }
    }
    // sprites (8x8; all 64, no 8-per-line limit; SMB never uses 8x16)
    if (u.mask & 0x10) {
        uint16_t sp_base = (u.ctrl & 0x08) ? 0x1000 : 0x0000;
        for (int s = 63; s >= 0; s--) {
            uint8_t sy = u.oam[s * 4 + 0];
            if (sy >= 0xEF) continue;
            uint8_t tid = u.oam[s * 4 + 1];
            uint8_t attr = u.oam[s * 4 + 2];
            uint8_t sx = u.oam[s * 4 + 3];
            uint8_t pal = 4 + (attr & 3);
            bool behind = attr & 0x20, fh = attr & 0x40, fv = attr & 0x80;
            for (int row = 0; row < 8; row++) {
                int y = sy + 1 + row;
                if (y >= 240) break;
                int pr = fv ? 7 - row : row;
                const uint8_t* pat = u.chr + sp_base + tid * 16 + pr;
                uint8_t lo = pat[0], hi = pat[8];
                uint8_t* line = fb + y * 256;
                for (int k = 0; k < 8; k++) {
                    int x = sx + k;
                    if (x >= 256) break;
                    int bit = fh ? k : 7 - k;
                    uint8_t ci = (uint8_t)(((lo >> bit) & 1)
                                 | (((hi >> bit) & 1) << 1));
                    if (!ci) continue;
                    if (behind && line[x] != ubg) continue;
                    line[x] = lut[u.palette[pal * 4 + ci] & 0x3F];
                }
            }
        }
    }
    // crop 256x240 -> 240x224 (8px off left/right, 8 off top/bottom)
    for (int y = 0; y < 224; y++)
        memcpy(out + y * 240, fb + (y + 8) * 256 + 8, 240);
}

static const uint8_t IDX_LUT[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
};
void render_gray(Core& c, uint8_t* out) { render_lut(c, out, GRAY_LUT); }
void render_idx(Core& c, uint8_t* out) { render_lut(c, out, IDX_LUT); }

}  // namespace

// ---------------------------------------------------------------- C API
extern "C" {

Core* smb_create(const uint8_t* rom, int rom_len) {
    if (rom_len < 16 + 0x8000 + 0x2000) return nullptr;
    if (memcmp(rom, "NES\x1a", 4) != 0) return nullptr;
    int prg_banks = rom[4];
    Core* c = new Core();
    memset(c->ram, 0xFF, sizeof(c->ram));   // NES power-on pattern (matches
                                            // the reference core; SMB never
                                            // writes parts of the stack page)
    const uint8_t* prg = rom + 16;
    if (prg_banks == 2) memcpy(c->prg, prg, 0x8000);
    else { memcpy(c->prg, prg, 0x4000); memcpy(c->prg + 0x4000, prg, 0x4000); }
    memcpy(c->ppu.chr, prg + prg_banks * 0x4000, 0x2000);
    // reset vector
    c->cpu.pc = c->prg[0x7FFC] | ((uint16_t)c->prg[0x7FFD] << 8);
    return c;
}

void smb_destroy(Core* c) { delete c; }

// run exactly one frame (until scanline wraps past 260->261 end)
void smb_frame(Core* c, uint8_t buttons) {
    c->pad_state = buttons;
    if (c->pad_strobe) c->pad_shift = buttons;
    uint64_t target = c->ppu.frame + 1;
    while (c->ppu.frame < target) {
        if (c->ppu.pending >= c->ppu.next_event) ppu_sync(*c);
        int cyc = cpu_step(*c);
        c->ppu.pending += cyc * 3;
    }
    c->frames_done++;
}

uint8_t* smb_ram(Core* c) { return c->ram; }

// render current frame to 240x224 grayscale (caller buffer)
void smb_render(Core* c, uint8_t* out) { render_gray(*c, out); }
uint8_t* smb_oam(Core* c) { return c->ppu.oam; }
uint8_t* smb_vram(Core* c) { return c->ppu.vram; }

int smb_state_size(void) { return (int)sizeof(Core); }
void smb_save(Core* c, uint8_t* out) { memcpy(out, c, sizeof(Core)); }
void smb_load(Core* c, const uint8_t* in) {
    Core tmp;
    memcpy(&tmp, in, sizeof(Core));
    memcpy(tmp.prg, c->prg, sizeof(tmp.prg));    // ROM stays
    memcpy(tmp.ppu.chr, c->ppu.chr, sizeof(tmp.ppu.chr));
    memcpy(c, &tmp, sizeof(Core));
}

void smb_set_ram(Core* c, const uint8_t* ram) { memcpy(c->ram, ram, 0x800); }

}  // extern "C"
