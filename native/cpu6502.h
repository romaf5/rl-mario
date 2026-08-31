// Minimal 6502 interpreter (documented opcodes; SMB uses no illegal ops).
// The bus is provided by the including translation unit via read8/write8.
#pragma once
#include <cstdint>

struct Cpu6502 {
    uint8_t a = 0, x = 0, y = 0, sp = 0xFD;
    uint16_t pc = 0;
    // status flags kept unpacked for speed
    bool c = false, z = false, i = true, d = false, v = false, n = false;
    uint64_t cycles = 0;
    bool nmi_pending = false;
};
