// Minimal 6502 interpreter (documented opcodes; SMB uses no illegal ops --
// one jams the CPU, see cpu_step).
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
    // halted on an undocumented opcode (see cpu_step). Occupies what was
    // padding after nmi_pending: sizeof(Cpu6502) and so the savestate
    // layout (sizeof(Core)) are unchanged; smb_load clears it.
    bool jammed = false;
};
