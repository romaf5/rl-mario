// Emulator access for the search: one Emu per worker thread over the native SMB
// core (the real game: no training hacks), compact states and RAM decoding.
#pragma once
#include <cstddef>
#include <cstdint>

namespace ss {

constexpr int kFrameSkip = 4;       // frames per decision (the policy's step)
constexpr int kNumActions = 12;     // COMPLEX_MOVEMENT
extern const uint8_t kActionButtons[kNumActions];

size_t full_state_size();           // the native savestate: sizeof(Core)
size_t compact_state_size();        // what changes: sizeof(Core) - PRG - CHR copies

class Emu {
public:
    Emu(const uint8_t* rom, int rom_len);
    ~Emu();
    Emu(const Emu&) = delete;
    Emu& operator=(const Emu&) = delete;
    bool ok() const { return core_ != nullptr; }
    void load_full(const uint8_t* full);
    void save_full(uint8_t* full) const;
    void load(const uint8_t* compact);
    void save(uint8_t* compact) const;
    void frame(uint8_t buttons);
    void step(int action);          // kFrameSkip frames holding the action
    // step + its 84x84 frame: max of the step's last two frames (the net's input)
    void step_obs(int action, uint8_t* obs84);
    void obs_now(uint8_t* obs84) const;   // the current frame only
    const uint8_t* ram() const;
    bool jammed() const;
private:
    void* core_;
};

// ---- SMB RAM decode ----
inline int level_gp(const uint8_t* r) { return r[0x75F] > 7 ? -1 : r[0x75F] * 4 + r[0x75C]; }
inline int mario_x(const uint8_t* r) { return r[0x6D] * 256 + r[0x86]; }
inline int mario_y(const uint8_t* r) { return r[0xB5] * 256 + r[0xCE]; }
inline int camera_x(const uint8_t* r) { return r[0x71A] * 256 + r[0x71C]; }
inline int lives(const uint8_t* r) { return r[0x75A]; }
inline int game_timer(const uint8_t* r) { return r[0x7F8] * 100 + r[0x7F9] * 10 + r[0x7FA]; }
inline bool in_control(const uint8_t* r) { return r[0x0E] == 0x08 && r[0x770] == 1; }
// dying animation / dead, or fallen below the screen while in control
inline bool dying(const uint8_t* r) {
    return r[0x0E] == 0x0B || r[0x0E] == 0x06 || (in_control(r) && r[0xB5] > 1);
}
// the coordinate system x lives in: level, area, sub-area, AreaType, swimming
inline uint32_t frame_id(const uint8_t* r) {
    int g = level_gp(r);
    if (g < 0) g = 63;
    return (uint32_t)((((g * 256 + r[0x760]) * 256 + r[0x74F]) * 8 + (r[0x74E] & 7)) * 2 + (r[0x704] & 1));
}

}  // namespace ss
