// AlphaZero-style MCTS on the real game, for a net that sees 84x84 frames and
// predicts a prior over the 12 actions and the frames left to the segment goal.
//
// Children are created lazily: a simulation descends by PUCT to an edge with no
// child, emulates that one step (the parent's compact state + action) and renders
// its frame; the net then scores the new node (prior, value), batched over every
// leaf of a wave across all trees of the forest. The game is deterministic, so the
// backup is min: b(leaf) = the net's frames to go (goal 0, dead v_death),
// b(n) = 4 + min over created children. q(c) = clamp(1 - (b(c) - best sibling) / scale).
#pragma once
#include <cstdint>
#include <vector>
#include "../core/pool.h"
#include "../emu/emu.h"
#include "../emu/obs.h"
#include "../route/route.h"

namespace ss {

struct MctsParams {
    float c_puct = 1.5f;
    float fpu = 0.5f;              // q of an unvisited child
    float scale = 32.0f;           // frames: a child this much slower than its best sibling has q = 0
    float v_death = 4096.0f;       // frames to go of a dead end (the top of the net's value range)
    int max_nodes = 1 << 16;       // per tree
};

enum : uint8_t { kRunning = 0, kGoal = 1, kDead = 2 };
constexpr int kStack = 4;          // frames per net input

struct MctsNode {
    int32_t parent;
    int32_t child[kNumActions];    // -1: not created
    float prior[kNumActions];
    int32_t n;                     // finished simulations through this node
    int32_t pending;               // simulations of the current wave through this node
    float b;                       // frames to go: the best found below, or the net's value
    uint8_t action;                // the edge from the parent
    uint8_t term;                  // kRunning / kGoal / kDead
    uint8_t evaluated;             // prior and value set (terminals: at creation)
    uint8_t inflight;              // waiting for the net this wave
};

struct MctsTree {
    std::vector<MctsNode> nodes;
    std::vector<uint8_t> states;   // compact state per node
    std::vector<uint8_t> frames;   // 84x84 per node
    std::vector<int32_t> free_ids;
    int32_t root = -1;
    Segment seg;
    uint8_t hist[kStack - 1][kObsSize];   // committed frames before the root, newest first
};

struct MctsLeaf { int32_t tree, node; };

class Forest {
public:
    Forest(Pool& pool, std::vector<Emu*>& emus, int n_trees, const MctsParams& p);
    int size() const { return (int)trees_.size(); }
    // new root from a full state; its frame (and the 3 before it) = the current screen
    bool reset(int t, const uint8_t* full, const std::vector<int>& route);
    // one wave: up to per_tree new leaves per listed tree; emulated and rendered in
    // parallel; terminal leaves are backed up at once. Returns the leaves for the net
    // with their input stacks (kStack x 84 x 84, oldest first) in obs.
    int select(const int32_t* trees, int n_trees, int per_tree, int max_leaves, MctsLeaf* leaves, uint8_t* obs);
    // the net's outputs for select's leaves: priors (probabilities), values (frames)
    void backup(int n, const MctsLeaf* leaves, const float* priors, const float* values);
    // root statistics per action: visits and b (-1: not created); returns the root's n
    int root_stats(int t, int32_t* visits, float* best, float* root_b) const;
    void root_noise(int t, const float* noise, float frac);
    // play the action in the tree's game: its child becomes the root (created if needed);
    // returns the new root's term
    int commit(int t, int action);
    // the root's input does nothing now: every (action, NOOP) reaches one exact state
    void forced(const int32_t* trees, int n, int32_t* out);
    void root_state(int t, uint8_t* full, uint8_t* ram, uint8_t* stack) const;
    int nodes(int t) const { return (int)(trees_[t].nodes.size() - trees_[t].free_ids.size()); }
    const MctsParams& params() const { return p_; }

private:
    int32_t alloc(MctsTree& T);
    void stack_of(const MctsTree& T, int32_t node, uint8_t* out) const;
    void refresh(MctsTree& T, int32_t node);
    void gc(MctsTree& T);
    Pool& pool_;
    std::vector<Emu*>& emus_;
    std::vector<MctsTree> trees_;
    MctsParams p_;
    size_t cs_;
};

}  // namespace ss
