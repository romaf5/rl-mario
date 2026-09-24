// AlphaZero-style MCTS on the real game, for a net that sees 84x84 frames and
// predicts a prior over the 12 actions and the frames left to the segment goal.
//
// Children are created lazily: a simulation descends by PUCT to an edge with no
// child, emulates that one step (the parent's compact state + action) and renders
// its frame; the net then scores the new node (prior, value), batched over every
// leaf of a wave across all trees of the forest. Backup is the mean (AlphaZero): a
// simulation reaching a leaf worth v frames to go (the net; goal 0, dead v_death)
// adds 4 * depth + v to each node on its path; b(n) = that sum / n. (A min backup
// was tried first: over noisy net values the minimum is optimistic and the most
// explored subtree looked best -- the agent stalled in 1-1.)
// q(c) = clamp(1 - (b(c) - best sibling) / scale). With exact leaf values (the route)
// min_backup keeps b(n) = 4 + min over children instead: the best line found.
//
// Leaf values: value_mix * net + (1 - value_mix) * route, where route is the frames to
// go along the level's route (the beam's progress rank, optimize/progress.h, carried
// from parent to child) when a route is registered for the level (set_route).
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include "../optimize/progress.h"
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
    float value_mix = 1.0f;        // leaf value: this x net + (1 - this) x route (1 without a route)
    int min_backup = 0;            // 1: b(n) = 4 + min over children (the best line; for exact values)
    int relative = 0;              // 1: leaf values are relative to the root (relvalue.py), so a
                                   // goal at depth k is worth -4k: the root's b is then exactly 0
};

enum : uint8_t { kRunning = 0, kGoal = 1, kDead = 2, kDup = 3 };   // kDup: same state as a sibling
constexpr int kStack = 4;          // frames per net input

struct MctsNode {
    int32_t parent;
    int32_t child[kNumActions];    // -1: not created
    float prior[kNumActions];
    int32_t n;                     // finished simulations through this node
    int32_t pending;               // simulations of the current wave through this node
    double w;                      // sum over simulations of the frames to go they found
    float b;                       // w / n: mean frames to go (the net's value at a new leaf)
    float v0;                      // the value this node was given when it was created
    uint8_t action;                // the edge from the parent
    uint8_t term;                  // kRunning / kGoal / kDead
    uint8_t evaluated;             // prior and value set (terminals: at creation)
    uint8_t inflight;              // waiting for the net this wave
    int32_t tau;                   // route progress: the latest reference step matched
    int64_t rank;                  // route progress: frames to go x 16 (progress.h units)
    uint64_t key;                  // exact state key (sibling duplicates)
};

struct MctsTree {
    std::vector<MctsNode> nodes;
    std::vector<uint8_t> states;   // compact state per node
    std::vector<uint8_t> frames;   // 84x84 per node
    std::vector<int32_t> free_ids;
    int32_t root = -1;
    Segment seg;
    uint8_t hist[kStack - 1][kObsSize];   // committed frames before the root, newest first
    const RefProgress* rp = nullptr;      // the segment level's route, if registered
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
    // survival check before a commit: for each candidate root action, is there a button
    // that, held from the child state for `horizon` steps, does not die (or reaches the
    // goal)? out[i] = 1: that action survives; 0: every constant continuation dies
    void safe(int t, const int32_t* actions, int n, int horizon, int32_t* out);
    int nodes(int t) const { return (int)(trees_[t].nodes.size() - trees_[t].free_ids.size()); }
    const MctsParams& params() const { return p_; }
    // the route for a level: the reference actions from their start state (full)
    void set_route(int level_gp, const uint8_t* start_full, const uint8_t* actions, int n);
    void set_value_mix(float mix) { p_.value_mix = mix; }
    float route_value(int t, int32_t node) const;
    // per leaf: the route's frames to go, and the depth from the root. A relative value
    // (D = value(leaf) - value(root)) is all the search needs: q compares siblings, so a
    // constant per tree cancels -- and unlike absolute frames to go, it is on the screen.
    void leaf_info(int n, const MctsLeaf* leaves, float* values, int32_t* depths) const;
    float root_value(int t) const { return route_value(t, trees_[t].root); }

private:
    int32_t alloc(MctsTree& T);
    void stack_of(const MctsTree& T, int32_t node, uint8_t* out) const;
    void add_path(MctsTree& T, int32_t leaf, float v, bool pending);
    float term_value(const MctsTree& T, int32_t node) const;
    void refresh_min(MctsTree& T, int32_t x);
    void gc(MctsTree& T);
    void route_root(MctsTree& T);
    Pool& pool_;
    std::map<int, std::unique_ptr<RefProgress>> routes_;
    std::vector<Emu*>& emus_;
    std::vector<MctsTree> trees_;
    MctsParams p_;
    size_t cs_;
};

}  // namespace ss
