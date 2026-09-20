#include "mcts.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include "../core/keys.h"

namespace ss {

namespace {

void init_node(MctsNode& n, int32_t parent, uint8_t action) {
    n.parent = parent;
    for (int a = 0; a < kNumActions; a++) { n.child[a] = -1; n.prior[a] = 1.0f / kNumActions; }
    n.n = 0; n.pending = 0; n.b = 0; n.action = action;
    n.term = kRunning; n.evaluated = 0; n.inflight = 0;
}

uint8_t term_of(Outcome o) { return o == Outcome::Goal ? kGoal : o == Outcome::Dead ? kDead : kRunning; }

}  // namespace

Forest::Forest(Pool& pool, std::vector<Emu*>& emus, int n_trees, const MctsParams& p)
    : pool_(pool), emus_(emus), trees_(n_trees), p_(p), cs_(compact_state_size()) {}

int32_t Forest::alloc(MctsTree& T) {
    if (!T.free_ids.empty()) {
        const int32_t id = T.free_ids.back();
        T.free_ids.pop_back();
        return id;
    }
    if ((int)T.nodes.size() >= p_.max_nodes) return -1;
    const int32_t id = (int32_t)T.nodes.size();
    T.nodes.emplace_back();
    T.states.resize(T.nodes.size() * cs_);
    T.frames.resize(T.nodes.size() * kObsSize);
    return id;
}

bool Forest::reset(int t, const uint8_t* full, const std::vector<int>& route) {
    MctsTree& T = trees_[t];
    T.nodes.clear(); T.states.clear(); T.frames.clear(); T.free_ids.clear();
    T.root = -1;
    Emu& e = *emus_[0];
    e.load_full(full);
    T.seg = Segment::make(route, e.ram());
    if (T.seg.start_gp < 0) return false;
    const int32_t r = alloc(T);
    init_node(T.nodes[r], -1, 0);
    e.save(&T.states[(size_t)r * cs_]);
    e.obs_now(&T.frames[(size_t)r * kObsSize]);
    for (auto& h : T.hist) memcpy(h, &T.frames[(size_t)r * kObsSize], kObsSize);
    T.nodes[r].term = term_of(T.seg.classify(e.ram()));
    T.root = r;
    return true;
}

// oldest first: [t-3, t-2, t-1, node]; above the root, the committed history
void Forest::stack_of(const MctsTree& T, int32_t node, uint8_t* out) const {
    int32_t x = node;
    int h = 0;
    for (int k = kStack - 1; k >= 0; k--) {
        const uint8_t* f;
        if (x >= 0) { f = &T.frames[(size_t)x * kObsSize]; x = T.nodes[x].parent; }
        else f = T.hist[h++];
        memcpy(out + (size_t)k * kObsSize, f, kObsSize);
    }
}

void Forest::refresh(MctsTree& T, int32_t x) {
    MctsNode& N = T.nodes[x];
    float best = 1e30f;
    for (int a = 0; a < kNumActions; a++) {
        const int32_t c = N.child[a];
        if (c >= 0 && T.nodes[c].evaluated) best = std::min(best, T.nodes[c].b);
    }
    if (best < 1e30f) N.b = std::min(p_.v_death, kFrameSkip + best);
}

int Forest::select(const int32_t* trees, int n_trees, int per_tree, int max_leaves, MctsLeaf* leaves,
                   uint8_t* obs) {
    struct Job { int32_t t, node; bool emulate; };
    std::vector<Job> jobs;
    jobs.reserve((size_t)max_leaves);
    std::vector<int32_t> path;
    for (int i = 0; i < n_trees; i++) {
        const int t = trees[i];
        MctsTree& T = trees_[t];
        if (T.root < 0 || T.nodes[T.root].term != kRunning) continue;
        if (!T.nodes[T.root].evaluated) {                 // the root first needs the net
            if (!T.nodes[T.root].inflight && (int)jobs.size() < max_leaves) {
                T.nodes[T.root].inflight = 1;
                jobs.push_back({t, T.root, false});
            }
            continue;
        }
        for (int k = 0; k < per_tree && (int)jobs.size() < max_leaves; k++) {
            path.clear();
            int32_t x = T.root, leaf = -1;
            bool terminal = false;
            for (;;) {
                const MctsNode& N = T.nodes[x];
                // PUCT; an edge whose child waits for the net is not taken again this wave
                float bstar = 1e30f;
                for (int a = 0; a < kNumActions; a++) {
                    const int32_t c = N.child[a];
                    if (c >= 0 && T.nodes[c].evaluated) bstar = std::min(bstar, T.nodes[c].b);
                }
                const float sq = std::sqrt((float)(N.n + N.pending + 1));
                int best = -1;
                float bs = -1e30f;
                for (int a = 0; a < kNumActions; a++) {
                    const int32_t c = N.child[a];
                    float q;
                    int nc = 0;
                    if (c < 0) {
                        q = p_.fpu;
                    } else {
                        const MctsNode& C = T.nodes[c];
                        if (!C.evaluated) continue;
                        q = C.term == kDead ? 0.f : std::min(1.f, std::max(0.f, 1.f - (C.b - bstar) / p_.scale));
                        nc = C.n + C.pending;
                        if (C.pending) q *= (float)C.n / (float)nc;   // in-flight visits count as q = 0
                    }
                    const float s = q + p_.c_puct * N.prior[a] * sq / (1.f + nc);
                    if (s > bs) { bs = s; best = a; }
                }
                if (best < 0) break;                      // every edge in flight
                path.push_back(x);
                const int32_t c = N.child[best];
                if (c < 0) {
                    const int32_t id = alloc(T);          // may move T.nodes
                    if (id < 0) break;                    // tree full
                    init_node(T.nodes[id], x, (uint8_t)best);
                    T.nodes[x].child[best] = id;
                    T.nodes[id].inflight = 1;
                    leaf = id;
                    break;
                }
                if (T.nodes[c].term != kRunning) { leaf = c; terminal = true; break; }
                x = c;
            }
            if (leaf < 0) continue;
            if (terminal) {                               // a known end: count it, no net
                T.nodes[leaf].n++;
                for (int j = (int)path.size() - 1; j >= 0; j--) { T.nodes[path[j]].n++; refresh(T, path[j]); }
                continue;
            }
            for (int32_t p : path) T.nodes[p].pending++;
            jobs.push_back({t, leaf, true});
        }
    }

    const int nj = (int)jobs.size();
    pool_.parallel_for(nj, [&](int64_t j, int w) {
        const Job& J = jobs[j];
        if (!J.emulate) return;
        MctsTree& T = trees_[J.t];
        MctsNode& C = T.nodes[J.node];
        Emu& e = *emus_[w];
        e.load(&T.states[(size_t)C.parent * cs_]);
        e.step_obs(C.action, &T.frames[(size_t)J.node * kObsSize]);
        e.save(&T.states[(size_t)J.node * cs_]);
        C.term = term_of(T.seg.classify(e.ram()));
    }, 1);

    int n_out = 0;
    for (const Job& J : jobs) {
        MctsTree& T = trees_[J.t];
        MctsNode& C = T.nodes[J.node];
        if (C.term == kRunning) { leaves[n_out++] = {J.t, J.node}; continue; }
        const float v = C.term == kGoal ? 0.f : p_.v_death;   // a new terminal: back it up now
        C.evaluated = 1; C.inflight = 0; C.b = v; C.n = 1;
        for (int32_t x = C.parent; x >= 0; x = T.nodes[x].parent) {
            T.nodes[x].pending--; T.nodes[x].n++; refresh(T, x);
        }
    }
    pool_.parallel_for(n_out, [&](int64_t i, int) {
        stack_of(trees_[leaves[i].tree], leaves[i].node, obs + (size_t)i * kStack * kObsSize);
    }, 8);
    return n_out;
}

void Forest::backup(int n, const MctsLeaf* leaves, const float* priors, const float* values) {
    for (int i = 0; i < n; i++) {
        MctsTree& T = trees_[leaves[i].tree];
        MctsNode& C = T.nodes[leaves[i].node];
        memcpy(C.prior, priors + (size_t)i * kNumActions, sizeof C.prior);
        C.b = std::min(p_.v_death, std::max(0.f, values[i]));
        C.evaluated = 1; C.inflight = 0; C.n++;
        for (int32_t x = C.parent; x >= 0; x = T.nodes[x].parent) {
            T.nodes[x].pending--; T.nodes[x].n++; refresh(T, x);
        }
    }
}

int Forest::root_stats(int t, int32_t* visits, float* best, float* root_b) const {
    const MctsTree& T = trees_[t];
    const MctsNode& R = T.nodes[T.root];
    for (int a = 0; a < kNumActions; a++) {
        const int32_t c = R.child[a];
        visits[a] = c >= 0 ? T.nodes[c].n : 0;
        best[a] = c >= 0 && T.nodes[c].evaluated ? T.nodes[c].b : -1.f;
    }
    if (root_b) *root_b = R.b;
    return R.n;
}

void Forest::root_noise(int t, const float* noise, float frac) {
    MctsTree& T = trees_[t];
    MctsNode& R = T.nodes[T.root];
    for (int a = 0; a < kNumActions; a++) R.prior[a] = (1 - frac) * R.prior[a] + frac * noise[a];
}

void Forest::gc(MctsTree& T) {
    std::vector<uint8_t> keep(T.nodes.size(), 0);
    std::vector<int32_t> st{T.root};
    while (!st.empty()) {
        const int32_t x = st.back();
        st.pop_back();
        keep[x] = 1;
        for (int32_t c : T.nodes[x].child)
            if (c >= 0) st.push_back(c);
    }
    T.free_ids.clear();
    for (int32_t i = (int32_t)T.nodes.size() - 1; i >= 0; i--)
        if (!keep[i]) T.free_ids.push_back(i);
}

int Forest::commit(int t, int action) {
    MctsTree& T = trees_[t];
    int32_t c = T.nodes[T.root].child[action];
    if (c < 0) {
        c = alloc(T);
        if (c < 0) {                                      // full: keep only the root
            for (int32_t& x : T.nodes[T.root].child) x = -1;
            gc(T);
            c = alloc(T);
        }
        init_node(T.nodes[c], T.root, (uint8_t)action);
        T.nodes[T.root].child[action] = c;
        Emu& e = *emus_[0];
        e.load(&T.states[(size_t)T.root * cs_]);
        e.step_obs(action, &T.frames[(size_t)c * kObsSize]);
        e.save(&T.states[(size_t)c * cs_]);
        MctsNode& C = T.nodes[c];
        C.term = term_of(T.seg.classify(e.ram()));
        if (C.term != kRunning) { C.evaluated = 1; C.b = C.term == kGoal ? 0.f : p_.v_death; }
    }
    memmove(T.hist[1], T.hist[0], (size_t)(kStack - 2) * kObsSize);
    memcpy(T.hist[0], &T.frames[(size_t)T.root * kObsSize], kObsSize);
    T.nodes[c].parent = -1;
    T.root = c;
    gc(T);
    return T.nodes[c].term;
}

void Forest::forced(const int32_t* trees, int n, int32_t* out) {
    std::vector<uint64_t> key((size_t)n * kNumActions);
    pool_.parallel_for((int64_t)n * kNumActions, [&](int64_t j, int w) {
        const MctsTree& T = trees_[trees[j / kNumActions]];
        Emu& e = *emus_[w];
        e.load(&T.states[(size_t)T.root * cs_]);
        e.step((int)(j % kNumActions));
        e.step(0);
        key[j] = exact_key(e.ram());
    }, 1);
    for (int i = 0; i < n; i++) {
        out[i] = 1;
        for (int a = 1; a < kNumActions; a++)
            if (key[(size_t)i * kNumActions + a] != key[(size_t)i * kNumActions]) { out[i] = 0; break; }
    }
}

void Forest::root_state(int t, uint8_t* full, uint8_t* ram, uint8_t* stack) const {
    const MctsTree& T = trees_[t];
    Emu& e = *emus_[0];
    e.load(&T.states[(size_t)T.root * cs_]);
    if (full) e.save_full(full);
    if (ram) memcpy(ram, e.ram(), 0x800);
    if (stack) stack_of(T, T.root, stack);
}

}  // namespace ss
