"""The latent search, many games at once.

latent.py grows one tree per decision and walks it in Python: ~1.1 s per decision, a gate of
32 games in an hour and a half, and far too slow to play the thousands of games a MuZero loop
learns from. Here K trees grow together: the walk down is compiled (numba), and each wave's
picks from every tree are imagined in one call to the model. The rules are latent.py's
exactly -- PUCT on W, death as a price, calibrated event heads, the depth bound, pending nodes
unselectable until imagined -- so a game played here is a game latent.py would play.

Oracles, to find which part of the model fails (each node then also carries the real game's
state, and every imagined step is also played for real):
  --real-events       deaths and finishes from the real game, not the event heads
  --real-value V.pt   W from the stage A/C value net on the real screens, not the value head
  --real-prior        the prior below the root from the policy net on the real screens, not the
                      model's distilled head (the root always has the net's)
  --dedup             a child whose real state equals a brother's is never visited again (the C++
                      search prunes these: in the air, B changes nothing)
  --real-veto H       stage A's commit check: the move must survive H steps of some held input
                      in the real game, else the next most visited
Both oracles together test the search's own rules with a perfect model.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.blatent --model smbzero/runs/wm13/wm.pt \
      --prior-net smbzero/runs/zero8/net.pt --deep-prior model --levels all --delays 5,20,35,50
"""
import argparse, json, os, time
import numpy as np
import numba
import torch
from .common import FPS, ROUTE, RUNS, Search, e2e_segments, gp
from .model import NO_PREV, load as load_model
from .relvalue import load as load_rel

HOPELESS = 512.0


@numba.njit(cache=True)
def _backup_p(k, x, child, parent, b, w, n, self_mode, term):
    while x >= 0:
        best, any_ = np.float32(1e30), False
        for a in range(12):
            c = child[k, x, a]
            if c >= 0 and term[k, c] != 3:
                any_ = True
                if b[k, c] < best:
                    best = b[k, c]
        if any_:
            b[k, x] = min(w[k, x], best) if self_mode else best
        n[k, x] += 1
        x = parent[k, x]


@numba.njit(cache=True)
def select_wave(active, budget, child, parent, depth, n, w, b, term, pend, prior, act_in, size,
                max_nodes, max_depth, fpu, c_puct, scale, per_wave, self_mode, out, allow):
    """Up to per_wave picks per active tree. out[i] = (tree, parent node, action, new node).
    A pick into a settled node (terminal, or at the depth bound) is a revisit: backed up at once."""
    cnt = 0
    score = np.empty(12, np.float32)
    for k in range(active.shape[0]):
        if not active[k]:
            continue
        picks = 0
        while budget[k] > 0 and picks < per_wave and size[k] < max_nodes - 1:
            x = 0
            blocked, revisit, a = False, -1, -1
            while True:
                if depth[k, x] >= max_depth:
                    revisit = x
                    break
                bstar, ready = np.float32(1e30), False
                all_pend = True
                for j in range(12):
                    c = child[k, x, j]
                    if not allow[k, x, j]:
                        continue
                    if c < 0:
                        all_pend = False
                    elif not pend[k, c] and term[k, c] != 3:
                        all_pend = False
                        ready = True
                        if b[k, c] < bstar:
                            bstar = b[k, c]
                if all_pend:
                    blocked = True
                    break
                sq = np.sqrt(np.float32(n[k, x]))
                best, a = np.float32(-1e30), -1
                for j in range(12):
                    c = child[k, x, j]
                    if (c >= 0 and (pend[k, c] or term[k, c] == 3)) or not allow[k, x, j]:
                        continue
                    q, nc = np.float32(fpu), np.float32(0.0)
                    if c >= 0:
                        nc = np.float32(n[k, c])
                        if term[k, c] == 2:
                            q = np.float32(0.0)
                        else:
                            q = np.float32(1.0) - (b[k, c] - bstar) / scale
                            q = min(max(q, np.float32(0.0)), np.float32(1.0))
                    s_ = q + c_puct * prior[k, x, j] * sq / (np.float32(1.0) + nc)
                    if s_ > best:
                        best, a = s_, j
                c = child[k, x, a]
                if c < 0:
                    break
                if term[k, c] != 0:
                    revisit = c
                    break
                x = c
            if blocked:
                break
            budget[k] -= 1
            if revisit >= 0:
                _backup_p(k, revisit, child, parent, b, w, n, self_mode, term)
                continue
            c = size[k]
            size[k] += 1
            child[k, x, a] = c
            parent[k, c] = x
            depth[k, c] = depth[k, x] + 1
            act_in[k, c] = a
            n[k, c] = 0
            b[k, c] = 0.0
            w[k, c] = 0.0
            pend[k, c] = True
            for j in range(12):
                child[k, c, j] = -1
            term[k, c] = 0
            out[cnt, 0] = k
            out[cnt, 1] = x
            out[cnt, 2] = a
            out[cnt, 3] = c
            cnt += 1
            picks += 1
    return cnt


@numba.njit(cache=True)
def apply_wave(out, cnt, wv, p_dead, p_goal, pri, uniform, child, parent, n, w, b, term, pend, prior,
               dead_p, goal_p, death_cost, self_mode, dup, floor, allow, topk):
    for i in range(cnt):
        k, c = out[i, 0], out[i, 3]
        pend[k, c] = False
        if dup[i]:                              # the same state as a brother: never again (term 3)
            term[k, c] = 3
            w[k, c] = b[k, c] = HOPELESS
            continue
        for j in range(12):
            prior[k, c, j] = np.float32(1.0 / 12) if uniform else pri[i, j]
        if topk >= 12:
            for j in range(12):
                allow[k, c, j] = True
        else:
            order = np.argsort(-pri[i])
            for j in range(12):
                allow[k, c, j] = False
            for j in range(topk):
                allow[k, c, order[j]] = True
        if p_dead[i] > dead_p:                  # certain enough to stop looking
            term[k, c] = 2
            w[k, c] = HOPELESS
        elif p_goal[i] > goal_p:                # the model says it finished
            term[k, c] = 1
            w[k, c] = 0.0
        else:                                   # otherwise death is a price
            w[k, c] = min(max(wv[i], floor), np.float32(HOPELESS)) + p_dead[i] * death_cost
        b[k, c] = w[k, c]
    for i in range(cnt):                        # every pick imagined before any is backed up
        if not dup[i]:
            _backup_p(out[i, 0], out[i, 3], child, parent, b, w, n, self_mode, term)


@numba.njit(cache=True)
def reroot(k, r, child, parent, depth, n, w, b, term, prior, act_in, keys, size, order):
    """Keep node r's subtree as tree k, r first (breadth-first order, written to `order`: the
    old index of each kept node). Costs move into r's frame: W from r = W from the old root
    minus r's own W (a death stays a death). Returns the number kept."""
    m = size[k]
    order[0] = r
    cnt, head = 1, 0
    while head < cnt:
        x = order[head]
        head += 1
        for a in range(12):
            c = child[k, x, a]
            if c >= 0:
                order[cnt] = c
                cnt += 1
    new = np.full(m, -1, np.int32)
    for i in range(cnt):
        new[order[i]] = i
    ch_t, par_t, dep_t = child[k, :m].copy(), parent[k, :m].copy(), depth[k, :m].copy()
    n_t, w_t, b_t, term_t = n[k, :m].copy(), w[k, :m].copy(), b[k, :m].copy(), term[k, :m].copy()
    pri_t, ai_t, key_t = prior[k, :m].copy(), act_in[k, :m].copy(), keys[k, :m].copy()
    delta = w_t[r]
    for i in range(cnt):
        o = order[i]
        for a in range(12):
            c = ch_t[o, a]
            child[k, i, a] = new[c] if c >= 0 else -1
        parent[k, i] = new[par_t[o]] if i > 0 else -1
        depth[k, i] = dep_t[o] - 1
        n[k, i] = n_t[o]
        term[k, i] = term_t[o]
        if term_t[o] >= 2 or w_t[o] >= HOPELESS:
            w[k, i] = w_t[o]
        else:
            w[k, i] = max(w_t[o] - delta, np.float32(0.0))
        b[k, i] = b_t[o] if b_t[o] >= HOPELESS else max(b_t[o] - delta, np.float32(0.0))
        for a in range(12):
            prior[k, i, a] = pri_t[o, a]
        act_in[k, i] = ai_t[o]
        keys[k, i] = key_t[o]
    size[k] = cnt
    return cnt


class Forest:
    """K latent trees. Node arrays on the CPU (numba walks them), latents on the GPU."""
    def __init__(self, model, K, max_nodes=2048, c_puct=1.5, scale=32.0, fpu=0.5, dead_p=0.95, goal_p=0.9,
                 death_cost=HOPELESS, calib=None, max_depth=12, backup='self', prior_net=None,
                 deep_prior='uniform', per_wave=32, device='cuda', emu=None, real_events=False, real_value=None,
                 real_prior=False, dedup=False, floor=0.0, topk=12):
        self.m, self.dev, self.K, self.N = model, device, K, max_nodes
        self.emu, self.real_events, self.real_value, self.real_prior = emu, real_events, real_value, real_prior
        self.dedup = dedup
        # W below 0 is a line doing better than the value expected from the root. latent.py
        # floored it at 0 -- then every good line ties at 0 and the prior alone decides; the C++
        # search keeps it ("relative: below 0 is progress"). -inf: no floor.
        self.floor = np.float32(floor)
        # topk < 12: below the root only each node's topk moves under the prior may be expanded --
        # twelve ways at every node kept a 1000-simulation tree 5-7 deep
        self.topk = int(topk)
        self.allow = np.ones((K, max_nodes, 12), np.bool_)
        self.keys = np.zeros((K, max_nodes), np.int64)
        if emu is not None:
            self.states = [[None] * max_nodes for _ in range(K)]
            self.frames = (np.zeros((K, max_nodes, 84, 84), np.uint8)
                           if (real_value is not None or real_prior) else None)
            self.root_stack = np.zeros((K, 4, 84, 84), np.uint8)
            self.lives0, self.level = np.zeros(K, np.int32), [None] * K
            self.root_e = {}
            import threading
            from concurrent.futures import ThreadPoolExecutor
            # ctypes releases the GIL, so the steps run in parallel -- but a context steps with its
            # one emulator, so every worker gets a context of its own (states are portable bytes)
            self.local = threading.local()
            self.pool = ThreadPoolExecutor(32)
        self.calib = None if calib is None else np.asarray(calib, np.float32)
        self.prior_net, self.uniform = prior_net, (prior_net is not None and deep_prior == 'uniform')
        self.params = dict(c_puct=np.float32(c_puct), scale=np.float32(scale), fpu=np.float32(fpu))
        self.dead_p, self.goal_p, self.death_cost = np.float32(dead_p), np.float32(goal_p), np.float32(death_cost)
        self.max_depth, self.self_mode, self.per_wave = max_depth, backup == 'self', per_wave
        c = model.g.conv.out_channels
        self.lat = torch.zeros((K * max_nodes, c, 11, 11), device=device, dtype=torch.float16)
        self.root_lat = torch.zeros((K, c, 11, 11), device=device, dtype=torch.float16)
        self.child = np.full((K, max_nodes, 12), -1, np.int32)
        self.parent = np.full((K, max_nodes), -1, np.int32)
        self.depth = np.zeros((K, max_nodes), np.int32)
        self.n = np.zeros((K, max_nodes), np.int32)
        self.w = np.zeros((K, max_nodes), np.float32)
        self.b = np.zeros((K, max_nodes), np.float32)
        self.term = np.zeros((K, max_nodes), np.uint8)
        self.pend = np.zeros((K, max_nodes), np.bool_)
        self.prior = np.zeros((K, max_nodes, 12), np.float32)
        self.act_in = np.full((K, max_nodes), NO_PREV, np.int64)
        self.size = np.zeros(K, np.int32)
        self.out = np.zeros((K * per_wave, 4), np.int32)

    @torch.no_grad()
    def reset(self, ks, stacks, prevs, states=None, levels=None, keep=None):
        """Trees ks from the real screens (len(ks), 4, 84, 84) and the moves before them (with an
        oracle, also the real states and level names). keep[i]: the tree was rerooted on this
        move and keeps its subtree -- the root itself is still encoded fresh from the screen."""
        ks = np.asarray(ks)
        keep = np.zeros(len(ks), bool) if keep is None else np.asarray(keep, bool)
        if self.emu is not None:
            for i, k in enumerate(ks):
                self.states[k][0] = states[i]
                self.root_stack[k] = stacks[i]
                self.lives0[k] = int(self.emu.ram(states[i])[0x75A])
                self.level[k] = levels[i]
            if self.real_value is not None:
                with torch.autocast('cuda', dtype=torch.float16):
                    e = self.real_value.embed(torch.from_numpy(np.ascontiguousarray(stacks)).to(self.dev))
                for i, k in enumerate(ks):
                    self.root_e[int(k)] = e[i]
        x = torch.from_numpy(np.ascontiguousarray(stacks)).to(self.dev)
        pv = torch.as_tensor(np.asarray(prevs, np.int64), device=self.dev)
        with torch.autocast('cuda', dtype=torch.float16):
            s, pi, _ = self.m.initial(x, pv)
            if self.prior_net is not None:
                pi, _ = self.prior_net(x)
        pri = torch.softmax(pi.float(), 1).cpu().numpy()
        self.lat[torch.as_tensor(ks * self.N, device=self.dev)] = s.half()
        self.root_lat[torch.as_tensor(ks, device=self.dev)] = s.half()
        for i, k in enumerate(ks):
            if not keep[i]:
                self.child[k, 0] = -1
                self.n[k, 0] = 1
                self.size[k] = 1
            self.parent[k, 0] = -1
            self.depth[k, 0] = 0
            self.n[k, 0] = max(self.n[k, 0], 1)
            self.w[k, 0] = self.b[k, 0] = 0.0
            self.term[k, 0] = 0
            self.pend[k, 0] = False
            self.prior[k, 0] = pri[i]
            self.act_in[k, 0] = prevs[i]
            self.allow[k, 0] = True

    def advance(self, k, a):
        """After move a: keep that child's subtree for the next decision (True), or nothing."""
        c = int(self.child[k, 0, a])
        if c < 0 or self.term[k, c] != 0 or self.pend[k, c]:
            return False
        order = np.zeros(self.size[k], np.int32)
        cnt = reroot(k, c, self.child, self.parent, self.depth, self.n, self.w, self.b, self.term, self.prior,
                     self.act_in, self.keys, self.size, order)
        o = torch.from_numpy(order[:cnt].astype(np.int64) + k * self.N).to(self.dev)
        self.lat[k * self.N: k * self.N + cnt] = self.lat[o].clone()
        if self.emu is not None:
            self.states[k][:cnt] = [self.states[k][x] for x in order[:cnt]]
            if self.frames is not None:
                self.frames[k, :cnt] = self.frames[k, order[:cnt]]
        return True

    @torch.no_grad()
    def run(self, ks, sims):
        active = np.zeros(self.K, np.bool_)
        active[np.asarray(ks)] = True
        budget = np.where(active, sims, 0).astype(np.int32)
        while True:
            cnt = select_wave(active, budget, self.child, self.parent, self.depth, self.n, self.w, self.b,
                              self.term, self.pend, self.prior, self.act_in, self.size, self.N, self.max_depth,
                              self.params['fpu'], self.params['c_puct'], self.params['scale'], self.per_wave,
                              self.self_mode, self.out, self.allow)
            if cnt == 0:
                if not (active & (budget > 0) & (self.size < self.N - 1)).any():
                    break
                continue                        # only revisits this wave: go again
            o = self.out[:cnt]
            kk, par, act, ch = o[:, 0].astype(np.int64), o[:, 1].astype(np.int64), o[:, 2], o[:, 3].astype(np.int64)
            src = torch.from_numpy(kk * self.N + par).to(self.dev)
            dst = torch.from_numpy(kk * self.N + ch).to(self.dev)
            a_t = torch.from_numpy(act.astype(np.int64)).to(self.dev)
            prev_t = torch.from_numpy(self.act_in[kk, par]).to(self.dev)
            dep_np = self.depth[kk, ch]
            with torch.autocast('cuda', dtype=torch.float16):
                s2, ev, _ = self.m.g(self.lat[src], a_t, prev_t)
                pi, w = self.m.f(s2, self.root_lat[torch.from_numpy(kk).to(self.dev)],
                                 torch.from_numpy(dep_np.astype(np.float32)).to(self.dev))
            self.lat[dst] = s2.half()
            lg = ev.float().cpu().numpy()
            if self.calib is not None:          # a price is only fair if the probability is honest
                ki = np.clip(dep_np - 1, 0, len(self.calib) - 1)
                lg = lg / self.calib[ki, :, 0] + self.calib[ki, :, 1]
            p_ev = (1.0 / (1.0 + np.exp(-np.clip(lg, -30.0, 30.0)))).astype(np.float32)
            pri = torch.softmax(pi.float(), 1).cpu().numpy().astype(np.float32)
            w = w.float().cpu().numpy().astype(np.float32)
            dup = np.zeros(cnt, np.bool_)
            if self.emu is not None:
                ev_real = self._play_picks(kk, par, act, ch)
                if self.dedup:                  # the C++ search's rule: a brother already there wins
                    seen = {}
                    for i in range(cnt):
                        k, x = int(kk[i]), int(par[i])
                        if (k, x) not in seen:
                            seen[(k, x)] = {int(self.keys[k, o]) for o in self.child[k, x]
                                            if o >= 0 and not self.pend[k, o] and self.term[k, o] != 3}
                        key = int(self.keys[k, ch[i]])
                        dup[i] = key in seen[(k, x)]
                        seen[(k, x)].add(key)
                if self.real_events:
                    p_ev[:, 0] = ev_real == 1
                    p_ev[:, 1] = ev_real == 2
                if self.real_value is not None or self.real_prior:
                    leaf = self._leaf_stacks(kk, ch)
                if self.real_value is not None:
                    w = self._real_w(kk, leaf, dep_np)
                if self.real_prior:
                    with torch.autocast('cuda', dtype=torch.float16):
                        lg_pi, _ = self.prior_net(torch.from_numpy(leaf).to(self.dev))
                    pri = torch.softmax(lg_pi.float(), 1).cpu().numpy().astype(np.float32)
            apply_wave(self.out, cnt, w, np.ascontiguousarray(p_ev[:, 1]),
                       np.ascontiguousarray(p_ev[:, 0]), pri, self.uniform and not self.real_prior, self.child, self.parent, self.n,
                       self.w, self.b, self.term, self.pend, self.prior, self.dead_p, self.goal_p,
                       self.death_cost, self.self_mode, dup, self.floor, self.allow, self.topk)

    def _play_picks(self, kk, par, act, ch):
        """Play each imagined step for real: 0 running, 1 goal, 2 dead -- the search engine's rule
        (emu.h dying()): the dying animation, a life lost, or below the screen while in control."""
        out = np.zeros(len(kk), np.int32)

        def step(i):
            e = getattr(self.local, 'emu', None)
            if e is None:
                e = self.local.emu = Search(threads=1)
            st, a = self.states[kk[i]][par[i]], np.array([act[i]], np.uint8)
            obs, tr, st2 = e.replay_obs(st, a) if self.frames is not None else (None,) + tuple(e.replay(st, a))
            key = 0
            if self.dedup:                      # emu.h exact_key: all game-state RAM
                m = e.ram(st2)
                m[0:8] = 0; m[9] = 0; m[0x100:0x300] = 0
                m[0x7DD:0x7E3] = 0; m[0x7ED] = m[0x7EE] = 0; m[0x7F8:0x7FB] = 0
                key = hash(m.tobytes())
            return obs, tr, st2, key

        for i, (obs, tr, st2, key) in enumerate(self.pool.map(step, range(len(kk)))):
            k = int(kk[i])
            self.keys[k, ch[i]] = key
            if obs is not None:
                self.frames[k, ch[i]] = obs[-1]
            self.states[k][ch[i]] = st2
            L = self.level[k]
            lvl, mode = int(tr[-1, 2]), int(tr[-1, 7])
            if lvl != gp(L) or mode == 2:
                j = ROUTE.index(L)
                out[i] = 1 if (mode == 2 if j == len(ROUTE) - 1 else lvl == gp(ROUTE[j + 1])) else 2
            elif (tr[-1, 6] in (0x0B, 0x06) or tr[-1, 9] < self.lives0[k]
                  or (tr[-1, 6] == 0x08 and mode == 1 and tr[-1, 1] >= 512)):
                out[i] = 2
        return out

    def _leaf_stacks(self, kk, ch):
        """Each new node's real last four screens."""
        leaf = np.zeros((len(kk), 4, 84, 84), np.uint8)
        for i in range(len(kk)):
            k, x, fr = int(kk[i]), int(ch[i]), []
            while x != 0 and len(fr) < 4:
                fr.append(self.frames[k, x])
                x = int(self.parent[k, x])
            got = fr[::-1]
            leaf[i] = np.stack(list(self.root_stack[k][len(got):]) + got) if len(got) < 4 else np.stack(got)
        return leaf

    @torch.no_grad()
    def _real_w(self, kk, leaf, dep):
        """W of each new node from the value net on its real last four screens."""
        with torch.autocast('cuda', dtype=torch.float16):
            v = self.real_value
            e_root = torch.stack([self.root_e[int(k)] for k in kk])
            w = v.fold(v.head(v.embed(torch.from_numpy(leaf).to(self.dev)), e_root,
                              torch.from_numpy(dep.astype(np.int64)).to(self.dev)))
        return w.float().cpu().numpy().astype(np.float32)

    def choose(self, k):
        """Most visited root move, ties to the best cost (latent.py's choose without the veto)."""
        kids = self.child[k, 0]
        n = np.array([self.n[k, c] if c >= 0 else 0 for c in kids])
        b = np.array([self.b[k, c] if c >= 0 else np.inf for c in kids])
        if not n.sum():
            return 1
        return int(np.lexsort((np.where(np.isinf(b), 1e9, b), -n))[0])

    def visits(self, k):
        kids = self.child[k, 0]
        return np.array([self.n[k, c] if c >= 0 else 0 for c in kids], np.int32)


def survives(s, state, a, horizon):
    """Does move a, then some held input, last `horizon` steps in the real game (search rule)?"""
    for c in range(12):
        acts = np.array([a] + [c] * (horizon - 1), np.uint8)
        out, n = s.classify_along(state, ROUTE, acts)
        if not n or np.asarray(out[:n])[-1] != 2:
            return True
    return False


def play(s, forest, starts, sims, caps, log=None, veto=0, reuse=False, veto_net=None, veto_p=0.5):
    """Play every start to its end; K at a time, a finished game's tree goes to the next start.
    starts: [(level, delay, state)]; caps: max decisions per start. The real game is stepped
    only by the moves chosen. Returns one dict per start."""
    K = forest.K
    queue = list(range(len(starts)))
    slot = [None] * K                               # the game each tree plays
    games = [dict(level=l, delay=d, actions=[], won=False, reason='too long') for l, d, _ in starts]
    state, stack, lives0 = {}, {}, {}
    t0 = time.time()

    def admit(k):
        if not queue:
            slot[k] = None
            return
        i = queue.pop(0)
        slot[k] = i
        st = starts[i][2]
        state[i], stack[i] = st, np.repeat(s.obs(st)[None], 4, 0)
        lives0[i] = int(s.ram(st)[0x75A])

    for k in range(K):
        admit(k)
    kept = np.zeros(K, bool)                        # the tree keeps the played move's subtree
    while any(x is not None for x in slot):
        ks = [k for k in range(K) if slot[k] is not None]
        forest.reset(ks, np.stack([stack[slot[k]] for k in ks]),
                     [games[slot[k]]['actions'][-1] if games[slot[k]]['actions'] else NO_PREV for k in ks],
                     states=[state[slot[k]] for k in ks], levels=[games[slot[k]]['level'] for k in ks],
                     keep=[kept[k] for k in ks])
        forest.run(ks, sims)
        picks = {k: forest.choose(k) for k in ks}
        if veto_net is not None:                    # stage A's commit check, learned: no game touched
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                lp = veto_net(torch.from_numpy(np.stack([stack[slot[k]] for k in ks])).cuda(),
                              torch.tensor([games[slot[k]]['actions'][-1] if games[slot[k]]['actions'] else NO_PREV
                                            for k in ks]).cuda())
            ps = torch.sigmoid(lp.float()).cpu().numpy()
            for j, k in enumerate(ks):
                n = forest.visits(k)
                order = [picks[k]] + [int(x) for x in np.argsort(-n, kind='stable') if n[x] > 0 and x != picks[k]]
                order += [x for x in range(12) if x not in order]
                picks[k] = next((x for x in order if ps[j, x] >= veto_p), picks[k])
        if veto:                                    # an oracle: stage A's commit check, in the real game
            import threading
            from concurrent.futures import ThreadPoolExecutor
            if not hasattr(play, 'pool'):
                play.pool, play.local = ThreadPoolExecutor(32), threading.local()

            def check(k):
                e = getattr(play.local, 'emu', None)
                if e is None:
                    e = play.local.emu = Search(threads=1)
                n = forest.visits(k)
                order = [int(x) for x in np.argsort(-n, kind='stable') if n[x] > 0]
                order = [picks[k]] + [x for x in order if x != picks[k]]
                order += [x for x in range(12) if x not in order]
                return next((x for x in order if survives(e, state[slot[k]], x, veto)), picks[k])
            picks = dict(zip(ks, play.pool.map(check, ks)))
        for k in ks:
            i = slot[k]
            g = games[i]
            a = picks[k]
            kept[k] = forest.advance(k, a) if reuse else False
            obs, tr, state[i] = s.replay_obs(state[i], np.array([a], np.uint8))
            g['actions'].append(a)
            stack[i] = np.concatenate([stack[i][1:], obs])
            lvl0 = g['level']
            lvl, mode = int(tr[-1, 2]), int(tr[-1, 7])
            end = None
            if lvl != gp(lvl0) or mode == 2:
                j = ROUTE.index(lvl0)
                won = mode == 2 if j == len(ROUTE) - 1 else lvl == gp(ROUTE[j + 1])
                end = ('goal' if won else 'dead at %s' % lvl0, won)
            elif tr[-1, 6] in (0x0B, 0x06) or tr[-1, 9] < lives0[i]:
                end = ('dead at %s' % lvl0, False)
            elif len(g['actions']) >= caps[i]:
                end = ('too long', False)
            if end:
                g['reason'], g['won'] = end
                g['decisions'] = len(g['actions'])
                g['seconds'] = round((g['delay'] + 4 * len(g['actions'])) / FPS, 1)
                if log:
                    log('[blatent] %s d=%02d %s: %d decisions, %.1f s game time (%.0f s wall)'
                        % (lvl0, g['delay'], 'WON' if g['won'] else g['reason'], g['decisions'], g['seconds'],
                           time.time() - t0))
                kept[k] = False
                admit(k)
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--levels', default='all')
    ap.add_argument('--delays', default='5,20,35,50')
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--parallel', type=int, default=32, help='trees growing together')
    ap.add_argument('--per-wave', type=int, default=32)
    ap.add_argument('--death-cost', type=float, default=HOPELESS)
    ap.add_argument('--max-depth', type=int, default=0, help="0: the model's training unroll")
    ap.add_argument('--raw', action='store_true', help='ignore the checkpoint calibration')
    ap.add_argument('--cap', type=float, default=1.5, help='most decisions, as a multiple of the route')
    ap.add_argument('--prior-net')
    ap.add_argument('--deep-prior', default='uniform', choices=('uniform', 'model'))
    ap.add_argument('--scale', type=float, default=32.0)
    ap.add_argument('--c-puct', type=float, default=1.5)
    ap.add_argument('--backup', default='self', choices=('self', 'children'))
    ap.add_argument('--real-events', action='store_true', help='oracle: deaths and finishes from the real game')
    ap.add_argument('--real-value', help='oracle: W from this relvalue.pt on the real screens')
    ap.add_argument('--real-prior', action='store_true', help='oracle: the net prior on the real screens below the root')
    ap.add_argument('--topk', type=int, default=12, help='below the root, expand only the topk moves under the prior')
    ap.add_argument('--no-floor', action='store_true', help='keep W below 0 (the C++ search does)')
    ap.add_argument('--reuse', action='store_true', help="keep the played move's subtree (the C++ search does); "
                    '--sims then counts new visits')
    ap.add_argument('--dedup', action='store_true', help="oracle: prune a child whose real state equals a brother's")
    ap.add_argument('--veto-net', help="a survival.py net: stage A's commit check, learned (no game touched)")
    ap.add_argument('--veto-p', type=float, default=0.5, help='the veto net refuses a move below this')
    ap.add_argument('--real-veto', type=int, default=0, help="oracle: stage A's commit check over this many steps")
    ap.add_argument('--out')
    a = ap.parse_args()
    s = Search(threads=4)
    segs = {g['level']: g for g in e2e_segments(s)}
    model, ck = load_model(a.model)
    model.eval()
    calib = None if a.raw else ck.get('calib')
    md = a.max_depth or int(ck.get('unroll', 12))
    pn = None
    if a.prior_net:
        from .net import load as load_net
        pn = load_net(a.prior_net)[0].eval()
    forest = Forest(model, a.parallel, max_nodes=(3 if a.reuse else 1) * a.sims + 2 * a.per_wave + 2, c_puct=a.c_puct, scale=a.scale,
                    death_cost=a.death_cost, calib=calib, max_depth=md, backup=a.backup, prior_net=pn,
                    deep_prior=a.deep_prior, per_wave=a.per_wave,
                    emu=s if (a.real_events or a.real_value or a.real_prior or a.dedup) else None, real_events=a.real_events,
                    real_prior=a.real_prior, dedup=a.dedup, floor=-1e9 if a.no_floor else 0.0, topk=a.topk,
                    real_value=load_rel(a.real_value)[0].cuda().eval() if a.real_value else None)
    levels = ROUTE if a.levels == 'all' else a.levels.split(',')
    delays = [int(x) for x in a.delays.split(',')]
    starts = [(l, d, s.frames(segs[l]['start'], d)) for l in levels for d in delays]
    caps = [int(a.cap * len(segs[l]['opt'])) for l, _, _ in starts]
    t0 = time.time()
    assert not (a.reuse and a.topk < 12), 'reroot does not carry the top-k masks'
    vnet = None
    if a.veto_net:
        from .survival import load as load_surv
        vnet = load_surv(a.veto_net)[0]
    games = play(s, forest, starts, a.sims, caps, log=lambda m: print(m, flush=True), veto=a.real_veto,
                 reuse=a.reuse, veto_net=vnet, veto_p=a.veto_p)
    for g in games:                                # how far through the level: the route is only the ruler
        seg = segs[g['level']]
        s.set_progress_route(ROUTE, seg['opt'], seg['start'])
        try:
            togo = s.progress_along(s.frames(seg['start'], g['delay']), ROUTE, seg['opt'],
                                    np.array(g['actions'], np.uint8), ref_start=seg['start'])
            g['progress'] = 1.0 if g['won'] else round(float(np.clip(1 - togo.min() / togo[0], 0, 1)), 3)
        except ValueError:
            g['progress'] = float('nan')
    for l in levels:
        gs = [g for g in games if g['level'] == l]
        print('[blatent] %s: won %d/%d  %s | mean %.0f%% of the level'
              % (l, sum(g['won'] for g in gs), len(gs),
                 ' '.join('%.1fs' % g['seconds'] if g['won'] else g['reason'].replace(' at %s' % l, '') for g in gs),
                 100 * np.nanmean([g['progress'] for g in gs])), flush=True)
    print('[blatent] %d/%d won, %d simulations, %.0f s wall, the game stepped only by the moves played'
          % (sum(g['won'] for g in games), len(games), a.sims, time.time() - t0), flush=True)
    if a.out:
        json.dump(games, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
