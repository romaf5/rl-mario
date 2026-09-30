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
    """self_mode: 0 children (b = the best child's), 1 self (min of the node's own and the best
    child's), 2 mean (AlphaZero's Q: the mean leaf value of the simulations through the node --
    the min over thousands of noisy leaves follows the luckiest error; a mean does not)."""
    if self_mode == 2:
        v = w[k, x]
        while x >= 0:
            n[k, x] += 1
            b[k, x] += (v - b[k, x]) / np.float32(n[k, x])
            x = parent[k, x]
        return
    while x >= 0:
        best, any_ = np.float32(1e30), False
        for a in range(12):
            c = child[k, x, a]
            if c >= 0 and term[k, c] != 3:
                any_ = True
                if b[k, c] < best:
                    best = b[k, c]
        if any_:
            b[k, x] = min(w[k, x], best) if self_mode == 1 else best
        n[k, x] += 1
        x = parent[k, x]


@numba.njit(cache=True)
def select_wave(active, budget, child, parent, depth, n, w, b, term, pend, prior, act_in, size,
                max_nodes, max_depth, fpu, c_puct, scale, per_wave, self_mode, out, allow, vl, use_vl):
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
                # use_vl: the C++ search's virtual loss -- a simulation in flight counts on every node
                # of its path, as a visit with q = 0, so one wave fans out over the brothers
                sq = np.sqrt(np.float32(n[k, x] + vl[k, x] + 1)) if use_vl else np.sqrt(np.float32(n[k, x]))
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
                        if use_vl and vl[k, c] > 0:
                            q = q * np.float32(n[k, c]) / np.float32(n[k, c] + vl[k, c])
                            nc = np.float32(n[k, c] + vl[k, c])
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
            vl[k, c] = 0
            if use_vl:
                y = x
                while y >= 0:
                    vl[k, y] += 1
                    y = parent[k, y]
            out[cnt, 0] = k
            out[cnt, 1] = x
            out[cnt, 2] = a
            out[cnt, 3] = c
            cnt += 1
            picks += 1
    return cnt


@numba.njit(cache=True)
def apply_wave(out, cnt, wv, p_dead, p_goal, pri, uniform, child, parent, n, w, b, term, pend, prior,
               dead_p, goal_p, death_cost, self_mode, dup, floor, allow, topk, v_death, vl, use_vl):
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
            w[k, c] = v_death
        elif p_goal[i] > goal_p:                # the model says it finished
            term[k, c] = 1
            w[k, c] = 0.0
        else:                                   # otherwise death is a price
            w[k, c] = min(max(wv[i], floor), np.float32(HOPELESS)) + p_dead[i] * death_cost
        b[k, c] = w[k, c]
    for i in range(cnt):                        # every pick imagined before any is backed up
        if not dup[i]:
            _backup_p(out[i, 0], out[i, 3], child, parent, b, w, n, self_mode, term)
        if use_vl:                              # the simulation has landed: its path is no longer in flight
            y = out[i, 1]
            while y >= 0:
                vl[out[i, 0], y] -= 1
                y = parent[out[i, 0], y]


@numba.njit(cache=True)
def reroot(k, r, child, parent, depth, n, w, b, term, prior, act_in, keys, size, order, decay, floor):
    """Keep node r's subtree as tree k, r first (breadth-first order, written to `order`: the
    old index of each kept node). Costs move into r's frame: W from r = W from the old root
    minus r's own W (a death stays a death), floored as new leaves are -- with no floor a line
    better than r's own estimate stays better (the C++ search keeps it; flooring here at 0 tied
    every such kept line at each move). Returns the number kept."""
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
        n[k, i] = max(int(n_t[o] * decay), 1) if i > 0 else n_t[o]
        term[k, i] = term_t[o]
        if term_t[o] >= 2 or w_t[o] >= HOPELESS:
            w[k, i] = w_t[o]
        else:
            w[k, i] = max(w_t[o] - delta, floor)
        b[k, i] = b_t[o] if b_t[o] >= HOPELESS else max(b_t[o] - delta, floor)
        for a in range(12):
            prior[k, i, a] = pri_t[o, a]
        act_in[k, i] = ai_t[o]
        keys[k, i] = key_t[o]
    size[k] = cnt
    return cnt


@numba.njit(cache=True)
def rebackup(k, cnt, child, w, b, term, self_mode, n):
    """b of nodes 0..cnt-1 of tree k from their (re-imagined) own values, leaves up: nodes are in
    breadth-first order, so every child comes after its parent. (Mean: the node's own value and
    its children's means, weighted by their visits.)"""
    for x in range(cnt - 1, -1, -1):
        if term[k, x] != 0:
            b[k, x] = w[k, x]
            continue
        if self_mode == 2:
            tot, cntv = w[k, x], np.float32(1.0)
            for a in range(12):
                c = child[k, x, a]
                if c >= 0 and term[k, c] != 3:
                    tot += b[k, c] * n[k, c]
                    cntv += n[k, c]
            b[k, x] = tot / cntv
            continue
        best, any_ = np.float32(1e30), False
        for a in range(12):
            c = child[k, x, a]
            if c >= 0 and term[k, c] != 3:
                any_ = True
                if b[k, c] < best:
                    best = b[k, c]
        if any_:
            b[k, x] = min(w[k, x], best) if self_mode == 1 else best
        else:
            b[k, x] = w[k, x]


class Forest:
    """K latent trees. Node arrays on the CPU (numba walks them), latents on the GPU."""
    def __init__(self, model, K, max_nodes=2048, c_puct=1.5, scale=32.0, fpu=0.5, dead_p=0.95, goal_p=0.9,
                 death_cost=HOPELESS, calib=None, max_depth=12, backup='self', prior_net=None,
                 deep_prior='uniform', per_wave=32, device='cuda', emu=None, real_events=False, real_value=None,
                 real_prior=False, dedup=False, floor=0.0, topk=12, value='w', noise=0.0, alpha=0.3, seed=0,
                 imagine=False, v_death=4096.0, vloss=False, real_doom=False, reuse_decay=1.0, fresh_votes=False):
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
        # value 'tg': MuZero's absolute value -- a leaf costs W = tg(leaf) + 4 depth - tg(root)
        self.value = value
        self.tg_root = np.zeros(K, np.float32)
        self.noise, self.alpha, self.rng = noise, alpha, np.random.default_rng(seed)
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
        # imagine: every expansion's frame drawn by the model's decoder; --real-value / --real-prior
        # then read imagined stacks (the real frames above the root, the drawn ones below it)
        self.imagine = imagine
        if imagine:
            assert emu is None and model.dec is not None, 'imagined frames need a model with a decoder, no emulator'
            self.frames = np.zeros((K, max_nodes, 84, 84), np.uint8)
            self.root_stack = np.zeros((K, 4, 84, 84), np.uint8)
            self.root_e = {}
        self.calib = None if calib is None else np.asarray(calib, np.float32)
        self.prior_net, self.uniform = prior_net, (prior_net is not None and deep_prior == 'uniform')
        self.params = dict(c_puct=np.float32(c_puct), scale=np.float32(scale), fpu=np.float32(fpu))
        self.dead_p, self.goal_p, self.death_cost = np.float32(dead_p), np.float32(goal_p), np.float32(death_cost)
        # A death must cost more than any living line. It cost HOPELESS (512) while a living leaf
        # costs up to W (<= 512) + p(dead) x death_cost (<= 1024): a line the model thought 90% fatal
        # looked worse than a certain death, and even with the real game's deaths a dead end tied the
        # worst living leaf -- the all-oracle search died 3/32 where the C++ search (4096) won 24.
        self.v_death = np.float32(v_death)
        self.use_vl = bool(vloss)
        # reuse_decay < 1: a kept subtree keeps its shape (the depth reuse buys) but its visits count
        # for less -- they were cast on the last screen's futures, which the model may now see otherwise
        self.reuse_decay = float(reuse_decay)
        # fresh_votes: the move is chosen by this decision's own visits -- the kept ones still steer
        # the search down the kept tree, but a line the new screen shows worse is not played on votes
        # cast before it was seen
        self.fresh_votes = fresh_votes
        self.n_kept = np.zeros((K, 12), np.int64)
        self.real_doom = real_doom
        self.vl = np.zeros((K, max_nodes), np.int32)
        self.max_depth, self.self_mode, self.per_wave = max_depth, {'children': 0, 'self': 1, 'mean': 2}[backup], per_wave
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
    def reset(self, ks, stacks, prevs, states=None, levels=None, keep=None, rams=None):
        """Trees ks from the real screens (len(ks), 4, 84, 84) and the moves before them (with an
        oracle, also the real states and level names). keep[i]: the tree was rerooted on this
        move and keeps its subtree -- the root itself is still encoded fresh from the screen."""
        ks = np.asarray(ks)
        keep = np.zeros(len(ks), bool) if keep is None else np.asarray(keep, bool)
        if self.emu is not None:
            for i, k in enumerate(ks):
                self.states[k][0] = states[i]
                self.lives0[k] = int(self.emu.ram(states[i])[0x75A])
                self.level[k] = levels[i]
        if self.emu is not None or self.imagine:
            for i, k in enumerate(ks):
                self.root_stack[k] = stacks[i]
            if self.real_value is not None:
                with torch.autocast('cuda', dtype=torch.float16):
                    e = self.real_value.embed(torch.from_numpy(np.ascontiguousarray(stacks)).to(self.dev))
                for i, k in enumerate(ks):
                    self.root_e[int(k)] = e[i]
        x = torch.from_numpy(np.ascontiguousarray(stacks)).to(self.dev)
        pv = torch.as_tensor(np.asarray(prevs, np.int64), device=self.dev)
        xin = torch.from_numpy(np.ascontiguousarray(rams)).to(self.dev) if self.m.ram_in else x
        with torch.autocast('cuda', dtype=torch.float16):
            s, pi, _ = self.m.initial(xin, pv)
            if self.prior_net is not None:
                pi, _ = self.prior_net(x)
        pri = torch.softmax(pi.float(), 1).cpu().numpy()
        if self.noise:                      # self-play explores: Dirichlet noise on the root's prior
            pri = (1 - self.noise) * pri + self.noise * self.rng.dirichlet([self.alpha] * 12, len(ks))
        if self.value == 'tg':
            with torch.autocast('cuda', dtype=torch.float16):
                self.tg_root[ks] = self.m.f.tgv(s).float().cpu().numpy()
        self.lat[torch.as_tensor(ks * self.N, device=self.dev)] = s.half()
        self.root_lat[torch.as_tensor(ks, device=self.dev)] = s.half()
        for i, k in enumerate(ks):
            kids = self.child[k, 0]
            self.n_kept[k] = [self.n[k, c] if keep[i] and c >= 0 else 0 for c in kids]
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

    @torch.no_grad()
    def reimagine(self, ks):
        """Kept subtrees, imagined again from the new root's real screen: every node's latent
        through the dynamics from its (re-imagined) parent, then its value, events and prior, and
        the best costs leaves up. The tree's shape and visits stay -- that is the depth reuse buys;
        the futures are the model's current ones, not those imagined from the last screen."""
        ks = [k for k in ks if self.size[k] > 1]
        if not ks:
            return
        dmax = max(int(self.depth[k, :self.size[k]].max()) for k in ks)
        for d in range(1, dmax + 1):
            kk, xx = [], []
            for k in ks:
                idx = np.nonzero(self.depth[k, :self.size[k]] == d)[0]
                idx = idx[self.term[k, idx] != 3]
                kk.append(np.full(len(idx), k)); xx.append(idx)
            kk, xx = np.concatenate(kk).astype(np.int64), np.concatenate(xx).astype(np.int64)
            if not len(xx):
                continue
            par = self.parent[kk, xx].astype(np.int64)
            src = torch.from_numpy(kk * self.N + par).to(self.dev)
            dst = torch.from_numpy(kk * self.N + xx).to(self.dev)
            a_t = torch.from_numpy(self.act_in[kk, xx]).to(self.dev)
            prev_t = torch.from_numpy(self.act_in[kk, par]).to(self.dev)
            with torch.autocast('cuda', dtype=torch.float16):
                s2, ev, _ = self.m.g(self.lat[src], a_t, prev_t)
                pi, w = self.m.f(s2, self.root_lat[torch.from_numpy(kk).to(self.dev)],
                                 torch.full((len(xx),), float(d), device=self.dev))
                if self.value == 'tg':
                    w = self.m.f.tgv(s2).float() + 4.0 * d - torch.from_numpy(self.tg_root[kk]).to(self.dev)
            self.lat[dst] = s2.half()
            lg = ev.float().cpu().numpy()
            if self.calib is not None:
                ki = min(d - 1, len(self.calib) - 1)
                lg = lg / self.calib[ki, :, 0] + self.calib[ki, :, 1]
            p = 1.0 / (1.0 + np.exp(-np.clip(lg, -30.0, 30.0)))
            wv = np.clip(w.float().cpu().numpy(), self.floor, HOPELESS) + p[:, 1] * self.death_cost
            pri = torch.softmax(pi.float(), 1).cpu().numpy()
            dead, goal = p[:, 1] > self.dead_p, p[:, 0] > self.goal_p
            # an oracle's part is the real game's and does not move with imagination: only the
            # model's parts are imagined again (the real values were moved into this root's frame)
            if self.real_events:
                dead, goal = self.term[kk, xx] == 2, self.term[kk, xx] == 1
            else:
                self.term[kk, xx] = np.where(dead, 2, np.where(goal, 1, 0))
            if self.real_value is not None:
                wv = self.w[kk, xx]
            self.w[kk, xx] = np.where(dead, self.v_death, np.where(goal, 0.0, wv)).astype(np.float32)
            if not self.real_prior:
                self.prior[kk, xx] = np.full(12, 1.0 / 12, np.float32) if self.uniform else pri
        for k in ks:
            rebackup(k, int(self.size[k]), self.child, self.w, self.b, self.term, self.self_mode, self.n)

    def advance(self, k, a):
        """After move a: keep that child's subtree for the next decision (True), or nothing."""
        c = int(self.child[k, 0, a])
        if c < 0 or self.term[k, c] != 0 or self.pend[k, c]:
            return False
        order = np.zeros(self.size[k], np.int32)
        cnt = reroot(k, c, self.child, self.parent, self.depth, self.n, self.w, self.b, self.term, self.prior,
                     self.act_in, self.keys, self.size, order, np.float32(self.reuse_decay), self.floor)
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
                              self.self_mode, self.out, self.allow, self.vl, self.use_vl)
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
            if self.value == 'tg':
                with torch.autocast('cuda', dtype=torch.float16):
                    tgl = self.m.f.tgv(s2).float().cpu().numpy()
                w = torch.from_numpy(tgl + 4.0 * dep_np.astype(np.float32) - self.tg_root[kk])
            self.lat[dst] = s2.half()
            if self.imagine:
                with torch.autocast('cuda', dtype=torch.float16):
                    fr = self.m.dec(s2)
                self.frames[kk, ch] = fr.float().round().clamp(0, 255).to(torch.uint8).cpu().numpy()
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
            if self.emu is not None or self.imagine:
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
                       self.death_cost, self.self_mode, dup, self.floor, self.allow, self.topk, self.v_death,
                       self.vl, self.use_vl)

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
            doomed = False
            if self.real_doom and not (tr[-1, 6] in (0x0B, 0x06)):
                # doom, not death: the state is lost if no move then held input lasts 24 steps
                doomed = not any(survives(e, st2, x, 24) for x in range(12))
            key = 0
            if self.dedup:                      # emu.h exact_key: all game-state RAM
                m = e.ram(st2)
                m[0:8] = 0; m[9] = 0; m[0x100:0x300] = 0
                m[0x7DD:0x7E3] = 0; m[0x7ED] = m[0x7EE] = 0; m[0x7F8:0x7FB] = 0
                key = hash(m.tobytes())
            return obs, tr, st2, key, doomed

        for i, (obs, tr, st2, key, doomed) in enumerate(self.pool.map(step, range(len(kk)))):
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
                  or (tr[-1, 6] == 0x08 and mode == 1 and tr[-1, 1] >= 512) or doomed):
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
        """Most visited root move, ties to the best cost (latent.py's choose without the veto) --
        never a move known to die while one is not: with reuse a kept child can hold thousands of
        visits cast before the new screen showed it dead (the C++ search never has such a child:
        the emulator tells death when the child is made, before any visit)."""
        kids = self.child[k, 0]
        n = self.visits(k).astype(np.int64)
        b = np.array([self.b[k, c] if c >= 0 else np.inf for c in kids])
        dead = np.array([c >= 0 and self.term[k, c] == 2 for c in kids])
        live = (n > 0) & ~dead
        if live.any():
            n = np.where(live, n, -1)
        elif not n.sum():
            return 1
        return int(np.lexsort((np.where(np.isinf(b), 1e9, b), -n))[0])

    def root_value(self, k):
        """The search's frames to go at the root: the model's own guess plus the best line found."""
        return float(max(self.tg_root[k] + self.b[k, 0], 0.0))

    def lines(self, k, n, rng):
        """n root-to-leaf lines of tree k, each walked down by visit counts (a child drawn in
        proportion to its visits): the futures the search spent its simulations imagining."""
        out = []
        for _ in range(n):
            x, line = 0, []
            while True:
                kids = [(a, c) for a, c in enumerate(self.child[k, x]) if c >= 0 and self.term[k, c] != 3]
                if not kids:
                    break
                v = np.array([max(int(self.n[k, c]), 1) for _, c in kids], np.float64)
                a, c = kids[int(rng.choice(len(kids), p=v / v.sum()))]
                line.append(int(a))
                x = c
            if line:
                out.append(line)
        return out

    def visits(self, k):
        """Root visits by move (with fresh_votes, only this decision's)."""
        kids = self.child[k, 0]
        n = np.array([self.n[k, c] if c >= 0 else 0 for c in kids], np.int64)
        if self.fresh_votes:
            n = np.maximum(n - self.n_kept[k], 0)
        return n.astype(np.int32)


def survives(s, state, a, horizon):
    """Does move a, then some held input, last `horizon` steps in the real game (search rule)?"""
    for c in range(12):
        acts = np.array([a] + [c] * (horizon - 1), np.uint8)
        out, n = s.classify_along(state, ROUTE, acts)
        if not n or np.asarray(out[:n])[-1] != 2:
            return True
    return False


def play(s, forest, starts, sims, caps, log=None, veto=0, reuse=False, veto_net=None, veto_p=0.5,
         reimagine=False,
         temp=0.0, record=False, lines=0):
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
                     keep=[kept[k] for k in ks],
                     rams=np.stack([s.ram(state[slot[k]]) for k in ks]) if forest.m.ram_in else None)
        if reimagine:
            forest.reimagine([k for k in ks if kept[k]])
        forest.run(ks, sims)
        picks = {k: forest.choose(k) for k in ks}
        if temp > 0:                                # self-play: a move drawn from the visit counts
            for k in ks:
                n = forest.visits(k).astype(np.float64)
                if n.sum() > 0:
                    p = n ** (1.0 / temp); p /= p.sum()
                    picks[k] = int(forest.rng.choice(12, p=p))
        if lines:                                   # treedata: the lines each tree imagined
            for k in ks:
                g = games[slot[k]]
                g.setdefault('lines', []).append([len(g['actions']), forest.lines(k, lines, forest.rng)])
        if record:
            for k in ks:
                g = games[slot[k]]
                g.setdefault('visits', []).append(forest.visits(k).tolist())
                g.setdefault('root_tg', []).append(round(forest.root_value(k), 1))
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
    ap.add_argument('--real-doom', action='store_true', help="with --real-events: a node is dead once its state is "
                    "doomed (no move then held input survives 24 steps), not when the death registers")
    ap.add_argument('--vloss', action='store_true', help="the C++ search's virtual loss: a simulation in flight "
                    "counts on its path (q x n/(n+pending)), so a wave fans out; use with --per-wave 128")
    ap.add_argument('--v-death', type=float, default=4096.0, help='what a node known to die costs (the C++ search: '
                    '4096); must exceed any living leaf, HOPELESS + death cost')
    ap.add_argument('--max-depth', type=int, default=0, help="0: the model's training unroll")
    ap.add_argument('--raw', action='store_true', help='ignore the checkpoint calibration')
    ap.add_argument('--cap', type=float, default=1.5, help='most decisions, as a multiple of the route')
    ap.add_argument('--prior-net')
    ap.add_argument('--deep-prior', default='uniform', choices=('uniform', 'model'))
    ap.add_argument('--scale', type=float, default=32.0)
    ap.add_argument('--c-puct', type=float, default=1.5)
    ap.add_argument('--backup', default='self', choices=('self', 'children', 'mean'))
    ap.add_argument('--real-events', action='store_true', help='oracle: deaths and finishes from the real game')
    ap.add_argument('--real-value', help='oracle: W from this relvalue.pt on the real screens')
    ap.add_argument('--real-prior', action='store_true', help='oracle: the net prior on the real screens below the root')
    ap.add_argument('--value', default='w', choices=('w', 'tg'), help="the leaf's price: the W head, or "
                    "MuZero's frames to go (W = tg(leaf) + 4 depth - tg(root))")
    ap.add_argument('--noise', type=float, default=0.0, help='self-play: Dirichlet noise share on the root prior')
    ap.add_argument('--temp', type=float, default=0.0, help='self-play: draw moves from visits^(1/temp) (0: best)')
    ap.add_argument('--record', action='store_true', help='keep each decision\'s visits and root value in --out')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--topk', type=int, default=12, help='below the root, expand only the topk moves under the prior')
    ap.add_argument('--no-floor', action='store_true', help='keep W below 0 (the C++ search does)')
    ap.add_argument('--reimagine', action='store_true', help='with --reuse: imagine the kept subtree again '
                    'from the new real screen every decision (its visits kept, its futures fresh)')
    ap.add_argument('--reuse-decay', type=float, default=1.0, help="with --reuse: kept visits x this (0: keep the "
                    "tree's shape, not its votes)")
    ap.add_argument('--fresh-votes', action='store_true', help="with --reuse: the move by this decision's visits, "
                    "not the kept ones")
    ap.add_argument('--max-nodes', type=int, default=0, help='nodes per tree (0: 3x --sims with --reuse, else --sims)')
    ap.add_argument('--reuse', action='store_true', help="keep the played move's subtree (the C++ search does); "
                    '--sims then counts new visits')
    ap.add_argument('--dedup', action='store_true', help="oracle: prune a child whose real state equals a brother's")
    ap.add_argument('--veto-net', help="a survival.py net: stage A's commit check, learned (no game touched)")
    ap.add_argument('--veto-p', type=float, default=0.5, help='the veto net refuses a move below this')
    ap.add_argument('--imagine', action='store_true', help="the model draws every expanded frame; "
                    "--real-value / --real-prior then judge imagined stacks (no emulator)")
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
    # A kept subtree fills the tree: at 3x the simulations, 8-3's kept trees filled all 966 nodes and
    # the search got almost no new simulations (the C++ search keeps 32768 and its root grows past 2000)
    forest = Forest(model, a.parallel, max_nodes=a.max_nodes or ((3 if a.reuse else 1) * a.sims + 2 * a.per_wave + 2), c_puct=a.c_puct, scale=a.scale,
                    death_cost=a.death_cost, calib=calib, max_depth=md, backup=a.backup, prior_net=pn,
                    deep_prior=a.deep_prior, per_wave=a.per_wave,
                    emu=s if (a.real_events or a.dedup or ((a.real_value or a.real_prior) and not a.imagine)) else None, real_events=a.real_events,
                    real_prior=a.real_prior, dedup=a.dedup, floor=-1e9 if a.no_floor else 0.0, topk=a.topk,
                    value=a.value, noise=a.noise, seed=a.seed, imagine=a.imagine, v_death=a.v_death, vloss=a.vloss,
                    real_doom=a.real_doom, reuse_decay=a.reuse_decay, fresh_votes=a.fresh_votes,
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
                 reuse=a.reuse, reimagine=a.reimagine, veto_net=vnet, veto_p=a.veto_p, temp=a.temp, record=a.record)
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
