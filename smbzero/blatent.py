"""The latent search, many games at once.

latent.py grows one tree per decision and walks it in Python: ~1.1 s per decision, a gate of
32 games in an hour and a half, and far too slow to play the thousands of games a MuZero loop
learns from. Here K trees grow together: the walk down is compiled (numba), and each wave's
picks from every tree are imagined in one call to the model. The rules are latent.py's
exactly -- PUCT on W, death as a price, calibrated event heads, the depth bound, pending nodes
unselectable until imagined -- so a game played here is a game latent.py would play.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.blatent --model smbzero/runs/wm13/wm.pt \
      --prior-net smbzero/runs/zero8/net.pt --deep-prior model --levels all --delays 5,20,35,50
"""
import argparse, json, os, time
import numpy as np
import numba
import torch
from .common import FPS, ROUTE, RUNS, Search, e2e_segments, gp
from .model import NO_PREV, load as load_model

HOPELESS = 512.0


@numba.njit(cache=True)
def _backup_p(k, x, child, parent, b, w, n, self_mode):
    while x >= 0:
        best, any_ = np.float32(1e30), False
        for a in range(12):
            c = child[k, x, a]
            if c >= 0:
                any_ = True
                if b[k, c] < best:
                    best = b[k, c]
        if any_:
            b[k, x] = min(w[k, x], best) if self_mode else best
        n[k, x] += 1
        x = parent[k, x]


@numba.njit(cache=True)
def select_wave(active, budget, child, parent, depth, n, w, b, term, pend, prior, act_in, size,
                max_nodes, max_depth, fpu, c_puct, scale, per_wave, self_mode, out):
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
                    if c < 0:
                        all_pend = False
                    elif not pend[k, c]:
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
                    if c >= 0 and pend[k, c]:
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
                _backup_p(k, revisit, child, parent, b, w, n, self_mode)
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
               dead_p, goal_p, death_cost, self_mode):
    for i in range(cnt):
        k, c = out[i, 0], out[i, 3]
        pend[k, c] = False
        for j in range(12):
            prior[k, c, j] = np.float32(1.0 / 12) if uniform else pri[i, j]
        if p_dead[i] > dead_p:                  # certain enough to stop looking
            term[k, c] = 2
            w[k, c] = HOPELESS
        elif p_goal[i] > goal_p:                # the model says it finished
            term[k, c] = 1
            w[k, c] = 0.0
        else:                                   # otherwise death is a price
            w[k, c] = min(max(wv[i], np.float32(0.0)), np.float32(HOPELESS)) + p_dead[i] * death_cost
        b[k, c] = w[k, c]
    for i in range(cnt):                        # every pick imagined before any is backed up
        _backup_p(out[i, 0], out[i, 3], child, parent, b, w, n, self_mode)


class Forest:
    """K latent trees. Node arrays on the CPU (numba walks them), latents on the GPU."""
    def __init__(self, model, K, max_nodes=2048, c_puct=1.5, scale=32.0, fpu=0.5, dead_p=0.95, goal_p=0.9,
                 death_cost=HOPELESS, calib=None, max_depth=12, backup='self', prior_net=None,
                 deep_prior='uniform', per_wave=32, device='cuda'):
        self.m, self.dev, self.K, self.N = model, device, K, max_nodes
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
    def reset(self, ks, stacks, prevs):
        """Fresh trees ks from the real screens (len(ks), 4, 84, 84) and the moves before them."""
        ks = np.asarray(ks)
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
            self.child[k, 0] = -1
            self.parent[k, 0] = -1
            self.depth[k, 0] = 0
            self.n[k, 0] = 1
            self.w[k, 0] = self.b[k, 0] = 0.0
            self.term[k, 0] = 0
            self.pend[k, 0] = False
            self.prior[k, 0] = pri[i]
            self.act_in[k, 0] = prevs[i]
            self.size[k] = 1

    @torch.no_grad()
    def run(self, ks, sims):
        active = np.zeros(self.K, np.bool_)
        active[np.asarray(ks)] = True
        budget = np.where(active, sims, 0).astype(np.int32)
        while True:
            cnt = select_wave(active, budget, self.child, self.parent, self.depth, self.n, self.w, self.b,
                              self.term, self.pend, self.prior, self.act_in, self.size, self.N, self.max_depth,
                              self.params['fpu'], self.params['c_puct'], self.params['scale'], self.per_wave,
                              self.self_mode, self.out)
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
            apply_wave(self.out, cnt, w.float().cpu().numpy().astype(np.float32), np.ascontiguousarray(p_ev[:, 1]),
                       np.ascontiguousarray(p_ev[:, 0]), pri, self.uniform, self.child, self.parent, self.n,
                       self.w, self.b, self.term, self.pend, self.prior, self.dead_p, self.goal_p,
                       self.death_cost, self.self_mode)

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


def play(s, forest, starts, sims, caps, log=None):
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
    while any(x is not None for x in slot):
        ks = [k for k in range(K) if slot[k] is not None]
        forest.reset(ks, np.stack([stack[slot[k]] for k in ks]),
                     [games[slot[k]]['actions'][-1] if games[slot[k]]['actions'] else NO_PREV for k in ks])
        forest.run(ks, sims)
        for k in ks:
            i = slot[k]
            g = games[i]
            a = forest.choose(k)
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
    forest = Forest(model, a.parallel, max_nodes=a.sims + 2 * a.per_wave + 2, c_puct=a.c_puct, scale=a.scale,
                    death_cost=a.death_cost, calib=calib, max_depth=md, backup=a.backup, prior_net=pn,
                    deep_prior=a.deep_prior, per_wave=a.per_wave)
    levels = ROUTE if a.levels == 'all' else a.levels.split(',')
    delays = [int(x) for x in a.delays.split(',')]
    starts = [(l, d, s.frames(segs[l]['start'], d)) for l in levels for d in delays]
    caps = [int(a.cap * len(segs[l]['opt'])) for l, _, _ in starts]
    t0 = time.time()
    games = play(s, forest, starts, a.sims, caps, log=lambda m: print(m, flush=True))
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
