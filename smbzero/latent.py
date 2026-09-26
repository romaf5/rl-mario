"""A search that never touches the game to think.

The screen is encoded once per decision; from there the tree is grown inside the world model:
the dynamics imagines each step, the prediction head scores it, and the event heads say when a
line has died or finished. The real game is stepped only by the move actually played -- four
frames per decision, the same as a person at a controller.

Scores are W, the frames a node has thrown away since the root, so a node's value already
includes its depth and the backup is a plain minimum over children. The tree is not allowed
to grow deeper than the model was trained to unroll: at 1000 simulations PUCT drives a line
31 steps long, and a model honest to 12 steps will happily price a 31-step fantasy as cheap. Every decision re-encodes
the real screen, so the model never has to stay honest for longer than one lookahead.

A model that is sure a line dies is rare; one that is 30% worried is common. So death is not
a verdict here but a price: a node costs W + p(dead) x 512 frames. That only works if p is a
probability and not a mood -- the event heads are trained with a positive weight of up to 50
and say 0.82 where they are right 0.39 of the time. `tools.wmcal` fits a temperature and bias
per head per depth, and the price is charged on the calibrated number.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.latent --model smbzero/runs/wm3/wm.pt \
      --level 1-1 --delays 5,20 --sims 600
"""
import argparse, json, os, time
import numpy as np
import torch
from .common import FPS, MAX_DELAY, ROUTE, RUNS, THREADS, Search, e2e_segments, gp, level_name, load_state
from .model import load as load_model

HOPELESS = 512.0


class LatentTree:
    """One tree, grown in the model. Nodes live in tensors; a wave expands many leaves at once."""
    def __init__(self, model, max_nodes=4096, c_puct=1.5, scale=32.0, fpu=0.5,
                 dead_p=0.95, goal_p=0.9, death_cost=HOPELESS, calib=None, max_depth=12,
                 backup='self', device='cuda'):
        self.m, self.dev = model, device
        self.calib = calib                    # (K, 3, 2): temperature and bias per depth per head
        self.max_depth = max_depth            # as far as the model was trained to imagine
        # 'self': b = min(own estimate, children) -- the node's own first guess shields it from
        # everything found below it, so a move that dies two steps later still looks as cheap
        # as its first step. 'children': once expanded, a node is worth its best child, as in
        # the C++ search (b = 4 + min over children; W here already counts the steps).
        self.backup_mode = backup
        self.max_nodes, self.c_puct, self.scale, self.fpu = max_nodes, c_puct, scale, fpu
        self.dead_p, self.goal_p, self.death_cost = dead_p, goal_p, death_cost
        c = model.g.conv.out_channels
        self.lat = torch.zeros((max_nodes, c, 11, 11), device=device)
        self.prior = torch.zeros((max_nodes, 12), device=device)
        self.w = np.zeros(max_nodes, np.float32)          # the node's own estimate
        self.b = np.zeros(max_nodes, np.float32)          # best found below it
        self.n = np.zeros(max_nodes, np.int32)
        self.child = np.full((max_nodes, 12), -1, np.int32)
        self.parent = np.full(max_nodes, -1, np.int32)
        self.depth = np.zeros(max_nodes, np.int32)
        self.term = np.zeros(max_nodes, np.uint8)         # 1 goal, 2 dead
        self.size = 0

    @torch.no_grad()
    def reset(self, stack):
        """stack: (4, 84, 84) uint8 -- the real screen, encoded fresh every decision."""
        self.child[:] = -1; self.parent[:] = -1; self.n[:] = 0; self.term[:] = 0
        self.depth[:] = 0
        x = torch.from_numpy(stack[None]).to(self.dev)
        with torch.autocast('cuda', dtype=torch.float16):
            s, pi, w = self.m.initial(x)
        self.lat[0] = s[0].float()
        self.prior[0] = torch.softmax(pi.float(), 1)[0]
        self.root_lat = self.lat[0:1].clone()
        self.w[0] = self.b[0] = 0.0                       # the root wastes nothing against itself
        self.n[0] = 1
        self.size = 1

    def _select(self):
        """Walk down by PUCT. Returns (node, action) for an edge to expand, or (None, c) for a
        child already settled -- visiting it again is what lets PUCT divert to its brothers."""
        x = 0
        while True:
            if self.depth[x] >= self.max_depth:
                return None, x             # as deep as the model is honest: widen, do not dream
            kids = self.child[x]
            made = kids >= 0
            bstar = self.b[kids[made]].min() if made.any() else 0.0
            q = np.full(12, self.fpu)
            nc = np.zeros(12)
            if made.any():
                bb = self.b[kids[made]]
                q[made] = np.clip(1.0 - (bb - bstar) / self.scale, 0.0, 1.0)
                q[made] = np.where(self.term[kids[made]] == 2, 0.0, q[made])
                nc[made] = self.n[kids[made]]
            score = q + self.c_puct * self.prior[x].cpu().numpy() * np.sqrt(self.n[x]) / (1 + nc)
            a = int(np.argmax(score))
            c = self.child[x, a]
            if c < 0:
                return x, a
            if self.term[c]:
                return None, int(c)                        # settled: count the visit, look elsewhere
            x = int(c)

    def _backup(self, node):
        x = node
        while x >= 0:
            kids = self.child[x][self.child[x] >= 0]
            if len(kids):
                best = self.b[kids].min()
                self.b[x] = best if self.backup_mode == 'children' else min(self.w[x], best)
            self.n[x] += 1
            x = int(self.parent[x])

    @torch.no_grad()
    def run(self, sims, per_wave=32):
        done = 0
        while done < sims and self.size < self.max_nodes - per_wave:
            picks, revisits = [], 0
            for _ in range(min(per_wave, sims - done)):
                x, a = self._select()
                if x is None:                  # a settled child: its visit raises the brothers' pull
                    self._backup(a)
                    done += 1; revisits += 1
                    continue
                c = self.size
                self.size += 1
                self.child[x, a] = c
                self.parent[c] = x
                self.depth[c] = self.depth[x] + 1
                self.n[c] = 0
                self.b[c] = self.w[c] = 0.0
                picks.append((x, a, c))
            if not picks:
                if not revisits:
                    break
                continue
            src = torch.tensor([p[0] for p in picks], device=self.dev)
            act = torch.tensor([p[1] for p in picks], device=self.dev)
            dep = torch.tensor([float(self.depth[p[2]]) for p in picks], device=self.dev)
            with torch.autocast('cuda', dtype=torch.float16):
                s2, ev, _ = self.m.g(self.lat[src], act)
                pi, w = self.m.f(s2, self.root_lat.expand(len(picks), -1, -1, -1), dep)
            lg = ev.float().cpu().numpy()
            if self.calib is not None:        # a price is only fair if the probability is honest
                k = np.clip(np.array([self.depth[q[2]] for q in picks]) - 1, 0, len(self.calib) - 1)
                lg = lg / self.calib[k, :, 0] + self.calib[k, :, 1]
            p_ev = 1.0 / (1.0 + np.exp(-np.clip(lg, -30.0, 30.0)))
            w = w.float().cpu().numpy()
            pri = torch.softmax(pi.float(), 1)
            for i, (x, a, c) in enumerate(picks):
                self.lat[c] = s2[i].float()
                self.prior[c] = pri[i]
                if p_ev[i, 1] > self.dead_p:                       # certain enough to stop looking
                    self.term[c] = 2; self.w[c] = HOPELESS
                elif p_ev[i, 0] > self.goal_p:                     # the model says it finished
                    self.term[c] = 1; self.w[c] = 0.0
                else:                                              # otherwise death is a price
                    self.w[c] = float(np.clip(w[i], 0.0, HOPELESS) + p_ev[i, 1] * self.death_cost)
                self.b[c] = self.w[c]
                self._backup(c)
                done += 1

    def pv(self):
        """How deep the line the search actually believes in goes, and how deep the tree got."""
        x, d = 0, 0
        while True:
            kids = self.child[x][self.child[x] >= 0]
            if not len(kids):
                return d, int(self.depth[:self.size].max())
            x = int(kids[np.argmin(self.b[kids])]); d += 1

    def visits(self):
        kids = self.child[0]
        return np.array([self.n[c] if c >= 0 else 0 for c in kids]), \
               np.array([self.b[c] if c >= 0 else np.inf for c in kids])


def play(search_engine, tree, start, route, sims, max_decisions, per_wave=32):
    """One game. The game is stepped only by the move chosen -- nothing else."""
    s = search_engine
    state = start
    stack = np.repeat(s.obs(state)[None], 4, 0)
    acts, reason = [], 'too long'
    pvd, maxd = [], []
    start_level = s.level(state)
    for _ in range(max_decisions):
        tree.reset(stack)
        tree.run(sims, per_wave)
        p, m = tree.pv()
        pvd.append(p); maxd.append(m)
        n, b = tree.visits()
        a = int(np.lexsort((np.where(np.isinf(b), 1e9, b), -n))[0]) if n.sum() else 1
        obs, tr, state = s.replay_obs(state, np.array([a], np.uint8))
        acts.append(a)
        stack = np.concatenate([stack[1:], obs])
        lvl, mode = int(tr[-1, 2]), int(tr[-1, 7])
        if lvl != gp(start_level) or mode == 2:
            k = ROUTE.index(start_level)
            won = mode == 2 if k == len(ROUTE) - 1 else lvl == gp(ROUTE[k + 1])
            return acts, ('goal' if won else 'dead at %s' % start_level), won, (pvd, maxd)
        if tr[-1, 6] in (0x0B, 0x06) or tr[-1, 9] < s.ram(start)[0x75A]:
            return acts, 'dead at %s' % start_level, False, (pvd, maxd)
    return acts, reason, False, (pvd, maxd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--level', default='1-1')
    ap.add_argument('--delays', default='5,20,35,50')
    ap.add_argument('--sims', type=int, default=600)
    ap.add_argument('--per-wave', type=int, default=32)
    ap.add_argument('--max-nodes', type=int, default=4096)
    ap.add_argument('--death-cost', type=float, default=HOPELESS, help='frames charged per unit of p(dead)')
    ap.add_argument('--max-depth', type=int, default=0,
                    help="deepest node the tree may grow (0: the model's training unroll)")
    ap.add_argument('--raw', action='store_true', help='ignore the checkpoint calibration')
    ap.add_argument('--cap', type=float, default=2.5, help='most decisions, as a multiple of the route')
    ap.add_argument('--backup', default='self', choices=('self', 'children'),
                    help="'children': an expanded node is worth its best child, not min(itself, them)")
    ap.add_argument('--out')
    a = ap.parse_args()
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    model, ck = load_model(a.model)
    model.eval()
    calib = None if a.raw else ck.get('calib')
    if calib is None and not a.raw:
        print('[latent] no calibration in the checkpoint: run tools.wmcal first', flush=True)
    md = a.max_depth or int(ck.get('unroll', 12))
    print('[latent] tree may grow %d deep (the model was trained to unroll %s)'
          % (md, ck.get('unroll', 'unknown')), flush=True)
    tree = LatentTree(model, max_nodes=a.max_nodes, death_cost=a.death_cost, calib=calib,
                      max_depth=md, backup=a.backup)
    cap = int(a.cap * len(segs[a.level]['opt']))
    opt, ref_start = segs[a.level]['opt'], segs[a.level]['start']
    s.set_progress_route(ROUTE, opt, ref_start)
    res = []
    for d in [int(x) for x in a.delays.split(',')]:
        t0 = time.time()
        start = s.frames(segs[a.level]['start'], d)
        acts, reason, won, (pvd, maxd) = play(s, tree, start, ROUTE, a.sims, cap, a.per_wave)
        res.append(dict(delay=d, won=won, reason=reason, decisions=len(acts), actions=[int(x) for x in acts],
                        seconds=round((d + 4 * len(acts)) / FPS, 1), wall=round(time.time() - t0)))
        res[-1].update(pv=round(float(np.mean(pvd)), 1), max_depth=int(np.max(maxd)) if maxd else 0)
        # How far through the level: one death ends a game, so won/lost on a few starts cannot
        # rank two models (one model went 197 decisions on one start and 1500 on another).
        # The route is only the ruler here.
        try:
            togo = s.progress_along(start, ROUTE, opt, np.array(acts, np.uint8), ref_start=ref_start)
            res[-1]['progress'] = 1.0 if won else round(float(np.clip(1 - togo.min() / togo[0], 0, 1)), 3)
        except ValueError:
            res[-1]['progress'] = float('nan')
        print('[latent] %s d=%02d %s: %d decisions, %.1f s game time, %.0f ms per decision, '
              'believed line %.1f deep (tree reaches %d), %.0f%% of the level'
              % (a.level, d, 'WON' if won else reason, len(acts), res[-1]['seconds'],
                 1000 * (time.time() - t0) / max(len(acts), 1), res[-1]['pv'],
                 res[-1]['max_depth'], 100 * res[-1]['progress']), flush=True)
    pr = [r['progress'] for r in res if r['progress'] == r['progress']]
    print('[latent] %s: won %d/%d, mean %.0f%% of the level (min %.0f%%, max %.0f%%), '
          'the game stepped only by the moves played'
          % (a.level, sum(r['won'] for r in res), len(res), 100 * np.mean(pr) if pr else 0,
             100 * min(pr) if pr else 0, 100 * max(pr) if pr else 0))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
