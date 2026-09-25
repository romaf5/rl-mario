"""A search that never touches the game to think.

The screen is encoded once per decision; from there the tree is grown inside the world model:
the dynamics imagines each step, the prediction head scores it, and the event heads say when a
line has died or finished. The real game is stepped only by the move actually played -- four
frames per decision, the same as a person at a controller.

Scores are W, the frames a node has thrown away since the root, so a node's value already
includes its depth and the backup is a plain minimum over children. Every decision re-encodes
the real screen, so the model never has to stay honest for longer than one lookahead.

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
                 dead_p=0.5, goal_p=0.5, device='cuda'):
        self.m, self.dev = model, device
        self.max_nodes, self.c_puct, self.scale, self.fpu = max_nodes, c_puct, scale, fpu
        self.dead_p, self.goal_p = dead_p, goal_p
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
        """Walk down by PUCT to an edge with no child. Returns (node, action) or None."""
        x = 0
        while True:
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
                return None                                # a settled line: nothing to expand
            x = int(c)

    def _backup(self, node):
        x = node
        while x >= 0:
            kids = self.child[x][self.child[x] >= 0]
            if len(kids):
                self.b[x] = min(self.w[x], self.b[kids].min())
            self.n[x] += 1
            x = int(self.parent[x])

    @torch.no_grad()
    def run(self, sims, per_wave=32):
        done = 0
        while done < sims and self.size < self.max_nodes - per_wave:
            picks = []
            for _ in range(min(per_wave, sims - done)):
                p = self._select()
                if p is None:
                    break
                x, a = p
                c = self.size
                self.size += 1
                self.child[x, a] = c
                self.parent[c] = x
                self.depth[c] = self.depth[x] + 1
                self.n[c] = 0
                self.b[c] = self.w[c] = 0.0
                picks.append((x, a, c))
            if not picks:
                break
            src = torch.tensor([p[0] for p in picks], device=self.dev)
            act = torch.tensor([p[1] for p in picks], device=self.dev)
            dep = torch.tensor([float(self.depth[p[2]]) for p in picks], device=self.dev)
            with torch.autocast('cuda', dtype=torch.float16):
                s2, ev, _ = self.m.g(self.lat[src], act)
                pi, w = self.m.f(s2, self.root_lat.expand(len(picks), -1, -1, -1), dep)
            p_ev = ev.float().sigmoid().cpu().numpy()
            w = w.float().cpu().numpy()
            pri = torch.softmax(pi.float(), 1)
            for i, (x, a, c) in enumerate(picks):
                self.lat[c] = s2[i].float()
                self.prior[c] = pri[i]
                if p_ev[i, 1] > self.dead_p:                       # the model says this line dies
                    self.term[c] = 2; self.w[c] = HOPELESS
                elif p_ev[i, 0] > self.goal_p:                     # the model says it finished
                    self.term[c] = 1; self.w[c] = 0.0
                else:
                    self.w[c] = float(np.clip(w[i], 0.0, HOPELESS))
                self.b[c] = self.w[c]
                self._backup(c)
                done += 1

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
    start_level = s.level(state)
    for _ in range(max_decisions):
        tree.reset(stack)
        tree.run(sims, per_wave)
        n, b = tree.visits()
        a = int(np.lexsort((np.where(np.isinf(b), 1e9, b), -n))[0]) if n.sum() else 1
        obs, tr, state = s.replay_obs(state, np.array([a], np.uint8))
        acts.append(a)
        stack = np.concatenate([stack[1:], obs])
        lvl, mode = int(tr[-1, 2]), int(tr[-1, 7])
        if lvl != gp(start_level) or mode == 2:
            k = ROUTE.index(start_level)
            won = mode == 2 if k == len(ROUTE) - 1 else lvl == gp(ROUTE[k + 1])
            return acts, ('goal' if won else 'dead at %s' % start_level), won
        if tr[-1, 6] in (0x0B, 0x06) or tr[-1, 9] < s.ram(start)[0x75A]:
            return acts, 'dead at %s' % start_level, False
    return acts, reason, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--level', default='1-1')
    ap.add_argument('--delays', default='5,20,35,50')
    ap.add_argument('--sims', type=int, default=600)
    ap.add_argument('--per-wave', type=int, default=32)
    ap.add_argument('--max-nodes', type=int, default=4096)
    ap.add_argument('--out')
    a = ap.parse_args()
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    model, _ = load_model(a.model)
    model.eval()
    tree = LatentTree(model, max_nodes=a.max_nodes)
    cap = int(2.5 * len(segs[a.level]['opt']))
    res = []
    for d in [int(x) for x in a.delays.split(',')]:
        t0 = time.time()
        start = s.frames(segs[a.level]['start'], d)
        acts, reason, won = play(s, tree, start, ROUTE, a.sims, cap, a.per_wave)
        res.append(dict(delay=d, won=won, reason=reason, decisions=len(acts),
                        seconds=round((d + 4 * len(acts)) / FPS, 1), wall=round(time.time() - t0)))
        print('[latent] %s d=%02d %s: %d decisions, %.1f s game time, %.0f ms per decision'
              % (a.level, d, 'WON' if won else reason, len(acts), res[-1]['seconds'],
                 1000 * (time.time() - t0) / max(len(acts), 1)), flush=True)
    print('[latent] %s: won %d/%d, the game stepped only by the moves played'
          % (a.level, sum(r['won'] for r in res), len(res)))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
