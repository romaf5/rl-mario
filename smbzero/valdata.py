"""Training data for a learned value: leaves of the agent's own searches, labelled with the
route's frames to go relative to their tree's root.

Why relative: the search ranks a node by b, and PUCT's q compares a node with its siblings,
so a constant per tree cancels -- only differences within one tree matter. Absolute frames
to go are not on a cropped 84x84 screen (1-1 repeats pipes and hills: the old value head
ordered nearby states at chance), but "how much further along than the root" is.

Each sample: the root's 4 frames, the leaf's 4 frames, the depth from the root, and
    D = route(leaf) - route(root)      (frames; negative = closer to the goal than the root)
    W = D + 4 * depth                  (frames wasted against perfect play from the root, >= 0)
The net predicts W (well conditioned, >= 0, mostly small) and the search uses D = W - 4 depth.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.valdata --net smbzero/runs/zero8/net.pt
"""
import argparse, os, time
import numpy as np
from .common import DATA, MAX_DELAY, ROUTE, THREADS, Search, e2e_segments, route_values
from .net import Evaluator, load
from .play import Game, Player


class Collector:
    """Samples leaves of each wave; flushes shards of `shard` samples."""
    def __init__(self, out, per_wave=6, shard=8192, seed=0):
        self.out, self.per_wave, self.shard = out, per_wave, shard
        self.rng = np.random.default_rng(seed)
        os.makedirs(out, exist_ok=True)
        self.reset()
        self.files = 0
        self.total = 0

    def reset(self):
        self.roots, self.root_key = [], {}
        self.leaf, self.ridx, self.depth, self.dval, self.rval, self.rlvl = [], [], [], [], [], []

    def __call__(self, f, n, trees, step):
        vals, depths = f.leaf_info(n)
        take = self.rng.choice(n, min(self.per_wave, n), replace=False)
        for i in take:
            t = int(f.leaves[i, 0])
            key = (t, step)
            j = self.root_key.get(key)
            if j is None:
                j = self.root_key[key] = len(self.roots)
                _, ram, stack = f.state(t)
                self.roots.append(stack.copy())
                self.rval.append(f.root_value(t))
                self.rlvl.append(int(ram[0x75F]) * 4 + int(ram[0x75C]))
            self.leaf.append(f.stacks[i].copy())
            self.ridx.append(j); self.depth.append(int(depths[i])); self.dval.append(float(vals[i]))
        if len(self.leaf) >= self.shard:
            self.flush()

    def flush(self):
        if not self.leaf:
            return
        rv = np.array(self.rval, np.float32)
        ridx = np.array(self.ridx, np.int32)
        d = np.array(self.dval, np.float32) - rv[ridx]                 # D = route(leaf) - route(root)
        np.savez_compressed(os.path.join(self.out, 'shard%04d.npz' % self.files),
                            roots=np.stack(self.roots), leaves=np.stack(self.leaf), root_idx=ridx,
                            depth=np.array(self.depth, np.int32), d=d, root_value=rv,
                            root_level=np.array(self.rlvl, np.int32))
        self.files += 1
        self.total += len(self.leaf)
        self.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--out', default=os.path.join(DATA, 'value'))
    ap.add_argument('--levels', default=','.join(ROUTE))
    ap.add_argument('--games', type=int, default=8, help='games per round (one per tree)')
    ap.add_argument('--sims', type=int, default=500)
    ap.add_argument('--per-tree', type=int, default=64)
    ap.add_argument('--per-wave', type=int, default=6, help='leaves sampled per wave')
    ap.add_argument('--samples', type=int, default=250000)
    ap.add_argument('--noise', type=float, default=0.25, help='root noise: the data must cover mistakes too')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    levels = a.levels.split(',')
    net, _ = load(a.net)
    ev = Evaluator(net, a.games * a.per_tree)
    pl = Player(s, ev, a.games, per_tree=a.per_tree, routes=route_values(segs), value_mix=0.0, min_backup=True)
    col = Collector(a.out, per_wave=a.per_wave, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    t0, rounds = time.time(), 0
    while col.total < a.samples:
        rounds += 1
        lv = rng.choice(levels, a.games)
        games, caps = [], []
        for l in lv:
            d = int(rng.integers(0, MAX_DELAY + 1))
            games.append(Game(s.frames(segs[l]['start'], d), tag=(l, d)))
            caps.append(int(1.5 * len(segs[l]['opt'])))
        pl.play(games, sims=a.sims, noise=a.noise, segment_limit=1, rng=rng, max_decisions=caps, on_wave=col)
        col.flush()
        print('[valdata] round %d: %d samples in %d shards (%.0f s); %s' % (
            rounds, col.total, col.files, time.time() - t0,
            ' '.join('%s%s' % (g.tag[0], '+' if g.won else '-') for g in games)), flush=True)
    print('[valdata] done: %d samples, %d shards' % (col.total, col.files))


if __name__ == '__main__':
    main()
