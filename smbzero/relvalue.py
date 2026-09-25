"""A value the search can actually use: how much time a node has wasted, relative to its
tree's root.

The search ranks a node by b and PUCT's q compares siblings, so a constant per tree cancels:
only differences within one tree matter. Absolute frames to go are not on a cropped 84x84
screen (1-1 repeats pipes and hills), but "how much further along than the root" is. The net
sees the root's 4 frames, the leaf's 4 frames and the depth, and predicts

    W = D + 4 * depth,  D = frames to go (leaf) - frames to go (root)

W is the time wasted against perfect play from the root: near 0 on a good line, large in a
dead end. The search uses D = W - 4 * depth.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.relvalue --steps 20000 --out smbzero/runs/relv0
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import DATA, RUNS, V_SCALE
from .net import _Stage


class RelValue(nn.Module):
    """W from the root's frames, the leaf's frames and the depth.

    fusion='late':  each is embedded on its own (the root's embedding is cached per decision).
    fusion='early': both go through one trunk as 8 channels -- how far one screen has moved
                    on from the other is exactly what a convolution over the pair can see.
    """
    def __init__(self, channels=(16, 32, 32), hidden=256, fusion='late'):
        super().__init__()
        self.fusion = fusion
        stages, cin = [], 8 if fusion == 'early' else 4
        for c in channels:
            stages.append(_Stage(cin, c)); cin = c
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(cin * 11 * 11, hidden)
        self.h1 = nn.Linear((hidden if fusion == 'early' else 3 * hidden) + 2, hidden)
        self.h2 = nn.Linear(hidden, 1)

    def trunk(self, x):
        h = self.stages(x.float() / 255.0)
        return F.relu(self.fc(F.relu(h).flatten(1)))

    def embed(self, x):
        return x if self.fusion == 'early' else self.trunk(x)

    def head(self, e_leaf, e_root, depth):
        """-> W in frames (wasted against perfect play from the root)"""
        d = depth.float().unsqueeze(1)
        if self.fusion == 'early':
            e = self.trunk(torch.cat([e_leaf, e_root], 1))
        else:
            e = torch.cat([e_leaf, e_root, e_leaf - e_root], 1)
        return self.h2(F.relu(self.h1(torch.cat([e, d / 32, (d / 32) ** 2], 1)))).squeeze(1) * 16.0

    def forward(self, leaf, root, depth):
        return self.head(self.embed(leaf), self.embed(root), depth)


def load(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = RelValue(fusion=ck.get('fusion', 'late'))
    net.load_state_dict(ck['state'])
    return net.to(device).eval(), ck


class Data:
    """The shards of smbzero/data/value, with root groups kept whole across the split."""
    def __init__(self, pattern, device='cpu'):
        roots, leaves, ridx, depth, d, lvl, shard = [], [], [], [], [], [], []
        off = 0
        for i, p in enumerate(sorted(glob.glob(pattern)) if isinstance(pattern, str) else pattern):
            z = np.load(p)
            roots.append(z['roots']); leaves.append(z['leaves'])
            ridx.append(z['root_idx'] + off); depth.append(z['depth']); d.append(z['d'])
            lvl.append(z['root_level'][z['root_idx']])
            shard.append(np.full(len(z['roots']), i, np.int32))
            off += len(z['roots'])
        self.shard = np.concatenate(shard)
        self.roots = torch.from_numpy(np.concatenate(roots))
        self.leaves = torch.from_numpy(np.concatenate(leaves))
        self.ridx = torch.from_numpy(np.concatenate(ridx).astype(np.int64))
        self.depth = torch.from_numpy(np.concatenate(depth).astype(np.int64))
        self.d = torch.from_numpy(np.concatenate(d).astype(np.float32))
        self.level = np.concatenate(lvl)
        # past a few hundred frames the search only needs "hopeless"; exact magnitudes would
        # swamp the regression (a rollout into a pit can read thousands)
        self.w = (self.d + 4.0 * self.depth).clamp(max=512.0)

    def __len__(self):
        return len(self.d)

    def split(self, frac=0.08, seed=0):
        """Hold out whole shards. Roots of one game a few decisions apart are nearly the same
        state, so a scattered split would let the answer leak; a shard is a slice of play."""
        rng = np.random.default_rng(seed)
        ids = np.unique(self.shard)
        hold = set(rng.choice(ids, max(1, int(round(frac * len(ids)))), replace=False).tolist())
        m = np.isin(self.shard[self.ridx.numpy()], list(hold))
        return np.nonzero(~m)[0], np.nonzero(m)[0]

    def batch(self, idx, device):
        return (self.leaves[idx].to(device, non_blocking=True), self.roots[self.ridx[idx]].to(device),
                self.depth[idx].to(device), self.w[idx].to(device))


def sibling_pairs(data, idx, min_gap=8.0, max_pairs=20000, seed=0, same_depth=True):
    """Pairs of leaves of one root whose route values differ by >= min_gap frames: the
    comparison the search makes. same_depth: siblings of one parent (what PUCT's q ranks);
    otherwise any two leaves of the tree (what the root's b is a minimum over, ranked by
    D = W - 4 depth). Returns (i, j) with the better node first."""
    rng = np.random.default_rng(seed)
    ridx, depth = data.ridx.numpy()[idx], data.depth.numpy()[idx]
    key = data.w.numpy()[idx] if same_depth else data.d.numpy()[idx]
    order = np.lexsort((depth, ridx)) if same_depth else np.argsort(ridx, kind='stable')
    pairs = []
    k = 0
    while k < len(order):
        e = k
        while (e + 1 < len(order) and ridx[order[e + 1]] == ridx[order[k]]
               and (not same_depth or depth[order[e + 1]] == depth[order[k]])):
            e += 1
        g = order[k:e + 1]
        for _ in range(min(3 * len(g), 24) if len(g) > 1 else 0):
            a, b = rng.choice(g, 2, replace=False)
            if abs(key[a] - key[b]) >= min_gap:
                pairs.append((idx[a], idx[b]) if key[a] < key[b] else (idx[b], idx[a]))
        k = e + 1
    pairs = np.unique(np.array(pairs, np.int64).reshape(-1, 2), axis=0)
    if len(pairs) > max_pairs:
        pairs = pairs[rng.choice(len(pairs), max_pairs, replace=False)]
    return pairs


@torch.no_grad()
def pair_score(pred_fn, data, pairs, device='cuda', batch=2048):
    """Share of pairs the predictor orders like the route. pred_fn(idx) -> value per sample."""
    good = np.zeros(len(pairs), bool)
    flat = pairs.reshape(-1)
    vals = np.concatenate([pred_fn(flat[i:i + batch]) for i in range(0, len(flat), batch)])
    vals = vals.reshape(-1, 2)
    good = vals[:, 0] < vals[:, 1]
    return good, vals


def rel_pred(net, data, device='cuda'):
    def f(idx):
        leaf, root, dep, _ = data.batch(torch.from_numpy(idx), device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return net(leaf, root, dep).float().cpu().numpy()
    return f


def abs_pred(net, data, device='cuda'):
    """The old absolute head: frames to go from the leaf's frames alone."""
    def f(idx):
        leaf = data.leaves[torch.from_numpy(idx)].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, v = net(leaf)
        return (v.float() * V_SCALE).cpu().numpy()
    return f


def report(name, good, level, pairs, log):
    per = []
    lv = level[pairs[:, 0]]
    for g in sorted(set(lv.tolist())):
        m = lv == g
        per.append('%d-%d %.0f%%' % (g // 4 + 1, g % 4 + 1, 100 * good[m].mean()))
    log('[relvalue] %-22s %.1f%% of %d sibling pairs   %s' % (name, 100 * good.mean(), len(good), ' '.join(per)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(DATA, 'value', '*.npz'))
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--fusion', default='late', choices=('late', 'early'))
    ap.add_argument('--precision', default='bf16', choices=('bf16', 'fp32'))
    ap.add_argument('--holdout', type=float, default=0.08)
    ap.add_argument('--baseline', default='smbzero/runs/zero8/net.pt', help='net whose absolute value head to compare')
    ap.add_argument('--out', default=os.path.join(RUNS, 'relv0'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'train.log'), 'a')
    log = lambda m: (print(m, flush=True), logf.write(m + '\n'), logf.flush())

    data = Data(a.data)
    tr, va = data.split(frac=a.holdout)
    pairs = sibling_pairs(data, va)
    tree_pairs = sibling_pairs(data, va, same_depth=False, seed=1)
    log('[relvalue] %d samples (%d roots), %d train / %d validation, %d sibling + %d tree pairs; '
        'W: median %.0f, 90pct %.0f frames' % (len(data), len(data.roots), len(tr), len(va), len(pairs),
                                               len(tree_pairs), np.median(data.w.numpy()),
                                               np.percentile(data.w.numpy(), 90)))
    if a.baseline and os.path.exists(a.baseline):
        from .net import load as load_net
        bnet, _ = load_net(a.baseline)
        f = abs_pred(bnet.eval(), data)
        report('absolute head (old)', pair_score(f, data, pairs)[0], data.level, pairs, log)
        report('absolute head, whole tree', pair_score(f, data, tree_pairs)[0], data.level, tree_pairs, log)
        del bnet
        torch.cuda.empty_cache()

    net = RelValue(fusion=a.fusion).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    t0, hist = time.time(), []
    amp = torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.precision == 'bf16')
    for step in range(1, a.steps + 1):
        net.train()
        idx = torch.from_numpy(rng.choice(tr, a.batch))
        leaf, root, dep, w = data.batch(idx, 'cuda')
        with amp:
            pred = net(leaf, root, dep)
            loss = F.smooth_l1_loss(pred.float() / 16, w / 16)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if step % 1000 == 0 or step == a.steps:
            net.eval()
            good, _ = pair_score(rel_pred(net, data), data, pairs)
            vi = torch.from_numpy(rng.choice(va, 4096))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                l2, r2, d2, w2 = data.batch(vi, 'cuda')
                mae = (net(l2, r2, d2).float() - w2).abs().mean().item()
            hist.append(dict(step=step, loss=loss.item(), pair=float(good.mean()), mae_frames=mae,
                             s=round(time.time() - t0)))
            log('[relvalue] step %5d loss %.4f  pairs %.1f%%  W error %.1f frames  (%.0f s)'
                % (step, loss.item(), 100 * good.mean(), mae, time.time() - t0))
    f = rel_pred(net, data)
    report('relative value (new)', pair_score(f, data, pairs)[0], data.level, pairs, log)
    fd = lambda idx: f(idx) - 4.0 * data.depth.numpy()[idx]        # D = W - 4 depth
    report('relative value, whole tree', pair_score(fd, data, tree_pairs)[0], data.level, tree_pairs, log)
    torch.save(dict(state=net.state_dict(), hist=hist, fusion=a.fusion), os.path.join(a.out, 'relvalue.pt'))
    json.dump(hist, open(os.path.join(a.out, 'hist.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
