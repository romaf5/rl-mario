"""Value pairs from rollouts: the states a search never shows you.

The leaves an MCTS hands out are the ones its prior liked, so 0.04% of them are badly
wasted -- the value learned to grade good play and had never seen a trap. And the search's
own backed-up verdicts cannot fill the gap: only shallow nodes are ever revisited (median
depth 5), while the value is asked about leaves at depth 25.

So walk into the bad states on purpose: from a state the agent could be in, roll out with
random, sticky or route-with-noise actions, and label points along the way with the route's
verdict. Same shard format as valdata, so both train the same net.

  venv_retro/bin/python -m smbzero.valroll --pairs 150000
"""
import argparse, os, time
import numpy as np
from .common import DATA, MAX_DELAY, ROUTE, THREADS, Search, e2e_segments, gp


def actions_for(rng, opt, t, length, mode):
    if mode == 'route':                                   # good play with a few mistakes
        a = np.array(opt[t:t + length], np.uint8).copy()
        if len(a) < length:
            a = np.concatenate([a, rng.integers(0, 12, length - len(a)).astype(np.uint8)])
        flip = rng.random(length) < 0.2
        a[flip] = rng.integers(0, 12, flip.sum()).astype(np.uint8)
        return a
    if mode == 'sticky':                                  # a held input: how a body moves
        out = []
        while len(out) < length:
            out += [int(rng.integers(0, 12))] * int(rng.integers(1, 9))
        return np.array(out[:length], np.uint8)
    return rng.integers(0, 12, length).astype(np.uint8)   # anything at all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DATA, 'valroll'))
    ap.add_argument('--pairs', type=int, default=150000)
    ap.add_argument('--length', type=int, default=72, help='decisions per rollout')
    ap.add_argument('--per-rollout', type=int, default=16, help='pairs sampled per rollout')
    ap.add_argument('--rollouts-per-root', type=int, default=3)
    ap.add_argument('--shard', type=int, default=8192)
    ap.add_argument('--modes', default='route,sticky,random,random')
    ap.add_argument('--levels', default=','.join(ROUTE))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    levels = a.levels.split(',')
    for l in levels:
        s.set_progress_route(ROUTE, segs[l]['opt'], segs[l]['start'])
    modes = a.modes.split(',')
    rng = np.random.default_rng(a.seed)
    roots, leaves, ridx, depth, dval, rval, rlvl = [], [], [], [], [], [], []
    t0, total, files, per_level = time.time(), 0, 0, {}

    def flush():
        nonlocal roots, leaves, ridx, depth, dval, rval, rlvl, files
        if not leaves:
            return
        np.savez_compressed(os.path.join(a.out, 'roll%04d.npz' % files),
                            roots=np.stack(roots), leaves=np.stack(leaves),
                            root_idx=np.array(ridx, np.int32), depth=np.array(depth, np.int32),
                            d=np.array(dval, np.float32) - np.array(rval, np.float32)[np.array(ridx)],
                            root_value=np.array(rval, np.float32),
                            root_level=np.array(rlvl, np.int32))
        files += 1
        roots, leaves, ridx, depth, dval, rval, rlvl = [], [], [], [], [], [], []

    while total < a.pairs:
        lvl = levels[int(rng.integers(len(levels)))]
        g = segs[lvl]
        if rng.random() < 0.3:      # the level's entry: the search fills its history with one frame too
            start, t = s.frames(g['start'], int(rng.integers(0, MAX_DELAY + 1))), 0
            root_stack = np.repeat(s.obs(start)[None], 4, 0)
        else:                       # mid-route: the root's real last four frames, as the search sees them
            t = int(rng.integers(4, max(len(g['opt']) - 8, 5)))
            _, pre = s.replay(g['start'], g['opt'][:t - 4])
            robs, _, start = s.replay_obs(pre, g['opt'][t - 4:t])
            root_stack = robs[-4:]
        root_obs = root_stack[-1]
        r0 = None
        j = None
        for _ in range(a.rollouts_per_root):
            mode = modes[int(rng.integers(len(modes)))]
            acts = actions_for(rng, g['opt'], t, a.length, mode)
            out, n = s.classify_along(start, ROUTE, acts)      # stop where the segment ends
            if n < 4:
                continue
            acts = acts[:n - 1] if out[n - 1] else acts[:n]    # the terminal step is not a leaf
            if len(acts) < 4:
                continue
            prog = s.progress_along(start, ROUTE, g['opt'], acts, ref_start=g['start'])
            obs, _, _ = s.replay_obs(start, acts)
            if j is None:                                      # the root is shared by its rollouts
                j = len(roots)
                roots.append(root_stack)
                rval.append(float(prog[0])); rlvl.append(gp(lvl))
            frames = np.concatenate([root_obs[None], obs])
            pick = np.unique(rng.integers(1, len(acts) + 1, a.per_rollout))
            for k in pick:
                leaves.append(frames[np.clip(np.arange(k - 3, k + 1), 0, None)])
                ridx.append(j); depth.append(int(k))
                dval.append(float(prog[k]))                    # route frames to go at the leaf
                per_level[lvl] = per_level.get(lvl, 0) + 1
                total += 1
        if len(leaves) >= a.shard:
            flush()
            print('[valroll] %d pairs, %d shards (%.0f s) | %s' % (
                total, files, time.time() - t0,
                ' '.join('%s %d' % (k, v) for k, v in sorted(per_level.items()))), flush=True)
    flush()
    print('[valroll] done: %d pairs in %d shards' % (total, files))


if __name__ == '__main__':
    main()
