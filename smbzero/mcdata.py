"""Value pairs with no teacher: the agent races itself.

The value learns W, the frames a position has thrown away against the best the agent knows
from the root. Until now that came from the search's route -- a solution handed to it. Here
it comes from play alone:

    from a state s, the agent plays to the end of the level        -> it took T_s frames
    from a branch of s (a few random steps, then the agent again)  -> it took T_b frames
    the branch at depth d wasted   W = (4 d + T_b) - T_s

One continuation labels every state along it, so a pair costs a fraction of a level. Nothing
is read from the game but the screen, the level ending and the death.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.mcdata --net NET.pt --relvalue V.pt --pairs 120000
"""
import argparse, os, time
import numpy as np
from .common import DATA, MAX_DELAY, ROUTE, THREADS, Search, e2e_segments, gp
from .net import RelEvaluator, load as load_net
from .play import Game, Player
from .relvalue import load as load_rel

HOPELESS = 512.0


def play_batch(player, starts, caps, sims, rng):
    games = [Game(st, tag=i) for i, st in enumerate(starts)]
    player.play(games, sims=sims, segment_limit=1, rng=rng, max_decisions=caps)
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--relvalue', required=True)
    ap.add_argument('--out', default=os.path.join(DATA, 'mc'))
    ap.add_argument('--pairs', type=int, default=120000)
    ap.add_argument('--games', type=int, default=16)
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--per-tree', type=int, default=48)
    ap.add_argument('--branches', type=int, default=4, help='branches per root (siblings)')
    ap.add_argument('--prefix', type=int, default=24, help='longest random prefix of a branch')
    ap.add_argument('--depth', type=int, default=60, help='deepest pair taken from a branch')
    ap.add_argument('--shard', type=int, default=8192)
    ap.add_argument('--levels', default=','.join(ROUTE))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}          # only for each level's entry state
    levels = a.levels.split(',')
    net, _ = load_net(a.net)
    rel, _ = load_rel(a.relvalue)
    ev = RelEvaluator(net, rel, a.games * a.per_tree)
    player = Player(s, ev, a.games, per_tree=a.per_tree, value_mix=1.0, min_backup=True, relative=True)
    rng = np.random.default_rng(a.seed)
    roots, leaves, ridx, depth, dval, rval, rlvl = [], [], [], [], [], [], []
    t0, total, files, stats = time.time(), 0, 0, dict(spines=0, spine_wins=0, branches=0, branch_wins=0)

    def flush():
        nonlocal roots, leaves, ridx, depth, dval, rval, rlvl, files
        if not leaves:
            return
        np.savez_compressed(os.path.join(a.out, 'mc%04d.npz' % files),
                            roots=np.stack(roots), leaves=np.stack(leaves), root_idx=np.array(ridx, np.int32),
                            depth=np.array(depth, np.int32), d=np.array(dval, np.float32),
                            root_value=np.array(rval, np.float32), root_level=np.array(rlvl, np.int32))
        files += 1
        roots, leaves, ridx, depth, dval, rval, rlvl = [], [], [], [], [], [], []

    pool = []          # winning lines: (level, start, actions, T). A line is expensive to earn,
    while total < a.pairs:                          # so it is kept and branched from many times
        if len(pool) < a.games:
            lv = rng.choice(levels, a.games)
            starts = [s.frames(segs[l]['start'], int(rng.integers(0, MAX_DELAY + 1))) for l in lv]
            caps = [int(2.0 * len(segs[l]['opt'])) for l in lv]
            spines = play_batch(player, starts, caps, a.sims, rng)
            stats['spines'] += len(spines)
            for g in spines:
                if g.won:
                    pool.append((lv[g.tag], starts[g.tag], np.array(g.actions, np.uint8), len(g.actions)))
            stats['spine_wins'] = len(pool)
            if not pool:
                continue

        # Several branches from ONE root: leaves of equal depth are then true siblings, the
        # comparison the search actually makes when PUCT ranks a node against its brothers.
        nb = max(a.branches, 1)
        groups = []
        for _ in range(max(1, a.games // nb)):
            l, st0, acts, T = pool[int(rng.integers(len(pool)))]
            groups.append((l, st0, acts, T, int(rng.integers(0, max(T - 8, 1)))))
        bstarts, bcaps, prefixes, owner, root_states = [], [], [], [], []
        for gi, (l, st0, acts, T, t) in enumerate(groups):
            _, root_state = s.replay(st0, acts[:t])
            root_states.append(root_state)
            for _ in range(nb):
                k = int(rng.integers(2, a.prefix + 1))
                pre = (rng.integers(0, 12, k).astype(np.uint8) if rng.random() < 0.5
                       else np.full(k, rng.integers(0, 12), np.uint8))   # a held input, or anything
                out, n = s.classify_along(root_state, ROUTE, pre)
                pre = pre[:n - 1] if n and out[n - 1] else pre[:n]
                prefixes.append(pre)
                _, after = s.replay(root_state, pre)
                bstarts.append(after); bcaps.append(int(2.0 * len(segs[l]['opt']))); owner.append(gi)
        branches = play_batch(player, bstarts, bcaps, a.sims, rng) if bstarts else []
        stats['branches'] += len(branches)

        jroot = {}                       # one stored root per group, shared by all its branches
        for bg in branches:
            b = bg.tag
            l, st0, acts, T, t = groups[owner[b]]
            stats['branch_wins'] += bool(bg.won)
            if bg.won and len(pool) < 200:      # a branch that finished is a line of its own
                pool.append((l, bstarts[b], np.array(bg.actions, np.uint8), len(bg.actions)))
            if owner[b] not in jroot:
                rs = root_states[owner[b]]                 # already replayed once, above
                if t >= 4:                                 # render only the root's last four frames
                    _, before = s.replay(st0, acts[:t - 4])
                    root_stack = s.replay_obs(before, acts[t - 4:t])[0][-4:]
                else:
                    root_stack = np.repeat(s.obs(rs)[None], 4, 0)
                jroot[owner[b]] = (len(roots), rs)
                roots.append(root_stack); rval.append(4.0 * (T - t)); rlvl.append(gp(l))
            j, root_state = jroot[owner[b]]
            r_root = 4.0 * (T - t)
            branch_acts = np.concatenate([prefixes[b], np.array(bg.actions, np.uint8)])
            obs, _, _ = s.replay_obs(root_state, branch_acts[:a.depth])
            frames = np.concatenate([roots[j], obs])
            Tb = len(branch_acts)
            for d in range(1, min(a.depth, len(branch_acts)) + 1):
                r_leaf = 4.0 * (Tb - d) if bg.won else HOPELESS + r_root
                leaves.append(frames[d:d + 4])                 # the stack ending at that step
                ridx.append(j); depth.append(d); dval.append(r_leaf - r_root)
                total += 1
        if len(leaves) >= a.shard:
            flush()
            print('[mcdata] %d pairs, %d shards (%.0f s) | spines %d/%d won, branches %d/%d won' % (
                total, files, time.time() - t0, stats['spine_wins'], stats['spines'],
                stats['branch_wins'], stats['branches']), flush=True)
    flush()
    print('[mcdata] done: %d pairs in %d shards' % (total, files))


if __name__ == '__main__':
    main()
