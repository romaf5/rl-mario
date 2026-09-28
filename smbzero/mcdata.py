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
    ap.add_argument('--spine-tries', type=int, default=6, help='attempts per level before giving up on it')
    ap.add_argument('--seed-lines', help="glob of JSON winning lines (level in the name, _4-2.json) to start the pool with")
    ap.add_argument('--spine-sims', type=int, default=0, help='simulations for a spine (0: same as --sims); '
                    'a winning line is rare and is branched from many times, so it is worth more search')
    ap.add_argument('--frontier', type=float, default=0.0,
                    help="share of a seeded level's roots taken from its seeded lines just before their frontier, "
                         "the earliest root a branch has won from (Go-Explore's backward algorithm)")
    ap.add_argument('--frontier-window', type=int, help='frontier roots from this many decisions before it '
                    '(default: --depth). The value learns the hard step from roots within the search\'s horizon of it')
    ap.add_argument('--seed-copies', type=int, default=1, help="a seeded line's continuation this many times per root, "
                    'to weigh as much as the agent branches beside it')
    ap.add_argument('--frontier-init', help='comma list, one per seeded line in order: frontiers to start from '
                    '(the last ones a previous run printed)')
    ap.add_argument('--min-pool', type=int, help='play spines while the pool is smaller than this (default: --games)')
    ap.add_argument('--cap-by-line', action='store_true', help="stop a branch once it cannot beat its line by "
                    "less than HOPELESS: its label is HOPELESS either way")
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
    import glob as _g                                # a second run into the same directory adds shards
    t0, total, files, stats = time.time(), 0, len(_g.glob(os.path.join(a.out, 'mc*.npz'))), dict(spines=0, spine_wins=0, branches=0, branch_wins=0)

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
    tries = {l: 0 for l in levels}                  # so it is kept and branched from many times
    made = {l: 0 for l in levels}                   # roots kept per level: keep the mix even
    bwin = {l: [0, 0] for l in levels}              # branches won / tried: keep them hard enough
    if a.seed_lines:                                # first ways the agent cannot find (Go-Explore)
        import glob as _glob, json as _json, re as _re
        for f in sorted(_glob.glob(a.seed_lines)):
            m_ = _re.search(r'_(\d-\d)\.json$', f)
            if m_ and m_.group(1) in levels:
                for g in _json.load(open(f)):
                    if g.get('won'):
                        pool.append((m_.group(1), s.frames(segs[m_.group(1)]['start'], g['delay']),
                                     np.array(g['actions'], np.uint8), len(g['actions'])))
        print('[mcdata] %d seeded lines: %s' % (len(pool), ','.join(sorted({q[0] for q in pool}))), flush=True)
    n_seed = len(pool)
    front = [q[3] - 8 for q in pool]               # per seeded line: earliest root a branch won from
    if a.frontier_init:
        front = [int(x) for x in a.frontier_init.split(',')]
        assert len(front) == n_seed, (front, n_seed)
    min_pool = a.games if a.min_pool is None else a.min_pool
    while total < a.pairs:
        # A pool that fills with the easy levels never tries the hard ones again, and the
        # value then goes into the gate never having seen them. Ask for what is missing.
        have = {q[0] for q in pool}
        want = [l for l in levels if l not in have and tries[l] < a.spine_tries]
        if len(pool) < min_pool or want:
            lv = rng.choice(want if want else levels, a.games)
            for l in set(lv.tolist()):            # one round, not one game, per attempt
                tries[l] += 1
            starts = [s.frames(segs[l]['start'], int(rng.integers(0, MAX_DELAY + 1))) for l in lv]
            caps = [int(2.0 * len(segs[l]['opt'])) for l in lv]
            spines = play_batch(player, starts, caps, a.spine_sims or a.sims, rng)
            stats['spines'] += len(spines)
            for g in spines:
                if g.won:
                    pool.append((lv[g.tag], starts[g.tag], np.array(g.actions, np.uint8), len(g.actions)))
            stats['spine_wins'] = len(pool)
            print('[mcdata] spines %s -> pool %d (%s) | %.0f s' % (
                ','.join(sorted(set(lv.tolist()))), len(pool),
                ','.join('%s=%d' % (k, v) for k, v in sorted(made.items()) if v) or 'no roots yet',
                time.time() - t0), flush=True)
            if not pool:
                continue

        # Several branches from ONE root: leaves of equal depth are then true siblings, the
        # comparison the search actually makes when PUCT ranks a node against its brothers.
        nb = max(a.branches, 1)
        groups = []
        # A level whose spines are easy to win fills the pool and crowds the rest out, and the
        # value then has almost nothing of the hard levels. Favour what we have least of.
        pw = np.array([1.0 / (1 + made[q[0]]) for q in pool])
        for gi in rng.choice(len(pool), max(1, a.games // nb), p=pw / pw.sum()):
            gi = int(gi)
            seeded = [k for k in range(n_seed) if pool[k][0] == pool[gi][0]]
            if seeded and rng.random() < a.frontier:
                # Where the agent's own branches stop winning is where the level is hard (the
                # hidden vine, a maze pipe). Roots just before it: the seeded line's continuation
                # there shows the way within one branch's depth, beside the agent's lines that miss it.
                gi = int(rng.choice(seeded))
                t = int(rng.integers(max(front[gi] - (a.frontier_window or a.depth), 0), max(front[gi], 1)))
            else:
                t = int(rng.integers(0, max(pool[gi][3] - 8, 1)))
            l, st0, acts, T = pool[gi]
            groups.append((l, st0, acts, T, t, gi))
        bstarts, bcaps, prefixes, owner, root_states = [], [], [], [], []
        for gi, (l, st0, acts, T, t, pidx) in enumerate(groups):
            _, root_state = s.replay(st0, acts[:t])
            root_states.append(root_state)
            won_, tried_ = bwin[l]
            # Where a branch almost always recovers, the labels are nearly all the same number
            # and there is nothing to rank: 1-1's random branches cost 52 frames at the 90th
            # percentile and its gate collapsed while 4-1, twice as spread, cleared 4/4.
            reach = a.prefix if tried_ < 24 or won_ / tried_ < 0.9 else a.prefix * 3
            for _ in range(nb):
                k = int(rng.integers(2, reach + 1))
                pre = (rng.integers(0, 12, k).astype(np.uint8) if rng.random() < 0.5
                       else np.full(k, rng.integers(0, 12), np.uint8))   # a held input, or anything
                out, n = s.classify_along(root_state, ROUTE, pre)
                pre = pre[:n - 1] if n and out[n - 1] else pre[:n]
                prefixes.append(pre)
                _, after = s.replay(root_state, pre)
                cap = int(2.0 * len(segs[l]['opt']))
                if a.cap_by_line:                  # W = 4 (prefix + agent) - 4 (T - t) >= HOPELESS past this
                    cap = max(1, min(cap, (T - t) + int(HOPELESS // 4) + 1 - len(pre)))
                bstarts.append(after); bcaps.append(cap); owner.append(gi)
        branches = play_batch(player, bstarts, bcaps, a.sims, rng) if bstarts else []
        stats['branches'] += len(branches)

        jroot = {}                       # one stored root per group, shared by all its branches
        for bg in branches:
            b = bg.tag
            l, st0, acts, T, t, pidx = groups[owner[b]]
            stats['branch_wins'] += bool(bg.won)
            if bg.won and pidx < n_seed:
                front[pidx] = min(front[pidx], t)
            bwin[l][1] += 1; bwin[l][0] += bool(bg.won)
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
                made[l] += 1
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
        # A seeded line's own continuation, as one more sibling. From a root before the vine the
        # agent's branches almost never bump the hidden block -- that is why it needed seeding --
        # so without this the value would never see, at equal depth, the one line that does.
        for gi, (l, st0, acts, T, t, pidx) in enumerate(groups):
            if pidx >= n_seed or gi not in jroot:
                continue
            j, root_state = jroot[gi]
            cont = acts[t:]
            obs, _, _ = s.replay_obs(root_state, cont[:a.depth])
            frames = np.concatenate([roots[j], obs])
            for d in range(1, min(a.depth, len(cont)) + 1):
                for _ in range(a.seed_copies):
                    leaves.append(frames[d:d + 4]); ridx.append(j); depth.append(d)
                    dval.append(4.0 * (len(cont) - d) - 4.0 * (T - t))     # the line itself: nothing lost
                    total += 1
            stats['seeded'] = stats.get('seeded', 0) + 1
        if len(leaves) >= a.shard:
            flush()
            print('[mcdata] %d pairs, %d shards (%.0f s) | spines %d/%d won, branches %d/%d won%s' % (
                total, files, time.time() - t0, stats['spine_wins'], stats['spines'],
                stats['branch_wins'], stats['branches'],
                (' | frontiers %s' % ' '.join('%s:%d/%d' % (pool[k][0], front[k], pool[k][3]) for k in range(n_seed)))
                if a.frontier else ''), flush=True)
    flush()
    print('[mcdata] done: %d pairs in %d shards' % (total, files))


if __name__ == '__main__':
    main()
