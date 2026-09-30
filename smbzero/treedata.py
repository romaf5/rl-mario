"""World-model trajectories along the lines the latent search imagines -- replayed for real.

Every part of the model measures well on the data it was trained on (paths from the route,
explore, the agents' games), and each part alone still breaks the deep search inside it: the
search imagines 48 steps down lines no training trajectory took, and the model's value and prior
on those lines are what the min backup and the tree's shape follow. So show the model those lines.
The search plays from random points of each level (the agent dies early, so its own starts would
only ever show the levels' first seconds); at each decision a few root-to-leaf lines are drawn
from its tree by visit counts, extended with held random moves to --length, and replayed in the
emulator -- frames, RAM, events -- as wmdata trajectories. The game is stepped here only to make
training data; the agent still never steps it to think.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.treedata --model smbzero/runs/wm29/wm.pt --out smbzero/data/tree/t0
"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import ROUTE, RUNS, Search, e2e_segments
from . import blatent as B
from .model import load as load_model


def extend(line, length, rng):
    """The tree's line, then random moves each held 1-8 steps, to `length`."""
    out = list(line[:length])
    while len(out) < length:
        out += [int(rng.integers(0, 12))] * int(rng.integers(1, 9))
    return np.array(out[:length], np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--prior-net', default=os.path.join(RUNS, 'zero8', 'net.pt'))
    ap.add_argument('--starts', type=int, default=256, help='games, from random points of the 8 levels')
    ap.add_argument('--decisions', type=int, default=40, help='decisions played per game')
    ap.add_argument('--lines', type=int, default=4, help='lines drawn from each tree')
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--depth', type=int, default=48)
    ap.add_argument('--length', type=int, default=48, help='steps per trajectory')
    ap.add_argument('--temp', type=float, default=0.5, help='moves drawn from the visit counts: more states')
    ap.add_argument('--parallel', type=int, default=32)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--shard', type=int, default=16384, help='steps per shard')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    s = Search(threads=4)
    segs = {g['level']: g for g in e2e_segments(s)}
    starts = []
    for i in range(a.starts):
        lvl = ROUTE[i % len(ROUTE)]
        seg = segs[lvl]
        if rng.random() < 0.3:                     # the level's entry after a start delay
            st = s.frames(seg['start'], int(rng.integers(0, 61)))
        else:                                      # a random point of its route
            t = int(rng.integers(0, max(len(seg['opt']) - 8, 1)))
            _, st = s.replay(seg['start'], seg['opt'][:t])
        starts.append((lvl, 0, st))
    model, ck = load_model(a.model)
    model.eval()
    from .net import load as load_net
    pn = load_net(a.prior_net)[0].eval()
    forest = B.Forest(model, a.parallel, max_nodes=16384, calib=ck.get('calib'), max_depth=a.depth,
                      backup='children', prior_net=pn, deep_prior='model', per_wave=128, floor=-1e9,
                      vloss=True, seed=a.seed)
    t0 = time.time()
    games = B.play(s, forest, starts, a.sims, [a.decisions] * len(starts), reuse=True, reimagine=True,
                   temp=a.temp, lines=a.lines)
    nl = sum(len(ls) for g in games for _, ls in g.get('lines', []))
    print('[treedata] %d games, %d decisions, %d lines drawn (%.0f s)'
          % (len(games), sum(len(g['actions']) for g in games), nl, time.time() - t0), flush=True)
    json.dump([dict(level=g['level'], actions=g['actions'], lines=g.get('lines', [])) for g in games],
              open(os.path.join(a.out, 'games.json'), 'w'))

    local = threading.local()

    def one(i):
        e = getattr(local, 'emu', None)
        if e is None:
            e = local.emu = Search(threads=1)
        g, (lvl, _, st) = games[i], starts[i]
        seg = segs[lvl]
        r = np.random.default_rng(a.seed * 100003 + i)
        recs, t_at = [], 0
        acts_played = np.array(g['actions'], np.uint8)
        for t, ls in g.get('lines', []):
            if t > t_at:
                _, st = e.replay(st, acts_played[t_at:t])
                t_at = t
            for line in ls:
                action = extend(line, a.length, r)
                out, n = e.classify_along(st, ROUTE, action)
                if n <= 1:
                    continue
                action = action[:n]
                try:
                    prog = e.progress_along(st, ROUTE, seg['opt'], action, ref_start=seg['start'])
                except ValueError:
                    continue
                obs, _, _ = e.replay_obs(st, action)
                st_, rr = st, [e.ram(st)]
                for j in range(n):
                    _, st_ = e.replay(st_, action[j:j + 1]); rr.append(e.ram(st_))
                recs.append(dict(frames=np.concatenate([e.obs(st)[None], obs]), acts=action,
                                 outcome=np.asarray(out[:n], np.uint8), forced=e.forced_along(st, action),
                                 prog=prog, ram=np.stack(rr), meta=(ROUTE.index(lvl), 10),
                                 tree=min(len(line), n)))
        return recs

    buf, files, done, steps_tree, steps = [], 0, 0, 0, 0

    def flush():
        nonlocal buf, files
        if not buf:
            return
        offs, foffs = [0], [0]
        for r in buf:
            offs.append(offs[-1] + len(r['acts'])); foffs.append(foffs[-1] + len(r['frames']))
        np.savez_compressed(os.path.join(a.out, 'traj%04d.npz' % files),
                            frames=np.concatenate([r['frames'] for r in buf]),
                            acts=np.concatenate([r['acts'] for r in buf]),
                            outcome=np.concatenate([r['outcome'] for r in buf]),
                            forced=np.concatenate([r['forced'] for r in buf]),
                            prog=np.concatenate([r['prog'] for r in buf]),
                            ram=np.concatenate([r['ram'] for r in buf]),
                            offs=np.array(offs, np.int64), foffs=np.array(foffs, np.int64),
                            meta=np.array([r['meta'] for r in buf], np.int32))
        files += 1
        buf = []

    dead = 0
    with ThreadPoolExecutor(a.threads) as pool:
        for recs in pool.map(one, range(len(games))):
            for r in recs:
                buf.append(r)
                done += 1
                steps += len(r['acts']); steps_tree += r['tree']
                dead += int(r['outcome'][-1] == 2)
            if sum(len(x['acts']) for x in buf) >= a.shard:
                flush()
    flush()
    print('[treedata] %d trajectories (%d end in a death), %d steps -- %.0f%% of them the tree\'s own -- '
          'in %d shards (%.0f s)' % (done, dead, steps, 100 * steps_tree / max(steps, 1), files, time.time() - t0),
          flush=True)


if __name__ == '__main__':
    main()
