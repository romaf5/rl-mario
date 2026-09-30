"""MuZero's training data from the latent agent's own games (blatent --record).

Each game is replayed in the emulator for what really happened -- the screens, deaths,
finishes, forced steps -- and stored as world-model trajectories (wmtrain's format) with two
more targets:

  pi   per decision: the search's visit counts, normalised -- the policy learns the search
  tgt  per frame: frames to go, n-step: 4 n + the search's own root value n decisions later;
       exact where the game ends within n (the goal: 0 left; a death: +512, the search's price)

  venv_retro/bin/python -m smbzero.mzdata --games smbzero/runs/mz/sp00.json --out smbzero/data/mz/it00
"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import ROUTE, Search, e2e_segments

DEATH = 512.0


def targets(n, end, root_tg, nstep):
    """Frames to go at frames 0..n of a game of n decisions that ended in `end`."""
    if end == 'goal':
        last = 0.0
    elif end == 'dead':
        last = DEATH
    else:                                   # stopped at the cap: the search's last word, one step on
        last = max(root_tg[-1] - 4.0, 0.0) if len(root_tg) else 0.0
    tg = np.zeros(n + 1, np.float32)
    tg[n] = last
    for j in range(n):
        if n - j <= nstep:
            tg[j] = 4.0 * (n - j) + last
        else:
            tg[j] = 4.0 * nstep + root_tg[j + nstep]
    return tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', required=True, help='comma-separated blatent --record JSONs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nstep', type=int, default=10)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--shard', type=int, default=8192, help='decisions per shard')
    ap.add_argument('--ram', action='store_true', help="also keep the game's RAM at every frame (for --ram models)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    games = []
    for f in a.games.split(','):
        games += [g for g in json.load(open(f)) if 'visits' in g and len(g['actions']) > 1]
    s = Search(threads=2)
    segs = {g['level']: g for g in e2e_segments(s)}
    local = threading.local()

    def one(g):
        e = getattr(local, 'emu', None)
        if e is None:
            e = local.emu = Search(threads=1)
        seg = segs[g['level']]
        start = e.frames(seg['start'], g['delay'])
        acts = np.array(g['actions'], np.uint8)
        out, m = e.classify_along(start, ROUTE, acts)
        n = max(m, 1)                        # the engine's rule may end the game a step early
        acts = acts[:n]
        obs, _, _ = e.replay_obs(start, acts)
        frames = np.concatenate([e.obs(start)[None], obs])
        outc = np.asarray(out[:n], np.uint8)
        end = 'goal' if outc[-1] == 1 else 'dead' if outc[-1] == 2 else 'cap'
        prog = e.progress_along(start, ROUTE, seg['opt'], acts, ref_start=seg['start'])
        v = np.array(g['visits'][:n], np.float32)
        pi = v / np.maximum(v.sum(1, keepdims=True), 1)
        tg = targets(n, end, np.array(g['root_tg'][:n], np.float32), a.nstep)
        r = dict(frames=frames, acts=acts, outcome=outc, forced=e.forced_along(start, acts), prog=prog,
                 pi=pi, tgt=tg, meta=(ROUTE.index(g['level']), 9), end=end)
        if a.ram:                                   # the RAM at each of the n + 1 frames
            st_, rr = start, [e.ram(start)]
            for i in range(n):
                _, st_ = e.replay(st_, acts[i:i + 1]); rr.append(e.ram(st_))
            r['ram'] = np.stack(rr)
        return r

    t0 = time.time()
    buf, files, ends = [], 0, {'goal': 0, 'dead': 0, 'cap': 0}

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
                            pi=np.concatenate([r['pi'] for r in buf]),
                            tgt=np.concatenate([r['tgt'] for r in buf]),
                            offs=np.array(offs, np.int64), foffs=np.array(foffs, np.int64),
                            meta=np.array([r['meta'] for r in buf], np.int32),
                            **({'ram': np.concatenate([r['ram'] for r in buf])} if a.ram else {}))
        files += 1
        buf = []

    with ThreadPoolExecutor(a.threads) as pool:
        for r in pool.map(one, games):
            buf.append(r)
            ends[r['end']] += 1
            if sum(len(x['acts']) for x in buf) >= a.shard:
                flush()
    flush()
    print('[mzdata] %d games (%s) -> %d shards in %s (%.0f s)'
          % (len(games), ' '.join('%s %d' % kv for kv in ends.items()), files, a.out, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
