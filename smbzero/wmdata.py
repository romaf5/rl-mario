"""Trajectories for the world model: (frames, action) -> next frames, and the events that
matter (goal, dead, forced).

A trajectory starts anywhere the agent can find itself -- a level's entry with a random start
delay, or a random point along the route -- and is played by one of:
  route    the search's own actions from that point, with a few random ones mixed in
  sticky   a random action held for a few decisions (how a body moves)
  random   independent random actions (deaths, walls, the game's ugly corners)
It stops when the segment ends (the search's own rule), so the last step carries the event.

  venv_retro/bin/python -m smbzero.wmdata --trajectories 6000
"""
import argparse, os, time
import numpy as np
from .common import DATA, MAX_DELAY, ROUTE, THREADS, Search, e2e_segments


def rollout(s, seg, rng, mode, length):
    """One trajectory from a random point of a level: (start state, actions)."""
    opt = seg['opt']
    t = int(rng.integers(0, max(len(opt) - 8, 1)))
    if rng.random() < 0.35:                                  # from the level's entry, after a start delay
        start, t = s.frames(seg['start'], int(rng.integers(0, MAX_DELAY + 1))), 0
    else:                                                    # from a point along the route
        _, start = s.replay(seg['start'], opt[:t])
    if mode == 'route':
        acts = np.array(opt[t:t + length], np.uint8).copy()
        if len(acts) < length:
            acts = np.concatenate([acts, rng.integers(0, 12, length - len(acts)).astype(np.uint8)])
        flip = rng.random(length) < 0.15
        acts[flip] = rng.integers(0, 12, flip.sum()).astype(np.uint8)
    elif mode == 'sticky':
        acts, n = [], 0
        while n < length:
            k = int(rng.integers(1, 9))
            acts += [int(rng.integers(0, 12))] * k
            n += k
        acts = np.array(acts[:length], np.uint8)
    else:
        acts = rng.integers(0, 12, length).astype(np.uint8)
    return start, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DATA, 'world'))
    ap.add_argument('--trajectories', type=int, default=6000)
    ap.add_argument('--length', type=int, default=48, help='decisions per trajectory')
    ap.add_argument('--shard', type=int, default=500)
    ap.add_argument('--modes', default='route,route,sticky,random')
    ap.add_argument('--levels', default=','.join(ROUTE))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    levels = a.levels.split(',')
    for l in levels:                                   # keep each level's reference: one replay per call
        s.set_progress_route(ROUTE, segs[l]['opt'], segs[l]['start'])
    modes = a.modes.split(',')
    rng = np.random.default_rng(a.seed)
    t0, done, files = time.time(), 0, 0
    frames, acts, outcome, forced, prog, offs, foffs, meta = [], [], [], [], [], [0], [0], []
    while done < a.trajectories:
        lvl = levels[int(rng.integers(len(levels)))]
        mode = modes[int(rng.integers(len(modes)))]
        start, action = rollout(s, segs[lvl], rng, mode, a.length)
        out, n = s.classify_along(start, ROUTE, action)
        if n <= 1:
            continue
        action = action[:n]
        obs, _, _ = s.replay_obs(start, action)
        frames.append(np.concatenate([s.obs(start)[None], obs]))     # n + 1 frames
        prog.append(s.progress_along(start, ROUTE, segs[lvl]['opt'], action, ref_start=segs[lvl]['start']))
        acts.append(action); outcome.append(out); forced.append(s.forced_along(start, action))
        offs.append(offs[-1] + n); foffs.append(foffs[-1] + n + 1)
        meta.append((ROUTE.index(lvl), modes.index(mode)))
        done += 1
        if len(acts) >= a.shard:
            np.savez_compressed(os.path.join(a.out, 'traj%04d.npz' % files),
                                frames=np.concatenate(frames), acts=np.concatenate(acts),
                                outcome=np.concatenate(outcome), forced=np.concatenate(forced),
                                prog=np.concatenate(prog),
                                offs=np.array(offs, np.int64), foffs=np.array(foffs, np.int64),
                                meta=np.array(meta, np.int32))
            files += 1
            frames, acts, outcome, forced, prog, offs, foffs, meta = [], [], [], [], [], [0], [0], []
            print('[wmdata] %d trajectories, %d shards (%.0f s)' % (done, files, time.time() - t0), flush=True)
    print('[wmdata] done: %d trajectories in %d shards' % (done, files))


if __name__ == '__main__':
    main()
