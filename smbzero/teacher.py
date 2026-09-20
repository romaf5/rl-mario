"""Teacher data: the search's routes from start variants of every route level.

A variant is a level's entry state (first in-control frame, from the verified e2e
route) plus d NOOP frames. d = 0 uses the e2e optimised route; other delays run
the beam from the delayed state along that level's explore reference (replayed
from its own start). Each route becomes an episode (frames, one-hot policy,
frames to go, forced mask) in smbzero/data/teacher/.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.teacher
"""
import argparse, os, time
import numpy as np
from .common import (DATA, MAX_DELAY, ROUTE, THREADS, Search, e2e_segments, episode, gp, save_episode)


def route_episode(s, start, actions, **meta):
    obs, tr, _ = s.replay_obs(start, actions)
    n = len(actions)
    frames = np.concatenate([s.obs(start)[None], obs])
    policy = np.zeros((n, 12), np.float32)
    policy[np.arange(n), actions] = 1
    value = 4.0 * (n - np.arange(n))
    return episode(frames, policy, value, s.forced_along(start, actions), **meta), tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--beam', type=int, default=1000)
    ap.add_argument('--levels', default=','.join(ROUTE))
    ap.add_argument('--delays-11', default=','.join(str(d) for d in range(0, MAX_DELAY + 1, 6)))
    ap.add_argument('--delays', default='0,13,27,41,55')
    ap.add_argument('--out', default=os.path.join(DATA, 'teacher'))
    a = ap.parse_args()
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    for lvl in a.levels.split(','):
        g = segs[lvl]
        for d in map(int, (a.delays_11 if lvl == '1-1' else a.delays).split(',')):
            path = os.path.join(a.out, '%s_d%02d.npz' % (lvl, d))
            if os.path.exists(path):
                continue
            t = time.time()
            start = s.frames(g['start'], d)
            if d == 0:
                acts = g['opt']
            else:
                r = s.optimize(start, ROUTE, g['ref'], beam=a.beam, per_cell=16, ref_start=g['start'])
                if not r.found:
                    print('[teacher] %s d=%d: no route (%.0f s)' % (lvl, d, time.time() - t), flush=True)
                    continue
                acts = r.actions
            ep, tr = route_episode(s, start, acts, level=lvl, delay=d, source='teacher', start_state=np.frombuffer(start, np.uint8))
            k = ROUTE.index(lvl)
            ok = tr[-1, 7] == 2 if k == len(ROUTE) - 1 else int(tr[-1, 2]) == gp(ROUTE[k + 1])
            save_episode(path, ep)
            print('[teacher] %s d=%02d: %d steps (%d forced), %.0f s%s' % (
                lvl, d, len(acts), ep['forced'].sum(), time.time() - t, '' if ok else '  GOAL NOT REACHED'), flush=True)


if __name__ == '__main__':
    main()
