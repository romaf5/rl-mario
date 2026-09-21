"""Can the net's value rank nearby states? The test for handing leaf values to the net.

From random states on the search's route of each level, two random 8-step plans;
the local teacher's frames to go orders the two end states (pairs it separates by
>= 8 frames); the score is how often the net orders them the same way. value_mix
can move toward the net once this passes ~90% on every level.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.value_test --net smbzero/runs/zero1/net.pt
"""
import argparse
import numpy as np
from .common import ROUTE, THREADS, Search, e2e_segments
from .net import Evaluator, load


def rank_score(s, ev, seg, pairs=30, seed=0):
    rng = np.random.default_rng(seed)
    opt = seg['opt']
    ok = tot = tries = 0
    while tot < pairs and tries < pairs * 6:
        tries += 1
        t = int(rng.integers(3, len(opt) - 20))
        obs0, _, st = s.replay_obs(seg['start'], opt[:t])
        if s.forced_along(st, np.zeros(1, np.uint8))[0]:
            continue
        vals = []
        for _ in range(2):
            plan = rng.integers(0, 12, 8).astype(np.uint8)
            obs, tr, end = s.replay_obs(st, plan)
            if tr[-1, 6] != 8:                  # the end state must be in control
                break
            ev.stacks[0] = np.concatenate([obs0, obs])[-4:]
            _, v = ev(1)
            _, est, _ = s.lookahead(end, ROUTE, opt, ref_start=seg['start'], beam=100, horizon=30)
            vals.append((float(v[0]), est))
        if len(vals) < 2 or abs(vals[0][1] - vals[1][1]) < 8:
            continue
        tot += 1
        ok += np.sign(vals[0][0] - vals[1][0]) == np.sign(vals[0][1] - vals[1][1])
    return ok / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--pairs', type=int, default=30)
    ap.add_argument('--levels', default=','.join(ROUTE))
    a = ap.parse_args()
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    net, _ = load(a.net)
    ev = Evaluator(net, 4)
    res = {l: rank_score(s, ev, segs[l], a.pairs) for l in a.levels.split(',')}
    for l, (sc, n) in res.items():
        print('[value_test] %s: net orders %.0f%% of %d pairs like the local teacher' % (l, 100 * sc, n))
    print('[value_test] mean %.0f%%' % (100 * np.mean([sc for sc, _ in res.values()])))


if __name__ == '__main__':
    main()
