"""Does the net help? The same MCTS with (A) the net's prior, (B) a uniform prior, and
(C) the net alone (its top move, no search, no survival check), per level over start delays.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.ablation --net NET.pt
"""
import argparse, json
import numpy as np
from ..common import ROUTE, THREADS, Search, e2e_segments, route_values
from ..net import Evaluator, load
from ..play import Game, Player


class Uniform:
    """An Evaluator stand-in: the same input buffer, a uniform prior, a value of 0 (unused: route values)."""
    def __init__(self, max_leaves):
        self.stacks = np.zeros((max_leaves, 4, 84, 84), np.uint8)

    def __call__(self, n):
        return np.full((n, 12), 1 / 12, np.float32), np.zeros(n, np.float32)


def level_games(s, segs, ev, level, delays, sims, safe):
    pl = Player(s, ev, len(delays), per_tree=128, routes=route_values(segs), value_mix=0.0, min_backup=True)
    games = [Game(s.frames(segs[level]['start'], d), tag=d) for d in delays]
    pl.play(games, sims=sims, segment_limit=1, max_decisions=int(2.5 * len(segs[level]['opt'])),
            safe_horizon=24 if safe else 0)
    return [(g.tag, g.won, g.reason, round((g.tag + g.frames()) / 50.007, 1)) for g in games]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--delays', default='5,20,35,50')
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--out', default='smbzero/runs/ablation.json')
    a = ap.parse_args()
    delays = [int(d) for d in a.delays.split(',')]
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    net, _ = load(a.net)
    n = len(delays) * 128
    arms = [('A net prior + MCTS', Evaluator(net, n), a.sims, True),
            ('B uniform prior + MCTS', Uniform(n), a.sims, True),
            ('C net alone', Evaluator(net, n), 2, False)]
    res = {}
    for name, ev, sims, safe in arms:
        for lvl in ROUTE:
            r = level_games(s, segs, ev, lvl, delays, sims, safe)
            res['%s %s' % (name, lvl)] = r
            print('[ablation] %-24s %s: won %d/%d  %s' % (name, lvl, sum(x[1] for x in r), len(r),
                  ' '.join(('%.1fs' % x[3]) if x[1] else x[2].replace('dead at ', 'dead ') for x in r)), flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
