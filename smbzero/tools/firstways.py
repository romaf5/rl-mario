"""First ways through the levels the agent cannot find its own way through, by Go-Explore.

Stage C learns only from the agent's own play, and so it can only learn a level it can already
finish. 4-2 (a vine hidden in an invisible block) and 8-4 (a maze of pipes) it never has:
120+ attempts each at 2000 simulations, none finished. Its search is steered by a policy and a
value that have never seen the way, so it never tries it.

Go-Explore does not need a reason to try something: it keeps an archive of every distinct place
reached (cells: area, position, camera, tiles on screen), returns to them by restoring the
emulator's state, and explores from there at random, so a bumped hidden block or a new pipe is a
new cell and is kept. Its only input besides the level is the order of the levels, so it knows
4-2 should lead to world 8. The raw route it finds -- not the one the beam search then polishes
-- is written out in the format `latent --out` uses, for mcdata to take as a winning line.

  venv_retro/bin/python -m smbzero.tools.firstways --levels 4-2,8-4 --delays 0,20,40
"""
import argparse, json, os
import numpy as np
from ..common import ROUTE, RUNS, THREADS, Search, e2e_segments, FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', default='4-2,8-4')
    ap.add_argument('--delays', default='0,20,40')
    ap.add_argument('--budget', type=float, default=600.0, help='seconds of exploration per start')
    ap.add_argument('--settle', type=float, default=120.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(RUNS, 'firstways'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    for lvl in a.levels.split(','):
        games = []
        for d in [int(x) for x in a.delays.split(',')]:
            start = s.frames(segs[lvl]['start'], d)
            ref = s.explore(start, ROUTE, budget_s=a.budget, settle_s=a.settle, seed=a.seed + d)
            acts = np.asarray(ref.actions, np.uint8)
            won = False
            if ref.found and len(acts):
                out, n = s.classify_along(start, ROUTE, acts)
                won = bool(n and np.asarray(out[:n])[-1] == 1)     # the real game agrees it reached the goal
            print('[firstways] %s delay %d: %s -- %d decisions (%.1f s), %d cells, %d walks'
                  % (lvl, d, 'FOUND' if won else 'not found', len(acts), (d + 4 * len(acts)) / FPS,
                     ref.stats['cells'], ref.stats['walks']), flush=True)
            if won:
                games.append(dict(delay=d, won=True, reason='goal', decisions=len(acts),
                                  seconds=round((d + 4 * len(acts)) / FPS, 1), actions=[int(x) for x in acts]))
        json.dump(games, open(os.path.join(a.out, 'firstways_%s.json' % lvl), 'w'))
        print('[firstways] %s: %d of %d starts found a way' % (lvl, len(games), len(a.delays.split(','))), flush=True)


if __name__ == '__main__':
    main()
