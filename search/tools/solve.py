"""Solve route segments: explore -> optimise -> settle, chained.

  venv_retro/bin/python search/tools/solve.py --start 4-2 --segments 1 --out search/out/4-2
  venv_retro/bin/python search/tools/solve.py --start FullGame --segments 8 --out search/out/e2e

Each segment starts where the previous one's optimised route settled (first
in-control frame of the next level). Writes route.npz and stats.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
from smbsearch import Search, load_state, ROUTE, gp, level_name, FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='4-2')
    ap.add_argument('--segments', type=int, default=1)
    ap.add_argument('--route', default=','.join(ROUTE))
    ap.add_argument('--threads', type=int, default=40)
    ap.add_argument('--explore-budget', type=float, default=600)
    ap.add_argument('--explore-settle', type=float, default=120)
    ap.add_argument('--beam', type=int, default=20000)
    ap.add_argument('--per-cell', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    route = a.route.split(',')
    s = Search(threads=a.threads)
    state = load_state(a.start)
    n0, state = s.settle(state)
    assert n0 >= 0, 'start state never reaches player control'
    all_actions = [np.zeros(n0, np.uint8)]
    rec = dict(start=a.start, levels=[], seg_actions=[], seg_settle=[], lead_in=n0)
    stats = dict(segments=[], threads=s.threads, lead_in_steps=n0)
    for k in range(a.segments):
        lvl = s.level(state)
        t0 = time.time()
        ref = s.explore(state, route, budget_s=a.explore_budget, settle_s=a.explore_settle, seed=a.seed + k, verbose=1)
        t1 = time.time()
        if not ref.found:
            print('[solve] %s: explore found no exit in %.0f s (%d cells, %d walks)'
                  % (lvl, t1 - t0, ref.stats['cells'], ref.stats['walks'])); break
        print('[solve] %s: reference %d steps (%d cells, %d walks, %.0f s)'
              % (lvl, len(ref.actions), ref.stats['cells'], ref.stats['walks'], t1 - t0), flush=True)
        opt = s.optimize(state, route, ref.actions, beam=a.beam, per_cell=a.per_cell, verbose=1)
        t2 = time.time()
        best = opt.actions if opt.found and len(opt.actions) <= len(ref.actions) else ref.actions
        print('[solve] %s: optimised %d steps (reference %d, %.0f s)' % (lvl, len(best), len(ref.actions), t2 - t1), flush=True)
        tr, end = s.replay(state, best)
        nxt = level_name(int(tr[-1, 2])) if int(tr[-1, 2]) >= 0 else '?'
        settle, nstate = (0, end) if int(tr[-1, 7]) == 2 else s.settle(end)
        rec['levels'].append(lvl); rec['seg_actions'].append(len(best)); rec['seg_settle'].append(max(settle, 0))
        rec['ref_actions_%d' % k] = ref.actions
        all_actions += [best, np.zeros(max(settle, 0), np.uint8)]
        stats['segments'].append(dict(level=lvl, next=nxt, reference_steps=len(ref.actions),
                                      optimised_steps=len(best), settle_steps=settle,
                                      explore_s=round(t1 - t0, 1), optimise_s=round(t2 - t1, 1),
                                      explore=ref.stats, optimise=opt.stats))
        state = nstate
        if int(tr[-1, 7]) == 2:
            break
    rec['actions'] = np.concatenate(all_actions)
    np.savez(os.path.join(a.out, 'route.npz'), **{k: np.asarray(v) for k, v in rec.items()})
    stats['total_steps'] = int(len(rec['actions'])); stats['total_frames'] = int(len(rec['actions'])) * 4
    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=1, default=float)
    print('[solve] total %d steps = %d frames (%.2f s)' % (stats['total_steps'], stats['total_frames'],
                                                         stats['total_frames'] / FPS))


if __name__ == '__main__':
    main()
