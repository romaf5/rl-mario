"""Clear rate per level at a fixed search budget -- the comparison every change is judged by.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.levels --net NET.pt --sims 1000
  ... --relvalue RELVALUE.pt        # the learned value, no route at play time
"""
import argparse, json, os
import numpy as np
from ..common import ROUTE, RUNS, THREADS, Search
from ..eval import run
from ..net import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--relvalue', help='play on the learned value instead of the search route')
    ap.add_argument('--delays', default='5,20,35,50')
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--levels', default=','.join(ROUTE))
    ap.add_argument('--out')
    a = ap.parse_args()
    delays = [int(d) for d in a.delays.split(',')]
    s = Search(threads=THREADS)
    net, _ = load(a.net)
    rv = None
    if a.relvalue:
        from ..relvalue import load as load_rel
        rv, _ = load_rel(a.relvalue)
    tag = 'learned value' if rv is not None else 'route value'
    res, won, tot = {}, 0, 0
    for lvl in a.levels.split(','):
        summary, r = run(net, delays, sims=a.sims, parallel=len(delays), level=lvl, s=s, log=lambda m: None,
                         value_mix=0.0, min_backup=True, relvalue=rv)
        res[lvl] = [(x['delay'], x['won'], x['reason'], round(x['seconds'], 1)) for x in r]
        won += summary['won']; tot += len(r)
        print('[levels] %-14s %s: won %d/%d  %s' % (tag, lvl, summary['won'], len(r),
              ' '.join(('%.1fs' % x['seconds']) if x['won'] else x['reason'].replace('dead at ', 'dead ') for x in r)),
              flush=True)
    print('[levels] %s, %d simulations: %d/%d' % (tag, a.sims, won, tot))
    if a.out:
        json.dump(dict(net=a.net, relvalue=a.relvalue, sims=a.sims, won=won, total=tot, per_level=res),
                  open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
