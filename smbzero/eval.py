"""Evaluate SMBZero on the full game: FullGame (first control in 1-1) after d NOOP frames.

  # live: 80 ms of wall clock per decision, one game at a time
  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.eval --net smbzero/runs/sup0/net.pt --delays 0,30 --budget-ms 80
  # batch: fixed simulations per decision, many games at once
  ... --delays all --sims 600 --parallel 16
  # one level from its entry state (e2e route) + delays
  ... --level 4-2 --delays 0,20,40

Game time = the delay + 4 frames per decision, first control -> the axe. Winners are
replayed in stable-retro (--verify) and saved as route files for render_demo.
"""
import argparse, json, os, sys, time
import numpy as np
from .common import (FPS, MAX_DELAY, REPO, RUNS, THREADS, Search, e2e_segments, load_state, route_values)
from .net import Evaluator, load
from .play import Game, Player

sys.path.insert(0, os.path.join(REPO, 'search', 'tools'))


def run(net, delays, sims=None, budget_ms=None, parallel=1, per_tree=None, level=None, verify=False, out=None,
        log=print, s=None, c_puct=1.5, value_mix=0.0, min_backup=False):
    s = s or Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    if level:
        base, limit = segs[level]['start'], 1
        cap = int(2.5 * len(segs[level]['opt']))
    else:
        base, limit = load_state('FullGame'), None
        cap = int(2.5 * sum(len(g['opt']) for g in segs.values())) + 2000      # + the forced transitions
    per_tree = per_tree or (256 if budget_ms else 128)
    n_par = 1 if budget_ms else parallel
    ev = Evaluator(net, max_leaves=max(per_tree * n_par, 256))
    player = Player(s, ev, n_par, per_tree=per_tree, c_puct=c_puct, routes=route_values(segs), value_mix=value_mix,
                    min_backup=min_backup)
    results = []
    for i in range(0, len(delays), n_par):
        chunk = delays[i:i + n_par]
        games = [Game(s.frames(base, d), tag=d) for d in chunk]
        t = time.time()
        player.play(games, sims=sims, budget_s=budget_ms / 1000 if budget_ms else None, segment_limit=limit,
                    max_decisions=cap)
        for g in games:
            ds = np.array(g.decision_s) if g.decision_s else np.zeros(1)
            r = dict(delay=g.tag, won=g.won, reason=g.reason, decisions=len(g.actions),
                     frames=int(g.tag) + g.frames(), seconds=(int(g.tag) + g.frames()) / FPS,
                     decision_ms_p50=float(np.percentile(ds, 50) * 1e3), decision_ms_p99=float(np.percentile(ds, 99) * 1e3),
                     searched=len(g.decision_s))
            if out:
                np.savez(os.path.join(out, 'game_d%02d.npz' % g.tag), start=level or 'FullGame', lead_frames=g.tag,
                         actions=np.array(g.actions, np.uint8))
            if verify and g.won and not level:
                from verify_retro import verify as vr
                v = vr(load_state('FullGame'), np.array(g.actions, np.uint8), retro_state='FullGame', lead_frames=g.tag)
                r['verified'] = bool(v['ok'])
            results.append(r)
            log('[eval] d=%02d %s %s: %d decisions, %.2f s game time, decision p50 %.1f ms p99 %.1f ms%s' % (
                g.tag, 'WON' if g.won else 'lost', g.reason, len(g.actions), r['seconds'], r['decision_ms_p50'],
                r['decision_ms_p99'], (', stable-retro %s' % ('PASS' if r.get('verified') else 'FAIL')) if 'verified' in r else ''))
        log('[eval] %d/%d games done (%.0f s)' % (len(results), len(delays), time.time() - t))
    won = [r for r in results if r['won']]
    summary = dict(games=len(results), won=len(won), mean_seconds_won=float(np.mean([r['seconds'] for r in won])) if won else None,
                   sims=sims, budget_ms=budget_ms, level=level, value_mix=value_mix)
    return summary, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--delays', default='0')
    ap.add_argument('--sims', type=int)
    ap.add_argument('--budget-ms', type=float)
    ap.add_argument('--parallel', type=int, default=16)
    ap.add_argument('--per-tree', type=int)
    ap.add_argument('--c-puct', type=float, default=1.5)
    ap.add_argument('--value-mix', type=float, default=0.0, help='leaf value: this x net + (1 - this) x route')
    ap.add_argument('--min-backup', action='store_true', help='b = 4 + min over children (exact route values)')
    ap.add_argument('--level')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    delays = list(range(MAX_DELAY + 1)) if a.delays == 'all' else [int(d) for d in a.delays.split(',')]
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.net)), 'eval_%s' % time.strftime('%H%M%S'))
    os.makedirs(out, exist_ok=True)
    net, _ = load(a.net)
    logf = open(os.path.join(out, 'eval.log'), 'a')
    log = lambda m: (print(m, flush=True), logf.write(m + '\n'), logf.flush())
    summary, results = run(net, delays, sims=a.sims, budget_ms=a.budget_ms, parallel=a.parallel, per_tree=a.per_tree,
                           level=a.level, verify=a.verify, out=out, log=log, c_puct=a.c_puct, value_mix=a.value_mix,
                           min_backup=a.min_backup)
    log('[eval] summary %s' % json.dumps(summary))
    json.dump(dict(summary=summary, results=results), open(os.path.join(out, 'eval.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
