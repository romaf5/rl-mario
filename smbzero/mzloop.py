"""MuZero's loop, in the world model: play with the search inside the model, learn from what
the search chose and what really happened, keep the model only if it plays better.

  per iteration:
    self-play   blatent --value tg --noise --temp --record: 8 levels x 8 random start delays
    data        mzdata: the real screens and outcomes, visit counts, n-step frames to go
    train       wmtrain --init <best> --tg --sp-frac 0.5: self-play (the last few iterations)
                with the world data for grounding
    calibrate   tools.wmcal: the event heads' temperatures, which the search prices death with
    gate        blatent --value tg, the 32 fixed games (8 levels x 5,20,35,50): wins, then progress

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.mzloop --init smbzero/runs/wm20/wm.pt --out smbzero/runs/mz0
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
import numpy as np
from .common import DATA, ROUTE, RUNS

PY = sys.executable
WORLD = ','.join(os.path.join(DATA, d, '*.npz') for d in ('world', 'world_agent', 'world_sib2', 'world_app'))
WORLD_RAM = os.path.join(DATA, 'world_ram', 'p*', '*.npz')


def run(args, log, tries=2):
    """One step of an iteration; a failed step is tried once more a minute later (a CUDA out of
    memory when another job briefly filled the GPU stopped the first loop at its first self-play)."""
    t = time.time()
    for k in range(tries):
        with open(log, 'w') as f:
            r = subprocess.run([PY, '-m'] + args, stdout=f, stderr=subprocess.STDOUT)
        if not r.returncode:
            return time.time() - t
        if k + 1 < tries:
            time.sleep(60)
    raise RuntimeError('%s failed (%s)' % (args[0], log))


def score(path):
    games = json.load(open(path))
    won = sum(g['won'] for g in games)
    prog = float(np.nanmean([g.get('progress', np.nan) for g in games]))
    return won, prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True, help='a world model with a frames-to-go head')
    ap.add_argument('--prior-net', default=os.path.join(RUNS, 'zero8', 'net.pt'))
    ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--games-per-level', type=int, default=8)
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--noise', type=float, default=0.25)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--search', default='--vloss --per-wave 128 --no-floor',
                    help="the search's flags, for self-play and gates alike")
    ap.add_argument('--value-teacher', default=os.path.join(RUNS, 'relv3', 'relvalue.pt'),
                    help="the W head's teacher on every row ('' for the route)")
    ap.add_argument('--value', default='tg', choices=('w', 'tg'), help="the search's leaf price")
    ap.add_argument('--w-tgdiff', type=float, default=4.0, help='frames-to-go differences within an unroll')
    ap.add_argument('--window', type=int, default=5, help='iterations of self-play kept for training')
    ap.add_argument('--ram', action='store_true', help='a model reading the RAM (world_ram data, RAM in the shards)')
    ap.add_argument('--accept', default='gate', choices=('gate', 'always'),
                    help="gate: a candidate replaces the model only if it gates at least as well; always: "
                         "MuZero's way, every candidate goes on (best.pt still keeps the best gate)")
    ap.add_argument('--out', default=os.path.join(RUNS, 'mz0'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'loop.log'), 'a')
    log = lambda m: (print(m, flush=True), logf.write(m + '\n'), logf.flush())
    blat = ['smbzero.blatent', '--prior-net', a.prior_net, '--deep-prior', 'model', '--backup', 'children',
            '--value', a.value, '--sims', str(a.sims)] + a.search.split()
    # With the W search the root value is not frames to go, so self-play teaches the policy (the
    # search's visit counts) and the dynamics and events (what really happened where the search
    # goes) -- and the W and frames-to-go heads keep their route targets from the world data.
    tg_w = ['--tg', '--w-tg', '1', '--w-tgdiff', str(a.w_tgdiff)] if a.value == 'tg' else []
    world = WORLD_RAM if a.ram else WORLD
    ram = ['--ram'] if a.ram else []
    best = os.path.join(a.out, 'best.pt')
    cur = os.path.join(a.out, 'cur.pt') if a.accept == 'always' else best    # what plays and trains on
    if not os.path.exists(best):
        shutil.copy(a.init, best)
        if cur != best:
            shutil.copy(a.init, cur)
        dt = run(blat + ['--model', best, '--cap', '2.5', '--out', os.path.join(a.out, 'gate_init.json')],
                 os.path.join(a.out, 'gate_init.log'))
        w0, p0 = score(os.path.join(a.out, 'gate_init.json'))
        log('[mz] start: %s -- gate %d/32, %.0f%% of the level (%.0f s)' % (a.init, w0, 100 * p0, dt))
    best_score = score(os.path.join(a.out, 'gate_init.json'))
    rng = np.random.default_rng(len(glob.glob(os.path.join(a.out, 'it*'))))
    for it in range(a.iters):
        d = os.path.join(a.out, 'it%02d' % it)
        if os.path.exists(os.path.join(d, 'done')):
            best_score = tuple(json.load(open(os.path.join(d, 'done')))['best'])
            continue
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        # self-play: every level, random start delays (the gate's own delays are not special)
        delays = ','.join(str(x) for x in sorted(rng.choice(61, a.games_per_level, replace=False)))
        sp = os.path.join(d, 'selfplay.json')
        run(blat + ['--model', cur, '--cap', '2.5', '--delays', delays, '--noise', str(a.noise), '--temp',
                    str(a.temp), '--record', '--seed', str(it), '--out', sp], os.path.join(d, 'selfplay.log'))
        spw, spp = score(sp)
        shards = os.path.join(DATA, 'mz', os.path.basename(a.out), 'it%02d' % it)
        run(['smbzero.mzdata', '--games', sp, '--out', shards] + ram, os.path.join(d, 'mzdata.log'))
        keep = ','.join(os.path.join(DATA, 'mz', os.path.basename(a.out), 'it%02d' % j, '*.npz')
                        for j in range(max(0, it - a.window + 1), it + 1))
        cand = os.path.join(d, 'wm')
        run(['smbzero.wmtrain', '--data', world + ',' + keep, '--steps', str(a.steps), '--lr', str(a.lr),
             '--unroll', '12', '--w-cons', '2.0', '--transform', '--edge', '--pi-mlp'] + tg_w + ram + [
             '--distill', a.prior_net, '--w-policy', '4', '--sp-frac', '0.5', '--init', cur, '--out', cand]
            + (['--value-teacher', a.value_teacher] if a.value_teacher else []),
            os.path.join(d, 'train.log'))
        run(['smbzero.tools.wmcal', '--model', os.path.join(cand, 'wm.pt'), '--data', world, '--unroll', '12',
             '--rows', '8192'], os.path.join(d, 'wmcal.log'))
        gate = os.path.join(d, 'gate.json')
        run(blat + ['--model', os.path.join(cand, 'wm.pt'), '--cap', '2.5', '--out', gate], os.path.join(d, 'gate.log'))
        sc = score(gate)
        kept = sc >= best_score
        if kept:
            shutil.copy(os.path.join(cand, 'wm.pt'), best)
            best_score = sc
        if cur != best:
            shutil.copy(os.path.join(cand, 'wm.pt'), cur)
        log('[mz] it %02d: self-play %d/%d won, %.0f%% | gate %d/32, %.0f%% of the level -> %s (best %d/32, %.0f%%) (%.0f s)'
            % (it, spw, 8 * a.games_per_level, 100 * spp, sc[0], 100 * sc[1],
               'best' if kept else 'sent back' if cur == best else 'kept (not best)',
               best_score[0], 100 * best_score[1], time.time() - t0))
        json.dump(dict(score=list(sc), best=list(best_score), kept=kept), open(os.path.join(d, 'done'), 'w'))


if __name__ == '__main__':
    main()
