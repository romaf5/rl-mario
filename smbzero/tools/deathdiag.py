"""Why did the latent search die here? Replay one game and look at its last decisions.

For each decision before the death, three columns:
  truth   which of the 12 moves can still survive: the move, then any of the 12 inputs held
          for 20 steps, played in the real game -- the emulator is the ground truth here,
          not the model and not the route
  choice  the move the search played
  model   the search's view of every move: visits, and the best cost found below it

The point is to tell apart three failures that look identical from outside: the model did
not see the danger, the model saw it and preferred the risk anyway, or the game was already
lost before the danger was visible.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.deathdiag \
      --model smbzero/runs/wm7/wm.pt --game run.json --level 8-1 --last 16
"""
import argparse, json
import numpy as np
from ..common import ROUTE, THREADS, Search, e2e_segments
from ..latent import LatentTree
from ..model import load as load_model

HOLD = 20


def survivors(s, state):
    """-> bool per move: can it be followed by any held input for HOLD steps without dying?"""
    ok = np.zeros(12, bool)
    for a in range(12):
        for c in range(12):
            out, n = s.classify_along(state, ROUTE, np.array([a] + [c] * HOLD, np.uint8))
            if not (np.asarray(out[:n]) == 2).any():
                ok[a] = True
                break
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--game', required=True, help="latent --out json; its first game is replayed")
    ap.add_argument('--level', default='8-1')
    ap.add_argument('--last', type=int, default=16, help='decisions to inspect before the end')
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--backup', default='self', choices=('self', 'children'))
    a = ap.parse_args()
    game = json.load(open(a.game))[0]
    acts = np.array(game['actions'], np.uint8)
    s = Search(threads=THREADS)
    seg = {g['level']: g for g in e2e_segments(s)}[a.level]
    start = s.frames(seg['start'], game['delay'])
    model, ck = load_model(a.model)
    model.eval()
    tree = LatentTree(model, max_nodes=8192, calib=ck.get('calib'), max_depth=int(ck.get('unroll', 12)),
                      backup=a.backup)

    obs, _, _ = s.replay_obs(start, acts)
    frames = np.concatenate([np.repeat(s.obs(start)[None], 4, 0), obs])
    T = len(acts)
    print('[deathdiag] %s delay %d: %d decisions, %s' % (a.level, game['delay'], T, game['reason']))
    print('  t   played  can survive (moves)            search: most visited .. least, (visits, best cost)')
    for t in range(max(0, T - a.last), T):
        _, st = s.replay(start, acts[:t])
        ok = survivors(s, st)
        tree.reset(frames[t:t + 4])
        tree.run(a.sims)
        n, b = tree.visits()
        order = np.argsort(-n)
        view = ' '.join('%d%s(%d,%.0f)' % (m, '' if ok[m] else 'x', n[m], b[m] if np.isfinite(b[m]) else -1)
                        for m in order[:6])
        flag = '' if ok[acts[t]] else '  <-- played a move that cannot survive'
        if not ok.any():
            flag = '  (already lost: no move survives)'
        print('  %3d  %2d%s     %-30s %s%s' % (t, acts[t], '' if ok[acts[t]] else 'x',
                                               ','.join(str(m) for m in np.flatnonzero(ok)) or '-',
                                               view, flag))
    print('  (x: that move dies within %d steps whatever follows it)' % HOLD)


if __name__ == '__main__':
    main()
