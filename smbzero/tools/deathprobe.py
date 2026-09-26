"""Can the world model see an enemy coming, or only a pit?

deathdiag found one death the model was blind to: walking right into an enemy, priced the
same as jumping over it. This asks whether that is general. From many states along each
level, play a few plain lines (walk, run, stand, jump-then-walk, hold run-jump) for ten
steps in the real game, and label every line by what happened: survived, killed by an
enemy (the engine's dying state, $000E = 0x0B), or lost another way (a pit, mostly). Then
score each line by the model's calibrated p(dead), highest over the ten steps, and ask how
well that score separates each kind of death from the lines that survived (AUC: 0.5 is a
coin, 1.0 is perfect).

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.deathprobe --model smbzero/runs/wm7/wm.pt
"""
import argparse
import numpy as np
import torch
from ..common import ROUTE, THREADS, Search, e2e_segments
from ..model import load as load_model

LINES = {'walk': [1] * 10, 'run': [3] * 10, 'stand': [0] * 10,
         'jump, walk': [2] + [1] * 9, 'run-jump': [4] * 10}
DYING = 0x0B


def auc(pos, neg):
    """P(a random positive scores above a random negative); ties count half."""
    if not len(pos) or not len(neg):
        return float('nan')
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='one checkpoint, or several comma-separated to average')
    ap.add_argument('--levels', default='1-1,4-1,8-1,8-2,8-3')
    ap.add_argument('--states', type=int, default=96)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    models = []                      # several: average their calibrated p -- misses that are
    for path in a.model.split(','):  # one model's quirks need not be the others'
        m, ck = load_model(path)
        models.append((m.eval(), ck.get('calib')))
    rng = np.random.default_rng(a.seed)
    allrows = []
    for lvl in a.levels.split(','):
        seg = segs[lvl]
        opt = seg['opt']
        rows = []
        for _ in range(a.states):
            t = int(rng.integers(4, max(len(opt) - 12, 5)))
            _, before = s.replay(seg['start'], opt[:t - 4])
            obs, _, state = s.replay_obs(before, opt[t - 4:t])
            x = torch.from_numpy(obs[-4:][None]).cuda()
            for name, line in LINES.items():
                line = np.array(line, np.uint8)
                out, n = s.classify_along(state, ROUTE, line)
                out = np.asarray(out[:n])
                _, tr, _ = s.replay_obs(state, line[:max(n, 1)])
                dead = (out == 2).any()
                cause = 'survived' if not dead else ('enemy' if (tr[:, 6] == DYING).any() else 'other')
                ps = []
                for model, cal in models:
                    with torch.autocast('cuda', dtype=torch.float16):
                        _, evs, _, _, _ = model.unroll(x, torch.from_numpy(line.astype(np.int64))[None].cuda())
                    lg = torch.stack(evs, 1).float()[0, :, 1].cpu().numpy()
                    if cal is not None:
                        k = np.arange(len(lg)).clip(max=len(cal) - 1)
                        lg = lg / cal[k, 1, 0] + cal[k, 1, 1]
                    ps.append(1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30))))
                p = np.mean(ps, 0)
                rows.append((lvl, name, cause, float(p.max())))
        allrows += rows
        c = [r[2] for r in rows]
        sv = [r[3] for r in rows if r[2] == 'survived']
        print('[deathprobe] %s: %d lines, %d survived, %d killed by an enemy, %d other deaths | '
              'AUC enemy %.2f, other %.2f'
              % (lvl, len(rows), c.count('survived'), c.count('enemy'), c.count('other'),
                 auc([r[3] for r in rows if r[2] == 'enemy'], sv),
                 auc([r[3] for r in rows if r[2] == 'other'], sv)), flush=True)
    sv = [r[3] for r in allrows if r[2] == 'survived']
    en = [r[3] for r in allrows if r[2] == 'enemy']
    ot = [r[3] for r in allrows if r[2] == 'other']
    print('[deathprobe] all levels: enemy deaths %d, AUC %.2f (mean p %.2f) | other deaths %d, AUC %.2f '
          '(mean p %.2f) | survived %d (mean p %.2f)'
          % (len(en), auc(en, sv), np.mean(en) if en else 0, len(ot), auc(ot, sv),
             np.mean(ot) if ot else 0, len(sv), np.mean(sv) if sv else 0))


if __name__ == '__main__':
    main()
