"""What the search will need from the world model, measured on held-out trajectories.

Per unroll depth: how well an unrolled latent still knows that a line reaches the goal, dies,
or is forced (average precision -- no threshold to pick), how far off its per-step waste is,
and how close its latent stays to the one the real frames produce.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.wmgate --model smbzero/runs/wm1/wm.pt
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from ..common import DATA
from ..model import load
from ..wmtrain import EVENTS, Trajectories


def average_precision(score, label):
    """Area under the precision/recall curve, for a rare event."""
    if label.sum() == 0:
        return float('nan')
    o = np.argsort(-score)
    t = label[o].astype(np.float64)
    tp = np.cumsum(t)
    prec = tp / np.arange(1, len(t) + 1)
    return float((prec * t).sum() / t.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', default=os.path.join(DATA, 'world', '*.npz'))
    ap.add_argument('--unroll', type=int, default=10)
    ap.add_argument('--rows', type=int, default=8192)
    ap.add_argument('--out')
    a = ap.parse_args()
    data = Trajectories(a.data, a.unroll)
    rng = np.random.default_rng(0)
    rows = rng.permutation(len(data))[:a.rows]                 # the same held-out slice wmtrain uses
    model, _ = load(a.model)
    model.eval()
    K = a.unroll
    P, T, R, RT, V, cons = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            obs, acts, ev, tgt, r, valid = data.batch(rows[i:i + 256], 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lat, evs, rs, _, _ = model.unroll(obs, acts)
                t = model.proj(model.h(tgt))
                p = model.proj.predict(torch.cat(lat[1:], 0))
            P.append(torch.stack(evs, 1).float().sigmoid().cpu().numpy())
            T.append(ev.cpu().numpy())
            R.append(torch.stack(rs, 1).float().cpu().numpy()); RT.append(r.cpu().numpy()); V.append(valid.cpu().numpy())
            cons.append(F.cosine_similarity(p.float(), t.float(), dim=1).mean().item())
    P, T = np.concatenate(P), np.concatenate(T)
    R, RT, V = np.concatenate(R), np.concatenate(RT), np.concatenate(V)
    res = dict(model=a.model, consistency=float(np.mean(cons)), depth={})
    print('[wmgate] %s, %d unrolls, latent consistency %.3f' % (a.model, len(P), np.mean(cons)))
    print('[wmgate] depth |  goal AP  dead AP  forced AP | waste error (frames) | share of steps dead')
    for k in range(K):
        aps = [average_precision(P[:, k, j], T[:, k, j] > 0.5) for j in range(3)]
        m = V[:, k] > 0
        err = float(np.abs(R[:, k][m] - RT[:, k][m]).mean()) if m.any() else float('nan')
        res['depth'][k + 1] = dict(zip(EVENTS, aps), waste_mae=err)
        print('[wmgate] %5d | %8.3f %8.3f %9.3f | %13.1f | %.1f%%'
              % (k + 1, aps[0], aps[1], aps[2], err, 100 * (T[:, k, 1] > 0.5).mean()))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
