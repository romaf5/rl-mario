"""Make the world model's event probabilities mean what they say.

The event heads are trained with a positive weight of up to 50, because a death is rare and
a model that never predicts one scores well. That weight buys recall and destroys
calibration: the model calls 0.4 what happens one time in twenty. A search that prices a
node at p(dead) x 512 frames then flinches away from every line, which is exactly what the
latent search did.

So fit one temperature and one bias per event head, per unroll depth, on held-out
trajectories -- the cheapest honest correction there is, and it needs no retraining. The
fitted numbers go into the checkpoint as `calib`; `latent.py` applies them.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.wmcal --model smbzero/runs/wm4/wm.pt
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from ..common import DATA
from ..model import load as load_model
from ..wmtrain import EVENTS, Trajectories


@torch.no_grad()
def collect(model, data, rows, device='cuda', batch=256):
    """-> logits (N, K, 3) and truths (N, K, 3) over held-out unroll starts."""
    L, T = [], []
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        b = data.batch(r, device)
        obs, acts, ev, prev = b[0], b[1], b[2], b[8]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, evs, _, _, _ = model.unroll(obs, acts, prev)
        L.append(torch.stack(evs, 1).float().cpu())
        T.append(ev.float().cpu())
    return torch.cat(L), torch.cat(T)


def fit(logit, truth, iters=200):
    """One temperature and bias minimising the negative log likelihood. Returns (T, b)."""
    t = torch.zeros(1, requires_grad=True)          # log-temperature, so T = exp(t) > 0
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t, b], lr=0.3, max_iter=iters)

    def step():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logit * torch.exp(-t) + b, truth)
        loss.backward()
        return loss
    opt.step(step)
    return float(torch.exp(t).detach()), float(b.detach())


def report(name, p, truth, thr=0.5):
    pos = truth > 0.5
    hit = p > thr
    tp = (hit & pos).sum(); fp = (hit & ~pos).sum(); fn = (~hit & pos).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    brier = ((p - truth.astype(np.float32)) ** 2).mean()
    conf = p[hit].mean() if hit.any() else 0.0
    print('  %-34s P%3.0f/R%3.0f  Brier %.4f  says %.2f where it fires, is right %.2f'
          % (name, 100 * prec, 100 * rec, brier, conf,
             pos[hit].mean() if hit.any() else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', default='smbzero/data/world*/*.npz')
    ap.add_argument('--unroll', type=int, default=6)
    ap.add_argument('--rows', type=int, default=8192, help='held-out unroll starts to fit on')
    a = ap.parse_args()
    model, ck = load_model(a.model)
    model.eval()
    data = Trajectories(a.data, a.unroll)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(data))
    va = perm[:4096]                                 # the same held-out split wmtrain used
    extra = perm[4096:4096 + a.rows]                 # and more of it, to fit on
    logit, truth = collect(model, data, np.concatenate([va, extra]))
    half = len(va)
    K = logit.shape[1]

    calib = np.zeros((K, 3, 2), np.float32)
    print('[wmcal] fitting on %d unroll starts, measuring on %d held out' % (len(extra), half))
    for k in range(K):
        for e in range(3):
            T, b = fit(logit[half:, k, e], truth[half:, k, e])
            calib[k, e] = (T, b)
    for e, name in enumerate(EVENTS):
        print('[wmcal] %s' % name)
        for k in (0, K - 1):
            raw = torch.sigmoid(logit[:half, k, e]).numpy()
            T, b = calib[k, e]
            cal = torch.sigmoid(logit[:half, k, e] / T + b).numpy()
            t = truth[:half, k, e].numpy()
            report('%d step raw' % (k + 1), raw, t)
            report('%d step calibrated (T %.2f b %+.2f)' % (k + 1, T, b), cal, t)

    ck = dict(ck) if isinstance(ck, dict) else {}
    torch.save(dict(state=model.state_dict(), calib=calib, **{k: v for k, v in ck.items()
                                                              if k not in ('state', 'calib')}), a.model)
    print('[wmcal] saved calibration into %s' % a.model)


if __name__ == '__main__':
    main()
