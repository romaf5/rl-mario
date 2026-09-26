"""Train the world model on the agent's trajectories, and measure what the search will need
from it: does an unrolled latent still know, k steps ahead, that the line dies, reaches the
goal, or is forced -- and does its latent still match the real frames?

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.wmtrain --steps 30000 --out smbzero/runs/wm0
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .common import DATA, RUNS
from .model import WorldModel, consistency, save

EVENTS = ('goal', 'dead', 'forced')


class Trajectories:
    """Shards of smbzero/data/world, with an index of usable unroll starts."""
    def __init__(self, pattern, unroll):
        F_, A, O, Fo, P, starts, lvl = [], [], [], [], [], [], []
        fo = ao = 0
        for p in sorted(sum((glob.glob(q) for q in pattern.split(',')), [])):
            z = np.load(p)
            F_.append(z['frames']); A.append(z['acts']); O.append(z['outcome']); Fo.append(z['forced'])
            P.append(z['prog'])                      # n + 1 per trajectory: aligned with the frames
            offs, foffs = z['offs'], z['foffs']
            for i in range(len(offs) - 1):
                n = offs[i + 1] - offs[i]
                for t in range(n):        # every position, not only those a whole unroll fits in:
                    starts.append((fo + foffs[i] + t, ao + offs[i] + t,      # a trajectory stops at its
                                   fo + foffs[i], ao + offs[i] + n - 1,      # death, so windows that must
                                   fo + foffs[i + 1] - 1))                   # fit whole would only ever
                    lvl.append(z['meta'][i, 0])                              # show it at the last step
            fo += len(z['frames']); ao += len(z['acts'])
        self.frames = torch.from_numpy(np.concatenate(F_))
        self.acts = torch.from_numpy(np.concatenate(A).astype(np.int64))
        self.out = torch.from_numpy(np.concatenate(O).astype(np.int64))
        self.forced = torch.from_numpy(np.concatenate(Fo).astype(np.float32))
        self.prog = torch.from_numpy(np.concatenate(P).astype(np.float32))
        self.starts = np.array(starts, np.int64)
        self.level = np.array(lvl, np.int32)
        self.unroll = unroll

    def __len__(self):
        return len(self.starts)

    def batch(self, rows, device):
        """-> obs (B,4,84,84), actions (B,K), events (B,K,3), the frames each step really led to.
        Past the end of a trajectory everything is held at its last step: the end absorbs, so a
        death is a death at every depth after it."""
        fi, ai = self.starts[rows, 0], self.starts[rows, 1]
        f0, aend, fend = self.starts[rows, 2], self.starts[rows, 3], self.starts[rows, 4]
        K = self.unroll
        stack = np.clip(fi[:, None] + np.arange(-3, 1)[None], f0[:, None], None)      # pad at the start
        obs = self.frames[torch.from_numpy(stack.reshape(-1))].view(len(rows), 4, 84, 84)
        step = np.minimum(ai[:, None] + np.arange(K)[None], aend[:, None])
        stp = torch.from_numpy(step)
        acts = self.acts[stp]
        out, forced = self.out[stp], self.forced[stp]
        ev = torch.stack([(out == 1).float(), (out == 2).float(), forced], -1)
        now = np.minimum(fi[:, None] + np.arange(K)[None], fend[:, None])
        nxt = np.minimum(fi[:, None] + np.arange(1, K + 1)[None], fend[:, None])
        r = (self.prog[torch.from_numpy(nxt)] - self.prog[torch.from_numpy(now)] + 4.0).clamp(min=0)
        valid = torch.from_numpy((ai[:, None] + np.arange(K)[None] <= aend[:, None]).astype(np.float32))
        # W at depth k: the frames thrown away since the unroll's root (depth 0 is 0 by definition)
        at = np.minimum(fi[:, None] + np.arange(K + 1)[None], fend[:, None])
        w = (self.prog[torch.from_numpy(at)] - self.prog[torch.from_numpy(fi)][:, None]
             + 4.0 * torch.arange(K + 1)).clamp(0, 512)
        wvalid = torch.cat([torch.ones(len(rows), 1), valid], 1)
        tgt_stack = np.clip(nxt[:, :, None] + np.arange(-3, 1)[None, None], f0[:, None, None], None)
        tgt_obs = self.frames[torch.from_numpy(tgt_stack.reshape(-1))].view(len(rows) * K, 4, 84, 84)
        return (obs.to(device), acts.to(device), ev.to(device), tgt_obs.to(device),
                r.to(device), valid.to(device), w.to(device), wvalid.to(device))


def h(x, eps=1e-3):
    """MuZero's value transform. Every decision the search makes is between lines a few
    frames apart, and on the raw scale those differences are invisible to the loss: smooth L1
    on W/16 is quadratic below 16 frames, so pricing a perfect line at 4 instead of 0 costs
    0.03 while a death-sized error costs 30. The model duly priced the route at 4.3 against a
    true 0.3 and a one-action slip at 4.8 against 3.7 -- having seen both kinds of target
    tens of thousands of times. Through h, 0 and 4 frames are 1.2 apart and 512 is 21.6."""
    return torch.sign(x) * (torch.sqrt(x.abs() + 1) - 1) + eps * x


def losses(model, obs, acts, ev, tgt_obs, r, valid, wt, wvalid,
           w_event=1.0, w_cons=1.0, w_waste=1.0, w_value=1.0, transform=False):
    lat, evs, rs, _, ws = model.unroll(obs, acts)
    e = torch.stack(evs, 1)                                  # (B, K, 3)
    pos = ev.sum((0, 1)).clamp(min=1)
    weight = (ev.numel() / 3 / pos).clamp(max=50)            # events are rare: weight them up
    le = F.binary_cross_entropy_with_logits(e, ev, pos_weight=weight)
    # (B, K, ...) flattened batch-major, the order tgt_obs is built in -- cat(dim=0)
    # would be depth-major and pair each latent with another sample's frames.
    lc = consistency(model, torch.stack(lat[1:], 1).flatten(0, 1), tgt_obs)
    pred_r = torch.stack(rs, 1)                              # the frames each step throws away
    if transform:
        lr = (F.smooth_l1_loss(h(pred_r.float()), h(r), reduction='none') * valid).sum() / valid.sum().clamp(min=1)
    else:
        lr = (F.smooth_l1_loss(pred_r.float() / 16, r / 16, reduction='none') * valid).sum() / valid.sum().clamp(min=1)
    pred_w = torch.stack(ws, 1)                              # W: thrown away since the root
    if transform:
        lw = (F.smooth_l1_loss(h(pred_w.float()), h(wt), reduction='none') * wvalid).sum() / wvalid.sum().clamp(min=1)
    else:
        lw = (F.smooth_l1_loss(pred_w.float() / 16, wt / 16, reduction='none') * wvalid).sum() / wvalid.sum().clamp(min=1)
    with torch.no_grad():
        wmae = ((pred_w.float() - wt).abs() * wvalid).sum() / wvalid.sum().clamp(min=1)
    return (w_event * le + w_cons * lc + w_waste * lr + w_value * lw,
            le.item(), lc.item(), wmae.item())


@torch.no_grad()
def gate(model, data, rows, device='cuda', batch=256):
    """Per unroll depth: does the model still know the event? Precision/recall for goal, dead,
    forced, over held-out trajectories."""
    K = data.unroll
    tp = np.zeros((K, 3)); fp = np.zeros((K, 3)); fn = np.zeros((K, 3)); n = 0
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        obs, acts, ev = data.batch(r, device)[:3]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, evs, _, _, _ = model.unroll(obs, acts)
        p = (torch.stack(evs, 1).float().sigmoid() > 0.5).cpu().numpy()
        t = ev.cpu().numpy() > 0.5
        tp += (p & t).sum(0); fp += (p & ~t).sum(0); fn += (~p & t).sum(0)
        n += len(r)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    return prec, rec, tp + fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(DATA, 'world', '*.npz'))
    ap.add_argument('--unroll', type=int, default=6, help='MuZero trains 5; the gate still unrolls further')
    ap.add_argument('--w-cons', type=float, default=2.0, help='weight on keeping the unrolled latent real')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--w-event', type=float, default=1.0,
                    help='weight on the goal/dead/forced heads: a classifier on one 84x84 frame '
                         'predicts an enemy death better (0.8 AUC on 8-1) than this model on four (0.7)')
    ap.add_argument('--transform', action='store_true',
                    help="train W and the per-step waste through MuZero's value transform")
    ap.add_argument('--out', default=os.path.join(RUNS, 'wm0'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'train.log'), 'a')
    log = lambda m: (print(m, flush=True), logf.write(m + '\n'), logf.flush())

    data = Trajectories(a.data, a.unroll)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(data))
    va, tr = perm[:4096], perm[4096:]
    log('[wm] %d unroll starts (%d frames), %d train / %d validation; events per step: %s'
        % (len(data), len(data.frames), len(tr), len(va),
           ' '.join('%s %.2f%%' % (e, 100 * v) for e, v in zip(EVENTS,
                    [(data.out == 1).float().mean(), (data.out == 2).float().mean(), data.forced.mean()]))))
    model = WorldModel().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    t0, hist = time.time(), []
    for step in range(1, a.steps + 1):
        model.train()
        obs, acts, ev, tgt, r, valid, wt, wv = data.batch(rng.choice(tr, a.batch), 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, le, lc, rmae = losses(model, obs, acts, ev, tgt, r, valid, wt, wv, w_cons=a.w_cons,
                                        w_event=a.w_event, transform=a.transform)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 2000 == 0 or step == a.steps:
            model.eval()
            prec, rec, tot = gate(model, data, va)
            hist.append(dict(step=step, event=le, consistency=lc, w_mae=rmae, s=round(time.time() - t0),
                             prec1=prec[0].tolist(), rec1=rec[0].tolist(),
                             precK=prec[-1].tolist(), recK=rec[-1].tolist()))
            log('[wm] step %5d event %.4f cons %.4f W error %.1f frames | 1 step ahead %s | %d steps ahead %s (%.0f s)'
                % (step, le, lc, rmae,
                   ' '.join('%s P%.0f/R%.0f' % (e, 100 * prec[0, i], 100 * rec[0, i]) for i, e in enumerate(EVENTS)),
                   a.unroll,
                   ' '.join('%s P%.0f/R%.0f' % (e, 100 * prec[-1, i], 100 * rec[-1, i]) for i, e in enumerate(EVENTS)),
                   time.time() - t0))
            save(model, os.path.join(a.out, 'wm.pt'), hist=hist, unroll=a.unroll)
    json.dump(hist, open(os.path.join(a.out, 'hist.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
