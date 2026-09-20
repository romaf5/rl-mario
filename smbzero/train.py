"""Training on episodes: the policy learns the target distribution (teacher one-hot or
MCTS visits; not at forced decisions), the value learns frames to go (Huber in units
of 64 frames; NaN targets are skipped).

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.train --steps 20000 --out smbzero/runs/sup0
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .common import DATA, RUNS, V_SCALE
from .net import Net, save


class Replay:
    """Episodes in one frame store; sample() builds 4-frame stacks by index."""
    def __init__(self, capacity=3_000_000, gpu_bytes=6e9):
        self.capacity = capacity
        self.gpu_bytes = gpu_bytes
        self.eps = []                     # (episode dict, first frame offset)
        self._dirty = True

    def add(self, ep, keep=False):
        ep = dict(ep)
        ep['keep'] = keep
        self.eps.append(ep)
        while sum(len(e['frames']) for e in self.eps) > self.capacity:
            i = next((i for i, e in enumerate(self.eps) if not e['keep']), None)
            if i is None:
                break
            self.eps.pop(i)
        self._dirty = True

    def load_dir(self, pattern, keep=False):
        for p in sorted(glob.glob(pattern)):
            z = np.load(p, allow_pickle=False)
            self.add({k: z[k] for k in z.files}, keep=keep)
        return self

    def _build(self):
        frames, idx, pol, val, forced, src = [], [], [], [], [], []
        off = 0
        for e in self.eps:
            n = len(e['policy'])
            t = np.arange(n)
            st = off + np.clip(t[:, None] + np.arange(-3, 1)[None], 0, None)   # frames t-3..t (frames[0] pads)
            frames.append(e['frames']); idx.append(st); pol.append(e['policy']); val.append(e['value'])
            forced.append(e['forced']); src.append(np.full(n, 1 if str(e.get('source', '')) == 'teacher' else 0))
            off += len(e['frames'])
        self.frames = torch.from_numpy(np.concatenate(frames))
        self.idx = torch.from_numpy(np.concatenate(idx).astype(np.int64))
        self.pol = torch.from_numpy(np.concatenate(pol).astype(np.float32))
        self.val = torch.from_numpy(np.concatenate(val).astype(np.float32))
        self.forced = torch.from_numpy(np.concatenate(forced).astype(bool))
        self.src = np.concatenate(src)
        self._dirty = False

    def __len__(self):
        if self._dirty:
            self._build()
        return len(self.pol)

    def sample(self, batch, rng, device):
        if self._dirty:
            self._build()
            if self.frames.numel() < self.gpu_bytes:      # small enough: gather on the GPU
                for k in ('frames', 'idx', 'pol', 'val', 'forced'):
                    setattr(self, k, getattr(self, k).to(device))
        i = torch.from_numpy(rng.integers(0, len(self.pol), batch)).to(self.pol.device)
        x = self.frames[self.idx[i].flatten()].view(batch, 4, 84, 84)
        return (x.to(device, non_blocking=True), self.pol[i].to(device), self.val[i].to(device),
                self.forced[i].to(device))


def losses(net, x, pol, val, forced):
    logits, v = net(x)
    pmask = ~forced
    lp = -(pol * F.log_softmax(logits.float(), 1)).sum(1)
    lp = (lp * pmask).sum() / pmask.sum().clamp(min=1)
    vmask = ~torch.isnan(val)
    lv = F.smooth_l1_loss((v.float() * V_SCALE / 64)[vmask], (val / 64)[vmask]) if vmask.any() else v.sum() * 0
    with torch.no_grad():
        acc = ((logits.argmax(1) == pol.argmax(1)) & pmask).sum() / pmask.sum().clamp(min=1)
        mae = ((v.float() * V_SCALE - val).abs()[vmask]).mean() if vmask.any() else torch.zeros(())
    return lp, lv, acc, mae


def train(net, replay, steps, lr=3e-4, batch=512, val_replay=None, log=print, seed=0, device='cuda', opt=None):
    net.train()
    opt = opt or torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    t0, hist = time.time(), []
    for step in range(1, steps + 1):
        x, pol, val, forced = replay.sample(batch, rng, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lp, lv, acc, mae = losses(net, x, pol, val, forced)
        loss = lp + lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if step % 500 == 0 or step == steps:
            rec = dict(step=step, policy=lp.item(), value=lv.item(), acc=acc.item(), mae_frames=mae.item(),
                       s=round(time.time() - t0))
            if val_replay is not None:
                rec.update(evaluate(net, val_replay, device=device))
                net.train()
            hist.append(rec)
            log('[train] ' + ' '.join('%s %s' % (k, ('%.4g' % v) if isinstance(v, float) else v) for k, v in rec.items()))
    net.eval()
    return opt, hist


@torch.no_grad()
def evaluate(net, replay, n=4096, device='cuda'):
    net.eval()
    rng = np.random.default_rng(123)
    x, pol, val, forced = replay.sample(n, rng, device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lp, lv, acc, mae = losses(net, x, pol, val, forced)
    return dict(val_policy=lp.item(), val_acc=acc.item(), val_mae_frames=mae.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(DATA, 'teacher', '*.npz'))
    ap.add_argument('--holdout', default='_d27,_d30', help='episode names held out for validation')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--out', default=os.path.join(RUNS, 'sup0'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hold = a.holdout.split(',') if a.holdout else []
    tr, va = Replay(), Replay()
    for p in sorted(glob.glob(a.data)):
        z = np.load(p)
        (va if any(h in os.path.basename(p) for h in hold) else tr).add({k: z[k] for k in z.files}, keep=True)
    logf = open(os.path.join(a.out, 'train.log'), 'a')
    log = lambda m: (print(m, flush=True), logf.write(m + '\n'), logf.flush())
    log('[train] %d train samples (%d episodes), %d validation samples (%d episodes)'
        % (len(tr), len(tr.eps), len(va) if va.eps else 0, len(va.eps)))
    net = Net().cuda()
    _, hist = train(net, tr, a.steps, lr=a.lr, batch=a.batch, val_replay=va if va.eps else None, log=log)
    save(net, os.path.join(a.out, 'net.pt'), hist=hist)
    json.dump(hist, open(os.path.join(a.out, 'hist.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
