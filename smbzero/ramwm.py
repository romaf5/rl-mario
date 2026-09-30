"""A world model of the game's RAM: the next 2 KB from the RAM now and the move.

A day of pixel world models stayed at 0-4/32 whatever was changed, and every failure came back to
imagined futures not being accurate enough -- sprites blurred, deaths missed, the move that matters
unseen. The RAM is the game's exact state, and its step is deterministic: learn that instead. The
agent still plans only in its head; it reads the RAM at play time rather than the screen.

The model sees the game-state bytes as bits (the stack and the sprite buffer, which the screen
redraws from the rest, are left out) and predicts the next step's bits; a learned per-bit skip
carries over what does not change. Trained unrolled on its own predictions, like the pixel model,
and measured on held-out trajectories where it matters: the bytes of Mario, the engine and the
enemies, exactly right k steps ahead.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.ramwm --data 'smbzero/data/world_ram/p*/*.npz' --out smbzero/runs/ram0
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import RUNS
from .model import NO_PREV

KEEP = np.r_[0:0x100, 0x300:0x800]                  # game-state bytes: not the stack, not the sprite buffer
NBITS = len(KEEP) * 8
KEY = {                                            # what the search will care about
    'mario x': [0x6D, 0x86], 'mario y': [0xB5, 0xCE], 'engine': [0x0E], 'lives': [0x75A],
    'enemy x': [0x6E, 0x6F, 0x70, 0x71, 0x72, 0x87, 0x88, 0x89, 0x8A, 0x8B],
    'enemy y': [0xCF, 0xD0, 0xD1, 0xD2, 0xD3], 'enemy flags': [0x0F, 0x10, 0x11, 0x12, 0x13],
}
_POS = {int(b): i for i, b in enumerate(KEEP)}


def to_bits(ram):
    """(..., 2048) uint8 -> (..., NBITS) float {0, 1}"""
    r = ram[..., torch.as_tensor(KEEP, device=ram.device)].long()
    shifts = torch.arange(8, device=ram.device)
    return ((r[..., None] >> shifts) & 1).flatten(-2).float()


def to_bytes(bits):
    """(..., NBITS) {0,1} -> (..., len(KEEP)) values"""
    b = bits.view(*bits.shape[:-1], len(KEEP), 8).long()
    return (b << torch.arange(8, device=bits.device)).sum(-1)


class RamModel(nn.Module):
    def __init__(self, hidden=2048):
        super().__init__()
        self.inp = nn.Linear(NBITS + 12 + NO_PREV + 1, hidden)
        self.h1, self.h2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, NBITS)
        self.keep = nn.Parameter(torch.full((NBITS,), 4.0))     # the skip: most bits carry over

    def forward(self, bits, a, prev):
        """bits (B, NBITS) in [0, 1], the move, the move before -> logits of the next bits"""
        x = torch.cat([bits, F.one_hot(a.long(), 12).float(), F.one_hot(prev.long(), NO_PREV + 1).float()], 1)
        h = F.relu(self.inp(x))
        h = F.relu(self.h1(h)) + h
        h = F.relu(self.h2(h)) + h
        return self.out(h) + self.keep * (2 * bits - 1)


class RamData:
    """RAM trajectories (wmdata --ram shards), split into train / held-out by trajectory."""
    def __init__(self, pattern, unroll, held=0.05, seed=0):
        R, A, starts, traj = [], [], [], []
        ro = ao = 0
        for ti, p in enumerate(sorted(sum((glob.glob(q) for q in pattern.split(',')), []))):
            z = np.load(p)
            R.append(z['ram']); A.append(z['acts'])
            offs, foffs = z['offs'], z['foffs']
            for i in range(len(offs) - 1):
                n = offs[i + 1] - offs[i]
                tid = len(traj)
                traj.append(tid)
                for t in range(n):
                    starts.append((ro + foffs[i] + t, ao + offs[i] + t, ao + offs[i], ao + offs[i] + n - 1, tid))
            ro += len(z['ram']); ao += len(z['acts'])
        self.ram = torch.from_numpy(np.concatenate(R))
        self.acts = torch.from_numpy(np.concatenate(A).astype(np.int64))
        self.starts = np.array(starts, np.int64)
        rng = np.random.default_rng(seed)
        held_t = set(rng.choice(len(traj), max(1, int(len(traj) * held)), replace=False).tolist())
        isva = np.array([s[4] in held_t for s in starts])
        self.tr, self.va = np.where(~isva)[0], np.where(isva)[0]
        self.unroll = unroll

    def batch(self, rows, device):
        s = self.starts[rows]
        ri, ai, a0, aend = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        K = self.unroll
        steps = np.arange(K)[None]
        valid = torch.from_numpy((ai[:, None] + steps <= aend[:, None]).astype(np.float32))
        acts = self.acts[torch.from_numpy(np.minimum(ai[:, None] + steps, aend[:, None]))]
        # the RAM after each step (held at the trajectory's end)
        n_left = (aend - ai + 1)[:, None]
        ram_idx = ri[:, None] + np.minimum(np.arange(K + 1)[None], n_left)
        rams = self.ram[torch.from_numpy(ram_idx)]
        prev = torch.from_numpy(np.where(ai - 1 >= a0, self.acts.numpy()[np.maximum(ai - 1, 0)], NO_PREV))
        return rams.to(device), acts.to(device), prev.to(device), valid.to(device)


def unroll(model, bits0, acts, prev, hard=False):
    """-> logits of the bits after each of K steps, fed on the model's own predictions"""
    bits, out = bits0, []
    for k in range(acts.shape[1]):
        lg = model(bits, acts[:, k], prev if k == 0 else acts[:, k - 1])
        out.append(lg)
        p = torch.sigmoid(lg.float())
        bits = (p > 0.5).float() if hard else p
    return torch.stack(out, 1)


@torch.no_grad()
def evaluate(model, data, rows, device='cuda'):
    """Per depth, on held-out trajectories, imagined with hard bits (as the search will): the share of
    the bits that changed since the start still wrong, and each key quantity exactly right."""
    K = data.unroll
    wrong = np.zeros(K); changed = np.zeros(K); cnt = np.zeros(K)
    key_ok = {k: np.zeros(K) for k in KEY}
    for i in range(0, len(rows), 512):
        rams, acts, prev, valid = data.batch(rows[i:i + 512], device)
        bits = to_bits(rams)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = unroll(model, bits[:, 0], acts, prev, hard=True)
        pred = (lg.float() > 0).float()
        true = bits[:, 1:]
        ch = (true != bits[:, :1]).float()                       # bits that moved since the start
        v = valid.bool()
        wrong += (((pred != true).float() * ch).sum(-1) * valid).sum(0).cpu().numpy()
        changed += ((ch.sum(-1)) * valid).sum(0).cpu().numpy()
        pb, tb = to_bytes(pred), to_bytes(true)
        for name, addrs in KEY.items():
            idx = [_POS[x] for x in addrs]
            ok = (pb[..., idx] == tb[..., idx]).all(-1)
            key_ok[name] += (ok & v).sum(0).cpu().numpy()
        cnt += valid.sum(0).cpu().numpy()
    return wrong / np.maximum(changed, 1), {k: v / np.maximum(cnt, 1) for k, v in key_ok.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='smbzero/data/world_ram/p*/*.npz')
    ap.add_argument('--unroll', type=int, default=12)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--hidden', type=int, default=2048)
    ap.add_argument('--out', default=os.path.join(RUNS, 'ram0'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'train.log'), 'a')
    log = lambda m: (print(m, flush=True), logf.write(m + '\n'), logf.flush())
    data = RamData(a.data, a.unroll)
    log('[ram] %d unroll starts (%d train, %d held out by trajectory), %d RAM frames; %d bits a state'
        % (len(data.starts), len(data.tr), len(data.va), len(data.ram), NBITS))
    model = RamModel(a.hidden).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    rng = np.random.default_rng(0)
    va = rng.choice(data.va, min(len(data.va), 4096), replace=False)
    t0, hist = time.time(), []
    for step in range(1, a.steps + 1):
        rams, acts, prev, valid = data.batch(rng.choice(data.tr, a.batch), 'cuda')
        bits = to_bits(rams)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = unroll(model, bits[:, 0], acts, prev)
        loss = (F.binary_cross_entropy_with_logits(lg.float(), bits[:, 1:], reduction='none').mean(-1) * valid).sum() \
            / valid.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 2000 == 0 or step == a.steps:
            model.eval()
            wr, ko = evaluate(model, data, va)
            model.train()
            d = [0, 3, 7, 11] if a.unroll >= 12 else list(range(a.unroll))
            hist.append(dict(step=step, loss=float(loss.detach()), wrong=wr.tolist(), key={k: v.tolist() for k, v in ko.items()}))
            log('[ram] step %5d loss %.5f | changed bits still wrong at depth 1/4/8/12: %s | exactly right at 1/4/8/12: %s (%.0f s)'
                % (step, loss.item(), '/'.join('%.1f%%' % (100 * wr[i]) for i in d),
                   '  '.join('%s %s' % (k, '/'.join('%.0f' % (100 * v[i]) for i in d)) for k, v in ko.items()),
                   time.time() - t0))
            torch.save(dict(state=model.state_dict(), hidden=a.hidden, unroll=a.unroll, hist=hist),
                       os.path.join(a.out, 'ram.pt'))


if __name__ == '__main__':
    main()
