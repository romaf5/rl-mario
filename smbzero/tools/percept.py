"""Is the enemy in the picture? The same classifier, five views of the same states.

The world model misses the enemy that kills it at the start of 8-1, and nothing done to its
loss, data or size moves its death detection there (about 0.7 AUC, against 0.9 on 8-2). One
explanation fits all of that: at 84x84 in gray, an enemy is a few pixels of mid-gray among
tree trunks and fence posts, and no model can predict a collision with what it cannot see.

This tests that directly, holding everything else fixed. States along a level (off the
route by a few random steps, for variety), each labelled by the real game: does walking
right get it killed by an enemy within six steps? Every state is rendered, from one step of
the emulator, five ways -- the engine's own 84x84 gray (what the models see), and 84x84 or
full 240x224, gray or color -- and the same small classifier is trained on each. Train and
test are split by position in the level, so neighbouring states cannot leak across.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.percept --level 8-1
"""
import argparse, ctypes, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ..common import ROUTE, THREADS, Search, e2e_segments
from smbsearch import ACTION_BUTTONS, REPO, ROM

H, W = 224, 240
DYING = 0x0B
WALK = [1] * 6


class Renderer:
    """batchenv, one core: load a state, step once, get the engine's obs and the RGB frames."""
    def __init__(self):
        L = self.L = ctypes.CDLL(os.path.join(REPO, 'native', 'libbatchenv.so'))
        P, I = ctypes.c_void_p, ctypes.c_int
        L.benv_create.restype = P; L.benv_create.argtypes = [ctypes.c_char_p, I, I, I, I]
        L.benv_load.argtypes = [P, I, ctypes.c_char_p]
        L.benv_step_raw_rgb4.argtypes = [P, I, I, P, P, P]
        rom = open(ROM, 'rb').read()
        self.env = L.benv_create(rom, len(rom), 1, 1, 0)
        self.obs = np.zeros((84, 84), np.uint8)
        self.ram = np.zeros(0x800, np.uint8)
        self.rgb = np.zeros((4, H, W, 3), np.uint8)

    def step(self, state, action):
        self.L.benv_load(self.env, 0, bytes(state))
        self.L.benv_step_raw_rgb4(self.env, 0, ACTION_BUTTONS[action], self.obs.ctypes.data,
                                  self.ram.ctypes.data, self.rgb.ctypes.data)
        return self.obs.copy(), self.rgb[-1].copy()


def views(obs84, rgb):
    """-> dict of (C, H, W) uint8 arrays: the five ways of seeing one moment."""
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    small = lambda a: np.asarray(Image.fromarray(a).resize((84, 84), Image.BOX))
    return {'engine gray 84': obs84[None],
            'gray 84': small(gray)[None],
            'color 84': small(rgb).transpose(2, 0, 1),
            'gray full': gray[None],
            'color full': rgb.transpose(2, 0, 1)}


class Clf(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(cin, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        self.h = nn.Linear(64, 1)

    def forward(self, x):
        return self.h(self.c(x.float() / 255.0).amax((2, 3))).squeeze(1)   # max-pool: a small sprite counts


def auc(p, y):
    pos, neg = p[y == 1][:, None], p[y == 0][None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / max(pos.size * neg.size, 1))


def train_eval(X, y, tr, te, steps, seed=0):
    torch.manual_seed(seed)
    m = Clf(X.shape[1]).cuda()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    rng = np.random.default_rng(seed)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), device='cuda')
    for _ in range(steps):
        b = rng.choice(tr, 64)
        xb = torch.from_numpy(X[b]).cuda()
        if rng.random() < 0.5:                       # no free lunch from memorising positions
            xb = torch.roll(xb, int(rng.integers(-4, 5)), dims=3)
        loss = F.binary_cross_entropy_with_logits(m(xb), torch.from_numpy(y[b]).float().cuda(), pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = np.concatenate([m(torch.from_numpy(X[te[i:i + 128]]).cuda()).float().cpu().numpy()
                            for i in range(0, len(te), 128)])
    return auc(p, y[te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='8-1')
    ap.add_argument('--samples', type=int, default=3000)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seeds', type=int, default=2)
    a = ap.parse_args()
    s = Search(threads=THREADS)
    seg = {g['level']: g for g in e2e_segments(s)}[a.level]
    opt = np.asarray(seg['opt'], np.uint8)
    rnd = Renderer()
    rng = np.random.default_rng(0)
    data, ys, pos = {}, [], []
    while len(ys) < a.samples:
        t = int(rng.integers(8, len(opt) - 8))
        line = np.concatenate([opt[:t], rng.integers(0, 12, int(rng.integers(0, 7))).astype(np.uint8)])
        _, before = s.replay(seg['start'], line[:-1])
        out, n = s.classify_along(before, ROUTE, line[-1:])
        if np.asarray(out[:n]).any():               # the step itself ended the game: not a decision
            continue
        _, now = s.replay(before, line[-1:])
        out, n = s.classify_along(now, ROUTE, np.array(WALK, np.uint8))
        out = np.asarray(out[:n])
        if (out == 1).any():
            continue
        _, tr_, _ = s.replay_obs(now, np.array(WALK[:max(n, 1)], np.uint8))
        dead = (out == 2).any()
        enemy = dead and (tr_[:, 6] == DYING).any()
        if dead and not enemy:                      # a pit: not the question asked here
            continue
        obs84, rgb = rnd.step(before, int(line[-1]))
        for k, v in views(obs84, rgb).items():
            data.setdefault(k, []).append(v)
        ys.append(int(enemy)); pos.append(t)
    y, pos = np.array(ys), np.array(pos)
    block = (pos * 20 // len(opt)) % 4                  # every fourth stretch of the level is held out
    tr, te = np.flatnonzero(block != 3), np.flatnonzero(block == 3)
    print('[percept] %s: %d states, %d killed by an enemy within %d steps of walking right '
          '(train %d / test %d)' % (a.level, len(y), y.sum(), len(WALK), len(tr), len(te)), flush=True)
    for k in data:
        X = np.stack(data[k])
        r = [train_eval(X, y, tr, te, a.steps, seed) for seed in range(a.seeds)]
        print('[percept]   %-15s %-14s AUC %.3f  (seeds: %s)'
              % (k, 'x'.join(str(d) for d in X.shape[1:]), np.mean(r), ' '.join('%.3f' % v for v in r)),
              flush=True)


if __name__ == '__main__':
    main()
