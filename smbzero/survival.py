"""Stage A's commit check, learned: does this move leave some held input alive for 24 steps?

Before committing a move, the stage A agent asks the real game whether the move, then some
button held, survives 24 steps, and takes the next favourite if not. In the latent search that
check is worth more than a perfect model: with it, deaths nearly stop (tools: blatent
--real-veto). But stage B may not touch the game to think. So learn it -- a net that reads the
real screen and the move before it and says, for each of the 12 moves, whether it survives.
Training may use the emulator; play may not.

The states are the agent's own (the latent agent's games), and short random branches off them,
with the last decisions before each death drawn more often.

  venv_retro/bin/python -m smbzero.survival make --games 'smbzero/runs/chB_*.json' --states 60000
  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.survival train --out smbzero/runs/surv0
"""
import argparse, glob, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import DATA, ROUTE, RUNS, Search, e2e_segments
from .model import NO_PREV
from .net import _Stage

HORIZON = 24


class SurvNet(nn.Module):
    """(4 x 84 x 84 frames, the move before them) -> 12 logits: does each move survive?"""
    def __init__(self, channels=(16, 32, 32), hidden=256):
        super().__init__()
        stages, cin = [], 4
        for c in channels:
            stages.append(_Stage(cin, c)); cin = c
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(cin * 11 * 11, hidden)
        self.prev = nn.Embedding(NO_PREV + 1, hidden)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 12))

    def forward(self, x, prev):
        h = F.relu(self.stages(x.float() / 255.0)).flatten(1)
        return self.out(self.fc(h) + self.prev(prev.long()))


def load(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = SurvNet()
    net.load_state_dict(ck['state'])
    return net.to(device).eval(), ck


def make(a):
    from .blatent import survives
    s = Search(threads=4)
    segs = {g['level']: g for g in e2e_segments(s)}
    games = []
    for fi, f in enumerate(sorted(glob.glob(a.games))):
        for gi, g in enumerate(json.load(open(f))):
            if len(g.get('actions', [])) > 8 and (not a.dead_only or g['reason'].startswith('dead')):
                g['gid'] = (a.seed * 1000 + fi) * 1000 + gi
                games.append(g)
    print('[survival] %d games from %s' % (len(games), a.games), flush=True)
    rng = np.random.default_rng(a.seed)
    local = threading.local()

    def emu():
        e = getattr(local, 'emu', None)
        if e is None:
            e = local.emu = Search(threads=1)
        return e

    def boundary(g):
        """The last decision of a lost game from which some move still survives 24 steps: the
        decisive states are here -- before it everything lives, after it everything dies."""
        e, acts = emu(), np.array(g['actions'], np.uint8)
        n, lo = len(acts), max(4, len(acts) - 40)
        _, st = e.replay(e.frames(segs[g['level']]['start'], g['delay']), acts[:lo])
        states = [st]
        for t in range(lo, n - 1):
            _, st = e.replay(st, acts[t:t + 1])
            states.append(st)
        for t in range(n - 1, lo - 1, -1):
            if any(survives(e, states[t - lo], x, HORIZON) for x in range(12)):
                return t
        return None

    jobs = []
    if a.boundary:
        dead = [g for g in games if g['reason'].startswith('dead')]
        with ThreadPoolExecutor(a.threads) as pool:
            bounds = list(pool.map(boundary, dead))
        dead = [(g, b) for g, b in zip(dead, bounds) if b is not None]
        print('[survival] %d lost games with a last savable decision' % len(dead), flush=True)
        for _ in range(a.states):
            g, b = dead[int(rng.integers(len(dead)))]
            t = max(4, b - int(rng.integers(0, 7)) + 1)
            k = int(rng.integers(1, 5)) if rng.random() < 0.5 else 0
            pre = (rng.integers(0, 12, k) if rng.random() < 0.5 else np.full(k, rng.integers(0, 12))).astype(np.uint8)
            jobs.append((g['level'], g['delay'], np.array(g['actions'][:t], np.uint8), pre, g['gid']))
    for _ in range(0 if a.boundary else a.states):
        g = games[int(rng.integers(len(games)))]
        n = len(g['actions'])
        near_end = g['reason'].startswith('dead') and rng.random() < a.near_frac
        t = int(rng.integers(max(4, n - a.near_window), n)) if near_end else int(rng.integers(4, n))
        k = int(rng.integers(1, 9)) if rng.random() < 0.5 else 0
        pre = (rng.integers(0, 12, k) if rng.random() < 0.5 else np.full(k, rng.integers(0, 12))).astype(np.uint8)
        jobs.append((g['level'], g['delay'], np.array(g['actions'][:t], np.uint8), pre, g['gid']))
    def label(job):
        e = emu()
        lvl, delay, acts, pre, gid = job
        out, m = e.classify_along(e.frames(segs[lvl]['start'], delay), ROUTE, np.concatenate([acts, pre]))
        path = np.concatenate([acts, pre])[:m]
        if m and out[m - 1] != 0:                  # the branch ended the game: stop just before
            path = path[:m - 1]
        if len(path) < 4:
            return None
        st0 = e.frames(segs[lvl]['start'], delay)
        _, before = e.replay(st0, path[:-4])
        obs, _, st = e.replay_obs(before, path[-4:])
        return obs[-4:], int(path[-1]), np.array([survives(e, st, x, HORIZON) for x in range(12)], bool), gid

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    X, P, Y, G, shard = [], [], [], [], 0
    with ThreadPoolExecutor(a.threads) as pool:
        for i, r in enumerate(pool.map(label, jobs)):
            if r is not None:
                X.append(r[0]); P.append(r[1]); Y.append(r[2]); G.append(r[3])
            if len(X) >= a.shard or (i == len(jobs) - 1 and X):
                np.savez_compressed(os.path.join(a.out, 'surv%04d.npz' % shard), x=np.stack(X),
                                    prev=np.array(P, np.int64), y=np.stack(Y), gid=np.array(G, np.int64))
                y = np.stack(Y)
                print('[survival] shard %d: %d states, %.1f%% of moves die, %.1f%% of states have one '
                      'that does (%.0f s)' % (shard, len(X), 100 * (1 - y.mean()), 100 * (~y).any(1).mean(),
                                               time.time() - t0), flush=True)
                X, P, Y, G, shard = [], [], [], [], shard + 1


def train(a):
    files = sorted(f for d in a.data.split(',') for f in glob.glob(os.path.join(d, '*.npz')))
    zs = [np.load(f) for f in files]
    x = np.concatenate([z['x'] for z in zs]); pv = np.concatenate([z['prev'] for z in zs])
    y = np.concatenate([z['y'] for z in zs]).astype(np.float32)
    rng = np.random.default_rng(0)
    if all('gid' in z for z in zs):
        # Held-out states must come from games the net never saw: states drawn around one moment
        # of one game are near twins, and a random split scored 94% where unseen games gave 43%.
        gid = np.concatenate([z['gid'] for z in zs])
        ug = rng.permutation(np.unique(gid))
        vg = set(ug[:max(len(ug) // 10, 1)].tolist())
        is_va = np.array([g in vg for g in gid])
        va, tr = np.where(is_va)[0], np.where(~is_va)[0]
    else:
        idx = rng.permutation(len(x)); nv = max(len(x) // 20, 1)
        va, tr = idx[:nv], idx[nv:]
    # Most states are fine whatever the move, or lost whatever the move -- the screen alone says
    # which. The veto's work is the few where one move dies and another lives: there a net that
    # learned "danger" but not "which move" lets the fatal one through (9 of 10 in play, surv0).
    s = y.sum(1)
    mixed = (s > 0) & (s < 12)
    tr_mixed = tr[mixed[tr]]
    va = va[mixed[va]] if a.eval_mixed else va
    nv = len(va)
    print('[survival] %d states (%d held out%s), %.1f%% of moves die, %.1f%% of states mixed'
          % (len(x), nv, ', mixed only' if a.eval_mixed else '', 100 * (1 - y.mean()), 100 * mixed.mean()), flush=True)
    net = SurvNet().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    live = float((y[tr] > 0.5).mean())
    die_w = min(live / max(1.0 - live, 1e-3), 50.0)     # dying moves are rare: weigh them up to the living
    X, PV, Y = torch.from_numpy(x), torch.from_numpy(pv), torch.from_numpy(y)
    os.makedirs(a.out, exist_ok=True)

    def evaluate():
        net.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, nv, 512):
                b = va[i:i + 512]
                with torch.autocast('cuda', dtype=torch.float16):
                    ps.append(torch.sigmoid(net(X[b].cuda(), PV[b].cuda()).float()).cpu().numpy())
        net.train()
        p, t = np.concatenate(ps).ravel(), y[va].ravel() > 0.5
        dies, lives = p[~t], p[t]
        # the veto's two errors: a death let through, a good move refused
        caught = (dies < 0.5).mean() if len(dies) else float('nan')
        refused = (lives < 0.5).mean() if len(lives) else float('nan')
        sub_d = dies[np.random.default_rng(1).integers(0, len(dies), min(len(dies), 3000))] if len(dies) else dies
        sub_l = lives[np.random.default_rng(2).integers(0, len(lives), min(len(lives), 3000))] if len(lives) else lives
        auc = float((sub_l[:, None] > sub_d[None, :]).mean()) if len(sub_d) and len(sub_l) else float('nan')
        return auc, caught, refused

    t0 = time.time()
    for step in range(1, a.steps + 1):
        nm = int(a.batch * a.mixed_frac) if len(tr_mixed) else 0
        b = torch.from_numpy(np.concatenate([rng.choice(tr_mixed, nm), rng.choice(tr, a.batch - nm)]) if nm
                             else rng.choice(tr, a.batch))
        xb, pb, yb = X[b].cuda(non_blocking=True), PV[b].cuda(), Y[b].cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            logit = net(xb, pb).float()
        wt = torch.where(yb > 0.5, torch.ones_like(yb), torch.full_like(yb, die_w))
        loss = F.binary_cross_entropy_with_logits(logit, yb, weight=wt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == a.steps:
            auc, caught, refused = evaluate()
            print('[survival] step %d loss %.4f | held out: AUC %.3f, deaths caught %.1f%%, good moves refused '
                  '%.1f%% (%.0f s)' % (step, loss.item(), auc, 100 * caught, 100 * refused, time.time() - t0),
                  flush=True)
            torch.save(dict(state=net.state_dict(), step=step, auc=auc, caught=caught, refused=refused),
                       os.path.join(a.out, 'surv.pt'))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('make')
    m.add_argument('--games', default=os.path.join(RUNS, 'chB_*.json'))
    m.add_argument('--states', type=int, default=60000)
    m.add_argument('--threads', type=int, default=32)
    m.add_argument('--shard', type=int, default=8192)
    m.add_argument('--seed', type=int, default=0)
    m.add_argument('--near-frac', type=float, default=0.5, help="share of a dead game's states taken near its end")
    m.add_argument('--near-window', type=int, default=30, help='how near: the last this many decisions')
    m.add_argument('--boundary', action='store_true', help="states around each lost game's last savable decision")
    m.add_argument('--dead-only', action='store_true', help='states from games that died only (where the '
                   'decisive, mixed states are)')
    m.add_argument('--out', default=os.path.join(DATA, 'surv'))
    t = sub.add_parser('train')
    t.add_argument('--data', default=os.path.join(DATA, 'surv'))
    t.add_argument('--steps', type=int, default=20000)
    t.add_argument('--batch', type=int, default=256)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--mixed-frac', type=float, default=0.0, help='share of each batch from states where some '
                   'moves die and some live')
    t.add_argument('--eval-mixed', action='store_true', help='report on held-out mixed states only')
    t.add_argument('--out', default=os.path.join(RUNS, 'surv0'))
    a = ap.parse_args()
    make(a) if a.cmd == 'make' else train(a)


if __name__ == '__main__':
    main()
