"""SMBZero's own loop: play, learn from the search, repeat. No teacher, no route.

Every target comes from the agent's own search:
  prior  <- the visit counts at each root it searched
  value  <- the value the search backed up into the nodes of its own tree (relative to that
            root, so W = b + 4 depth), which is exactly what it will predict next time

The only thing the game is asked for is what a player is told: the screen, and whether the
level ended (the next route level, or a death).

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.zeroloop --net smbzero/runs/zero8/net.pt \
      --relvalue smbzero/runs/relv1/relvalue.pt --out smbzero/runs/z1 --hours 4
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .common import DATA, MAX_DELAY, ROUTE, RUNS, THREADS, Search, e2e_segments, load_state
from .net import RelEvaluator, load as load_net, save as save_net
from .play import Game, Player
from .relvalue import RelValue, load as load_rel


class Buffer:
    """Frames the agent saw, with what its search concluded about them."""
    def __init__(self, cap=400000):
        self.cap = cap
        self.pol_x, self.pol_y = [], []                     # stack, visit distribution
        self.val_leaf, self.val_root, self.val_d, self.val_w = [], [], [], []

    def add_policy(self, stack, visits):
        self.pol_x.append(stack); self.pol_y.append(visits)

    def add_values(self, root, stacks, depth, b):
        for i in range(len(b)):
            self.val_leaf.append(stacks[i]); self.val_root.append(root)
            self.val_d.append(int(depth[i])); self.val_w.append(float(b[i]) + 4.0 * int(depth[i]))

    def trim(self):
        for a in (self.pol_x, self.pol_y):
            del a[:max(0, len(a) - self.cap // 8)]
        for a in (self.val_leaf, self.val_root, self.val_d, self.val_w):
            del a[:max(0, len(a) - self.cap)]

    def policy_batch(self, rng, n, device):
        i = rng.integers(0, len(self.pol_x), n)
        x = torch.from_numpy(np.stack([self.pol_x[k] for k in i])).to(device)
        y = torch.from_numpy(np.stack([self.pol_y[k] for k in i])).to(device)
        return x, y

    def value_batch(self, rng, n, device):
        i = rng.integers(0, len(self.val_w), n)
        leaf = torch.from_numpy(np.stack([self.val_leaf[k] for k in i])).to(device)
        root = torch.from_numpy(np.stack([self.val_root[k] for k in i])).to(device)
        dep = torch.tensor([self.val_d[k] for k in i], device=device)
        w = torch.tensor([self.val_w[k] for k in i], device=device, dtype=torch.float32)
        return leaf, root, dep, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--relvalue', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hours', type=float, default=4.0)
    ap.add_argument('--trees', type=int, default=16)
    ap.add_argument('--sims', type=int, default=600)
    ap.add_argument('--per-tree', type=int, default=64)
    ap.add_argument('--noise', type=float, default=0.25)
    ap.add_argument('--dump', type=int, default=8, help='value targets kept per decision')
    ap.add_argument('--full-game', type=int, default=2)
    ap.add_argument('--train-steps', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--levels', default=','.join(ROUTE))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'loop.log'), 'a')
    log = lambda m: (print(m, flush=True), logf.write(m + '\n'), logf.flush())
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)

    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}         # only for the starts: entry states
    levels = a.levels.split(',')
    net, _ = load_net(a.net)
    rel, _ = load_rel(a.relvalue)
    ev = RelEvaluator(net, rel, a.trees * a.per_tree)
    player = Player(s, ev, a.trees, per_tree=a.per_tree, value_mix=1.0, min_backup=True, relative=True)
    buf = Buffer()
    rng = np.random.default_rng(a.seed)
    opt_p = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    opt_v = torch.optim.AdamW(rel.parameters(), lr=a.lr, weight_decay=1e-4)
    t_end = time.time() + a.hours * 3600
    it = 0
    while time.time() < t_end:
        it += 1
        t0 = time.time()
        games, caps, tags = [], [], []
        for i in range(a.trees):
            if i < a.full_game:
                d = int(rng.integers(0, MAX_DELAY + 1))
                games.append(Game(s.frames(load_state('FullGame'), d), tag=('game', d)))
                caps.append(12000)
            else:
                l = levels[int(rng.integers(len(levels)))]
                d = int(rng.integers(0, MAX_DELAY + 1))
                games.append(Game(s.frames(segs[l]['start'], d), tag=(l, d)))
                caps.append(int(2.5 * len(segs[l]['opt'])))

        def after_decision(f, t, step):                    # the tree is whole: take its verdict
            b, dep, vis, st = f.dump(t, a.dump, seed=int(rng.integers(1 << 62)))
            if len(b):
                buf.add_values(f.state(t)[2], st, dep, b)

        player.play(games, sims=a.sims, noise=a.noise, rng=rng, max_decisions=caps,
                    segment_limit=[None if g.tag[0] == 'game' else 1 for g in games],
                    on_decision=after_decision)
        t1 = time.time()
        for g in games:
            for ep in g.episodes:
                for k in range(len(ep['policy'])):
                    if not ep['forced'][k]:                # a forced move teaches the prior nothing
                        buf.add_policy(ep['frames'][np.clip(np.arange(k - 3, k + 1), 0, None)], ep['policy'][k])
        buf.trim()
        won = {}
        for g in games:
            won.setdefault(g.tag[0], []).append(g.won)
        # train the prior on the visits and the value on the search's own backed-up values
        net.train(); rel.train()
        for _ in range(a.train_steps):
            x, y = buf.policy_batch(rng, a.batch, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = net(x)
                lp = -(y * F.log_softmax(logits.float(), 1)).sum(1).mean()
            opt_p.zero_grad(set_to_none=True); lp.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt_p.step()
            leaf, root, dep, w = buf.value_batch(rng, a.batch, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lv = F.smooth_l1_loss(rel(leaf, root, dep).float() / 16, w / 16)
            opt_v.zero_grad(set_to_none=True); lv.backward()
            torch.nn.utils.clip_grad_norm_(rel.parameters(), 5.0); opt_v.step()
        net.eval(); rel.eval()
        log('[zero] it %d: %s | play %.0f s, train %.0f s | policy %.3f value %.3f | buffer %d policy / %d value'
            % (it, ' '.join('%s %d/%d' % (k, sum(v), len(v)) for k, v in sorted(won.items())),
               t1 - t0, time.time() - t1, lp.item(), lv.item(), len(buf.pol_x), len(buf.val_w)))
        save_net(net, os.path.join(a.out, 'net.pt'), it=it)
        torch.save(dict(state=rel.state_dict(), it=it), os.path.join(a.out, 'relvalue.pt'))


if __name__ == '__main__':
    main()
