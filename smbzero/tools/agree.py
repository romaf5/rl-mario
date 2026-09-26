"""Decision quality: does planning inside the world model choose better than the prior alone?

Whole games are a noisy judge -- one death ends a game, and a level has one decisive hazard --
so ask about single decisions instead. At states the latent agent really visits, three pickers
choose a move:

  prior     the policy net's favourite, no search
  latent    the search inside the world model (the stage B agent)
  emulator  the MCTS that steps the real game (the stage A agent, 26/32) -- a strong reference

and each pick is scored three ways: does it survive in the real game (the move, then any held
input for 20 steps -- the emulator decides, not a model), the share of the emulator search's
visits it got, and whether it is the emulator's own choice. Reported over all states and over
the dangerous ones, where some move dies. If the latent search is no better than the prior
there, the world model's judgement is not yet adding anything.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.agree --model smbzero/runs/wm13/wm.pt
"""
import argparse, glob, json, re
import numpy as np
import torch
from ..common import ROUTE, THREADS, Search, e2e_segments
from ..latent import LatentTree
from ..model import load as load_model
from ..net import RelEvaluator, load as load_net
from ..play import Game, Player
from ..relvalue import load as load_rel
from .deathdiag import survivors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--net', default='smbzero/runs/zero8/net.pt')
    ap.add_argument('--relvalue', default='smbzero/runs/relv3/relvalue.pt', help="the emulator agent's value")
    ap.add_argument('--games', default='smbzero/runs/best_wm13_*.json,smbzero/runs/new_wm13_*.json')
    ap.add_argument('--states', type=int, default=192)
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    rng = np.random.default_rng(a.seed)

    pool = []
    for pat in a.games.split(','):
        for f in sorted(glob.glob(pat)):
            m = re.search(r'_(\d-\d)\.json$', f)
            if m:
                pool += [(m.group(1), g) for g in json.load(open(f)) if len(g.get('actions', [])) > 8]
    picks = []
    for _ in range(a.states):
        lvl, g = pool[int(rng.integers(len(pool)))]
        picks.append((lvl, g, int(rng.integers(4, len(g['actions'])))))

    net = load_net(a.net)[0].eval()
    wm, ck = load_model(a.model)
    wm.eval()
    tree = LatentTree(wm, max_nodes=8192, calib=ck.get('calib'), max_depth=int(ck.get('unroll', 12)),
                      prior_net=net, deep_prior='model')
    rel = load_rel(a.relvalue)[0]
    player = Player(s, RelEvaluator(net, rel, a.batch * 48), a.batch, per_tree=48, value_mix=1.0,
                    min_backup=True, relative=True)

    rows = []
    for i in range(0, len(picks), a.batch):
        chunk = picks[i:i + a.batch]
        states, stacks, prevs = [], [], []
        for lvl, g, t in chunk:
            acts = np.array(g['actions'], np.uint8)
            st0 = s.frames(segs[lvl]['start'], g['delay'])
            obs, _, _ = s.replay_obs(st0, acts[:t])
            frames = np.concatenate([np.repeat(s.obs(st0)[None], 4, 0), obs])
            states.append(s.replay(st0, acts[:t])[1]); stacks.append(frames[t:t + 4]); prevs.append(int(acts[t - 1]))
        # the emulator search: one decision per state, its visits read as it chooses
        emu_visits = {}
        def grab(f, k, step):
            if k not in emu_visits:
                emu_visits[k] = f.root(k)[0].astype(np.float64)
        games = [Game(st, tag=k) for k, st in enumerate(states)]
        player.play(games, sims=a.sims, segment_limit=1, rng=rng, max_decisions=1, on_decision=grab)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            prior = net(torch.from_numpy(np.stack(stacks)).cuda())[0].float().argmax(1).cpu().numpy()
        for k, (lvl, g, t) in enumerate(chunk):
            tree.reset(stacks[k], prevs[k])
            tree.run(a.sims)
            n, b = tree.visits()
            latent = int(np.lexsort((np.where(np.isinf(b), 1e9, b), -n))[0]) if n.sum() else int(prior[k])
            ev = emu_visits.get(k, np.zeros(12))
            share = ev / ev.sum() if ev.sum() else np.full(12, 1 / 12)
            emu = int(games[k].actions[0]) if games[k].actions else int(np.argmax(ev))
            ok = survivors(s, states[k])
            rows.append(dict(level=lvl, danger=not ok.all(), prior=int(prior[k]), latent=latent, emu=emu,
                             ok_prior=bool(ok[prior[k]]), ok_latent=bool(ok[latent]), ok_emu=bool(ok[emu]),
                             share_prior=float(share[prior[k]]), share_latent=float(share[latent])))
        print('[agree] %d/%d states' % (len(rows), len(picks)), flush=True)

    def report(name, rs):
        if not rs:
            return
        f = lambda key: 100 * np.mean([r[key] for r in rs])
        print('[agree] %-26s %3d states | survives: prior %3.0f%%  latent %3.0f%%  emulator %3.0f%% | '
              'emulator visit share: prior %.2f  latent %.2f | same move as emulator: prior %3.0f%%  latent %3.0f%%'
              % (name, len(rs), f('ok_prior'), f('ok_latent'), f('ok_emu'),
                 np.mean([r['share_prior'] for r in rs]), np.mean([r['share_latent'] for r in rs]),
                 100 * np.mean([r['prior'] == r['emu'] for r in rs]), 100 * np.mean([r['latent'] == r['emu'] for r in rs])))
    report('all', rows)
    report('danger (some move dies)', [r for r in rows if r['danger']])
    json.dump(rows, open(a.model.replace('wm.pt', 'agree.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
