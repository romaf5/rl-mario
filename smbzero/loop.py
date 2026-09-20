"""Expert iteration with DAgger: self-play with net-guided MCTS -> label -> train -> repeat.

Each iteration plays one segment per tree with root noise: a --full share of the games
start at a random route level's entry state (+ a random 0-60 frame delay) and play the
whole level; the rest start at a random state along a teacher route of that level and
play --rollout decisions (roll-in by the teacher, roll-out by the agent: every part of
every level gets the agent's own states each iteration). Every k-th searched state the agent
visits is labelled by the local teacher (a short beam along the level's optimised
route: its first action and its frames to go) -- the net learns on its own states,
not only on the teacher's routes. Won segments also add their MCTS visit counts and
the frames they actually took. Every --eval-every iterations: full games from 1-1.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.loop --init smbzero/runs/sup0/net.pt --out smbzero/runs/zero1
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from .common import (DATA, MAX_DELAY, ROUTE, THREADS, Search, e2e_segments, episode, route_values, save_episode)
from .eval import run as run_eval
from .net import Evaluator, load, save
from .play import Game, Player
from .train import Replay, train


def label(s, ep, seg, beam, horizon, max_labels):
    """The local teacher on (at most max_labels, evenly spaced) recorded states of an
    episode -> a DAgger episode (policy and value targets only where labelled)."""
    n = len(ep['policy'])
    pol = np.zeros((n, 12), np.float32)
    val = np.full(n, np.nan, np.float32)
    unlabelled = np.ones(n, bool)
    k = len(ep['label_idx'])
    pick = np.unique(np.linspace(0, k - 1, min(k, max_labels)).round().astype(int)) if k else []
    for j in pick:
        i, st = ep['label_idx'][j], ep['label_states'][j]
        a, est, _ = s.lookahead(st.tobytes(), ROUTE, seg['opt'], ref_start=seg['start'], beam=beam, horizon=horizon)
        if len(a):
            pol[i, a[0]] = 1; val[i] = est; unlabelled[i] = False
    return episode(ep['frames'], pol, val, unlabelled, level=str(ep['level']), source='dagger')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hours', type=float, default=4.0)
    ap.add_argument('--levels', default=','.join(ROUTE))
    ap.add_argument('--trees', type=int, default=16)
    ap.add_argument('--sims', type=int, default=200)
    ap.add_argument('--per-tree', type=int, default=64)
    ap.add_argument('--noise', type=float, default=0.25)
    ap.add_argument('--cap', type=float, default=2.5, help='decisions per full level: this x the teacher route')
    ap.add_argument('--full', type=float, default=0.25, help='share of games that play a whole level')
    ap.add_argument('--rollout', type=int, default=80, help='decisions of a game started on a teacher route')
    ap.add_argument('--label-every', type=int, default=6)
    ap.add_argument('--label-beam', type=int, default=100)
    ap.add_argument('--label-horizon', type=int, default=30)
    ap.add_argument('--max-labels', type=int, default=40, help='per segment')
    ap.add_argument('--train-steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--eval-delays', default='0,15,30,45,60')
    ap.add_argument('--eval-sims', type=int, default=400)
    ap.add_argument('--eval-level', default=None, help='evaluate one level instead of the full game')
    ap.add_argument('--value-mix', type=float, default=0.0, help='leaf value: this x net + (1 - this) x route')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.out.rstrip('/'))
    logf = open(os.path.join(a.out, 'loop.log'), 'a')
    log = lambda m: (print(m, flush=True), logf.write(m + '\n'), logf.flush())
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)

    s = Search(threads=THREADS)
    segs = {g['level']: g for g in e2e_segments(s)}
    levels = a.levels.split(',')
    net, _ = load(a.init)
    rep = Replay(capacity=2_000_000)
    rep.load_dir(os.path.join(DATA, 'teacher', '*.npz'), keep=True)
    rep.load_dir(os.path.join(DATA, 'dagger', '*.npz'), keep=True)
    log('[loop] %d teacher + dagger samples' % len(rep))
    routes = {}                                            # level -> [(start state, teacher actions)]
    for p in sorted(glob.glob(os.path.join(DATA, 'teacher', '*.npz'))):
        z = np.load(p)
        routes.setdefault(str(z['level']), []).append((z['start_state'].tobytes(), z['policy'].argmax(1).astype(np.uint8)))
    ev = Evaluator(net, max_leaves=a.trees * a.per_tree)
    player = Player(s, ev, a.trees, per_tree=a.per_tree, max_nodes=1 << 15, routes=route_values(segs),
                    value_mix=a.value_mix)
    rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    t_end = time.time() + a.hours * 3600
    it = 0
    while time.time() < t_end:
        it += 1
        t0 = time.time()
        lv = rng.choice(levels, a.trees)
        games, caps = [], []
        for i, l in enumerate(lv):
            if i < round(a.full * a.trees):                # the whole level from its entry
                d = int(rng.integers(0, MAX_DELAY + 1))
                games.append(Game(s.frames(segs[l]['start'], d), tag=(l, d, -1)))
                caps.append(int(a.cap * len(segs[l]['opt'])))
            else:                                          # from a teacher route state
                st, acts = routes[l][rng.integers(len(routes[l]))]
                t = int(rng.integers(0, len(acts)))
                games.append(Game(s.replay(st, acts[:t])[1], tag=(l, t, t)))
                caps.append(a.rollout)
        player.play(games, sims=a.sims, noise=a.noise, segment_limit=1, rng=rng, max_decisions=caps,
                    label_every=a.label_every)
        t1 = time.time()
        per, n_lab, frames_won = {}, 0, {}
        for g in games:
            l, d, t = g.tag
            if t < 0:                                      # whole-level games measure the agent
                per.setdefault(l, []).append(g.won)
                if g.won:
                    frames_won.setdefault(l, []).append(g.frames())
            for k, ep in enumerate(g.episodes):
                tag = 'it%04d_%s_d%02d_%d' % (it, l, d, k)
                dag = label(s, ep, segs[l], a.label_beam, a.label_horizon, a.max_labels)
                n_lab += int((~dag['forced']).sum())
                rep.add(dag, keep=True)
                save_episode(os.path.join(DATA, 'dagger', '%s_%s.npz' % (name, tag)), dag)
                if g.won and t < 0:                        # AlphaZero targets from levels that worked
                    sp = {k2: v for k2, v in ep.items() if k2 not in ('label_idx', 'label_states')}
                    rep.add(sp)
                    save_episode(os.path.join(DATA, 'selfplay', name, tag + '.npz'), sp)
        t2 = time.time()
        opt, hist = train(net, rep, a.train_steps, batch=512, log=lambda m: None, seed=it, opt=opt)
        t3 = time.time()
        log('[loop] it %d: whole levels won %d/%d (%s); play %.0f s, %d labels in %.0f s, train %.0f s '
            '(policy %.3f value %.3f acc %.2f mae %.0f f); replay %d' % (
                it, sum(sum(v) for v in per.values()), sum(len(v) for v in per.values()),
                ' '.join('%s %d/%d%s' % (l, sum(v), len(v), (' %.1fs' % (np.mean(frames_won[l]) / 50.007)) if l in frames_won else '')
                         for l, v in sorted(per.items())),
                t1 - t0, n_lab, t2 - t1, t3 - t2, hist[-1]['policy'], hist[-1]['value'], hist[-1]['acc'],
                hist[-1]['mae_frames'], len(rep)))
        save(net, os.path.join(a.out, 'net.pt'), it=it)
        if it % a.eval_every == 0:
            save(net, os.path.join(a.out, 'net_it%04d.pt' % it), it=it)
            summary, res = run_eval(net, [int(x) for x in a.eval_delays.split(',')], sims=a.eval_sims,
                                    parallel=len(a.eval_delays.split(',')), s=s, level=a.eval_level, log=lambda m: None,
                                    value_mix=a.value_mix)
            log('[loop] eval it %d (%s): won %d/%d, mean %.1f s (%s)' % (
                it, a.eval_level or 'full game', summary['won'], summary['games'], summary['mean_seconds_won'] or 0,
                ' '.join('d%d:%s' % (r['delay'], ('%.1fs' % r['seconds']) if r['won'] else r['reason']) for r in res)))


if __name__ == '__main__':
    main()
