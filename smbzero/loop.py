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
from .common import (DATA, MAX_DELAY, ROUTE, THREADS, Search, e2e_segments, episode, level_name, load_state,
                     route_values, save_episode)
from .eval import run as run_eval
from .net import Evaluator, load, save
from .play import Game, Player
from .train import Replay, train


def label(s, ep, seg, beam, horizon, max_labels, strong_last=0, strong_beam=1000, strong_horizon=80):
    """The local teacher on (at most max_labels, evenly spaced) recorded states of an
    episode -> a DAgger episode (policy and value targets only where labelled) plus,
    per labelled state, the teacher's whole path from it as a teacher episode: every
    state on the path with its action and frames to go (the sequences a prior needs:
    back off, run up, jump). The last `strong_last` states (where a failed game stalled or
    died) get a stronger teacher: puzzle steps like 8-4's hidden block need it."""
    n = len(ep['policy'])
    pol = np.zeros((n, 12), np.float32)
    val = np.full(n, np.nan, np.float32)
    unlabelled = np.ones(n, bool)
    paths = []
    k = len(ep['label_idx'])
    pick = np.unique(np.linspace(0, k - 1, min(k, max_labels)).round().astype(int)) if k else []
    for q, j in enumerate(pick):
        i, st = ep['label_idx'][j], ep['label_states'][j]
        strong = q >= len(pick) - strong_last
        a, est, _ = s.lookahead(st.tobytes(), ROUTE, seg['opt'], ref_start=seg['start'],
                                beam=strong_beam if strong else beam, horizon=strong_horizon if strong else horizon)
        if len(a):
            pol[i, a[0]] = 1; val[i] = est; unlabelled[i] = False
            obs, _, _ = s.replay_obs(st.tobytes(), a)
            m = len(a)
            pp = np.zeros((m, 12), np.float32); pp[np.arange(m), a] = 1
            paths.append(episode(np.concatenate([ep['frames'][i:i + 1], obs]), pp, est - 4.0 * np.arange(m),
                                 s.forced_along(st.tobytes(), a), level=str(ep['level']), source='teacher',
                                 acts=np.asarray(a, np.uint8)))
    return episode(ep['frames'], pol, val, unlabelled, level=str(ep['level']), source='dagger'), paths


def arrivals(s, patterns):
    """{level: [state]}: the states full games (eval / self-play route files) arrived in at
    each level's first frame -- the starts a full game really meets, which differ from the
    search route's entry (timing, RNG) and made full games fail where level games clear."""
    pool = {}
    for p in sorted(f for pat in patterns for f in glob.glob(pat)):
        z = np.load(p)
        if str(z['start']) != 'FullGame':
            continue
        st = s.frames(load_state('FullGame'), int(z['lead_frames']))
        tr, _ = s.replay(st, z['actions'])
        lv = tr[:, 2]
        for i in range(1, len(lv)):
            if lv[i] != lv[i - 1] and 0 <= lv[i] < 32:
                pool.setdefault(level_name(int(lv[i])), []).append(s.replay(st, z['actions'][:i + 1])[1])
    return pool


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
    ap.add_argument('--full-game', type=int, default=0, help='games per iteration that play from 1-1 (+ delay) to the end')
    ap.add_argument('--arrivals', default='', help='globs of full-game route files: whole-level games also start '
                                                  'where those games arrived (comma separated; + this run\'s games)')
    ap.add_argument('--arrival-share', type=float, default=0.5, help='share of whole-level games started from arrivals')
    ap.add_argument('--label-every', type=int, default=6)
    ap.add_argument('--label-beam', type=int, default=100)
    ap.add_argument('--label-horizon', type=int, default=30)
    ap.add_argument('--max-labels', type=int, default=40, help='per segment')
    ap.add_argument('--strong-last', type=int, default=4, help='failed whole levels: strong-teacher labels at the end')
    ap.add_argument('--train-steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--eval-delays', default='0,15,30,45,60')
    ap.add_argument('--eval-sims', type=int, default=400)
    ap.add_argument('--eval-level', default=None, help='evaluate one level instead of the full game')
    ap.add_argument('--value-mix', type=float, default=0.0, help='leaf value: this x net + (1 - this) x route')
    ap.add_argument('--min-backup', action='store_true', help='b = 4 + min over children (exact route values)')
    ap.add_argument('--visits-all', action='store_true', help='visit-count targets from every game, not only won levels')
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
                    value_mix=a.value_mix, min_backup=a.min_backup)
    rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    t_end = time.time() + a.hours * 3600
    it = 0
    pats = [x for x in a.arrivals.split(',') if x] + [os.path.join(a.out, 'games', '*.npz')]
    while time.time() < t_end:
        it += 1
        t0 = time.time()
        pool = arrivals(s, pats) if a.arrivals else {}
        lv = rng.choice(levels, a.trees)
        games, caps, limits = [], [], []
        for i, l in enumerate(lv):
            if i < a.full_game:                            # the whole game from 1-1: real arrivals
                d = int(rng.integers(0, MAX_DELAY + 1))
                games.append(Game(s.frames(load_state('FullGame'), d), tag=('game', d, -2)))
                caps.append(int(a.cap * sum(len(g['opt']) for g in segs.values())) + 2000)
                continue
            if i < a.full_game + round(a.full * a.trees):  # the whole level: from a full game's arrival or the entry
                if pool.get(l) and rng.random() < a.arrival_share:
                    st = pool[l][rng.integers(len(pool[l]))]
                    games.append(Game(st, tag=(l, 99, -1)))
                else:
                    d = int(rng.integers(0, MAX_DELAY + 1))
                    games.append(Game(s.frames(segs[l]['start'], d), tag=(l, d, -1)))
                caps.append(int(a.cap * len(segs[l]['opt'])))
            else:                                          # from a teacher route state
                st, acts = routes[l][rng.integers(len(routes[l]))]
                t = int(rng.integers(0, len(acts)))
                games.append(Game(s.replay(st, acts[:t])[1], tag=(l, t, t)))
                caps.append(a.rollout)
        player.play(games, sims=a.sims, noise=a.noise, segment_limit=[None if g.tag[2] == -2 else 1 for g in games],
                    rng=rng, max_decisions=caps, label_every=a.label_every)
        t1 = time.time()
        per, n_lab, frames_won = {}, 0, {}
        full_games = []
        for g in games:
            l, d, t = g.tag
            if t == -2:
                full_games.append('d%d: %s (%d levels)' % (d, 'WON %.1fs' % ((d + g.frames()) / 50.007) if g.won else g.reason,
                                                           len(g.episodes)))
                os.makedirs(os.path.join(a.out, 'games'), exist_ok=True)     # replayable: verify / render
                np.savez(os.path.join(a.out, 'games', 'it%04d_d%02d%s.npz' % (it, d, '_won' if g.won else '')),
                         start='FullGame', lead_frames=d, actions=np.array(g.actions, np.uint8))
            if t < 0:                                      # whole-level games measure the agent
                per.setdefault(l, []).append(g.won)
                if g.won:
                    frames_won.setdefault(l, []).append(g.frames())
            for k, ep in enumerate(g.episodes):
                el = str(ep['level'])                          # a full game has one episode per level
                tag = 'it%04d_%s_d%02d_%d' % (it, el, d, k)
                last = k == len(g.episodes) - 1
                dag, paths = label(s, ep, segs[el], a.label_beam, a.label_horizon, a.max_labels,
                                   strong_last=a.strong_last if (t < 0 and not g.won and last) else 0)
                n_lab += int((~dag['forced']).sum())
                rep.add(dag, keep=True)
                save_episode(os.path.join(DATA, 'dagger', '%s_%s.npz' % (name, tag)), dag)
                for q, pe in enumerate(paths):                 # the teacher's paths from those states
                    rep.add(pe, keep=True)
                    save_episode(os.path.join(DATA, 'dagger', '%s_%s_path%02d.npz' % (name, tag, q)), pe)
                if (g.won and t < 0) or a.visits_all:      # AlphaZero targets: the search's visit counts
                    sp = {k2: v for k2, v in ep.items() if k2 not in ('label_idx', 'label_states')}
                    rep.add(sp)
                    save_episode(os.path.join(DATA, 'selfplay', name, tag + '.npz'), sp)
        t2 = time.time()
        opt, hist = train(net, rep, a.train_steps, batch=512, log=lambda m: None, seed=it, opt=opt)
        t3 = time.time()
        if full_games:
            log('[loop] it %d: full games %s' % (it, '; '.join(full_games)))
        if pool:
            log('[loop] it %d: arrival starts: %s' % (it, ' '.join('%s %d' % (k, len(v)) for k, v in sorted(pool.items()))))
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
                                    value_mix=a.value_mix, min_backup=a.min_backup)
            log('[loop] eval it %d (%s): won %d/%d, mean %.1f s (%s)' % (
                it, a.eval_level or 'full game', summary['won'], summary['games'], summary['mean_seconds_won'] or 0,
                ' '.join('d%d:%s' % (r['delay'], ('%.1fs' % r['seconds']) if r['won'] else r['reason']) for r in res)))


if __name__ == '__main__':
    main()
