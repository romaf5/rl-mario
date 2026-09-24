"""Games with net-guided MCTS: many at once (one tree each, one net batch per wave).

Per decision: a forced root (the input does nothing) commits NOOP at once; other
roots get simulations (a fixed count, or a wall-clock budget for live play), then
the most visited action is played. Reaching the next route level starts a new
segment (a new tree); the axe ends the game; a death ends it as a failure.
Every segment becomes an episode (frames, visit distributions, frames to go).
"""
import time
import numpy as np
from .common import FRAME_SKIP, ROUTE, V_SCALE, Forest, episode


class Game:
    def __init__(self, start, tag=None):
        self.start, self.tag = start, tag
        self.actions = []             # every decision of the game
        self.episodes = []            # finished segments
        self.done = False
        self.won = False
        self.reason = ''
        self.decision_s = []          # wall clock per searched decision
        self._seg = None

    def frames(self):
        return len(self.actions) * FRAME_SKIP


class Player:
    def __init__(self, search, evaluator, n_trees, per_tree=128, max_nodes=1 << 15, routes=None, value_mix=1.0,
                 **mcts):
        """routes: {level: (start state, actions)} -- leaf values use frames to go along them
        (value_mix x net + (1 - value_mix) x route)."""
        self.s = search
        self.ev = evaluator
        self.n = n_trees
        self.per_tree = per_tree
        self.f = Forest(search, n_trees, max_nodes=max_nodes, max_leaves=evaluator.stacks.shape[0],
                        stacks=evaluator.stacks, value_mix=value_mix, **mcts)
        for level, (start, actions) in (routes or {}).items():
            self.f.set_route(level, start, actions)

    def _new_segment(self, t, g, state):
        assert self.f.reset(t, state, ROUTE), 'state is not on the route'
        _, ram, stack = self.f.state(t)
        g._seg = dict(level='%d-%d' % (ram[0x75F] + 1, ram[0x75C] + 1), frames=[stack[-1]], policy=[], acts=[],
                      root_b=[], forced=[], start_state=np.frombuffer(state, np.uint8), label_idx=[], label_states=[])

    def _end_segment(self, g, goal):
        seg = g._seg
        n = len(seg['policy'])
        if n:
            rb = np.array(seg['root_b'], np.float32)
            value = 4.0 * (n - np.arange(n)) if goal else np.minimum(rb, V_SCALE)
            ep = episode(np.array(seg['frames']), np.array(seg['policy']), value, np.array(seg['forced']),
                         level=seg['level'], source='selfplay', goal=goal, start_state=seg['start_state'],
                         acts=np.array(seg['acts'], np.uint8))
            ep['label_idx'] = np.array(seg['label_idx'], np.int32)          # states for the local teacher
            ep['label_states'] = (np.stack([np.frombuffer(x, np.uint8) for x in seg['label_states']])
                                  if seg['label_states'] else np.zeros((0, 0), np.uint8))
            g.episodes.append(ep)
        g._seg = None

    def play(self, games, sims=None, budget_s=None, noise=0.0, alpha=0.3, max_decisions=20000,
             segment_limit=None, rng=None, log=None, label_every=0, safe_horizon=24, on_wave=None):
        """Play the games to the end (axe, death or max_decisions). sims: simulations per
        searched decision; budget_s: wall-clock seconds per searched decision (live play).
        segment_limit: stop each game after this many finished segments (per-level play); an int,
        None (the whole game), or a list (one per game).
        label_every: keep the full state of every k-th searched decision (for the local teacher).
        max_decisions: an int, or a list (one per game).
        safe_horizon: before a commit, the chosen action must have a surviving continuation
        (some button held this many steps); else the next most visited (0: off).
        on_wave(forest, n, trees, step): called after each wave's net evaluation, while its
        leaves are still in forest.leaves / forest.stacks (training data for values)."""
        assert len(games) <= self.n and (sims or budget_s)
        rng = rng or np.random.default_rng()
        for t, g in enumerate(games):
            self._new_segment(t, g, g.start)
        live = [t for t, g in enumerate(games) if not g.done]
        step = 0
        while live:
            step += 1
            forced = self.f.forced(live)
            think = [t for t, fz in zip(live, forced) if not fz]
            t0 = time.perf_counter()
            if think:
                if hasattr(self.ev, 'new_decision'):     # a value relative to each root
                    self.ev.new_decision(self.f, think)
                base = {t: self.f.root(t)[3] for t in think}
                noised = set()
                while True:
                    todo = [t for t in think if self.f.root(t)[3] - base[t] < (sims or 1 << 30)]
                    if not todo or (budget_s and time.perf_counter() - t0 >= budget_s):
                        break
                    n = self.f.select(todo, self.per_tree)
                    if n == 0:
                        break
                    pri, val = self.ev(n, self.f)
                    if on_wave is not None:          # the leaves are still in self.f.stacks / leaves
                        on_wave(self.f, n, todo, step)
                    self.f.backup(n, pri, val)
                    if noise:                           # root noise once the root has its prior
                        for t in todo:
                            if t not in noised and self.f.root(t)[3] > 0:
                                self.f.noise(t, rng.dirichlet([alpha] * 12), noise); noised.add(t)
            dt = time.perf_counter() - t0
            for t, fz in zip(live, forced):
                g = games[t]
                if fz:
                    _, _, rb, rn = self.f.root(t)
                    a, pol, rb = 0, np.full(12, 1 / 12, np.float32), rb if rn > 0 else np.nan
                else:
                    visits, best, rb, _ = self.f.root(t)
                    tot = visits.sum()
                    pol = visits / tot if tot else np.full(12, 1 / 12, np.float32)
                    order = np.lexsort((np.where(best < 0, 1e9, best), -visits))
                    a = int(order[0])
                    if safe_horizon and not self.f.safe(t, [a], safe_horizon)[0]:
                        cand = [int(x) for x in order[1:4] if visits[x] > 0]
                        ok = self.f.safe(t, cand, safe_horizon) if cand else []
                        pick = next((c for c, o in zip(cand, ok) if o), None)
                        if pick is None:                    # none of the favourites: any action that survives
                            rest = [int(x) for x in order[4:]] + [int(x) for x in order[1:4] if visits[x] == 0]
                            ok = self.f.safe(t, rest, safe_horizon)
                            pick = next((c for c, o in zip(rest, ok) if o), a)
                        a = pick
                        g.unsafe = getattr(g, 'unsafe', 0) + 1
                    g.decision_s.append(time.perf_counter() - t0 if budget_s else dt)
                seg = g._seg
                if label_every and not fz and (len(g.decision_s) - 1) % label_every == 0:
                    seg['label_idx'].append(len(seg['policy'])); seg['label_states'].append(self.f.state(t)[0])
                seg['policy'].append(pol.astype(np.float32)); seg['root_b'].append(rb); seg['forced'].append(bool(fz))
                seg['acts'].append(a)                       # the action actually played (world-model data)
                g.actions.append(a)
                term = self.f.commit(t, a)
                seg['frames'].append(self.f.state(t)[2][-1])
                if term == Forest.GOAL:
                    self._end_segment(g, True)
                    state, ram, _ = self.f.state(t)
                    if ram[0x770] == 2:
                        g.done, g.won, g.reason = True, True, 'axe'
                    elif (segment_limit[t] if isinstance(segment_limit, (list, tuple)) else segment_limit) and \
                            len(g.episodes) >= (segment_limit[t] if isinstance(segment_limit, (list, tuple)) else segment_limit):
                        g.done, g.won, g.reason = True, True, 'segment'
                    else:
                        self._new_segment(t, g, state)
                elif term == Forest.DEAD:
                    self._end_segment(g, False)
                    g.done, g.reason = True, 'dead at %s' % seg['level']
                elif len(g.actions) >= (max_decisions[t] if isinstance(max_decisions, (list, tuple)) else max_decisions):
                    self._end_segment(g, False)
                    g.done, g.reason = True, 'too long'
            live = [t for t in live if not games[t].done]
            if log and step % 500 == 0:
                log('[play] decision %d: %d games live' % (step, len(live)))
        return games
