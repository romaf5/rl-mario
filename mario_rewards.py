"""Decomposed reward terms for the native Mario env.

The env decodes the game into a per-step `Signals` bundle (positions,
transitions, deaths, loops, level changes, timer, score ...). Every reward
is a `Term`: a small object that maps Signals -> per-env float32 array and
may keep its own per-env state (reset via `reset(idx, sig)`). `RewardSet`
composes terms from a config list and exposes the per-term breakdown, so
terms can be swapped in/out from YAML and inspected while playing.

Config example (env_config.reward):
    reward:
      - {type: clip, lo: -15, hi: 20, terms: [{type: progress_highwater, cap: 20}, {type: time}]}
      - {type: fail, cost: 100}          # death + idle timeout
      - {type: loop, cost: 100}
      - {type: offroute, cost: 100}
      - {type: stage, bonus: 500}
      - {type: victory, bonus: 500}
      - {type: score, scale: 0.1}
      - {type: novelty, bonus: 3.0, y_band: 24, global: true}
Without a `reward` list the env builds the legacy-equivalent set from its
flat kwargs (loop_penalty, fail_penalty, stage_bonus, ...).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Signals:
    """Per-step decoded game signals, all np arrays of length n."""
    n: int
    x: np.ndarray            # confirmed x this step (debounced)
    x_last: np.ndarray       # confirmed x last step
    prev_x: np.ndarray       # last step's x for idle/backtrack shaping
    xd: np.ndarray           # x - prev_x
    fwd: np.ndarray          # xd > 0
    new_ground: np.ndarray   # x > episode max_x
    ctx_change: np.ndarray   # life/area/swim/level change or legit transition
    t: np.ndarray            # game timer now
    t_last: np.ndarray       # game timer last step
    died: np.ndarray         # life decremented or game over this step
    game_over: np.ndarray
    loop: np.ndarray         # cycle: backward jump in-frame / visited
    legit: np.ndarray        # legitimate transition (frame change)
    off: np.ndarray          # confirmed entry into an off-route level
    level_up: np.ndarray     # confirmed on-route level advance (bool)
    level_delta: np.ndarray  # levels advanced on that step (int)
    newflag: np.ndarray      # flag grabbed (single-stage mode)
    victory_new: np.ndarray  # first victory step
    score_delta: np.ndarray  # game score delta (clipped)
    idle: np.ndarray         # steps without forward progress (updated)
    idle_to: np.ndarray      # idle timeout fired
    area: np.ndarray
    swim: np.ndarray
    ypix: np.ndarray
    held: np.ndarray         # x carried (debounce) this step
    gp: np.ndarray           # level index 0-31 (world*4 + stage)
    loop_count: np.ndarray = None    # teleports so far this life (incl. this step)
    loop_terminal: np.ndarray = None # this loop ends the episode
    single_stage: bool = False
    extra: dict = field(default_factory=dict)


class Term:
    """Base reward term: pure function of Signals, optional per-env state."""
    name = 'term'

    def __init__(self, n, **kw):
        self.n = n
        self.kw = kw

    def reset(self, idx, sig, hard=False):
        """Re-init per-env state for envs in idx. hard=True for a real
        episode reset (new start), False for a life-loss re-sync."""

    def __call__(self, s: Signals) -> np.ndarray:
        raise NotImplementedError

    def state(self):
        """Snapshot of per-env state (for rewind tools)."""
        return {}

    def restore(self, st):
        pass


class ProgressHighwater(Term):
    """+clip(x - highwater, 0, cap); highwater rebased on context change.
    Pays each pixel of a life's progress once; re-runs after a loop are
    never starved because loops rebase (or terminate) upstream."""
    name = 'progress'

    def __init__(self, n, cap=20.0, **kw):
        super().__init__(n, cap=cap)
        self.cap = float(cap)
        self.hw = np.zeros(n, dtype=np.int64)

    def reset(self, idx, sig, hard=False):
        if sig is not None:
            self.hw[idx] = sig.x[idx]

    def __call__(self, s):
        self.hw = np.where(s.ctx_change, s.x, self.hw)
        r = np.clip(s.x - self.hw, 0, self.cap).astype(np.float32)
        self.hw = np.maximum(self.hw, s.x)
        return r

    def state(self):
        return {'hw': self.hw.copy()}

    def restore(self, st):
        self.hw = st['hw'].copy()


class ProgressDelta(Term):
    """+dx per step (teleport-scale jumps zeroed). Legacy; loops become a
    treadmill unless they are terminal."""
    name = 'progress'

    def __init__(self, n, max_jump=24, **kw):
        super().__init__(n, max_jump=max_jump)
        self.max_jump = max_jump

    def __call__(self, s):
        dx = s.x - s.x_last
        return np.where(np.abs(dx) > self.max_jump, 0, dx).astype(np.float32)


class Time(Term):
    """Game timer ticks (about -0.17 per step)."""
    name = 'time'

    def __call__(self, s):
        return np.minimum(s.t - s.t_last, 0).astype(np.float32)


class Clip(Term):
    """Clipped sum of child terms (legacy base clip [-15, 20])."""
    name = 'base'

    def __init__(self, n, terms, lo=-15.0, hi=20.0, **kw):
        super().__init__(n, lo=lo, hi=hi)
        self.terms = terms
        self.lo, self.hi = float(lo), float(hi)
        self.last = {}

    def reset(self, idx, sig, hard=False):
        for t in self.terms:
            t.reset(idx, sig, hard)

    def __call__(self, s):
        tot = np.zeros(s.n, dtype=np.float32)
        self.last = {}
        for t in self.terms:
            v = t(s)
            self.last[t.name] = v
            tot = tot + v
        return np.clip(tot, self.lo, self.hi).astype(np.float32)

    def state(self):
        return {t.name: t.state() for t in self.terms}

    def restore(self, st):
        for t in self.terms:
            t.restore(st[t.name])


class Fail(Term):
    """-cost on death and on idle timeout (one flat attempt-ending cost)."""
    name = 'fail'

    def __init__(self, n, cost=100.0, death=True, idle=True, name=None, **kw):
        super().__init__(n, cost=cost, death=death, idle=idle)
        self.cost, self.death, self.idle = float(cost), death, idle
        self.name = name or ('fail' if (death and idle)
                             else 'fail_death' if death else 'fail_idle')

    def __call__(self, s):
        hit = np.zeros(s.n, dtype=bool)
        if self.death:
            hit |= s.died
        if self.idle:
            hit |= s.idle_to
        return np.where(hit, -self.cost, 0.0).astype(np.float32)


class Loop(Term):
    """-cost on every backward teleport (the game's loop). Whether it also
    ends the episode is the env's loop_terminal policy, not this term."""
    name = 'loop'

    def __init__(self, n, cost=100.0, **kw):
        super().__init__(n, cost=cost)
        self.cost = float(cost)

    def __call__(self, s):
        return np.where(s.loop, -self.cost, 0.0).astype(np.float32)


class OffRoute(Term):
    """-cost on entering a level outside the training set."""
    name = 'offroute'

    def __init__(self, n, cost=100.0, **kw):
        super().__init__(n, cost=cost)
        self.cost = float(cost)

    def __call__(self, s):
        return np.where(s.off, -self.cost, 0.0).astype(np.float32)


class Stage(Term):
    """+bonus x levels advanced (warps pay more); flag in single-stage."""
    name = 'stage'

    def __init__(self, n, bonus=500.0, **kw):
        super().__init__(n, bonus=bonus)
        self.bonus = float(bonus)

    def __call__(self, s):
        if s.single_stage:
            return np.where(s.newflag, self.bonus, 0.0).astype(np.float32)
        return np.where(s.level_up, self.bonus * s.level_delta, 0.0
                        ).astype(np.float32)


class Victory(Term):
    name = 'victory'

    def __init__(self, n, bonus=500.0, **kw):
        super().__init__(n, bonus=bonus)
        self.bonus = float(bonus)

    def __call__(self, s):
        return np.where(s.victory_new, self.bonus, 0.0).astype(np.float32)


class Score(Term):
    """+scale x game score delta (coins, stomps, hidden-block reveal +200)."""
    name = 'score'

    def __init__(self, n, scale=0.1, **kw):
        super().__init__(n, scale=scale)
        self.scale = float(scale)

    def __call__(self, s):
        return (s.score_delta * self.scale).astype(np.float32)


class Growth(Term):
    """Legacy: +min(xd,20) * k * x on new ground (position-scaled)."""
    name = 'growth'

    def __init__(self, n, k=0.001, **kw):
        super().__init__(n, k=k)
        self.k = float(k)

    def __call__(self, s):
        g = np.minimum(s.xd, 20) * self.k * s.x
        return np.where(s.fwd & s.new_ground, g, 0.0).astype(np.float32)


class Backtrack(Term):
    """Legacy: -cost per forward step over already-covered ground."""
    name = 'backtrack'

    def __init__(self, n, cost=0.15, **kw):
        super().__init__(n, cost=cost)
        self.cost = float(cost)

    def __call__(self, s):
        return np.where(s.fwd & ~s.new_ground, -self.cost, 0.0
                        ).astype(np.float32)


class IdleDrip(Term):
    """Legacy: -cost per idle step past a threshold, capped per episode."""
    name = 'idle'

    def __init__(self, n, cost=0.5, threshold=10, cap=15.0, **kw):
        super().__init__(n, cost=cost, threshold=threshold, cap=cap)
        self.cost, self.threshold, self.cap = float(cost), threshold, cap
        self.paid = np.zeros(n, dtype=np.float32)

    def reset(self, idx, sig, hard=False):
        self.paid[idx] = 0.0

    def __call__(self, s):
        hit = (s.idle > self.threshold) & (self.paid < self.cap)
        self.paid += np.where(hit, self.cost, 0.0)
        return np.where(hit, -self.cost, 0.0).astype(np.float32)

    def state(self):
        return {'paid': self.paid.copy()}

    def restore(self, st):
        self.paid = st['paid'].copy()


class Novelty(Term):
    """+bonus/sqrt(1+count) on the first visit of a spot (area, x//64,
    y-band) per episode; counts shared across envs and decayed (global)
    or per-episode only. Intentionally non-stationary: exploration."""
    name = 'novelty'

    def __init__(self, n, bonus=3.0, y_band=24, global_counts=True,
                 decay_every=3000, **kw):
        super().__init__(n, bonus=bonus, y_band=y_band,
                         global_counts=global_counts)
        self.bonus, self.y_band = float(bonus), int(y_band)
        self.global_counts, self.decay_every = global_counts, decay_every
        self.sets = [set() for _ in range(n)]
        self.counts = {}
        self.tick = 0

    def reset(self, idx, sig, hard=False):
        # every episode boundary, life losses included: with episode_life a
        # life IS an episode, and paying novelty only on the first life made
        # identical behaviour earn different reward depending on the life
        for i in idx:
            self.sets[i] = set()

    def __call__(self, s):
        r = np.zeros(s.n, dtype=np.float32)
        self.tick += 1
        if self.global_counts and self.tick % self.decay_every == 0:
            self.counts = {k: v * 0.5 for k, v in self.counts.items()
                           if v * 0.5 >= 0.1}
        for i in range(s.n):
            # the level belongs in the key: every x-1 level shares area 0,
            # so without it 1-1's traffic depleted 4-1's and 8-1's bonus
            cell = (int(s.gp[i]), int(s.area[i]), int(s.x[i]) // 64,
                    int(s.ypix[i]) // self.y_band)
            if cell in self.sets[i]:
                continue
            self.sets[i].add(cell)
            if self.global_counts:
                c = self.counts.get(cell, 0)
                self.counts[cell] = c + 1
                r[i] = self.bonus / (1 + c) ** 0.5
            else:
                r[i] = self.bonus
        return r

    def state(self):
        # counts and tick drive the bonus size and its decay: a rewind that
        # dropped them replayed the same game state for a different reward
        return {'sets': [set(x) for x in self.sets],
                'counts': dict(self.counts), 'tick': self.tick}

    def restore(self, st):
        self.sets = [set(x) for x in st['sets']]
        self.counts = dict(st.get('counts', {}))
        self.tick = st.get('tick', 0)


TERMS = {
    'progress_highwater': ProgressHighwater,
    'progress_delta': ProgressDelta,
    'time': Time,
    'clip': Clip,
    'fail': Fail,
    'loop': Loop,
    'offroute': OffRoute,
    'stage': Stage,
    'victory': Victory,
    'score': Score,
    'growth': Growth,
    'backtrack': Backtrack,
    'idle_drip': IdleDrip,
    'novelty': Novelty,
}


def build_term(n, spec):
    spec = dict(spec)
    typ = spec.pop('type')
    if typ == 'clip':
        children = [build_term(n, c) for c in spec.pop('terms')]
        return Clip(n, children, **spec)
    return TERMS[typ](n, **spec)


class RewardSet:
    """Ordered list of terms; total = sum. `last` holds the breakdown."""

    def __init__(self, n, specs):
        self.n = n
        self.terms = [build_term(n, s) for s in specs]
        self.last = {}

    def reset(self, idx, sig, hard=False):
        for t in self.terms:
            t.reset(idx, sig, hard)

    def __call__(self, s):
        tot = np.zeros(s.n, dtype=np.float32)
        self.last = {}
        for t in self.terms:
            v = t(s)
            self.last[t.name] = v
            if isinstance(t, Clip):
                for k, v2 in t.last.items():
                    self.last['base/' + k] = v2
            tot = tot + v
        return tot

    def has(self, name):
        """Is a term with this name present anywhere (including nested)?"""
        for t in self.terms:
            if t.name == name:
                return True
            if isinstance(t, Clip) and any(c.name == name for c in t.terms):
                return True
        return False

    def get(self, cls):
        for t in self.terms:
            if isinstance(t, cls):
                return t
            if isinstance(t, Clip):
                for c in t.terms:
                    if isinstance(c, cls):
                        return c
        return None

    def state(self):
        return [t.state() for t in self.terms]

    def restore(self, st):
        for t, s in zip(self.terms, st):
            t.restore(s)


def legacy_specs(x_reward='highwater', fail_penalty=15.0, loop_penalty=0.0,
                 offroute_penalty=0.0, stage_bonus=500.0, score_reward=0.0,
                 progress_reward=0.0, backtrack_penalty=0.0, idle_penalty=0.0,
                 idle_threshold=10, novelty_bonus=0.0, novelty_y_band=48,
                 novelty_global=False, **_):
    """The pre-refactor reward set expressed as term specs (same numerics,
    same order of summation)."""
    prog = ({'type': 'progress_highwater', 'cap': 20}
            if x_reward == 'highwater' else {'type': 'progress_delta'})
    specs = [{'type': 'clip', 'lo': -15, 'hi': 20,
              'terms': [prog, {'type': 'time'}]},
             {'type': 'fail', 'cost': fail_penalty, 'death': True,
              'idle': False, 'name': 'fail_death'}]
    if score_reward > 0:
        specs.append({'type': 'score', 'scale': score_reward})
    specs.append({'type': 'stage', 'bonus': stage_bonus})
    if offroute_penalty > 0:
        specs.append({'type': 'offroute', 'cost': offroute_penalty})
    specs.append({'type': 'victory', 'bonus': stage_bonus})
    if loop_penalty > 0:
        specs.append({'type': 'loop', 'cost': loop_penalty})
    if progress_reward > 0:
        specs.append({'type': 'growth', 'k': progress_reward})
    if backtrack_penalty > 0:
        specs.append({'type': 'backtrack', 'cost': backtrack_penalty})
    if idle_penalty > 0:
        specs.append({'type': 'idle_drip', 'cost': idle_penalty,
                      'threshold': idle_threshold})
    specs.append({'type': 'fail', 'cost': fail_penalty, 'death': False,
                  'idle': True, 'name': 'fail_idle'})
    if novelty_bonus > 0:
        specs.append({'type': 'novelty', 'bonus': novelty_bonus,
                      'y_band': novelty_y_band,
                      'global_counts': novelty_global})
    return specs
