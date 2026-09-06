"""Reward terms for the native Mario env -- positive-only design (2026-09-05).

The env decodes the game into a per-step `Signals` bundle; every reward is a
`Term` mapping Signals -> per-env float32 array, optionally with per-env state
(reset via `reset(idx, sig)`, snapshot via `state()/restore()` for rewind).
`RewardSet` composes the terms listed in `env_config.reward` and exposes the
per-term breakdown for the play tool.

Design (docs/reward_redesign_proposals.md, decided with the user):
  * there are NO negative rewards. Every failure (death, time-up, wrong exit,
    page reset, unpaid timeout) just ends the episode; its only cost is the
    future that is not collected. Penalties created preferences between ways
    of failing (suicide vs loop vs camping) whose sign depended on numbers
    nobody could justify.
  * progress pays for ground never covered in this life, once. Per frame
    (level, area byte, AreaType, swim flag) the term keeps the highwater x;
    a step pays clip(x - hw, 0, cap). A page reset / wrong pipe drops Mario
    into ground already paid, so re-runs pay nothing -- no farm, and no
    loop detection needed to prevent one.
  * a level clear pays 500 + 100 * (levels_gained - 1): the warp is still
    the best exit by far, but the bonus stays fittable by the critic (a
    500 * delta warp was +10..15 sd of the return distribution).

Config:
    reward:
      - {type: first_visit_progress, cap: 20}
      - {type: level_clear, base: 500, per_extra: 100}
      - {type: first_visit_cells, bonus: 2, x_bin: 64, y_bin: 32}   # optional: 2D first-visit
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Signals:
    """Per-step decoded game signals, np arrays of length n unless noted."""
    n: int
    x: np.ndarray            # level x this step (page*256 + sub)
    x_last: np.ndarray       # x last step
    frame: np.ndarray        # int64 frame id: (gp, area, atype, swim) packed
    frame_change: np.ndarray # frame differs from last step (transition)
    hold: np.ndarray         # teleport-scale x jump not yet confirmed: no pay
    t: np.ndarray            # game timer
    died: np.ndarray         # life lost / game over this step
    game_over: np.ndarray
    level_delta: np.ndarray  # confirmed ON-route levels gained this step (int)
    wrong_exit: np.ndarray   # confirmed entry into a level outside the route
    victory_new: np.ndarray  # 8-4 axe (full game) this step, first time
    page_reset: np.ndarray   # x fell far below this frame's highwater
    timeout: np.ndarray      # unpaid-steps cutoff fired
    gp: np.ndarray           # level index 0-31
    area: np.ndarray
    atype: np.ndarray
    swim: np.ndarray
    ypix: np.ndarray
    single_stage: bool = False
    extra: dict = field(default_factory=dict)


class Term:
    """Base reward term: function of Signals, optional per-env state."""
    name = 'term'

    def __init__(self, n, **kw):
        self.n = n
        self.kw = kw

    def reset(self, idx, sig, hard=False):
        """Re-init per-env state for envs in idx (new life or new episode).
        sig carries the fresh x / frame of those envs."""

    def __call__(self, s: Signals) -> np.ndarray:
        raise NotImplementedError

    def state(self):
        return {}

    def restore(self, st):
        pass


class FirstVisitProgress(Term):
    """+clip(x - hw[frame], 0, cap): pays each pixel of a life's ground once.

    hw is kept PER FRAME for the life, never rebased backwards: a page reset
    or a wrong pipe lands in ground already paid and earns nothing until new
    ground; a new frame (pipe to a new section, vine to a bonus area, water)
    starts its own highwater at the arrival x so every real transition pays
    from the first step."""
    name = 'progress'

    def __init__(self, n, cap=20.0, **kw):
        super().__init__(n, cap=cap)
        self.cap = float(cap)
        self.hw = [dict() for _ in range(n)]     # env -> {frame: highwater}

    def reset(self, idx, sig, hard=False):
        for i in idx:
            self.hw[i] = {}
            if sig is not None:
                self.hw[i][int(sig.frame[i])] = int(sig.x[i])

    def hw_for(self, frame, x):
        """Highwater of each env for the given frame (its x if unseen)."""
        out = np.empty(len(frame), dtype=np.int64)
        for i in range(len(frame)):
            out[i] = self.hw[i].get(int(frame[i]), int(x[i]))
        return out

    def __call__(self, s):
        r = np.zeros(s.n, dtype=np.float32)
        for i in range(s.n):
            if s.hold[i]:
                continue
            f = int(s.frame[i]); x = int(s.x[i]); d = self.hw[i]
            hw = d.get(f)
            if hw is None:          # first step in this frame: arrival pays 0
                d[f] = x
                continue
            if x > hw:
                r[i] = min(float(x - hw), self.cap)
                d[f] = x
        return r

    def state(self):
        return {'hw': [dict(d) for d in self.hw]}

    def restore(self, st):
        self.hw = [dict(d) for d in st['hw']]


class LevelClear(Term):
    """+base + per_extra * (delta - 1) on a confirmed on-route level advance;
    the full-game victory (8-4 axe) counts as one level."""
    name = 'clear'

    def __init__(self, n, base=500.0, per_extra=100.0, **kw):
        super().__init__(n, base=base, per_extra=per_extra)
        self.base = float(base); self.per_extra = float(per_extra)

    def __call__(self, s):
        d = s.level_delta.astype(np.float32)
        r = np.where(d > 0, self.base + self.per_extra * np.maximum(d - 1, 0),
                     0.0)
        r = r + np.where(s.victory_new, self.base, 0.0)
        return r.astype(np.float32)


class FirstVisitCells(Term):
    """+bonus for every (frame, x-bin, y-band) cell visited for the first
    time in this life: progress in 2D. Getting ON TOP of something at the
    same x is a new cell and pays; jumping in place does not (visited).
    Bounded per life by the number of cells, never negative."""
    name = 'cells'

    def __init__(self, n, bonus=2.0, x_bin=64, y_bin=32, **kw):
        super().__init__(n, bonus=bonus, x_bin=x_bin, y_bin=y_bin)
        self.bonus = float(bonus); self.xb = int(x_bin); self.yb = int(y_bin)
        self.seen = [set() for _ in range(n)]

    def reset(self, idx, sig, hard=False):
        for i in idx:
            self.seen[i] = set()

    def __call__(self, s):
        r = np.zeros(s.n, dtype=np.float32)
        for i in range(s.n):
            if s.hold[i]:
                continue
            key = (int(s.frame[i]), int(s.x[i]) // self.xb,
                   int(s.ypix[i]) // self.yb)
            if key not in self.seen[i]:
                self.seen[i].add(key)
                r[i] = self.bonus
        return r

    def state(self):
        return {'seen': [set(v) for v in self.seen]}

    def restore(self, st):
        self.seen = [set(v) for v in st['seen']]


TERMS = {
    'first_visit_progress': FirstVisitProgress,
    'level_clear': LevelClear,
    'first_visit_cells': FirstVisitCells,
}


def build_term(n, spec):
    spec = dict(spec)
    typ = spec.pop('type')
    if typ not in TERMS:
        raise ValueError('unknown reward term %r (known: %s)'
                         % (typ, ', '.join(sorted(TERMS))))
    return TERMS[typ](n, **spec)


DEFAULT_SPECS = [{'type': 'first_visit_progress', 'cap': 20},
                 {'type': 'level_clear', 'base': 500, 'per_extra': 100}]


class RewardSet:
    """Ordered list of terms; total = sum. `last` holds the breakdown."""

    def __init__(self, n, specs=None):
        self.n = n
        self.terms = [build_term(n, s) for s in (specs or DEFAULT_SPECS)]
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
        return tot

    def get(self, cls):
        for t in self.terms:
            if isinstance(t, cls):
                return t
        return None

    def state(self):
        return [t.state() for t in self.terms]

    def restore(self, st):
        for t, s in zip(self.terms, st):
            t.restore(s)
