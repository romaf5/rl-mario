# Backward Curriculum (Go-Explore phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PPO episodes on 4-2 start on the agent's own first door→warp route, near its end, and the start moves back 16 steps each time the policy warps from the current band ≥ 20% of the time.

**Architecture:** Everything lives in `MarioNativeVecEnv` (`mario_native_vecenv.py`): an action log per env gives every archived state variant its action prefix from the door (archive entry `[8]`); the first on-route clear with a known prefix becomes the route (verified by replay on a private 1-core `benv`); `_reset_env` draws 75% of starts from route states `[tau*, tau*+32]`; episode outcomes in `[tau*, tau*+16)` move `tau*`. The observer logs it, the auditor checks it.

**Tech Stack:** Python 3 / numpy, ctypes over `native/libbatchenv.so`, rl_games 1.6.5, standalone check scripts (`tests/*_bench.py`, `check()` style, no pytest).

**Spec:** `docs/superpowers/specs/2026-09-19-backward-curriculum-design.md`

## Global Constraints

- Defaults = off: `demo_start_prob` 0.0 leaves every existing config bitwise unchanged (no extra RNG draws, no entry `[8]`).
- The curriculum runs only when `demo_start_prob > 0 and self_restart_prob > 0 and not play_mode` (evals and tools set `self_restart_prob` 0, so they never see route starts).
- Asserted when on: `reset_noops == 0`, exactly one trained level (`random_stages` length 1).
- Run l values: `demo_start_prob 0.75`, `demo_window 32`, `demo_step 16`, `demo_success 0.2`, `demo_success_n 64`.
- Route starts: `start_cell None`, `is_door = (tau == 0)`, `was_restart False`.
- Success = a real on-route clear (`cleared > 0`).
- Every bench stays 0 FAIL; `tools/audit_env.py` 0 violations. Run with `venv_retro/bin/python`.
- Commits: repo style, end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`; commit and push at each green task.

---

## File Structure

| file | change |
|---|---|
| `mario_native_vecenv.py` | kwargs, action log, prefixes, route builder, curriculum, infos, sidecar |
| `callbacks.py` | `mario/demo_*`, `mario/clear_demo/<lvl>` |
| `train.py` | `fresh_archive_conflict` also refuses an existing sidecar |
| `grpo/train_grpo.py` | `demo_start_prob=0.0` in its env overrides |
| `tools/audit_env.py` | `--route`, demo labels, window, route-clears check |
| `tests/demo_bench.py` | new: all curriculum checks |
| `configs/mario_ppo_native_42.yaml` | run l |
| `CLAUDE.md`, `EXPERIMENTS.md` | one bullet each; run l entry |

---

### Task 1: Action prefixes for every archived state variant

**Files:**
- Modify: `mario_native_vecenv.py` (constructor kwargs ~L138-154, init ~L168-420, `_reset_env` ~L454, `_reset_explorer` ~L592, `step` ~L826, archive section ~L1015-1054, soft path ~L1315)
- Test: `tests/demo_bench.py` (create)

**Interfaces:**
- Produces: kwargs `demo_start_prob=0.0, demo_window=32, demo_step=16, demo_success=0.2, demo_success_n=64`; attrs `self.demo_on`, `self.demo_prob`, `self.act_log (n, ACT_LOG_MAX) int8`, `self.start_prefix: list[bytes|None]`, `self.demo_tau (n,) int32`, `self.route = None`, `self._rom: bytes`; methods `_episode_prefix(i) -> bytes|None` (valid before `ep_steps += 1` in `_after_step`), `_prefixes(ent) -> list` (entry `[8]`, padded, aligned with `ent[0]`). Module const `ACT_LOG_MAX = 8192`.

- [ ] **Step 1: Write the failing test** — create `tests/demo_bench.py`:

```python
"""Backward-curriculum checks (Go-Explore phase 2 on the agent's own route).
Run: venv_retro/bin/python tests/demo_bench.py

Properties:
  * every archived state variant carries its action prefix from the level's
    door, and replaying that prefix from the door reproduces the saved state
    byte for byte (door episodes, restarts, explorer walks; through reservoir
    rotation); archives without prefixes still load;
"""
import os, sys, tempfile
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import mario_native_vecenv as mnv
from mario_native_vecenv import MarioNativeVecEnv

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'mario_ppo_native_42.yaml')))['params']['config']
EC = dict(cfg['env_config']); [EC.pop(k, None) for k in ('name', 'action_type')]
EC.update(archive_path=None, explorer_envs=0, n_threads=1, dense_infos=True, seed=0,
          demo_start_prob=0.75, demo_window=32, demo_step=16, demo_success=0.2, demo_success_n=64)
VINE = np.load(os.path.join(ROOT, 'tests', 'data', 'vine_route_4-2.npy')).astype(np.int8)


def make(n=1, **kw):
    env = MarioNativeVecEnv('dbench', n, **dict(EC, **kw))
    return env, env.reset()


def replay_state(env, actions):
    """Independent replay: the level's door state, raw batch steps on a fresh
    1-core env, the savestate at the end."""
    lib = env.lib; rom = open(mnv.ROM, 'rb').read()
    e1 = lib.benv_create(rom, len(rom), 1, 1, int(env.single_stage))
    buf = __import__('ctypes').create_string_buffer(env.state_size)
    a = np.zeros(1, np.int32); o = np.zeros((1, 84, 84), np.uint8); r = np.zeros((1, 0x800), np.uint8)
    lib.benv_load(e1, 0, env.states[env.stages[0]])
    for x in np.frombuffer(actions, np.int8):
        a[0] = mnv._ACTION_BYTES[int(x)]
        lib.benv_step(e1, a.ctypes.data, o.ctypes.data, r.ctypes.data)
    lib.benv_save(e1, 0, buf); s = bytes(buf.raw); lib.benv_destroy(e1)
    return s


class _Always:                         # every refresh chance taken (rotations happen)
    def __init__(self, r): self.r = r
    def random_sample(self, *a): return np.zeros(a[0]) if a else 0.0
    def __getattr__(self, k): return getattr(self.r, k)


# ---------------------------------------------------------------- prefixes replay exactly
env, obs = make(n=4, self_restart_prob=0.6, explorer_envs=4, explore_pure=True)
rs = np.random.RandomState(1)
for s in range(300):
    env.step(rs.randint(0, 12, size=4))
env.rng = _Always(env.rng)             # force reservoir refreshes and rotations
for s in range(200):
    env.step(rs.randint(0, 12, size=4))
pairs = [(c, k, st, p) for c, e in env.archive.items() for k, (st, p) in enumerate(zip(e[0], env._prefixes(e)))]
known = [(c, k, st, p) for c, k, st, p in pairs if p is not None]
full = sum(len(e[0]) == 4 for e in env.archive.values())
bad = [(c, k) for c, k, st, p in known[:60] if replay_state(env, p) != st]
check('prefix: every archived variant has a prefix (%d of %d)' % (len(known), len(pairs)),
      len(pairs) >= 20 and len(known) == len(pairs), (len(known), len(pairs)))
check('prefix: %d variants replay byte-identically from the door (%d cells at 4 variants: rotated)'
      % (min(60, len(known)), full), not bad and full > 0, bad[:3])
check('prefix: variants and prefixes stay aligned (%d cells)' % len(env.archive),
      all(len(e[0]) == len(e[8]) for e in env.archive.values()))
env.close()

# ---------------------------------------------------------------- feature off = no prefixes, no extra draws
env0, _ = make(n=2, self_restart_prob=0.6, demo_start_prob=0.0)
for s in range(120):
    env0.step(rs.randint(0, 12, size=2))
check('off: demo_start_prob 0 stores no prefixes (entries of %d slots)' % max(len(e) for e in env0.archive.values()),
      not env0.demo_on and all(len(e) < 9 for e in env0.archive.values()))
env0.close()

# ---------------------------------------------------------------- old archives load
env, _ = make(n=1, self_restart_prob=0.6)
env.lib.benv_save(env.env, 0, env._sbuf)
old = [[bytes(env._sbuf.raw)], 0, 400, 0, 0, 0, 0, 0]
check('old entry: no [8] -> prefixes None, padded and aligned', env._prefixes(old) == [None] and len(old) == 9)
env.close()

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
```

- [ ] **Step 2: Run it to see it fail**

Run: `venv_retro/bin/python tests/demo_bench.py`
Expected: crash in `make` (`_prefixes` / kwargs unknown: `AttributeError: 'MarioNativeVecEnv' object has no attribute '_prefixes'`).

- [ ] **Step 3: Implement**

Module constant after `FRAME_STACK = 4`:
```python
# longest episode the action log records (the game timer bounds an episode at
# ~2400 agent steps); a longer one leaves its states without a prefix
ACT_LOG_MAX = 8192
```
Constructor signature: add `demo_start_prob=0.0, demo_window=32, demo_step=16, demo_success=0.2, demo_success_n=64,` before `**unknown`.
Right after `rom = open(ROM, 'rb').read()` add `self._rom = rom`.
After `self.sr_prob = self_restart_prob`:
```python
        # backward curriculum (Go-Explore phase 2, Salimans & Chen 2018): once
        # a door->clear route of the agent's own is known, demo_start_prob of
        # the resets start on it, near its end, and the start moves back
        # demo_step steps whenever the policy clears from the current band
        # often enough. Part of the self-restart machinery (evals and tools set
        # self_restart_prob 0 and never see it); needs exact replays.
        self.demo_prob = float(demo_start_prob)
        self.demo_on = self.demo_prob > 0 and self_restart_prob > 0 and not play_mode
        self.demo_window = int(demo_window); self.demo_step = int(demo_step)
        self.demo_success = float(demo_success); self.demo_success_n = int(demo_success_n)
        if self.demo_on:
            assert int(reset_noops) == 0, 'the route curriculum needs exact replays: reset_noops 0'
            assert random_stages and len(random_stages) == 1, 'the route curriculum trains one level'
        self.route = None
```
Next to `self.ep_steps = ...`:
```python
        # applied actions of the current episode (after every substitution) and
        # the prefix from the level's door to the episode's start state: an
        # archived state's prefix is start prefix + the episode's actions
        self.act_log = np.zeros((n, ACT_LOG_MAX), dtype=np.int8)
        self.start_prefix = [b''] * n
        self.demo_tau = np.full(n, -1, dtype=np.int32)   # route start index (-1: none)
```
New helpers (after `_forget_cell`):
```python
    def _prefixes(self, ent):
        """Entry [8]: the action prefix from the level's door (bytes, None if
        unknown) of each state variant in entry [0], index-aligned. Archives
        saved before prefixes existed are padded with None."""
        if not isinstance(ent[0], list):
            ent[0] = [ent[0]]
        while len(ent) < 9:
            ent.append(0 if len(ent) < 8 else None)
        if not isinstance(ent[8], list) or len(ent[8]) != len(ent[0]):
            ent[8] = [None] * len(ent[0])
        return ent[8]

    def _episode_prefix(self, i):
        """Actions from the level's door to env i's CURRENT state, or None.
        Valid inside _after_step before ep_steps is incremented."""
        p = self.start_prefix[i]; k = int(self.ep_steps[i]) + 1
        if p is None or k > ACT_LOG_MAX:
            return None
        return p + self.act_log[i, :k].tobytes()
```
`_reset_env`: at the top add `self.demo_tau[i] = -1; self.start_prefix[i] = b''`. In the restart branch replace `self.load_state(i, states[self.rng.randint(len(states))])` by:
```python
            k = self.rng.randint(len(states))
            self.load_state(i, states[k])
            self.start_prefix[i] = self._prefixes(ent)[k] if self.demo_on else None
```
`_reset_explorer`: top `self.start_prefix[i] = b''`; replace its `self.load_state(i, states[self.rng.randint(len(states))])` by the same three lines.
`step`: after `self.last_action[:] = acts`:
```python
        if self.demo_on:
            k = self.ep_steps; m = k < ACT_LOG_MAX
            self.act_log[np.nonzero(m)[0], k[m]] = acts[m].astype(np.int8)
```
Archive new cell: after `self.archive[cell] = [[bytes(self._sbuf.raw)], 0, int(t[i])]`:
```python
                    if self.demo_on:
                        self.archive[cell] += [0, 0, 0, 0, 0, [self._episode_prefix(i)]]
```
Archive refresh: after `early = self.cell_early.get(cell, 0)` add `pre = self._prefixes(ent) if self.demo_on else None`; after `ent[0].append(new)` add `if pre is not None: pre.append(self._episode_prefix(i))`; after `ent[0].pop(0)` (inside `if len(ent[0]) > 4:`) add `if pre is not None: pre.pop(0)`.
Soft path loop (continued lives): add `self.demo_tau[i] = -1` (the prefix stays valid: same game).

- [ ] **Step 4: Run the bench** — `venv_retro/bin/python tests/demo_bench.py` → `5/5 checks passed`.
- [ ] **Step 5: Existing benches** — `for b in env_bench explorer_bench novelty_bench reward_bench; do venv_retro/bin/python tests/$b.py | tail -1; done` → all pass.
- [ ] **Step 6: Commit** — `git add mario_native_vecenv.py tests/demo_bench.py && git commit -m "native env: every archived state variant carries its action prefix from the door (entry [8], demo curriculum groundwork)"` + trailer; push.

---

### Task 2: The route — built from the first clear, verified by replay, persisted

**Files:**
- Modify: `mario_native_vecenv.py` (helpers, `_after_step` before `self.ep_steps += 1`, `_save_archive`, end of `__init__`)
- Modify: `train.py` (`fresh_archive_conflict`)
- Test: `tests/demo_bench.py`

**Interfaces:**
- Consumes: `_episode_prefix`, `self._rom`, `self.states`, `self.stages`, `self.route_gps`.
- Produces: `_replay(actions: bytes) -> (states: list[bytes], gps: list[int], ram_last: np.ndarray)`; `set_route(actions, ram_check=None) -> bool`; `self.route = {'actions': bytes, 'states': list[bytes] (index tau = state after tau actions, only in the start level), 'last': int, 'tau_star': int, 'band': list[bool], 'rate': float|None, 'moves': int}`; sidecar `<archive_path>.demo.npz` (`actions` int8, `tau_star`, `level`).

- [ ] **Step 1: Failing tests** — append to `tests/demo_bench.py` before the final print (docstring: add "the first on-route clear with a known prefix becomes the route, verified by replay; a wrong exit never does; the route and tau* survive a save and reload, and a fresh run refuses the sidecar"):

```python
# ---------------------------------------------------------------- the first clear becomes the route
env, obs = make(n=1, self_restart_prob=1e-6)
k_done = None
for k, a in enumerate(VINE):
    obs, r, d, inf = env.step(np.array([a]))
    if d[0]:
        k_done = k; break
R = env.route
check('route: the vine run from the door becomes the route (%s actions, done at step %s)'
      % (None if R is None else len(R['actions']), k_done),
      R is not None and R['actions'] == VINE[:k_done + 1].tobytes())
check('route: start states stop before the level change (last %s, %d states, gp of last = 4-2)'
      % (None if R is None else R['last'], 0 if R is None else len(R['states'])),
      R is not None and len(R['states']) == R['last'] + 1 and R['last'] <= k_done - 1)
check('route: tau* starts demo_step before the last start state (%s)' % (None if R is None else R['tau_star']),
      R is not None and R['tau_star'] == R['last'] - 16)
check('route: state 0 is the door state', R is not None and R['states'][0] == env.states['4-2'])
env.close()
env, obs = make(n=1, self_restart_prob=1e-6)
flag = np.load(os.path.join(ROOT, 'tests', 'data', 'exit_flag_4-2.npy')).astype(np.int8)
check('route: a wrong exit (flag into 4-3) is rejected', not env.set_route(flag.tobytes()) and env.route is None)
check('route: set_route accepts the vine run', env.set_route(VINE.tobytes()) and env.route is not None)
env.close()

# ---------------------------------------------------------------- persistence
path = os.path.join(tempfile.mkdtemp(), 'archive.pkl')
env, obs = make(n=1, self_restart_prob=1e-6, archive_path=path)
for s in range(30):
    env.step(np.array([3]))
env.set_route(VINE.tobytes()); env.route['tau_star'] = 123
env.close()
env, obs = make(n=1, self_restart_prob=1e-6, archive_path=path)
check('persistence: route and tau* reload from the sidecar (tau* %s)' % (env.route and env.route['tau_star']),
      env.route is not None and env.route['tau_star'] == 123 and env.route['actions'] == VINE.tobytes())
check('persistence: archive entries keep their prefixes', all(len(e) == 9 for e in env.archive.values()))
env.close()
import train
os.remove(path)
cfgp = {'params': {'config': {'env_config': {'archive_path': path}}}}
check('persistence: a fresh run refuses an existing route sidecar', train.fresh_archive_conflict(cfgp, None, False))
```

- [ ] **Step 2: Run** — `venv_retro/bin/python tests/demo_bench.py` → the new checks FAIL (`route: ... None`, `set_route` AttributeError crash is acceptable as the failure).

- [ ] **Step 3: Implement** — helpers after `_episode_prefix`:

```python
    def _replay(self, actions):
        """Replay `actions` from the trained level's door on a private 1-core
        env with the training stepping. Returns the savestate before the first
        action and after each one, their levels, and the final RAM."""
        import ctypes as _ct
        e1 = self.lib.benv_create(self._rom, len(self._rom), 1, 1, int(self.single_stage))
        try:
            if self.skip != 4:
                self.lib.benv_set_skip(e1, self.skip)
            buf = _ct.create_string_buffer(self.state_size)
            a = np.zeros(1, np.int32); o = np.zeros((1, 84, 84), np.uint8)
            r = np.zeros((1, 0x800), np.uint8)
            self.lib.benv_load(e1, 0, self.states[self.stages[0]])
            self.lib.benv_obs(e1, 0, o.ctypes.data, r.ctypes.data)
            gp = lambda: min(max(int(r[0, 0x75F]) * 4 + int(r[0, 0x75C]), 0), 31)

            def snap():
                self.lib.benv_save(e1, 0, buf)
                return bytes(buf.raw)
            states, gps = [snap()], [gp()]
            for x in np.frombuffer(actions, np.int8):
                a[0] = _ACTION_BYTES[int(x)]
                self.lib.benv_step(e1, a.ctypes.data, o.ctypes.data, r.ctypes.data)
                states.append(snap()); gps.append(gp())
            return states, gps, r[0].copy()
        finally:
            self.lib.benv_destroy(e1)

    def set_route(self, actions, ram_check=None):
        """Make `actions` (applied actions from the trained level's door) the
        curriculum route if its replay ends in an on-route level advance (and,
        with ram_check, in exactly that RAM). Start states are the replay's
        states that are still in the start level; tau* starts demo_step
        before the last one. Returns True if accepted."""
        actions = actions if isinstance(actions, bytes) else np.asarray(actions, np.int8).tobytes()
        states, gps, ram = self._replay(actions)
        g0, g1 = gps[0], gps[-1]
        ok = g0 < g1 <= g0 + 15 and (self.route_gps is None or g1 in set(self.route_gps.tolist()))
        if ok and ram_check is not None and not np.array_equal(ram, ram_check):
            print('[demo] WARNING: route replay diverged from the live env; rejected', flush=True)
            ok = False
        if not ok:
            return False
        last = max(t for t, g in enumerate(gps) if g == g0)
        self.route = dict(actions=actions, states=states[:last + 1], last=last,
                          tau_star=max(0, last - self.demo_step), band=[], rate=None, moves=0)
        print(f'[demo] route: {len(actions)} actions, {last + 1} start states, '
              f'tau* {self.route["tau_star"]}', flush=True)
        return True
```
In `_after_step`, immediately after `done = done_pre`:
```python
        # the first on-route clear whose prefix is known becomes the route
        if self.demo_on and self.route is None and good.any():
            for i in np.nonzero(good)[0]:
                pre = self._episode_prefix(i)
                if pre is not None and self.set_route(pre, ram_check=self.ram[i]):
                    break
```
`_save_archive`, after `os.replace(tmp, self.archive_path)`:
```python
            if self.route is not None:
                dtmp = self.archive_path + '.demo.tmp'
                with open(dtmp, 'wb') as f:
                    np.savez(f, actions=np.frombuffer(self.route['actions'], np.int8),
                             tau_star=self.route['tau_star'], level=self.stages[0])
                os.replace(dtmp, self.archive_path + '.demo.npz')
```
End of `__init__` (after the `atexit` block):
```python
        side = (archive_path or '') + '.demo.npz'
        if self.demo_on and archive_path and os.path.exists(side):
            z = np.load(side)
            if self.set_route(z['actions'].tobytes()):
                self.route['tau_star'] = min(int(z['tau_star']), self.route['last'])
```
`train.py` `fresh_archive_conflict`: replace the condition by
```python
    found = [p for p in (path, (path or '') + '.demo.npz') if path and os.path.exists(p)]
    if found and not checkpoint and not resume_archive:
        return (f'{found[0]} exists: a fresh run would continue that archive. Move it '
                f'next to its run (runs_archive/<run>/), rename archive_path, or '
                f'pass --resume-archive to continue it on purpose')
```
- [ ] **Step 4: Run** — `venv_retro/bin/python tests/demo_bench.py` → all pass; `venv_retro/bin/python tests/env_bench.py | tail -1` → 30/30.
- [ ] **Step 5: Commit** — "native env: the first on-route clear with a known prefix becomes the curriculum route (replay-verified, persisted next to the archive)"; push.

---

### Task 3: Curriculum — route starts and the moving start point

**Files:**
- Modify: `mario_native_vecenv.py` (`_reset_env`, new `_reset_demo`, `_demo_outcome`, done loop, infos)
- Test: `tests/demo_bench.py`

**Interfaces:**
- Consumes: `self.route`, `self.demo_*`.
- Produces: `_reset_demo(i)`; `_demo_outcome(tau: int, success: bool)`; info keys `demo_start` (bool), `demo_tau` (int), and with a route `demo_len`, `demo_tau_star`, `demo_rate` (-1.0 before N outcomes).

- [ ] **Step 1: Failing tests** — append (docstring: "75% of resets start on the route inside [tau*, tau*+W]; tau 0 is a door episode, other route starts are neither door nor restart and credit no cell; tau* moves back demo_step at a 20% band success rate (13/64) and not below (12/64); a route start that plays the rest of the route clears and counts as a band success"):

```python
# ---------------------------------------------------------------- curriculum draws
env, obs = make(n=1, self_restart_prob=0.6)
env.set_route(VINE.tobytes()); R = env.route
taus = []
for k in range(2000):
    env._reset_env(0); taus.append(int(env.demo_tau[0]))
taus = np.array(taus); on = taus >= 0
check('draws: %.3f of resets start on the route (0.75)' % on.mean(), abs(on.mean() - 0.75) < 0.04)
check('draws: every route start is inside [tau*, tau*+W] (%d..%d, tau* %d)' % (taus[on].min(), taus[on].max(), R['tau_star']),
      taus[on].min() >= R['tau_star'] and taus[on].max() <= min(R['tau_star'] + 32, R['last']))
env._reset_env(0)
while env.demo_tau[0] < 0:
    env._reset_env(0)
check('draws: a route start is neither door nor restart and credits no cell',
      not env.is_door[0] and not env.was_restart[0] and env.start_cell[0] is None)
R['tau_star'] = 0
while env.demo_tau[0] != 0:
    env._reset_env(0)
check('draws: tau 0 is a door episode', env.is_door[0] and not env.was_restart[0])
env.close()

# ---------------------------------------------------------------- tau* moves at 20%, not below
env, obs = make(n=1, self_restart_prob=0.6)
env.set_route(VINE.tobytes()); R = env.route; t0 = R['tau_star']
for j in range(64):
    env._demo_outcome(t0, j >= 52)     # 52 failures then 12 successes (the window slides off a failure next)
check('tau*: 12/64 band successes keep tau* (%d)' % R['tau_star'], R['tau_star'] == t0 and abs(R['rate'] - 12 / 64) < 1e-9)
env._demo_outcome(t0 + 40, True)       # outside the band: ignored
check('tau*: outcomes outside the band are ignored', R['tau_star'] == t0)
env._demo_outcome(t0, True)            # sliding window: 13/64
check('tau*: 13/64 moves tau* back by demo_step (%d -> %d)' % (t0, R['tau_star']),
      R['tau_star'] == t0 - 16 and R['band'] == [] and R['moves'] == 1)
env.close()

# ---------------------------------------------------------------- a route start that plays the rest clears
env, obs = make(n=1, self_restart_prob=0.6, demo_start_prob=0.999999)
env.set_route(VINE.tobytes()); R = env.route
env._reset_env(0); env._fetch_obs(0); env._post_reset_init([0], env.ram)
tau = int(env.demo_tau[0]); res = None
for a in VINE[tau:]:
    obs, r, d, inf = env.step(np.array([a]))
    if d[0]:
        res = (float(r[0]), inf[0]['stages_cleared'], inf[0]['demo_start'], inf[0]['demo_tau']); break
check('suffix: from route state %d the rest of the route warps (%s)' % (tau, res),
      res is not None and res[0] >= 4700 and res[1] == 1 and res[2] and res[3] == tau)
check('suffix: the clear is a band success (%s)' % R['band'], (tau >= R['tau_star'] + 16) or R['band'][-1:] == [True])
env.close()
```

- [ ] **Step 2: Run** → new checks FAIL (`_demo_outcome` missing / draws 0.000).

- [ ] **Step 3: Implement** — `_reset_env` training branch: turn `if (self.sr_prob > 0 and self.archive and self.rng.random_sample() < self.sr_prob):` into an `elif` preceded by
```python
        if self.route is not None and self.rng.random_sample() < self.demo_prob:
            self._reset_demo(i)
        elif (self.sr_prob > 0 and self.archive
                and self.rng.random_sample() < self.sr_prob):
```
(the `else:` door branch and the common tail stay). New methods after `_reset_explorer`:
```python
    def _reset_demo(self, i):
        """Route start: a state on the curriculum route in [tau*, tau*+W]. It
        credits no archive cell; tau 0 is the level's door state (a door
        episode); every other start is neither door nor restart."""
        R = self.route
        hi = min(R['tau_star'] + self.demo_window, R['last'])
        tau = int(self.rng.randint(R['tau_star'], hi + 1))
        self.load_state(i, R['states'][tau])
        self.start_stage[i] = self.stages[0]
        self.demo_tau[i] = tau
        self.is_door[i] = tau == 0
        self.start_prefix[i] = R['actions'][:tau]

    def _demo_outcome(self, tau, success):
        """A route start ended. Outcomes of starts in the frontier band
        [tau*, tau*+demo_step) form a sliding window of demo_success_n; at a
        success rate >= demo_success tau* moves back demo_step (floor 0)."""
        R = self.route
        if not (R['tau_star'] <= tau < R['tau_star'] + self.demo_step):
            return
        R['band'].append(bool(success))
        if len(R['band']) > self.demo_success_n:
            R['band'].pop(0)
        if len(R['band']) < self.demo_success_n:
            return
        R['rate'] = sum(R['band']) / float(self.demo_success_n)
        if R['rate'] >= self.demo_success and R['tau_star'] > 0:
            R['tau_star'] = max(0, R['tau_star'] - self.demo_step)
            R['band'] = []; R['moves'] += 1
            print(f'[demo] tau* -> {R["tau_star"]} (band success {R["rate"]:.2f})', flush=True)
```
Done loop: first statement inside `for i in np.nonzero(done)[0]:`
```python
            if self.demo_tau[i] >= 0 and self.route is not None:
                self._demo_outcome(int(self.demo_tau[i]), self.cleared[i] > 0)
```
Infos: build the dict as `inf = {...}` (same keys) and before `infos.append(inf)`:
```python
            if self.demo_on:
                inf['demo_start'] = bool(self.demo_tau[i] >= 0)
                inf['demo_tau'] = int(self.demo_tau[i])
                if self.route is not None:
                    inf['demo_len'] = self.route['last'] + 1
                    inf['demo_tau_star'] = self.route['tau_star']
                    inf['demo_rate'] = -1.0 if self.route['rate'] is None else self.route['rate']
```
- [ ] **Step 4: Run** — demo_bench all pass; `env_bench`, `explorer_bench`, `novelty_bench` still pass.
- [ ] **Step 5: Commit** — "native env: backward curriculum -- route starts in [tau*, tau*+W], tau* steps back at a 20% band success rate"; push.

---

### Task 4: Observer metrics and tool guards

**Files:**
- Modify: `callbacks.py` (`__init__` buffers ~L47-64, `_process_single_info`, `after_print_stats`)
- Modify: `grpo/train_grpo.py:740` (env overrides)
- Test: `tests/demo_bench.py`

**Interfaces:**
- Consumes: info keys from Task 3.
- Produces: scalars `mario/demo_share`, `mario/clear_demo/<stage>`, `mario/demo_len`, `mario/demo_tau`, `mario/demo_frontier_success`.

- [ ] **Step 1: Failing test** — append:

```python
# ---------------------------------------------------------------- observer
import callbacks
class _W:
    def __init__(self): self.s = {}
    def add_scalar(self, k, v, e): self.s[k] = v
ob = callbacks.MarioObserver.__new__(callbacks.MarioObserver)
callbacks.MarioObserver.__init__(ob)
ob.writer = _W(); ob.algo = None; ob.video_freq = 0
base = dict(max_x_pos=100, game_progress=13, stages_cleared=0, start_stage='4-2', door=False, self_restart=False)
for j in range(8):
    ob._process_single_info(dict(base, demo_start=j < 4, demo_tau=500 if j < 4 else -1, stages_cleared=int(j == 0),
                                 demo_len=600, demo_tau_star=480, demo_rate=0.25))
ob.after_print_stats(0, 1, 0.0)
S = ob.writer.s
check('observer: demo share, clear rate of route starts and the curriculum state (%s)'
      % {k: v for k, v in S.items() if 'demo' in k},
      S.get('mario/demo_share') == 0.5 and S.get('mario/clear_demo/4-2') == 0.25 and S.get('mario/demo_len') == 600
      and S.get('mario/demo_tau') == 480 and S.get('mario/demo_frontier_success') == 0.25)
```
(If `MarioObserver.__init__` requires arguments, pass the defaults it declares; check its signature first with `grep -n "def __init__" callbacks.py`.)

- [ ] **Step 2: Run** → FAIL (no `mario/demo_share`).
- [ ] **Step 3: Implement** — `__init__` buffers: `self.episode_demo = []; self.demo_clears = {}; self.demo_state = None`. `_process_single_info` (after the `stage_records` block):
```python
        if 'demo_start' in info:
            # backward curriculum: share of route starts, their clear rate, and
            # the curriculum's state (route length, tau*, band success)
            self.episode_demo.append(float(info['demo_start']))
            if info['demo_start']:
                self.demo_clears.setdefault(info.get('start_stage', '?'), []).append(
                    float(info.get('stages_cleared', 0) > 0))
        if 'demo_tau_star' in info:
            self.demo_state = (info['demo_len'], info['demo_tau_star'], info.get('demo_rate', -1.0))
```
`after_print_stats` (before "Curriculum: sample unmastered stages"):
```python
        if self.episode_demo:
            self.writer.add_scalar('mario/demo_share', float(np.mean(self.episode_demo)), epoch_num)
        for stage, v in self.demo_clears.items():
            self.writer.add_scalar(f'mario/clear_demo/{stage}', float(np.mean(v)), epoch_num)
        if self.demo_state is not None:
            ln, ts, rate = self.demo_state
            self.writer.add_scalar('mario/demo_len', ln, epoch_num)
            self.writer.add_scalar('mario/demo_tau', ts, epoch_num)
            if rate >= 0:
                self.writer.add_scalar('mario/demo_frontier_success', rate, epoch_num)
```
Clear-buffers block: `self.episode_demo.clear(); self.demo_clears.clear()` (keep `demo_state`: the curriculum state persists).
`grpo/train_grpo.py:740` `ec.update(dict(...` add `demo_start_prob=0.0,` (GRPO prompts are its own start distribution).
- [ ] **Step 4: Run** — demo_bench all pass; `eval_bench`, `grpo_bench`, `train_bench` pass.
- [ ] **Step 5: Commit** — "Observer: backward-curriculum metrics (demo share, clear_demo, route length, tau*, band success); GRPO never draws route starts"; push.

---

### Task 5: The auditor knows route starts

**Files:**
- Modify: `tools/audit_env.py` (`record`, `check`, `main` args)

**Interfaces:**
- Consumes: `env.set_route`, `env.route`, `env.demo_tau`, `mario_native_vecenv._ACTION_BYTES`.
- Produces: `--route <npy>` (inject a route before the first reset); recorded `demo1` (n,) per step, `tau_star` per step, `route_actions`; violations `route_does_not_clear`, `demo_start_not_route_state`, `demo_tau_outside_window`.

- [ ] **Step 1: Implement** — `main`: `ap.add_argument('--route', default=None, help='action file (npy) from the door injected as the curriculum route')`. `record()`: after the `if args.archive:` block and before `obs = env.reset()`:
```python
    if getattr(args, 'route', None):
        assert env.set_route(np.load(args.route).astype(np.int8).tobytes()), 'the --route actions do not clear'
```
In the step loop after `R['restart1'].append(...)`:
```python
        R['demo1'].append(env.demo_tau[:nt].copy())
        R['tau_star'].append(env.route['tau_star'] if env.route is not None else -1)
```
After `D = {k: np.array(v) ...}`: `D['route_actions'] = np.frombuffer(env.route['actions'], np.int8).copy() if env.route is not None else np.zeros(0, np.int8)`.
`check()`: after the `doors` set is built:
```python
    route_ram, W = [], int(ec.get('demo_window', 32))
    if len(D['route_actions']):
        from mario_native_vecenv import _ACTION_BYTES
        renv = MarioNativeVecEnv('route', 1, random_stages=list(ec['random_stages']), full_game=True, n_threads=1)
        renv.load_state(0, renv.states[ec['random_stages'][0]])
        renv.lib.benv_obs(renv.env, 0, renv.obs_u8[0].ctypes.data, renv.ram[0].ctypes.data)
        route_ram.append(tuple(int(v) for v in renv.ram[0, ADDR]))
        ab = np.zeros(1, np.int32)
        for a in D['route_actions']:
            ab[0] = _ACTION_BYTES[int(a)]
            renv.lib.benv_step(renv.env, ab.ctypes.data, renv.obs_u8.ctypes.data, renv.ram.ctypes.data)
            route_ram.append(tuple(int(v) for v in renv.ram[0, ADDR]))
        renv.close()
        g = [min(max(r[A_[0x75F]] * 4 + r[A_[0x75C]], 0), 31) for r in (route_ram[0], route_ram[-1])]
        if not (g[0] < g[1] <= g[0] + 15 and (ROUTE is None or g[1] in ROUTE)):
            viol['route_does_not_clear'].append((0, 0, g))
```
(define `A_ = {a: k for k, a in enumerate(ADDR)}` just above; `A` is defined later in the function, so do not rely on it here.) Add `'demo1'` to the `L = {...}` key tuple and `TS = D['tau_star'].tolist()`. Replace the label block at `if done:`:
```python
                tau = L['demo1'][s][i]
                if cont:
                    ref = 'continuation'
                elif tuple(r0) in doors:
                    ref = 'door'
                elif tau >= 1:
                    ref = 'demo'
                    if not (tau < len(route_ram) and tuple(r0) == route_ram[tau]):
                        bad('demo_start_not_route_state', s, i, tau)
                else:
                    ref = 'restart'
                if tau >= 0 and not (TS[s] <= tau <= TS[s] + W):
                    bad('demo_tau_outside_window', s, i, (tau, TS[s]))
```
(the existing `exp_door, exp_restart` comparison and the `st = dict(label=ref, ...)` line stay as they are.)
- [ ] **Step 2: Run the audit with a route** — `CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python tools/audit_env.py --config configs/mario_ppo_native_42.yaml --route tests/data/vine_route_4-2.npy --steps 1500 --device cpu` with `demo_start_prob: 0.75` in the config (Task 6 sets it; for this step pass it by editing the config first or run Task 6 Step 1 before this step).
Expected: `none` under violations, an episodes row labelled `demo`.
- [ ] **Step 3: Commit** — "tools/audit_env.py: --route, route starts labelled and checked against an independent replay of the route"; push.

---

### Task 6: Run l config, docs, full check, launch

**Files:**
- Modify: `configs/mario_ppo_native_42.yaml`, `CLAUDE.md`, `EXPERIMENTS.md`

- [ ] **Step 1: Config** — header comment line 1:
```yaml
# 4-2 run l (2026-09-19): run k + the backward curriculum (Go-Explore phase 2 on the agent's own route): once a
# door->warp route is known (the first clear with a recorded action prefix), 75% of resets start on it in
# [tau*, tau*+32] and tau* steps back 16 when >= 20% of the last 64 band starts warp. Run k at epoch 2500: 0/446
# door episodes and 0/302 restarts at x 512-1023 reached the warp area; 53% of all steps loitered at the level end.
```
`name: Mario_PPO42l`; `archive_path: native/archive_42l.pkl`; after `cell_bonus_door_only: true` add:
```yaml
      # backward curriculum (demo_bench.py): share of resets on the route, draw width, band / step size, band
      # success rate to step back, band window
      demo_start_prob: 0.75
      demo_window: 32
      demo_step: 16
      demo_success: 0.2
      demo_success_n: 64
```
- [ ] **Step 2: Docs** — `CLAUDE.md` Native env list, new bullet after `explorer_envs`:
```markdown
- `demo_start_prob` (backward curriculum, Go-Explore phase 2): every archived state variant keeps its action prefix from the door (entry `[8]`); the first on-route clear with a known prefix becomes the route (replay-verified, `<archive>.demo.npz`); that share of resets starts on it in `[tau*, tau*+demo_window]`, and `tau*` steps back `demo_step` when `demo_success` of the last `demo_success_n` band starts clear. Metrics `mario/demo_*`, `mario/clear_demo/<lvl>`.
```
and add `tests/demo_bench.py` (backward curriculum) to the offline-checks list.
- [ ] **Step 3: Full check** — all benches (`reward env novelty grpo eval explorer vine train retro render demo`) 0 FAIL; the Task 5 audit with `--route` 0 violations; a plain audit (no route) 0 violations.
- [ ] **Step 4: Commit** — "4-2 config for run l (backward curriculum); CLAUDE.md"; push.
- [ ] **Step 5: Stop run k and archive it** — `pkill -f "train.py --config configs/mario_ppo_native_42.yaml"` in its own command (the pattern must not match the command doing it: run `kill <pid>` from `pgrep -f "venv_retro/bin/python train.py"`), wait for exit (SIGTERM saves the archive), then `mv runs/Mario_PPO42k_* runs_archive/`, `mv native/archive_42k.pkl runs_archive/Mario_PPO42k_19-14-17-38/archive.pkl`.
- [ ] **Step 6: Launch** — `setsid nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 venv_retro/bin/python train.py --config configs/mario_ppo_native_42.yaml --video-freq 200 > logs/train_ppo_42l_<MMDD-HHMM>.log 2>&1 < /dev/null & disown`; verify epochs, fps, no Traceback.
- [ ] **Step 7: EXPERIMENTS.md** — entry "2026-09-19 — Mario_PPO42l: backward curriculum" with the change, start time, the spec's prediction table; commit and push.
