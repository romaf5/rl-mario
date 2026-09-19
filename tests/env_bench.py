"""Env checks for the 2026-09-19 review fixes (training path of the 4-2 config).
Run: venv_retro/bin/python tests/env_bench.py

Properties:
  * the step's Signals survive the resets done inside that step (the trackers
    used to alias them: a reset rewrote .t / .frame of the step that ended);
  * rewards only for what the step did: a death step pays nothing (its RAM
    is already the respawn), neither do the hack-free path's death animation,
    pit fall and life-resume step, nor standing in the start cell; the term
    breakdown of a leaving step is zeroed like its reward;
  * a sub-area ($074F) is its own frame and archive spot: the coin-cache pipe
    into 4-2's coin room is a transition (not a held teleport paying +20),
    and 4-2's flag area and warp area are different frames;
  * life_loss_reset (per-life training): a life loss ends the game too and
    the next episode is a fresh draw -- a restart that died past 4-2's
    halfway point used to continue at x 1576 labelled a door episode; with
    it off the continued life is neither a door episode nor a restart; the
    multi-life eval still plays on after a death;
  * frontier practice goes where the policy can learn: with outcomes
    simulated, cells it never converts no longer take the frontier draws
    (the failure-rate weight grew with every failed try and locked the
    top-k onto them), untried proven winners and ~40% cells do;
  * the unpaid cutoff never fires on a level-change step: a cutoff landing on
    the warp's pending step (which pays 0 by design) lost the 4700;
  * archive hygiene: a cell in use by another env is not pruned, a cell that
    left the archive is never credited again (stale counters), and a refresh
    stores no byte-identical duplicate state;
  * the archive file keeps explorer walk counts and early deaths across a
    reload (every cell was 'fresh' again), and a fresh train.py run refuses
    to continue an existing archive file silently;
  * the per-spot tile-variant index equals a scan of the archive (after
    evictions too);
  * door counts of the relative novelty bonus decay across a GRPO freeze
    (applied at unfreeze; they never decayed); info coins has both digits;
"""
import os, sys
import numpy as np, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from mario_native_vecenv import MarioNativeVecEnv

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'mario_ppo_native_42.yaml')))['params']['config']
EC = dict(cfg['env_config']); [EC.pop(k, None) for k in ('name', 'action_type')]
EC.update(archive_path=None, explorer_envs=0, n_threads=1, dense_infos=True, seed=0)


def make(n=1, **kw):
    env = MarioNativeVecEnv('ebench', n, **dict(EC, **kw))
    return env, env.reset()


def kill(env, i=0):
    """Drop Mario below the screen: the game kills him on the next frame."""
    env.ram[i, 0xB5] = 2; env.ram[i, 0xCE] = 0xF0
    env.lib.benv_set_ram(env.env, int(i), env.ram[i].tobytes())


def replay(env, name):
    for a in np.load(os.path.join(ROOT, 'tests', 'data', name)).astype(int):
        obs, r, d, inf = env.step(np.array([a]))
        if d[0]:
            return obs, r, d, inf
    return None


# ---------------------------------------------------------------- signals are not aliased by resets
env, obs = make(self_restart_prob=0.0)
out = replay(env, 'exit_flag_4-2.npy')
sg = env.last_signals
check('signals: the wrong-exit step keeps its own timer (sig.t %d = info time %d, not the reset door timer %d)'
      % (sg.t[0], out[3][0]['time'], env.time_last[0]) if out else 'signals: flag run replayed',
      out is not None and int(sg.t[0]) == out[3][0]['time'] != int(env.time_last[0]),
      None if out is None else (int(sg.t[0]), out[3][0]['time'], int(env.time_last[0])))
check('signals: the wrong-exit step keeps its own frame (4-3), not the reset door frame',
      out is not None and bool(sg.wrong_exit[0]) and sg.frame[0] != env.prev_frame[0],
      None if out is None else (int(sg.frame[0]), int(env.prev_frame[0])))
env.close()

# ---------------------------------------------------------------- no pay for what the step did not do
from mario_native_vecenv import NativeEvalEnv
env, obs = make(self_restart_prob=0.0)
from mario_rewards import FirstVisitCells
cells = env.rewards.get(FirstVisitCells); r0 = env.ram[0]
start = (int(env.prev_frame[0]), int(env._x()[0]) // cells.xb, int(r0[0x3B8]) // cells.yb)
check('rewards: the cell a life starts in is already seen (not paid as a discovery)', start in cells.seen[0],
      (start, cells.seen[0]))
for s in range(40):                   # run right a little (death x within 600 px of the respawn)
    env.step(np.array([3]))
env._post_reset_init([0], env.ram)    # a life starting here, as an archive restart does: the respawn cell is new
kill(env)
for s in range(10):
    obs, r, d, inf = env.step(np.array([0]))
    if d[0]:
        break
terms = {k: float(v[0]) for k, v in env.last_terms.items()}
check('rewards: a death step pays nothing (r %.1f, terms %s)' % (r[0], terms),
      d[0] and float(r[0]) == 0 and all(v == 0 for v in terms.values()), (bool(d[0]), float(r[0]), terms))
env.close()
# hack-free path (clips): the pit fall, the death animation and the resume step score nothing
ev = NativeEvalEnv(raw_steps=True, **{k: v for k, v in dict(EC, self_restart_prob=0.0, episode_life=False).items()
                                      if k != 'dense_infos'})
ev.reset()
for s in range(40):
    ev.step(3)
x_kill = int(ev.v.x_last[0]); kill(ev.v)
paid_dying, lives0, resumed, paid_resume = 0.0, int(ev.v.ram[0, 0x75A]), False, None
for s in range(200):
    obs, r, d, inf = ev.step(0)
    st = int(ev.v.ram[0, 0x0E]); yv = int(ev.v.ram[0, 0xB5])
    if int(ev.v.ram[0, 0x75A]) == lives0 and (st in (0x0B, 0x06) or yv > 1):
        paid_dying += r
    if int(ev.v.ram[0, 0x75A]) < lives0 and st == 8 and not resumed:
        resumed, paid_resume = True, r
        break
check('rewards (hack-free): the pit fall and death animation pay nothing (%.1f)' % paid_dying, paid_dying == 0, paid_dying)
check('rewards (hack-free): the first control step of the new life pays nothing (%s)' % paid_resume,
      resumed and paid_resume == 0, (resumed, paid_resume))
ev.close()
env, obs = make(self_restart_prob=0.0)
log = []
for a in np.load(os.path.join(ROOT, 'tests', 'data', 'exit_flag_4-2.npy')).astype(int):
    obs, r, d, inf = env.step(np.array([a]))
    log.append((float(r[0]), sum(float(v[0]) for v in env.last_terms.values()), bool(getattr(env, 'last_leaving', np.zeros(1, bool))[0])))
    if d[0]:
        break
lv = [t for t in log if t[2]]
check('rewards: the leaving steps of the flag run (%d) pay 0 and their term breakdown sums to 0' % len(lv),
      len(lv) >= 2 and all(r == 0 and tot == 0 for r, tot, _ in lv), lv)
check('rewards: every step\'s term breakdown sums to its reward',
      all(abs(r - tot) < 1e-4 for r, tot, _ in log))
env.close()

# ---------------------------------------------------------------- sub-areas are frames
import gzip
env, obs = make(self_restart_prob=1e-6)            # archiving on (cell keys), practically no restarts
env.load_state(0, gzip.open(os.path.join(ROOT, 'tests', 'data', 'coin_cache_pipe_4-2.state')).read())
env._fetch_obs(0); env._post_reset_init([0], env.ram)
key0, x0 = env.cell_of(0), int(env._x()[0])
obs, r, d, inf = env.step(np.array([10]))           # DOWN into the coin-cache pipe
sg = env.last_signals
check('sub-area: the coin-cache pipe (x %d -> %d) is a frame change, not a held teleport' % (x0, sg.x[0]),
      bool(sg.frame_change[0]) and not bool(sg.hold[0]) and int(sg.x[0]) > x0 + 600,
      (bool(sg.frame_change[0]), bool(sg.hold[0]), int(sg.x[0])))
check('sub-area: arriving in the coin room pays no progress (%.1f)' % env.last_terms['progress'][0],
      float(env.last_terms['progress'][0]) == 0 and not bool(sg.page_reset[0]))
for s in range(6):
    env.step(np.array([0]))
key1 = env.cell_of(0)
check('sub-area: the coin room is its own archive spot (key slot 1 %d vs %d)' % (key1[1], key0[1]), key1[1] != key0[1],
      (key0, key1))
env.close()


def arrival_frame(name, atype):
    env, obs = make(self_restart_prob=0.0)
    for a in np.load(os.path.join(ROOT, 'tests', 'data', name)).astype(int):
        obs, r, d, inf = env.step(np.array([a]))
        if int(env.last_signals.atype[0]) == atype:
            f = int(env.last_signals.frame[0]); env.close(); return f
    env.close()


fw, ff = arrival_frame('vine_route_4-2.npy', 1), arrival_frame('exit_flag_4-2.npy', 1)
check('sub-area: 4-2 warp area and flag area are different frames (%s vs %s)' % (fw, ff),
      fw is not None and ff is not None and fw != ff)

# ---------------------------------------------------------------- life loss = fresh draw
env, obs = make(self_restart_prob=0.0)
for a in np.load(os.path.join(ROOT, 'tests', 'data', 'vine_route_4-2.npy')).astype(int):
    env.step(np.array([a]))
    if int(env.last_signals.atype[0]) == 1:
        break
for s in range(10):
    env.step(np.array([3]))
env.lib.benv_save(env.env, 0, env._sbuf); deep = bytes(env._sbuf.raw); deep_x = int(env._x()[0])
deep_cell = env.cell_of(0); env.close()


def restart_then_die(**kw):
    env = MarioNativeVecEnv('ebench', 1, **dict(EC, self_restart_prob=1.0, **kw))
    env.archive = {deep_cell: [[deep], 0, 400]}
    env.reset()
    x_start = int(env._x()[0]); lab0 = (bool(env.is_door[0]), bool(env.was_restart[0]))
    kill(env)
    for s in range(10):
        obs, r, d, inf = env.step(np.array([0]))
        if d[0]:
            break
    out = dict(x_start=x_start, lab0=lab0, done=bool(d[0]), info_door=inf[0].get('door'),
               x_next=int(env._x()[0]), door=bool(env.is_door[0]), restart=bool(env.was_restart[0]),
               cont=bool(getattr(env, 'continuation', np.zeros(1, bool))[0]))
    env.close()
    return out


o = restart_then_die()
check('life loss: a restart from the warp area (x %d) that dies starts a fresh draw (next x %d, restart %s, door %s)'
      % (o['x_start'], o['x_next'], o['restart'], o['door']),
      o['lab0'] == (False, True) and o['done'] and o['x_next'] == o['x_start'] and o['restart'] and not o['door'], o)
check('life loss: the ended restart reports door False', o['info_door'] is False, o['info_door'])
o = restart_then_die(life_loss_reset=False)
check('life loss (life_loss_reset off): the continued life (x %d) is neither a door episode nor a restart' % o['x_next'],
      o['done'] and not o['door'] and not o['restart'] and o['cont'] and o['x_next'] != o['x_start'], o)
ev = NativeEvalEnv(raw_steps=False, **{k: v for k, v in dict(EC, self_restart_prob=0.0, episode_life=False).items()
                                       if k != 'dense_infos'})
ev.reset()
for s in range(20):
    ev.step(3)
lives0 = int(ev.v.ram[0, 0x75A]); kill(ev.v); dd = False
for s in range(10):
    obs, r, d, inf = ev.step(0); dd |= d
check('life loss: the multi-life eval keeps playing after a death (lives %d -> %d, done %s)'
      % (lives0, int(ev.v.ram[0, 0x75A]), dd), not dd and int(ev.v.ram[0, 0x75A]) == lives0 - 1)
ev.close()

# ---------------------------------------------------------------- frontier practice weights
env, obs = make(self_restart_prob=1.0, self_restart_frontier_prob=1.0)
env.lib.benv_save(env.env, 0, env._sbuf); st = bytes(env._sbuf.raw)
rs = np.random.RandomState(0)
kinds = {}
for j in range(48):                     # 16 dead, 24 learnable (40%), 8 untried explorer-proven winners
    c = ('4-2', 2 * 256 + 1, 100 + j, 5, 0, 2, 0, 0)
    kind = 'dead' if j < 16 else ('learnable' if j < 40 else 'fresh')
    env.archive[c] = [[st], 0, 400]; kinds[c] = kind
    env.explore_wins[c] = 1
    if kind == 'dead':
        env.cell_tries[c] = 200
    elif kind == 'learnable':
        env.cell_tries[c] = 20; env.cell_wins[c] = 8
p_true = {'dead': 0.0, 'learnable': 0.4, 'fresh': 0.4}
draws = {'dead': 0, 'learnable': 0, 'fresh': 0}
for k in range(1500):
    env._reset_env(0)
    c = env.start_cell[0]
    if c in kinds:
        draws[kinds[c]] += 1
        if rs.random_sample() < p_true[kinds[c]]:
            env.cell_wins[c] = env.cell_wins.get(c, 0) + 1
env.close()
tot = sum(draws.values())
check('frontier: cells the policy never converts get few draws (%s)' % draws,
      tot > 1000 and draws['dead'] < 0.1 * tot, draws)
check('frontier: every untried proven winner gets practised', draws['fresh'] >= 8, draws)

# ---------------------------------------------------------------- no cutoff on a level-change step
acts = np.load(os.path.join(ROOT, 'tests', 'data', 'vine_route_4-2.npy')).astype(int)
env, obs = make(self_restart_prob=0.0)
pend = None
for k, a in enumerate(acts):
    env.step(np.array([a]))
    if pend is None and int(env.last_signals.gp[0]) > 13:
        pend = k                      # first step already inside 8-1: the pending step
        break
env.close()
env, obs = make(self_restart_prob=0.0)
res = None
for k, a in enumerate(acts):
    if k == pend:
        env.unpaid[0] = env.unpaid_timeout - 1     # the cutoff would fire exactly on the pending step
    obs, r, d, inf = env.step(np.array([a]))
    if d[0]:
        res = (k, float(r[0]), bool(inf.time_outs[0]), inf[0]['stages_cleared'])
        break
env.close()
check('cutoff: an unpaid cutoff due on the warp\'s pending step waits for the confirm step (end %s)' % (res,),
      pend is not None and res is not None and res[0] == pend + 1 and res[1] >= 4700 and not res[2] and res[3] == 1, res)

# ---------------------------------------------------------------- archive hygiene
env, obs = make(n=2, self_restart_prob=1.0, self_restart_frontier_prob=0.0)
X = ('4-2', 2 * 256 + 1, 99, 5, 0, 2, 0, 0)
env.lib.benv_save(env.env, 0, env._sbuf)
env.archive = {X: [[bytes(env._sbuf.raw)], 12, 400]}; env.cell_early = {X: 11}
for i in range(2):
    env._reset_env(i); env._fetch_obs(i)
env._post_reset_init([0, 1], env.ram)
kill(env, 0)
for s in range(3):
    env.step(np.array([0, 0]))
check('archive: an early death does not prune a cell another env is playing from', X in env.archive,
      (X in env.archive, env.cell_early.get(X)))
env.archive.pop(X, None)               # evicted while env 1's episode from it still runs
env.cell_early.pop(X, None); env.cell_wins.pop(X, None)
kill(env, 1)
for s in range(3):
    env.step(np.array([0, 0]))
check('archive: a cell that left the archive gets no counters back (early %s, wins %s)'
      % (env.cell_early.get(X), env.cell_wins.get(X)), X not in env.cell_early and X not in env.cell_wins)
env.close()
env, obs = make(n=4, self_restart_prob=1e-6)
class _Always:                         # every refresh chance taken (the env's other draws unchanged)
    def __init__(self, r): self.r = r
    def random_sample(self, *a): return np.zeros(a[0]) if a else 0.0
    def __getattr__(self, k): return getattr(self.r, k)
env.rng = _Always(env.rng)
for s in range(80):
    env.step(np.full(4, 3 if s % 8 else 4))
dups = sum(len(e[0]) - len(set(e[0])) for e in env.archive.values())
check('archive: identical envs on identical inputs store no duplicate states (%d duplicates in %d cells)'
      % (dups, len(env.archive)), len(env.archive) > 5 and dups == 0, dups)
env.close()

# ---------------------------------------------------------------- persistence
import tempfile
path = os.path.join(tempfile.mkdtemp(), 'archive.pkl')
env, obs = make(self_restart_prob=1e-6, archive_path=path)
for s in range(40):
    env.step(np.array([3]))
c0 = list(env.archive)[0]
env.explore_walks[c0] = 17; env.cell_early[c0] = 5
env.close()
env, obs = make(self_restart_prob=1e-6, archive_path=path)
check('persistence: explorer walks and early deaths survive a save and reload (%s, %s)'
      % (env.explore_walks.get(c0), env.cell_early.get(c0)), env.explore_walks.get(c0) == 17 and env.cell_early.get(c0) == 5)
env.close()
import train
cfg = {'params': {'config': {'env_config': {'archive_path': path}}}}
check('persistence: a fresh train.py run refuses an existing archive file', train.fresh_archive_conflict(cfg, None, False))
check('persistence: --checkpoint or --resume-archive continue it',
      not train.fresh_archive_conflict(cfg, 'x.pth', False) and not train.fresh_archive_conflict(cfg, None, True))

# ---------------------------------------------------------------- tile-variant index
env, obs = make(n=4, self_restart_prob=1e-6, self_restart_cells=5, cell_max_variants=3)
for s in range(200):
    env.step(np.array([3, 4, 1, 3 if s % 5 else 5]))
brute = lambda c: sum(1 for k in env.archive if k[:6] == c[:6] and k[7:] == c[7:])
probe_cells = list(env.archive) + [c[:6] + (12345,) + c[7:] for c in list(env.archive)[:5]]
ok = all(env._tile_variants(c) == brute(c) for c in probe_cells)
check('index: tile-variant counts equal a scan of the archive after evictions (%d cells, cap 5)' % len(env.archive),
      ok and len(env.archive) == 5, [(env._tile_variants(c), brute(c)) for c in probe_cells if env._tile_variants(c) != brute(c)][:5])
env.close()

# ---------------------------------------------------------------- frozen door counts decay; coins
env, obs = make(self_restart_prob=1e-6)
env.door_seen = {('x',): 5.0}
env.freeze_door_seen(True)
for s in range(256):
    env.step(np.array([0]))
env.freeze_door_seen(False)
v = env.door_seen[('x',)]
check('novelty: door counts decay across a freeze (5.0 -> %.2f after 256 steps, expect 5 x 0.9^8 = %.2f)' % (v, 5 * 0.9 ** 8),
      abs(v - 5 * 0.9 ** 8) < 1e-6, v)
env.ram[0, 0x7ED] = 1; env.ram[0, 0x7EE] = 7; env.lib.benv_set_ram(env.env, 0, env.ram[0].tobytes())
obs, r, d, inf = env.step(np.array([0]))
check('info: coins reads both digits (%s)' % inf[0].get('coins'), inf[0].get('coins') == 17, inf[0].get('coins'))
env.close()

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
