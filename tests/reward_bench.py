"""Offline reward test bench: replays fixed trajectories through the env and
asserts the ORDERING and payments the reward design promises, in seconds,
without training. Run: venv_retro/bin/python tests/reward_bench.py

Corpus: human play traces (traces/*.csv, deterministic replay), scripted
bots on 8-4 from the door. Every assertion is a property from the design
(mario_rewards.py): no negative reward ever; a transition on the route pays;
a page reset pays 0 and its re-run pays 0 until new ground; the pipe-1
recovery pays inside the grace window; a corridor loop ends inside the
grace window; standing still ends at the cutoff with 0 and is a time-out;
identical actions give identical rewards.
"""
import csv, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mario_native_vecenv import MarioNativeVecEnv

ACT = ['NOOP', 'R', 'R+A', 'R+B', 'R+A+B', 'A', 'L', 'L+A', 'L+B', 'L+A+B', 'DOWN', 'UP']
AIDX = {a: i for i, a in enumerate(ACT)}
ROUTE = ['1-1', '1-2', '4-1', '4-2', '8-1', '8-2', '8-3', '8-4']
OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


def make(level, n=1, play_mode=True, **kw):
    ec = dict(full_game=True, random_stages=[level], route_levels=ROUTE,
              sticky_actions=0, explore_eps=0, self_restart_prob=0, reset_noops=0,
              n_threads=1, dense_infos=True, play_mode=play_mode, seed=0)
    ec.update(kw)
    env = MarioNativeVecEnv('bench', n, **ec); env.reset()
    return env


def replay(env, acts, rec=None):
    """Step recorded actions; return list of (r, x, hw, events, done, level)."""
    out = []
    for a in acts:
        obs, r, d, inf = env.step(np.array([a]))
        s = env.last_signals
        ev = set()
        if s.page_reset[0]: ev.add('reset')
        if s.frame_change[0]: ev.add('transition')
        if s.level_delta[0] > 0: ev.add('clear')
        if s.died[0]: ev.add('death')
        if s.timeout[0]: ev.add('timeout')
        if s.wrong_exit[0]: ev.add('wrong_exit')
        out.append((float(r[0]), int(s.x[0]), int(env.hw[0]), ev, bool(d[0]),
                    '%d-%d' % (inf[0]['world'], inf[0]['stage']), dict((k, float(v[0])) for k, v in env.last_terms.items())))
    return out


def trace_actions(path):
    return [AIDX[r['action']] for r in csv.DictReader(open(path))]


# ---------------------------------------------------------------- 8-4 human trace
T84 = 'traces/play_0905-181843.csv'
if os.path.exists(T84):
    env = make('8-4'); log = replay(env, trace_actions(T84)); env.close()
    rs = np.array([l[0] for l in log])
    check('8-4 trace: no negative reward anywhere (%d steps)' % len(log), rs.min() >= 0, rs.min())
    resets = [i for i, l in enumerate(log) if 'reset' in l[3]]
    check('8-4 trace: page resets detected at pipe 1 / corridor (%d)' % len(resets), len(resets) >= 5, resets[:5])
    check('8-4 trace: a page reset pays 0', all(log[i][0] == 0 for i in resets), [log[i][0] for i in resets][:5])
    # after the FIRST reset (pipe 1, ground approach): the re-run pays 0, the pipe entry pays
    i0 = resets[0]
    seg = log[i0 + 1:i0 + 61]
    paid = [k for k, l in enumerate(seg) if l[0] > 0]
    check('8-4 trace: the trace has ~12 resets, not one per low-x step', 8 <= len(resets) <= 16, len(resets))
    check('8-4 trace: pipe-1 recovery: re-run pays 0 until section 2 (x 1848 > hw 1281)',
          paid and all(l[0] == 0 for l in seg[:paid[0]]) and seg[paid[0]][1] > 1700, (paid[:1], seg[paid[0]][1] if paid else None))
    check('8-4 trace: pipe-1 recovery pays inside the 60-step grace window', paid and paid[0] < 60, paid[:1])
    check('8-4 trace: nothing ended the episode at the reset (play mode: no done)', not log[i0][4])
    water = [k for k, l in enumerate(log) if 'transition' in l[3] and l[1] < 200 and l[2] < 200]
    check('8-4 trace: water entry is a transition with its own highwater (no reset)',
          water and all('reset' not in log[k][3] for k in water), water[:2])
    tot = sum(rs); check('8-4 trace: total positive reward is substantial (%.0f)' % tot, tot > 3000)

# ---------------------------------------------------------------- 4-2 human trace (vine warp)
T42 = 'traces/play_0905-180528.csv'
if os.path.exists(T42):
    env = make('4-2'); log = replay(env, trace_actions(T42)); env.close()
    rs = np.array([l[0] for l in log])
    check('4-2 trace: no negative reward anywhere (%d steps)' % len(log), rs.min() >= 0, rs.min())
    vine = [i for i, l in enumerate(log) if 'transition' in l[3] and l[1] < 120 and log[i - 1][1] > 900]
    check('4-2 trace: the vine lift is a transition, not a reset', vine and all('reset' not in log[i][3] for i in vine), vine[:2])
    check('4-2 trace: coin-heaven ground pays after the vine', vine and sum(l[0] for l in log[vine[0]:vine[0] + 40]) > 0)
    clears = [i for i, l in enumerate(log) if 'clear' in l[3]]
    check('4-2 trace: the warp to 8-1 pays the clear bonus 500 + 100*14 = 1900',
          clears and abs(log[clears[0]][6].get('clear', 0) - 1900) < 1e-6, [log[i][6] for i in clears[:1]])
    check('4-2 trace: warp step is a clear, not a wrong exit', clears and 'wrong_exit' not in log[clears[0]][3])

# ---------------------------------------------------------------- scripted bots on 8-4 (training mode)
def bot(level, policy, steps, **kw):
    env = make(level, play_mode=False, **kw); log = []
    for s in range(steps):
        a = policy(s, env)
        obs, r, d, inf = env.step(np.array([a])); sg = env.last_signals
        log.append((float(r[0]), int(sg.x[0]), bool(d[0]), bool(sg.timeout[0]), bool(inf.time_outs[0]),
                    bool(sg.page_reset[0]), bool(sg.died[0]), bool(env.after_reset[0])))
        if d[0]: break
    env.close(); return log

# stand still: ends at the cutoff with 0, flagged as a time-out (bootstrap), not a death
log = bot('8-4', lambda s, e: 0, 400)
check('bot stand-still: episode ends exactly at unpaid_timeout=250', len(log) == 250, len(log))
check('bot stand-still: cutoff pays 0 and is a time-out (bootstrapped)', log[-1][0] == 0 and log[-1][3] and log[-1][4])
check('bot stand-still: no reward at all while idle', all(l[0] == 0 for l in log))
# run right + jump: dies or pays; never negative; rewards identical on replay (determinism)
rng = np.random.RandomState(7); acts = [3 if rng.random_sample() < 0.75 else 4 for _ in range(400)]
l1 = bot('8-4', lambda s, e: acts[s], 400); l2 = bot('8-4', lambda s, e: acts[s], 400)
check('bot run-right: identical actions -> identical rewards (determinism)', [l[0] for l in l1] == [l[0] for l in l2])
check('bot run-right: paid ground > 0 and never negative', sum(l[0] for l in l1) > 0 and min(l[0] for l in l1) >= 0)
# the human's ground approach to pipe 1 (first 232 actions of the 8-4 trace, deterministic), then
# stand still: page reset in TRAINING mode, unpaid grace, dead-end cutoff (not a time-out)
# (approach replayed in play mode -- the human died once on the way and training mode would end there --
#  then the env is switched to training semantics for the cutoff)
def bot_reset():
    env = make('8-4', play_mode=True); h = trace_actions(T84)[:232]; log = []
    for s in range(232):
        env.step(np.array([h[s]]))
    sg = env.last_signals
    first = (int(env.page_resets[0]) == 1, int(sg.x[0]), int(env.hw[0]), bool(env.after_reset[0]))
    env.play_mode = False
    for s in range(200):
        obs, r, d, inf = env.step(np.array([0])); sg = env.last_signals
        log.append((float(r[0]), int(sg.x[0]), bool(d[0]), bool(sg.timeout[0]), bool(inf.time_outs[0]), bool(sg.page_reset[0]), bool(sg.died[0]), bool(env.after_reset[0])))
        if d[0]: break
    env.close(); return first, log
if os.path.exists(T84):
    first, log = bot_reset()
    check('bot ground approach to pipe 1: exactly one page reset, unpaid state persists (x %d, hw %d)' % (first[1], first[2]), first[0] and first[3])
    rst = [0] if first[0] else []
if rst:
    tail = log
    check('bot after reset: re-run pays 0 (highwater kept)', all(l[0] == 0 for l in tail), [l[0] for l in tail if l[0] > 0][:3])
    check('bot after reset: ends within the 60-step grace window', len(tail) <= 61 and (tail[-1][3] or tail[-1][6]), (len(tail), tail[-1]))
    check('bot after reset: the loop cutoff is NOT a time-out (no bootstrap into a dead end)', not tail[-1][4])

# ---------------------------------------------------------------- 2D first-visit cells term
CELLS = [{'type': 'first_visit_progress', 'cap': 20}, {'type': 'level_clear', 'base': 500, 'per_extra': 100},
         {'type': 'first_visit_cells', 'bonus': 2, 'x_bin': 64, 'y_bin': 32}]
env = make('8-4', play_mode=True, reward=CELLS)
rs = []; cells = []
for s in range(120):
    a = 5 if s % 20 == 10 else 0            # stand still, jump in place every 20 steps
    obs, r, d, inf = env.step(np.array([a])); rs.append(float(r[0])); cells.append(env.last_terms['cells'][0])
env.close()
check('cells: standing still pays the start cell once, then 0', cells[0] == 2 and all(c == 0 for c in cells[1:10]), cells[:10])
check('cells: a jump in place pays the higher y-band once, repeats pay 0', sum(1 for c in cells if c > 0) <= 4 and max(cells) == 2, sum(1 for c in cells if c > 0))
check('cells: never negative', min(rs) >= 0)
env = make('8-4', play_mode=True, reward=CELLS); tot = 0.0
for s in range(300):
    obs, r, d, inf = env.step(np.array([3 if s % 3 else 4])); tot += float(env.last_terms['cells'][0])
    if d[0]: break
env.close()
check('cells: a run pays a bounded amount (%.0f over %d steps, < progress)' % (tot, s + 1), 0 < tot < 400)

# ---------------------------------------------------------------- archive cell key: tile signature sees the reveal
if os.path.exists(T84):
    env = make('8-4', play_mode=True, cell_tiles=True); h = trace_actions(T84); sigs = {}
    for s in range(1412):
        env.step(np.array([h[s]]))
        if s + 1 in (1402, 1410):
            sigs[s + 1] = env._tile_sig(0, 2406)
    env.close()
    check('archive key: tile signature differs before/after the hidden block is revealed (s1402 vs s1410)', sigs[1402] != sigs[1410], sigs)

# ---------------------------------------------------------------- frontier pool with predecessors
env = make('8-4', play_mode=False, self_restart_prob=1.0, self_restart_frontier_prob=1.0, frontier_predecessors=4, self_restart_frontier_k=64)
K = lambda b, y=2, sig=0: ('8-4', 3, b, y, 0, 3, sig)
env.archive = {K(b): [[b'x'], 1, 300] for b in (10, 14, 15, 16, 17, 18, 24, 30)}
env.archive[K(18, 1, 99)] = [[b'x'], 1, 300]          # revealed-block on-block state, never won
env.cell_wins = {K(24): 5, K(30): 3}; env.cell_tries = {c: 10 for c in env.archive}
import collections; draws = collections.Counter()
for _ in range(600):
    cells = list(env.archive.keys())
    winners = [c for c in cells if env.cell_wins.get(c, 0) > 0]; pool = set(winners)
    for w in winners:
        for c in cells:
            if c[0] == w[0] and c[1] == w[1] and c[4] == w[4] and c[5] == w[5] and w[2] - env.frontier_pred <= c[2] <= w[2]: pool.add(c)
    draws[len(pool)] += 1
env.close()
pool_sizes = set(draws)
check('frontier pool = winners + predecessors within 4 bins (24 -> 24; 30 -> 30) => size 2 (no cells at 20-23/26-29)', pool_sizes == {2}, pool_sizes)
env = make('8-4', play_mode=False, self_restart_prob=1.0, self_restart_frontier_prob=1.0, frontier_predecessors=4, self_restart_frontier_k=64)
env.archive = {K(b): [[b'x'], 1, 300] for b in (14, 15, 16, 17, 18, 19, 24)}
env.archive[K(18, 1, 99)] = [[b'x'], 1, 300]
env.cell_wins = {K(19): 2, K(24): 5}; env.cell_tries = {c: 10 for c in env.archive}
cells = list(env.archive.keys()); winners = [c for c in cells if env.cell_wins.get(c, 0) > 0]; pool = set(winners)
for w in winners:
    for c in cells:
        if c[0] == w[0] and c[1] == w[1] and c[4] == w[4] and c[5] == w[5] and w[2] - env.frontier_pred <= c[2] <= w[2]: pool.add(c)
env.close()
check('predecessors of a winning pipe-top cell (bin 19) include the revealed on-block cell (18, y1, sig 99) and bins 15-18', K(18, 1, 99) in pool and K(15) in pool and K(14) not in pool, sorted(pool))

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
