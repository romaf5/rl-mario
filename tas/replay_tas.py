"""Replay a TAS input file (FCEUX .fm2) on the native SMB core and score it
with the training reward code, frame-exact.

Purpose: a reward audit along a known-optimal any% route. Every 4 frames
(one agent step) the env's own `_after_step` scores the RAM, so we see
exactly what the trained agent WOULD be paid for the speedrunner's path:
where the reward is positive, zero, or penalised (loops? off-route? pipes?).

Usage: venv_retro/bin/python tas/replay_tas.py [--fm2 FILE] [--offset K]
       [--max-frames N] [--csv out.csv]
"""
import argparse, sys, os, csv, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mario_native_vecenv import MarioNativeVecEnv

FM2 = os.path.join(os.path.dirname(__file__), 'happylee-supermariobros,warped.fm2')
# FM2 port string order: R L D U T(start) S(select) B A
BITS = {'R': 0x80, 'L': 0x40, 'D': 0x20, 'U': 0x10, 'T': 0x08, 'S': 0x04,
        'B': 0x02, 'A': 0x01}


def parse_fm2(path):
    frames, cmds = [], []
    for line in open(path, encoding='latin-1'):
        if not line.startswith('|'):
            continue
        parts = line.strip().split('|')
        cmds.append(int(parts[1] or 0))
        p0 = parts[2]
        b = 0
        for ch in p0:
            b |= BITS.get(ch, 0)
        frames.append(b)
    return np.array(frames, dtype=np.uint8), np.array(cmds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fm2', default=FM2)
    ap.add_argument('--offset', type=int, default=0,
                    help='idle frames inserted before the inputs (power-on alignment)')
    ap.add_argument('--drop', type=int, default=0,
                    help='leading movie frames to drop (alignment the other way)')
    ap.add_argument('--shift', type=int, default=0,
                    help='shift gameplay inputs (after frame 120) by k frames')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--csv', default='')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    inputs, cmds = parse_fm2(a.fm2)
    if a.drop:
        inputs = inputs[a.drop:]
    if a.shift:
        # shift only the gameplay inputs (after the Start press) by k frames,
        # keeping the Start frame -> isolates an input-latch phase difference
        # from the frame-rule phase at game start
        g = inputs[120:]
        if a.shift > 0:
            g = np.concatenate([np.zeros(a.shift, np.uint8), g[:-a.shift]])
        else:
            g = np.concatenate([g[-a.shift:], np.zeros(-a.shift, np.uint8)])
        inputs = np.concatenate([inputs[:120], g])
    if a.max_frames:
        inputs = inputs[:a.max_frames]
    # same reward semantics as the training configs (reward set v2)
    env = MarioNativeVecEnv('tas', 1, episode_life=False, full_game=True,
                            x_reward='highwater', loop_penalty=100,
                            fail_penalty=100, backtrack_penalty=0,
                            progress_reward=0, score_reward=0.1,
                            novelty_bonus=0, stage_bonus=500, idle_timeout=450,
                            idle_penalty=0, sticky_actions=0, explore_eps=0,
                            self_restart_prob=0, dense_infos=True, n_threads=1)
    # the core is at power-on after benv_create; do NOT load a savestate,
    # and never let the env reset/reload one (its zombie guard would fire
    # on the title screen and load 1-1, silently desyncing the movie)
    lib = env.lib
    env._reset_env = lambda i, first=False: None
    for _ in range(a.offset):
        lib.benv_frames(env.env, 0, 1, 0)

    rows, total, events = [], 0.0, []
    prev = dict(gp=-1, life=None, x=0)
    t0 = time.time()
    nsteps = len(inputs) // 4
    playing = False
    for s in range(nsteps):
        for k in range(4):
            lib.benv_frames(env.env, 0, 1, int(inputs[4 * s + k]))
        env._fetch_obs(0)
        if not playing:
            # title screen / demo: score nothing until gameplay starts
            r0 = env.ram[0]
            if r0[0x770] == 1 and r0[0x0E] == 8:
                playing = True
                env._post_reset_init([0], env.ram)
                events.append((a.offset + 4 * s + 4, 'GAME START', 0, 0))
            continue
        obs, r, d, infos = env._after_step()
        info = infos[0]
        r = float(r[0]); total += r
        frame = a.offset + 4 * s + 4
        gp = info['game_progress']; life = info['life']; x = info['x_pos']
        rows.append((frame, gp, x, r, total, info['looped'], life))
        if gp != prev['gp']:
            events.append((frame, 'LEVEL %d-%d' % (info['world'], info['stage']),
                           x, round(total)))
        if prev['life'] is not None and life < prev['life']:
            events.append((frame, 'DEATH', x, round(total)))
        if info['looped']:
            events.append((frame, 'LOOP %+.0f' % r, x, round(total)))
        if info.get('offroute'):
            events.append((frame, 'OFF-ROUTE', x, round(total)))
        if info['victory']:
            events.append((frame, 'VICTORY', x, round(total)))
            break
        prev = dict(gp=gp, life=life, x=x)
    dt = time.time() - t0
    if not a.quiet:
        print('frames %d  steps %d  (%.1fs)  total reward %.0f' % (
            len(inputs), nsteps, dt, total))
        print('events (frame, what, x, cumulative R):')
        for e in events:
            print('  ', e)
        neg = [(f, x, r) for f, gp, x, r, tot, lp, lf in rows if r < -20]
        print('steps with reward < -20:', len(neg), neg[:10])
        zero = sum(1 for row in rows if row[3] == 0)
        print('zero-reward steps: %d / %d' % (zero, len(rows)))
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'gp', 'x', 'reward', 'total', 'looped', 'life'])
            w.writerows(rows)
    env.close()
    return events


if __name__ == '__main__':
    main()
