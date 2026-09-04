#!/usr/bin/env python3
"""Play Mario yourself on the TRAINING env and watch the rewards.

Exactly the env the agent trains in (same step = 4 frames, same reward
terms, same terminals), minus exploration noise and archive restarts.
Every step shows the per-term reward breakdown; you can rewind, and a CSV
trace of everything is written for later analysis.

    venv_retro/bin/python tools/play.py --config configs/mario_ppo_native_84.yaml --level 8-4

Keys
  arrows      move / down = pipe / up        Z = A (jump)   X = B (run)
  R           rewind one step (hold to keep rewinding)   Shift+R rewinds 15
  Space       pause / resume        N   single step while paused
  [ ]         slower / faster       B   bookmark here   M  jump to bookmark
  Esc / Q     quit (trace is flushed on every step)
"""
import argparse
import copy
import csv
import ctypes
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mario_native_vecenv import MarioNativeVecEnv  # noqa: E402

ACT = ['NOOP', 'R', 'R+A', 'R+B', 'R+A+B', 'A', 'L', 'L+A', 'L+B', 'L+A+B',
       'DOWN', 'UP']


def keys_to_action(k, pg):
    right, left = k[pg.K_RIGHT], k[pg.K_LEFT]
    down, up = k[pg.K_DOWN], k[pg.K_UP]
    a, b = k[pg.K_z], k[pg.K_x]
    if down:
        return 10
    if up and not (right or left):
        return 11
    if right:
        return 4 if (a and b) else 2 if a else 3 if b else 1
    if left:
        return 9 if (a and b) else 7 if a else 8 if b else 6
    return 5 if a else 0


class Snap:
    """Full env snapshot (emulator state + python trackers + reward state)."""

    def __init__(self, env):
        env.lib.benv_save(env.env, 0, env._sbuf)
        self.state = bytes(env._sbuf.raw)
        n = env.num_actors
        self.arrays = {k: v.copy() for k, v in vars(env).items()
                       if isinstance(v, np.ndarray) and v.ndim >= 1
                       and v.shape[0] == n}
        self.lists = {k: copy.deepcopy(v) for k, v in vars(env).items()
                      if isinstance(v, list) and len(v) == n}
        self.scalars = {k: v for k, v in vars(env).items()
                        if k in ('_ptr',)}
        self.terms = env.rewards.state()

    def restore(self, env):
        env.lib.benv_load(env.env, 0, self.state)
        for k, v in self.arrays.items():
            setattr(env, k, v.copy())
        for k, v in self.lists.items():
            setattr(env, k, copy.deepcopy(v))
        for k, v in self.scalars.items():
            setattr(env, k, v)
        env.rewards.restore(self.terms)
        env._fetch_obs(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mario_ppo_native_84.yaml')
    ap.add_argument('--level', default=None, help='start level, e.g. 8-4')
    ap.add_argument('--scale', type=int, default=3)
    ap.add_argument('--fps', type=int, default=15, help='agent steps per second (15 = real time)')
    ap.add_argument('--trace', default=None, help='CSV path (default: traces/play_<time>.csv)')
    ap.add_argument('--history', type=int, default=4000, help='rewind depth (steps)')
    ap.add_argument('--steps', type=int, default=0, help='auto-quit after N steps (testing)')
    args = ap.parse_args()

    import pygame as pg
    cfg = yaml.safe_load(open(args.config))['params']['config']
    ec = dict(cfg['env_config'])
    for k in ('name', 'action_type', 'archive_path'):
        ec.pop(k, None)
    ec.update(sticky_actions=0.0, explore_eps=0.0, self_restart_prob=0.0,
              explore_episode_prob=0.0, reset_noops=0, n_threads=1,
              dense_infos=True, episode_life=True)
    if args.level:
        ec['random_stages'] = [args.level]
    env = MarioNativeVecEnv('play', 1, **ec)
    env.reset()
    lib = env.lib
    rgb = ctypes.create_string_buffer(224 * 240 * 3)

    os.makedirs('traces', exist_ok=True)
    trace_path = args.trace or os.path.join(
        'traces', 'play_%s.csv' % time.strftime('%m%d-%H%M%S'))
    term_names = None
    tf = open(trace_path, 'w', newline='')
    tw = csv.writer(tf)

    pg.init()
    W, H = 240 * args.scale, 224 * args.scale
    panel = 420
    screen = pg.display.set_mode((W + panel, H))
    pg.display.set_caption('Mario reward inspector')
    font = pg.font.SysFont('monospace', 15)
    big = pg.font.SysFont('monospace', 18, bold=True)
    clock = pg.time.Clock()

    hist = []                      # snapshots BEFORE each step
    hist.append(Snap(env))
    cum = {}
    total = 0.0
    step = 0
    events = []
    paused = False
    fps = args.fps
    bookmark = None
    last = dict(action=0, r=0.0, terms={}, info={}, sig=None, done=False)
    prev_life = None

    def do_step(action):
        nonlocal total, step, prev_life, term_names
        hist.append(Snap(env))
        if len(hist) > args.history:
            hist.pop(0)
        obs, r, d, infos = env.step(np.array([action]))
        info = infos[0]
        terms = {k: float(v[0]) for k, v in env.last_terms.items()}
        sig = env.last_signals
        r = float(r[0])
        total += r
        for k, v in terms.items():
            cum[k] = cum.get(k, 0.0) + v
        step += 1
        ev = []
        if sig is not None:
            if sig.loop[0]:
                ev.append('LOOP')
            if sig.died[0]:
                ev.append('DEATH')
            if sig.idle_to[0]:
                ev.append('IDLE TIMEOUT')
            if sig.off[0]:
                ev.append('OFF-ROUTE')
            if sig.level_up[0] or sig.newflag[0]:
                ev.append('LEVEL CLEAR')
            if sig.victory_new[0]:
                ev.append('VICTORY')
            if sig.legit[0]:
                ev.append('transition')
        if d[0]:
            ev.append('done')
        if ev:
            events.append('%5d %s %+.0f' % (step, ' '.join(ev), r))
        if term_names is None:
            term_names = list(terms)
            tw.writerow(['step', 'action', 'x', 'ypix', 'life', 'level',
                         'area', 'swim', 'hw', 'reward'] + term_names
                        + ['flag_loop', 'flag_legit', 'flag_died', 'flag_idle_to',
                           'flag_off', 'flag_level_up', 'done', 'events'])
        ram = env.ram[0]
        tw.writerow([step, ACT[action], info['x_pos'], int(ram[0x3B8]),
                     info['life'], '%d-%d' % (info['world'], info['stage']),
                     int(ram[0x760]), int(ram[0x704]), int(env.hw[0]),
                     round(r, 3)] + [round(terms.get(k, 0.0), 3) for k in term_names]
                    + [int(sig.loop[0]), int(sig.legit[0]), int(sig.died[0]),
                       int(sig.idle_to[0]), int(sig.off[0]), int(sig.level_up[0]),
                       int(d[0]), ' '.join(ev)])
        tf.flush()
        last.update(action=action, r=r, terms=terms, info=info, sig=sig, done=bool(d[0]))

    def rewind(k):
        nonlocal total, step
        for _ in range(k):
            if len(hist) <= 1:
                break
            snap = hist.pop()
            snap.restore(env)
            step = max(0, step - 1)
        # rewards already counted stay in the cumulative (trace keeps truth)
        events.append('%5d rewind' % step)

    running = True
    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
            elif e.type == pg.KEYDOWN:
                if e.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif e.key == pg.K_SPACE:
                    paused = not paused
                elif e.key == pg.K_n and paused:
                    do_step(keys_to_action(pg.key.get_pressed(), pg))
                elif e.key == pg.K_r:
                    rewind(15 if (e.mod & pg.KMOD_SHIFT) else 1)
                elif e.key == pg.K_LEFTBRACKET:
                    fps = max(2, fps - 3)
                elif e.key == pg.K_RIGHTBRACKET:
                    fps = min(60, fps + 3)
                elif e.key == pg.K_b:
                    bookmark = Snap(env); events.append('%5d bookmark' % step)
                elif e.key == pg.K_m and bookmark is not None:
                    bookmark.restore(env); events.append('%5d -> bookmark' % step)
        keys = pg.key.get_pressed()
        if keys[pg.K_r] and pg.key.get_mods() & pg.KMOD_SHIFT:
            rewind(1)
        elif not paused:
            do_step(keys_to_action(keys, pg))
        if args.steps and step >= args.steps:
            running = False

        # ---- draw ----
        lib.benv_render_rgb(env.env, 0, rgb)
        frame = np.frombuffer(rgb, dtype=np.uint8).reshape(224, 240, 3)
        surf = pg.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pg.transform.scale(surf, (W, H))
        screen.fill((16, 16, 16))
        screen.blit(surf, (0, 0))
        ram = env.ram[0]
        info = last['info']
        lines = [
            ('step %d   %s   %s' % (step, ACT[last['action']], 'PAUSED' if paused else '%d st/s' % fps), (255, 255, 255)),
            ('x=%d y=%d  hw=%d  life=%s  lvl=%s  t=%s' % (
                int(ram[0x6D]) * 256 + int(ram[0x86]), int(ram[0x3B8]), int(env.hw[0]),
                info.get('life', '?'), '%s-%s' % (info.get('world', '?'), info.get('stage', '?')),
                info.get('time', '?')), (200, 200, 200)),
            ('', None),
            ('reward this step  %+8.2f' % last['r'], (255, 255, 0)),
        ]
        for k, v in last['terms'].items():
            if k.startswith('base/'):
                lines.append(('    %-14s %+8.2f' % (k[5:], v), (170, 170, 170)))
            else:
                lines.append(('  %-16s %+8.2f' % (k, v), (255, 255, 255) if v else (120, 120, 120)))
        lines.append(('', None))
        lines.append(('cumulative  %+9.1f' % total, (255, 255, 0)))
        for k, v in cum.items():
            if not k.startswith('base/'):
                lines.append(('  %-16s %+9.1f' % (k, v), (200, 200, 200)))
        lines.append(('', None))
        lines.append(('events', (255, 160, 80)))
        for evl in events[-8:]:
            lines.append(('  ' + evl, (255, 160, 80)))
        lines.append(('', None))
        lines.append(('R rewind  Shift+R x15  Space pause  N step', (110, 110, 110)))
        lines.append(('[ ] speed  B/M bookmark  Q quit', (110, 110, 110)))
        y = 8
        for text, col in lines:
            if col is not None:
                screen.blit((big if text.startswith(('reward', 'cumul')) else font).render(text, True, col), (W + 10, y))
            y += 17
        pg.display.flip()
        clock.tick(fps)

    tf.close()
    env.close()
    print('trace written to', trace_path)


if __name__ == '__main__':
    main()
