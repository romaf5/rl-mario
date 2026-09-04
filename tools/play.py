#!/usr/bin/env python3
"""Reward inspector: play the TRAINING env yourself and see every reward.

Exactly the env the agent trains in (4-frame steps, same reward terms and
signals), without exploration noise or archive restarts. Loops, off-route
entries and idle timeouts are flagged and paid but never reset the game
(play mode), so you can watch what happens next.

    venv_retro/bin/python tools/play.py --config configs/mario_ppo_native_84.yaml --level 8-4

Controls
  W A S D        move (S = down / enter pipe, W = up)     arrows also work
  J              A (jump)          K   B (run)
  Space          play / pause
  R or , (hold)  rewind, one step per tick while held
  . (hold)       step forward, replaying your recorded actions after a rewind
  [  ]           slower / faster    -  =   timeline zoom
  1..9           plot that reward term in the timeline (0 = total)
  B  /  M        set / jump to bookmark
  Q / Esc        quit

Layout: game view (left), reward terms (right), timeline (bottom): reward
per step with events marked, cumulative reward, x position + highwater.
Everything is also written to traces/play_<time>.csv for offline analysis.
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
EVENT_COLORS = {'LOOP': (255, 70, 70), 'DEATH': (255, 90, 200),
                'IDLE': (255, 170, 60), 'OFF-ROUTE': (255, 120, 40),
                'CLEAR': (90, 255, 120), 'VICTORY': (90, 255, 120),
                'transition': (90, 200, 255), 'GAME OVER': (255, 60, 60)}
BG, PANEL, GRID = (14, 14, 18), (24, 24, 30), (48, 48, 58)
TEXT, DIM = (230, 230, 230), (120, 120, 130)


def keys_to_action(k, pg):
    right, left = k[pg.K_d] or k[pg.K_RIGHT], k[pg.K_a] or k[pg.K_LEFT]
    down, up = k[pg.K_s] or k[pg.K_DOWN], k[pg.K_w] or k[pg.K_UP]
    a, b = k[pg.K_j], k[pg.K_k]
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
    """Full env snapshot: emulator state + python trackers + reward state."""

    def __init__(self, env):
        env.lib.benv_save(env.env, 0, env._sbuf)
        self.state = bytes(env._sbuf.raw)
        n = env.num_actors
        self.arrays = {k: v.copy() for k, v in vars(env).items()
                       if isinstance(v, np.ndarray) and v.ndim >= 1
                       and v.shape[0] == n}
        self.lists = {k: copy.deepcopy(v) for k, v in vars(env).items()
                      if isinstance(v, list) and len(v) == n}
        self.scalars = {k: v for k, v in vars(env).items() if k in ('_ptr',)}
        # the env RNG picks the next level on a reset: without it a rewind
        # replays a DIFFERENT game after the next death
        self.rngst = env.rng.get_state()
        self.terms = env.rewards.state()

    def restore(self, env):
        env.lib.benv_load(env.env, 0, self.state)
        for k, v in self.arrays.items():
            setattr(env, k, v.copy())
        for k, v in self.lists.items():
            setattr(env, k, copy.deepcopy(v))
        for k, v in self.scalars.items():
            setattr(env, k, v)
        env.rng.set_state(self.rngst)
        env.rewards.restore(self.terms)
        env._fetch_obs(0)


class Inspector:
    def __init__(self, args):
        import pygame as pg
        self.pg = pg
        self.args = args
        cfg = yaml.safe_load(open(args.config))['params']['config']
        ec = dict(cfg['env_config'])
        for k in ('name', 'action_type', 'archive_path'):
            ec.pop(k, None)
        ec.update(sticky_actions=0.0, explore_eps=0.0, self_restart_prob=0.0,
                  explore_episode_prob=0.0, reset_noops=0, n_threads=1,
                  dense_infos=True, episode_life=True, play_mode=True,
                  idle_timeout=10 ** 9)
        # --level changes only the START level; the config's level list stays
        # the on-route set, so a warp out of 1-2 is still on-route
        if args.level:
            ec.setdefault('route_levels', list(ec.get('random_stages') or []))
            ec['random_stages'] = [args.level]
        self.env = MarioNativeVecEnv('play', 1, **ec)
        self.env.reset()
        self.rgb = ctypes.create_string_buffer(224 * 240 * 3)

        os.makedirs('traces', exist_ok=True)
        self.trace_path = args.trace or os.path.join(
            'traces', 'play_%s.csv' % time.strftime('%m%d-%H%M%S'))
        self.tf = open(self.trace_path, 'w', newline='')
        self.tw = csv.writer(self.tf)
        self.term_names = None

        pg.init()
        self.scale = args.scale
        self.GW, self.GH = 240 * self.scale, 224 * self.scale
        self.PW, self.TH = 430, 300
        self.W, self.H = self.GW + self.PW, self.GH + self.TH
        self.screen = pg.display.set_mode((self.W, self.H))
        pg.display.set_caption('Mario reward inspector')
        self.font = pg.font.SysFont('monospace', 14)
        self.small = pg.font.SysFont('monospace', 12)
        self.big = pg.font.SysFont('monospace', 20, bold=True)
        self.clock = pg.time.Clock()

        self.records = []          # one per executed step on the current timeline
        self.future = []           # records popped by a rewind (redo-able, drawn dimmed)
        self.snaps = [Snap(self.env)]
        self.total = 0.0
        self.cum = {}
        self.paused = False
        self.fps = args.fps
        self.view = 400
        self.plot_term = 0         # 0 = total, else 1-based index into the term list
        self.bookmark = None
        self.mode = 'playing'
        self.last = None

    # ------------------------------------------------------------ stepping
    def step(self, action):
        env = self.env
        self.snaps.append(Snap(env))
        if len(self.snaps) > self.args.history:
            self.snaps.pop(0)
        obs, r, d, infos = env.step(np.array([action]))
        info = infos[0]
        sig = env.last_signals
        terms = {k: float(v[0]) for k, v in env.last_terms.items()}
        r = float(r[0])
        ev = []
        if sig.loop[0]:
            ev.append('LOOP')
        if sig.died[0]:
            ev.append('GAME OVER' if sig.game_over[0] else 'DEATH')
        if sig.idle_to[0]:
            ev.append('IDLE')
        if sig.off[0]:
            ev.append('OFF-ROUTE')
        if sig.level_up[0] or sig.newflag[0]:
            ev.append('CLEAR')
        if sig.victory_new[0]:
            ev.append('VICTORY')
        if sig.legit[0]:
            ev.append('transition')
        ram = env.ram[0]
        rec = dict(step=len(self.records) + 1, action=action, x=info['x_pos'],
                   y=int(ram[0x3B8]), hw=int(env.hw[0]), life=info['life'],
                   level='%d-%d' % (info['world'], info['stage']),
                   area=int(ram[0x760]), swim=int(ram[0x704]), r=r, terms=terms,
                   events=ev, done=bool(d[0]), t=info['time'])
        self.records.append(rec)
        self.total += r
        for k, v in terms.items():
            self.cum[k] = self.cum.get(k, 0.0) + v
        self.last = rec
        self.write_trace(rec)

    def write_trace(self, rec, note=''):
        if self.term_names is None:
            self.term_names = list(rec['terms'])
            self.tw.writerow(['step', 'action', 'x', 'ypix', 'life', 'level', 'area',
                              'swim', 'hw', 'time', 'reward'] + self.term_names
                             + ['events', 'done', 'note'])
        self.tw.writerow([rec['step'], ACT[rec['action']], rec['x'], rec['y'], rec['life'],
                          rec['level'], rec['area'], rec['swim'], rec['hw'], rec['t'],
                          round(rec['r'], 3)]
                         + [round(rec['terms'].get(k, 0.0), 3) for k in self.term_names]
                         + [' '.join(rec['events']), int(rec['done']), note])
        self.tf.flush()

    def rewind(self, k=1):
        for _ in range(k):
            if len(self.snaps) <= 1 or not self.records:
                break
            self.snaps.pop().restore(self.env)
            rec = self.records.pop()
            self.future.insert(0, rec)
            self.total -= rec['r']
            for kk, v in rec['terms'].items():
                self.cum[kk] = self.cum.get(kk, 0.0) - v
        self.last = self.records[-1] if self.records else None
        self.mode = 'rewinding'
        if self.records:
            self.write_trace(self.records[-1], note='rewound to here')

    def redo(self):
        if self.future:
            rec = self.future.pop(0)
            self.step(rec['action'])
        else:
            self.step(0)

    # ------------------------------------------------------------- drawing
    def text(self, txt, x, y, col=TEXT, font=None):
        self.screen.blit((font or self.font).render(txt, True, col), (x, y))

    def draw(self, cur_action):
        pg = self.pg
        self.screen.fill(BG)
        self.env.lib.benv_render_rgb(self.env.env, 0, self.rgb)
        frame = np.frombuffer(self.rgb, dtype=np.uint8).reshape(224, 240, 3)
        surf = pg.transform.scale(
            pg.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (self.GW, self.GH))
        self.screen.blit(surf, (0, 0))
        self.draw_panel(cur_action)
        self.draw_timeline()
        pg.display.flip()

    def draw_panel(self, cur_action):
        pg = self.pg
        pg.draw.rect(self.screen, PANEL, (self.GW, 0, self.PW, self.GH))
        x0, y = self.GW + 12, 10
        rec = self.last
        mode_col = {'playing': (120, 255, 120), 'paused': (255, 220, 90),
                    'rewinding': (255, 120, 120)}[self.mode]
        label = {'playing': '>  PLAYING', 'paused': '||  PAUSED',
                 'rewinding': '<<  REWIND'}[self.mode]
        self.text('%s   %d steps/s' % (label, self.fps), x0, y, mode_col, self.big); y += 30
        ahead = '   (+%d ahead: . to redo)' % len(self.future) if self.future else ''
        self.text('step %d%s' % (len(self.records), ahead), x0, y, DIM); y += 20
        ram = self.env.ram[0]
        self.text('x %4d   y %3d   hw %4d' % (int(ram[0x6D]) * 256 + int(ram[0x86]),
                                              int(ram[0x3B8]), int(self.env.hw[0])), x0, y); y += 18
        if rec:
            self.text('level %s   lives %s   time %s   input %s'
                      % (rec['level'], rec['life'], rec['t'], ACT[cur_action]), x0, y, DIM)
        y += 26
        self.text('reward this step', x0, y, DIM)
        if rec:
            self.text('%+8.2f' % rec['r'], x0 + 200, y, (255, 255, 120), self.big)
        y += 28
        if rec:
            for k, v in rec['terms'].items():
                sub = k.startswith('base/')
                name = k[5:] if sub else k
                col = (150, 150, 160) if sub else (TEXT if v else DIM)
                self.text(('    ' if sub else '  ') + '%-11s' % name, x0, y, col)
                self.text('%+8.2f' % v, x0 + 150, y, col)
                lim = 20.0 if (sub or k == 'base') else 100.0
                bx, bw = x0 + 240, 160
                pg.draw.line(self.screen, GRID, (bx + bw // 2, y + 2), (bx + bw // 2, y + 14))
                if v:
                    w = int(min(abs(v), lim) / lim * (bw // 2))
                    if v > 0:
                        pg.draw.rect(self.screen, (90, 200, 90), (bx + bw // 2, y + 3, w, 11))
                    else:
                        pg.draw.rect(self.screen, (220, 80, 80), (bx + bw // 2 - w, y + 3, w, 11))
                y += 17
        y += 10
        self.text('cumulative', x0, y, DIM)
        self.text('%+9.1f' % self.total, x0 + 200, y, (255, 255, 120), self.big); y += 26
        for k, v in self.cum.items():
            if not k.startswith('base/'):
                self.text('  %-11s %+9.1f' % (k, v), x0, y, (190, 190, 200)); y += 16
        y += 10
        self.text('events (latest)', x0, y, DIM); y += 18
        evs = [(r['step'], e, r['r']) for r in self.records[-300:] for e in r['events']][-7:]
        for st, e, r in evs:
            self.text('  %5d  %-11s %+.0f' % (st, e, r), x0, y, EVENT_COLORS.get(e, TEXT)); y += 16
        hy = self.GH - 60
        for line in ('WASD move   J jump   K run   Space play/pause',
                     'hold R or , : rewind   hold . : forward   B/M bookmark',
                     '[ ] speed   - = zoom   1-9 term / 0 total   Q quit'):
            self.text(line, x0, hy, DIM, self.small); hy += 15

    def draw_timeline(self):
        pg = self.pg
        top = self.GH
        pg.draw.rect(self.screen, PANEL, (0, top, self.W, self.TH))
        L, R = 60, self.W - 12
        recs, fut, view = self.records, self.future, self.view
        n_now = len(recs)
        start = max(0, n_now - view + min(len(fut), view // 4))
        shown = (recs[start:] + fut)[:view]
        if not shown:
            self.text('timeline: play to fill it', L, top + 12, DIM)
            return
        px = (R - L) / float(view)
        cur_i = n_now - start
        names = ['total'] + (self.term_names or [])
        sel = names[self.plot_term] if self.plot_term < len(names) else 'total'
        vals = ([r['r'] for r in shown] if sel == 'total'
                else [r['terms'].get(sel, 0.0) for r in shown])

        # strip 1: reward per step (bars) + events
        y0, h = top + 8, 100
        self.text('reward / step   [%s]' % sel, L, y0 - 2, DIM, self.small)
        lim = max(1.0, min(200.0, max(abs(v) for v in vals)))
        mid = y0 + 12 + (h - 12) // 2
        pg.draw.line(self.screen, GRID, (L, mid), (R, mid))
        self.text('%+.0f' % lim, 4, y0 + 10, DIM, self.small)
        self.text('%+.0f' % -lim, 4, y0 + h - 10, DIM, self.small)
        self.text('(sqrt scale)', 4, y0 + h // 2 - 4, GRID, self.small)
        for i, v in enumerate(vals):
            x = int(L + i * px)
            # sqrt scale: +7 running steps stay visible next to -100 penalties
            hh = int((min(abs(v), lim) / lim) ** 0.5 * ((h - 12) // 2))
            col = (90, 200, 90) if v >= 0 else (220, 80, 80)
            if i >= cur_i:
                col = tuple(c // 3 for c in col)
            if hh:
                pg.draw.rect(self.screen, col,
                             (x, mid - hh if v >= 0 else mid, max(1, int(px)), hh))
        for i, r in enumerate(shown):
            if r['events']:
                e = [e for e in r['events'] if e != 'transition'] or r['events']
                col = EVENT_COLORS.get(e[0], TEXT)
                if i >= cur_i:
                    col = tuple(c // 2 for c in col)
                x = int(L + i * px)
                pg.draw.line(self.screen, col, (x, y0 + 10), (x, top + self.TH - 16), 1)
                self.text(e[0], min(x + 2, R - 70), y0 + 10 + (i % 3) * 11, col, self.small)

        # strip 2: cumulative reward
        y0, h = top + 116, 80
        self.text('cumulative reward', L, y0 - 2, DIM, self.small)
        acc = sum(r['r'] for r in recs[:start])
        cum = []
        for r in shown:
            acc += r['r']; cum.append(acc)
        lo, hi = min(cum), max(cum)
        hi = hi if hi - lo >= 1 else lo + 1
        pts = [(int(L + i * px), int(y0 + h - (c - lo) / (hi - lo) * (h - 14)))
               for i, c in enumerate(cum)]
        if cur_i > 1:
            pg.draw.lines(self.screen, (255, 255, 120), False, pts[:cur_i], 2)
        if len(pts) - cur_i > 1:
            pg.draw.lines(self.screen, (90, 90, 60), False, pts[max(cur_i - 1, 0):], 1)
        self.text('%.0f' % hi, 4, y0 + 8, DIM, self.small)
        self.text('%.0f' % lo, 4, y0 + h - 12, DIM, self.small)

        # strip 3: x position + highwater
        y0, h = top + 204, 84
        self.text('x position (white)   highwater (yellow)', L, y0 - 2, DIM, self.small)
        xs = [r['x'] for r in shown]; hws = [r['hw'] for r in shown]
        lo, hi = min(xs + hws), max(xs + hws)
        hi = hi if hi - lo >= 50 else lo + 50

        def yy(v):
            return int(y0 + h - (v - lo) / float(hi - lo) * (h - 14))
        pts_x = [(int(L + i * px), yy(v)) for i, v in enumerate(xs)]
        pts_h = [(int(L + i * px), yy(v)) for i, v in enumerate(hws)]
        if cur_i > 1:
            pg.draw.lines(self.screen, (255, 230, 90), False, pts_h[:cur_i], 1)
            pg.draw.lines(self.screen, TEXT, False, pts_x[:cur_i], 2)
        if len(pts_x) - cur_i > 1:
            pg.draw.lines(self.screen, (70, 70, 80), False, pts_x[max(cur_i - 1, 0):], 1)
        self.text('%d' % hi, 4, y0 + 8, DIM, self.small)
        self.text('%d' % lo, 4, y0 + h - 12, DIM, self.small)

        # cursor + axis
        cx = int(L + max(cur_i - 1, 0) * px)
        pg.draw.line(self.screen, (255, 255, 255), (cx, top + 8), (cx, top + self.TH - 16), 1)
        self.text('step %d' % n_now, min(cx + 3, R - 70), top + self.TH - 14, TEXT, self.small)
        self.text('%d' % (start + 1), L, top + self.TH - 14, DIM, self.small)
        self.text('view %d steps' % view, R - 110, top + self.TH - 14, DIM, self.small)

    # ---------------------------------------------------------------- loop
    def run(self):
        pg = self.pg
        running = True
        while running:
            keys = pg.key.get_pressed()
            cur_action = keys_to_action(keys, pg)
            if self.args.demo:
                cur_action = 3 if (len(self.records) % 7) else 4
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running = False
                elif e.type == pg.KEYDOWN:
                    if e.key in (pg.K_ESCAPE, pg.K_q):
                        running = False
                    elif e.key == pg.K_SPACE:
                        self.paused = not self.paused
                        self.mode = 'paused' if self.paused else 'playing'
                    elif e.key == pg.K_LEFTBRACKET:
                        self.fps = max(2, self.fps - 3)
                    elif e.key == pg.K_RIGHTBRACKET:
                        self.fps = min(60, self.fps + 3)
                    elif e.key == pg.K_MINUS:
                        self.view = min(3000, int(self.view * 1.5))
                    elif e.key == pg.K_EQUALS:
                        self.view = max(60, int(self.view / 1.5))
                    elif e.key == pg.K_b:
                        self.bookmark = (Snap(self.env), list(self.records), dict(self.cum), self.total)
                    elif e.key == pg.K_m and self.bookmark is not None:
                        snap, recs, cum, total = self.bookmark
                        snap.restore(self.env)
                        self.records, self.cum, self.total = list(recs), dict(cum), total
                        self.future, self.snaps = [], [Snap(self.env)]
                        self.last = self.records[-1] if self.records else None
                        self.paused, self.mode = True, 'paused'
                    elif pg.K_0 <= e.key <= pg.K_9:
                        self.plot_term = e.key - pg.K_0
            if keys[pg.K_r] or keys[pg.K_COMMA]:
                self.paused = True
                self.rewind(1)
            elif keys[pg.K_PERIOD]:
                self.paused = True
                self.redo()
                self.mode = 'paused'
            elif not self.paused:
                if self.future:
                    self.future = []          # new input branches off the old future
                self.mode = 'playing'
                self.step(cur_action)
            self.draw(cur_action)
            if self.args.steps and len(self.records) >= self.args.steps:
                if self.args.demo:
                    self.rewind(60)
                    self.draw(cur_action)
                if self.args.screenshot:
                    pg.image.save(self.screen, self.args.screenshot)
                running = False
            self.clock.tick(self.fps)
        self.tf.close()
        self.env.close()
        print('trace written to', self.trace_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/mario_ppo_native_84.yaml')
    ap.add_argument('--level', default=None, help='start level, e.g. 8-4')
    ap.add_argument('--scale', type=int, default=3)
    ap.add_argument('--fps', type=int, default=15, help='steps per second (15 = real time)')
    ap.add_argument('--trace', default=None)
    ap.add_argument('--history', type=int, default=6000, help='rewind depth (steps)')
    ap.add_argument('--steps', type=int, default=0, help='auto-quit after N steps (testing)')
    ap.add_argument('--screenshot', default=None, help='save the final frame (testing)')
    ap.add_argument('--demo', action='store_true', help='scripted run/jump inputs, rewind 60 at the end (testing)')
    Inspector(ap.parse_args()).run()


if __name__ == '__main__':
    main()
