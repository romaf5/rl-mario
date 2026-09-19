"""Ghost replay: many recorded episodes rendered at once.

Every eval trace is (start state, actions); the emulator is deterministic, so
each trace replays in its own emulator in lockstep. The best run (largest
max x, or --main N) is drawn as the scene; every other run contributes only
Mario's sprite, cut from its own frame and pasted at its level x relative to
the main camera with lowered opacity. Ghosts vanish a moment after their
episode ends.

  python tools/ghosts.py --traces 'runs/<run>/eval_traces/epoch_2000/ep_*.npz' --out ghosts.gif
  python tools/ghosts.py --traces 'runs/<run>/eval_traces/epoch_*/video_8-4_*.npz' --out progress.mp4 --alpha 0.5

Trace sources (both replayable here and in tools/play.py --replay):
  <run>/eval_traces/epoch_N/ep_XX_<end>_x<max>.npz   clean door eval episodes
  <run>/eval_traces/epoch_N/video_<level>_NN.npz     the visualized clips

A trace replays exactly only in an env with the RECORDING's settings: the
run's config (--config: reward set, unpaid cutoff, page-reset rules, route)
under the evaluation overrides, plus the trace's own stepping (raw) and life
handling. Door eval episodes are single-life (episode_life on); clips play all
lives (episode_life off), where the eval ends a stuck life by zeroing the game
timer -- the same env rule fires again in the replay, so a clip with a forced
time-up replays exactly (with the defaults -- unpaid cutoff 250 instead of the
run's 500 -- the replay forced time-ups the recording never had).

  python tools/ghosts.py --check --traces 'runs/<run>/eval_traces/epoch_*/*.npz'
replays every trace and compares its x / lives per step with the recording
(the npz's x for door episodes, the clip's .csv) instead of rendering.
"""
import argparse, csv, ctypes, glob, os, re, sys
import numpy as np
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mario_native_vecenv import MarioNativeVecEnv

DEFAULT_CONFIG = os.path.join(ROOT, 'configs', 'mario_ppo_native_42.yaml')


def trace_env_kwargs(env_config, level, episode_life, unpaid_timeout=None):
    """Env kwargs of the evaluation env that recorded a trace: the run's
    env_config with the observer's eval overrides (no noise, no restarts or
    explorers, full game, no end_on_stage_exit, the config's route)."""
    ec = dict(env_config or {})
    route = list(ec.get('route_levels') or ec.get('random_stages') or [])
    for k in ('name', 'action_type', 'archive_path', 'video_levels',
              'video_level_steps', 'backend'):
        ec.pop(k, None)
    ec.update(sticky_actions=0.0, explore_eps=0.0, self_restart_prob=0.0,
              explore_episode_prob=0.0, reset_noops=0, n_threads=1,
              dense_infos=True, route_levels=route or None, full_game=True,
              explorer_envs=0, end_on_stage_exit=False,
              random_stages=[level] if level else None,
              episode_life=bool(episode_life))
    if unpaid_timeout is not None:
        ec['unpaid_timeout'] = int(unpaid_timeout)
    return ec


def load_trace(path):
    """Trace + the recording settings it implies. Newer traces store
    episode_life / unpaid_timeout; older ones: a door eval episode
    (ep_*.npz, with per-step x) was single-life, every clip multi-life."""
    z = np.load(path, allow_pickle=True)
    base = os.path.basename(path)[:-4]
    raw = int(z['raw']) if 'raw' in z else int('video' in base)
    door = 'x' in z.files or base.startswith('ep_')
    m = re.match(r'(?:ep|video)_(\d-\d)_', base)
    level = str(z['level']) if 'level' in z.files else (m.group(1) if m else None)
    return dict(name=base, path=path, state=bytes(np.asarray(z['state'], dtype=np.uint8)),
                actions=[int(a) for a in z['actions']], raw=raw, level=level,
                episode_life=bool(int(z['episode_life'])) if 'episode_life' in z.files else door,
                unpaid_timeout=int(z['unpaid_timeout']) if 'unpaid_timeout' in z.files else None,
                x=[int(v) for v in z['x']] if 'x' in z.files else None)


def make_replay_env(trace, env_config, level=None):
    """A single native env in the trace's start state, set up as the env
    that recorded it."""
    lvl = trace['level'] or level
    env = MarioNativeVecEnv('ghost', 1, **trace_env_kwargs(
        env_config, lvl, trace['episode_life'], trace['unpaid_timeout']))
    env.reset()
    env.load_state(0, trace['state']); env._fetch_obs(0)
    env._post_reset_init([0], env.ram)
    env._ring[0] = (env.obs_u8[0].astype(np.float32) / 255.0)[..., None]
    env._raw_steps = bool(trace['raw'])
    env.hold_on_done = True     # the terminal step stays (no fresh episode)
    return env


class Replayer:
    """One emulator per trace, stepped in lockstep."""

    def __init__(self, traces, level, env_config=None):
        self.tr = traces
        self.envs = [make_replay_env(t, env_config, level) for t in traces]
        self.buf = ctypes.create_string_buffer(224 * 240 * 3)
        self.t = 0
        self.alive = [True] * len(traces)
        self.ended_at = [None] * len(traces)

    def step(self):
        """Advance every unfinished trace by one action; return per-trace
        (x, ypix, screen_x, big, alive) for this step."""
        rows = []
        for i, (tr, env) in enumerate(zip(self.tr, self.envs)):
            if self.alive[i] and self.t < len(tr['actions']):
                env.step(np.array([tr['actions'][self.t]]))
            elif self.alive[i]:
                self.alive[i] = False; self.ended_at[i] = self.t
            r = env.ram[0]
            rows.append((int(r[0x6D]) * 256 + int(r[0x86]), int(r[0x3B8]), int(r[0x3AD]),
                         int(r[0x756]) > 0, self.alive[i]))
        self.t += 1
        return rows

    def frame(self, i):
        self.envs[i].lib.benv_render_rgb(self.envs[i].env, 0, self.buf)
        return np.frombuffer(self.buf, dtype=np.uint8).reshape(224, 240, 3).copy()

    def close(self):
        for e in self.envs:
            e.close()


def paste_ghost(base, crop, gx, gy, alpha, tint):
    h, w = crop.shape[:2]
    x0, y0 = max(gx, 0), max(gy, 0); x1, y1 = min(gx + w, base.shape[1]), min(gy + h, base.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    c = crop[y0 - gy:y1 - gy, x0 - gx:x1 - gx].astype(np.float32)
    bg = c[0, 0]                                  # corner = background of the ghost's own frame
    mask = (np.abs(c - bg).sum(axis=-1) > 60)
    if not mask.any():
        return
    region = base[y0:y1, x0:x1].astype(np.float32)
    col = c * 0.6 + np.array(tint, dtype=np.float32) * 0.4
    region[mask] = alpha * col[mask] + (1 - alpha) * region[mask]
    base[y0:y1, x0:x1] = region.astype(np.uint8)


def recorded_rows(trace):
    """Per-step (x, life) of the recording: the clip's .csv next to the npz
    (x, life), or the door episode's stored x (life unknown: None)."""
    csv_path = trace['path'][:-4] + '.csv'
    if os.path.exists(csv_path):
        return [(int(r['x']), int(r['life'])) for r in csv.DictReader(open(csv_path))]
    if trace['x'] is not None:
        return [(x, None) for x in trace['x']]
    return None


def check_traces(traces, env_config, level):
    """Replay each trace and compare x / lives step by step with the
    recording; returns the number of traces that diverged. A step matches
    when the lives agree and the recorded x equals the replay's reported x
    or its RAM x; x is not compared on frames out of player control ($0E in
    0-5 / 7: dying, pipes, the intermission). The env's reported x is HELD on
    such frames and on transition garbage, so a trace recorded under other
    hold rules differs there while the emulated game is identical; a real
    divergence moves the in-play RAM x as well."""
    bad = 0
    for t in traces:
        rec = recorded_rows(t)
        if rec is None:
            print('%-40s no recorded x to compare' % t['name']); continue
        env = make_replay_env(t, env_config, level)
        first, n_timeup, n_held = None, 0, 0
        for k, a in enumerate(t['actions'][:len(rec)]):
            _, _, _, infos = env.step(np.array([a]))
            n_timeup += bool(env.last_signals.timeout[0])
            info = infos[0]
            x, life = int(info['x_pos']), int(info['life'])
            r = env.ram[0]
            ram_x = int(r[0x6D]) * 256 + int(r[0x86])
            rx, rlife = rec[k]
            in_play = not (r[0x0E] <= 5 or r[0x0E] == 7)
            n_held += rx != x and (rx == ram_x or not in_play)
            if first is None and ((in_play and rx != x and rx != ram_x)
                                  or (rlife is not None and life != rlife)):
                first = (k, x, ram_x, rx, life, rlife)
        env.close()
        ok = first is None
        bad += not ok
        print('%-40s %s  %d steps, %d unpaid cutoffs (episode_life %s, raw %d)%s%s'
              % (t['name'], 'EXACT   ' if ok else 'DIVERGES', min(len(t['actions']), len(rec)),
                 n_timeup, t['episode_life'], t['raw'],
                 ', %d steps with another held x' % n_held if n_held else '',
                 '' if ok else '  first at step %d: x %d (RAM %d) vs %d, life %s vs %s' % first))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traces', required=True, nargs='+', help='npz files or globs')
    ap.add_argument('--config', default=DEFAULT_CONFIG,
                    help='the config of the run that recorded the traces (its env_config '
                         'is part of the replay: unpaid cutoff, reward set, route)')
    ap.add_argument('--check', action='store_true',
                    help='replay and compare with the recorded x / lives instead of rendering')
    ap.add_argument('--level', default=None, help='start level (default: from the trace, else 8-4)')
    ap.add_argument('--out', default='ghosts.gif')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--alpha', type=float, default=0.4)
    ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--main', default='best', help="'best' (largest max x) or a trace index")
    ap.add_argument('--linger', type=int, default=12, help='frames a finished ghost stays visible')
    ap.add_argument('--max-ghosts', type=int, default=48)
    args = ap.parse_args()
    paths = sorted(p for pat in args.traces for p in glob.glob(pat))[:args.max_ghosts]
    if not paths:
        sys.exit('no traces matched')
    traces = [load_trace(p) for p in paths]
    level = args.level or traces[0]['level'] or '8-4'
    env_config = yaml.safe_load(open(args.config))['params']['config']['env_config']
    if args.check:
        sys.exit(1 if check_traces(traces, env_config, level) else 0)
    T = min(args.max_steps, max(len(t['actions']) for t in traces))

    # pass 1: positions only -> pick the main run
    rp = Replayer(traces, level, env_config); pos = []
    for _ in range(T):
        pos.append(rp.step())
    rp.close()
    max_x = [max(pos[t][i][0] for t in range(T)) for i in range(len(traces))]
    main_i = int(np.argmax(max_x)) if args.main == 'best' else int(args.main)
    print('main = %s (max x %d); %d ghosts' % (traces[main_i]['name'], max_x[main_i], len(traces) - 1))

    # pass 2: render
    from PIL import Image, ImageDraw
    tints = [(255, 90, 90), (90, 160, 255), (90, 230, 120), (250, 200, 60), (220, 100, 230), (80, 220, 220)]
    rp = Replayer(traces, level, env_config); frames = []
    for t in range(T):
        rows = rp.step()
        base = rp.frame(main_i)
        xm, ym, sxm = rows[main_i][0], rows[main_i][1], rows[main_i][2]
        cam = xm - sxm
        shown = 0
        for i, (x, y, sx, big, alive) in enumerate(rows):
            if i == main_i:
                continue
            if not alive and (rp.ended_at[i] is None or t - rp.ended_at[i] > args.linger):
                continue
            gx = x - cam
            if gx < -16 or gx >= 240:
                continue
            fr = rp.frame(i); h = 32 if big else 16
            crop = fr[max(y, 0):max(y, 0) + h, max(sx, 0):max(sx, 0) + 16]
            if crop.size == 0:
                continue
            fade = 1.0 if alive else max(0.0, 1 - (t - rp.ended_at[i]) / args.linger)
            paste_ghost(base, crop, gx, y, args.alpha * fade, tints[i % len(tints)])
            shown += 1
        img = Image.fromarray(base).resize((480, 448), Image.NEAREST)
        d = ImageDraw.Draw(img)
        d.text((6, 428), 'step %4d  main %s x=%d  ghosts %d/%d' % (t, traces[main_i]['name'][:22], xm, shown, len(traces) - 1), fill=(255, 255, 255))
        frames.append(np.asarray(img))
        if not any(rp.alive):
            break
    rp.close()
    import imageio
    if args.out.endswith('.mp4'):
        imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=None)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps)
    print('wrote %s: %d frames at %d fps' % (args.out, len(frames), args.fps))


if __name__ == '__main__':
    main()
