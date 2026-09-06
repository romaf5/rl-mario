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
"""
import argparse, ctypes, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mario_native_vecenv import MarioNativeVecEnv

ENV_KW = dict(full_game=True, self_restart_prob=0, sticky_actions=0, explore_eps=0,
              reset_noops=0, n_threads=1, episode_life=False, dense_infos=True)


def load_trace(path):
    z = np.load(path, allow_pickle=True)
    raw = int(z['raw']) if 'raw' in z else int('video' in os.path.basename(path))
    return dict(name=os.path.basename(path)[:-4], state=bytes(np.asarray(z['state'], dtype=np.uint8)),
                actions=[int(a) for a in z['actions']], raw=raw,
                level=str(z['level']) if 'level' in z else None)


class Replayer:
    """One emulator per trace, stepped in lockstep."""

    def __init__(self, traces, level):
        self.tr = traces
        self.envs = []
        for t in traces:
            env = MarioNativeVecEnv('ghost', 1, random_stages=[level], **ENV_KW)
            env.reset()
            env.lib.benv_load(env.env, 0, t['state']); env._fetch_obs(0)
            env._post_reset_init([0], env.ram)
            env._raw_steps = bool(t['raw'])
            self.envs.append(env)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traces', required=True, nargs='+', help='npz files or globs')
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
    T = min(args.max_steps, max(len(t['actions']) for t in traces))

    # pass 1: positions only -> pick the main run
    rp = Replayer(traces, level); pos = []
    for _ in range(T):
        pos.append(rp.step())
    rp.close()
    max_x = [max(pos[t][i][0] for t in range(T)) for i in range(len(traces))]
    main_i = int(np.argmax(max_x)) if args.main == 'best' else int(args.main)
    print('main = %s (max x %d); %d ghosts' % (traces[main_i]['name'], max_x[main_i], len(traces) - 1))

    # pass 2: render
    from PIL import Image, ImageDraw
    tints = [(255, 90, 90), (90, 160, 255), (90, 230, 120), (250, 200, 60), (220, 100, 230), (80, 220, 220)]
    rp = Replayer(traces, level); frames = []
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
