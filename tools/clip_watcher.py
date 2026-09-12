"""Sidecar clip recorder for trainers that don't record video (grpo/).

Watches a checkpoint file; whenever it advances, plays a few clean door
episodes with the current policy (3 lives, no noise), writes the best one as
<run>/videos/clip_<step>.mp4 (+ .npz replay trace) and publishes the GIF to
the run's TensorBoard under gameplay/clip, like the rl_games runs get.

  python tools/clip_watcher.py --run runs/Mario_GRPO84_* --ckpt nn/grpo_last.pth --every 300
"""
import argparse, glob, os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
from render_ckpt import build
from mario_native_vecenv import NativeEvalEnv
from callbacks import MarioObserver


def record(model, cfg, level, episodes, max_steps, seed, route=None, stop_on_level_change=False):
    """route: the run's level list (so a warp/exit into another route level is
    a transition, not a wrong exit). stop_on_level_change: end the clip once
    the level is left (per-level clips); else play on (full game)."""
    ec = dict(cfg['env_config']); [ec.pop(k, None) for k in ('name', 'action_type', 'archive_path', 'video_levels')]
    ec.update(random_stages=[level], sticky_actions=0, explore_eps=0, self_restart_prob=0, reset_noops=0, episode_life=False)
    if route:
        ec['route_levels'] = list(route)
    best = None
    gen = torch.Generator().manual_seed(int(seed))      # local generator: reproducible, never touches the process RNG
    for ep in range(episodes):
        env = NativeEvalEnv(**ec); v = env.v; v._raw_steps = True; v.hold_on_done = True; obs = env.reset()     # hack-free, no reset after done
        v.lib.benv_save(v.env, 0, v._sbuf); start = bytes(v._sbuf.raw)
        frames, acts, total, info = [], [], 0.0, {}
        life_r, prev_life, per_frame_r = 0.0, None, []
        for step in range(max_steps):
            with torch.no_grad():
                lg = model({'obs': torch.from_numpy(obs[None]).float(), 'is_train': False})['logits']
            act = int(torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).item())   # sampled policy, seeded
            obs, r, done, info = env.step(act); total += r; acts.append(act); frames.extend(env.frames4)
            if prev_life is not None and info.get('life') != prev_life:
                life_r = 0.0
            prev_life = info.get('life'); life_r += r
            per_frame_r.extend([life_r] * len(env.frames4))
            if stop_on_level_change and step == 0:
                gp0 = info.get('game_progress')
            if stop_on_level_change and not done and info.get('game_progress') != gp0:
                for _ in range(30):                                   # a moment of the next level, then cut
                    obs, r, _d, _i = env.step(0); frames.extend(env.frames4); per_frame_r.extend([life_r] * len(env.frames4))
                break
            if done:
                for _ in range(240 if info.get('victory') else 90):    # ending / game-over screen keeps playing
                    obs, r, _d, _i = env.step(0); frames.extend(env.frames4); per_frame_r.extend([life_r] * len(env.frames4))
                break
        env.close()
        mx = info.get('max_x_pos', 0)
        if best is None or mx > best[0]:
            best = (mx, frames, acts, start, total, info, per_frame_r)
    return best


def publish(run_dir, step, frames, acts, start, mx, total, info, level, per_frame_r=None, tag='gameplay/clip', name=None):
    from PIL import Image, ImageDraw
    import imageio
    from tensorboardX import SummaryWriter
    try:
        from tensorboardX.proto.summary_pb2 import Summary
    except ImportError:
        from tensorboard.compat.proto.summary_pb2 import Summary
    vdir = os.path.join(run_dir, 'videos'); os.makedirs(vdir, exist_ok=True)
    base = os.path.join(vdir, (name or 'clip_%06d' % step) + '_x%d' % mx)
    imageio.mimsave(base + '.mp4', frames, fps=60, macro_block_size=None)
    np.savez_compressed(base + '.npz', state=np.frombuffer(start, dtype=np.uint8), actions=np.array(acts, dtype=np.int16), level=level, raw=1, step=step)
    # all 60 fps frames, like the observer's clips: _gif_bytes picks every 2nd
    # (30 fps at 33 ms = real time) or every 4th for long clips (15 fps at
    # 67 ms = real time). Feeding it pre-thinned frames played 2-4x too fast.
    pil = []
    for k, f in enumerate(frames):
        im = Image.fromarray(f); d = ImageDraw.Draw(im)
        lr = per_frame_r[k] if per_frame_r is not None and k < len(per_frame_r) else total
        d.rectangle((0, 214, 240, 224), fill=(0, 0, 0)); d.text((3, 213), 'step %d  x %d  R(life) %.0f' % (step, mx, lr), fill=(255, 255, 255))
        pil.append(im)
    gif = MarioObserver._gif_bytes(pil, per_step=4)
    w = SummaryWriter(os.path.join(run_dir, 'summaries'))
    w.file_writer.add_summary(Summary(value=[Summary.Value(tag=tag, image=Summary.Image(height=224, width=240, colorspace=3, encoded_image_string=gif))]), step)
    w.add_scalar(tag + '_max_x', mx, step); w.add_scalar(tag + '_reward', total, step); w.flush(); w.close()
    return base + '.mp4', len(gif)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True); ap.add_argument('--ckpt', default='nn/grpo_last.pth')
    ap.add_argument('--config', default='configs/mario_ppo_native_84.yaml'); ap.add_argument('--level', default='8-4')
    ap.add_argument('--every', type=int, default=300, help='seconds between checks'); ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--max-steps', type=int, default=3000); ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    run_dir = glob.glob(a.run)[0] if '*' in a.run else a.run
    ck = os.path.join(run_dir, a.ckpt); last = None
    while True:
        if os.path.exists(ck):
            try:
                stamp = os.path.getmtime(ck)
                if stamp != last:
                    model, cfg = build(a.config, ck)
                    step = int(torch.load(ck, map_location='cpu', weights_only=False).get('iter', 0))
                    mx, frames, acts, start, total, info, pfr = record(model, cfg, a.level, a.episodes, a.max_steps, seed=step)
                    path, n = publish(run_dir, step, frames, acts, start, mx, total, info, a.level, pfr)
                    print(time.strftime('%H:%M:%S'), 'step %d: clip %s (%d frames, max x %d, R %.0f, gif %dKB)' % (step, os.path.basename(path), len(frames), mx, total, n // 1024), flush=True)
                    last = stamp
            except Exception as e:
                print(time.strftime('%H:%M:%S'), 'clip failed:', e, flush=True)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == '__main__':
    main()
