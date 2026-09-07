"""Render a gameplay clip (mp4) + replayable trace from ANY checkpoint that
holds an rl_games actor-critic state dict (rl_games runs, grpo/train_grpo.py).

  python tools/render_ckpt.py runs/Mario_GRPO84_*/nn/grpo_last.pth --level 8-4 --out clip.mp4
  python tools/render_ckpt.py runs/<run>/nn/last_*.pth --greedy --episodes 3

Plays door episodes with 3 lives on the training (native) emulator, records
every emulated frame at 60 fps, and writes <out>.npz (start state + actions)
next to the video so the clip can be replayed in tools/play.py --replay or
overlaid with tools/ghosts.py.
"""
import argparse, glob, os, sys, yaml
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from mario_native_vecenv import NativeEvalEnv
from rl_games.algos_torch import model_builder


def build(cfg_path, ck_path):
    params = yaml.safe_load(open(cfg_path))['params']; cfg = params['config']
    net = model_builder.ModelBuilder().load(params)
    model = net.build({'actions_num': 12, 'input_shape': (84, 84, 4), 'num_seqs': 1, 'value_size': 1,
                       'normalize_value': cfg['normalize_value'], 'normalize_input': cfg['normalize_input']})
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('[ckpt] %s  iter/epoch %s  missing %d unexpected %d' % (ck_path, ck.get('iter', ck.get('epoch')), len(missing), len(unexpected)))
    model.eval(); return model, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ckpt'); ap.add_argument('--config', default='configs/mario_ppo_native_84.yaml')
    ap.add_argument('--level', default='8-4'); ap.add_argument('--out', default=None)
    ap.add_argument('--episodes', type=int, default=1, help='render the best of N sampled episodes')
    ap.add_argument('--greedy', action='store_true'); ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--outro', type=int, default=120, help='steps to keep recording after the episode ends (ending / game over)')
    a = ap.parse_args()
    ck = glob.glob(a.ckpt)[0] if '*' in a.ckpt else a.ckpt
    model, cfg = build(a.config, ck)
    ec = dict(cfg['env_config']); [ec.pop(k, None) for k in ('name', 'action_type', 'archive_path')]
    ec.update(random_stages=[a.level], sticky_actions=0, explore_eps=0, self_restart_prob=0, reset_noops=0,
              episode_life=False)      # a clip plays all 3 lives; the training config ends episodes per life
    torch.manual_seed(a.seed)
    best = None
    for ep in range(a.episodes):
        env = NativeEvalEnv(**ec); v = env.v; v._raw_steps = True      # hack-free: transitions and the ending are shown
        obs = env.reset(); v.lib.benv_save(v.env, 0, v._sbuf); start = bytes(v._sbuf.raw)
        frames, acts, total, info = [], [], 0.0, {}
        for step in range(a.max_steps):
            with torch.no_grad():
                logits = model({'obs': torch.from_numpy(obs[None]).float(), 'is_train': False})['logits']
            act = int(logits.argmax()) if a.greedy else int(torch.distributions.Categorical(logits=logits).sample())
            obs, r, done, info = env.step(act); total += r; acts.append(act); frames.extend(env.frames4)
            if done:
                for _ in range(a.outro):      # let the ending / game-over screen play
                    obs, r, _d, info2 = env.step(0); frames.extend(env.frames4)
                break
        cause = ('victory' if info.get('victory') else 'game over' if info.get('life') == 255 else 'loop' if info.get('loop_timeout')
                 else 'step limit' if step >= a.max_steps - 1
                 else 'timeout' if info.get('timeout') else 'running')
        print('episode %d: %d steps, max x %d, reward %.0f, ended by %s' % (ep, len(acts), info.get('max_x_pos', 0), total, cause))
        env.close()
        if best is None or info.get('max_x_pos', 0) > best[0]:
            best = (info.get('max_x_pos', 0), frames, acts, start, cause, total)
    max_x, frames, acts, start, cause, total = best
    run_dir = os.path.dirname(os.path.dirname(ck))
    out = a.out or os.path.join(run_dir, 'videos', 'manual_%s_x%d.mp4' % (os.path.basename(ck)[:-4], max_x))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    import imageio
    imageio.mimsave(out, frames, fps=60, macro_block_size=None)
    np.savez_compressed(out[:-4] + '.npz', state=np.frombuffer(start, dtype=np.uint8), actions=np.array(acts, dtype=np.int16),
                        level=a.level, raw=1, ckpt=ck)
    print('wrote %s (%d frames, %.0fs at 60fps) + %s | max x %d, %s, reward %.0f' % (out, len(frames), len(frames) / 60, out[:-4] + '.npz', max_x, cause, total))


if __name__ == '__main__':
    main()
