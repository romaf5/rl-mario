"""Find a WINNING door episode for a checkpoint and publish it as a clip.

Plays N door episodes in parallel (batched env, identical door start, no
noise) until one reaches the axe, replays that action sequence in a single
env for the frames, and publishes mp4 + npz + TensorBoard GIF like
tools/clip_watcher.py. Useful when the victory rate is a few percent.

  python tools/find_win.py runs/Mario_GRPO84_*/nn/grpo_last.pth --batch 32 --rounds 6
"""
import argparse, glob, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
from render_ckpt import build
from clip_watcher import publish
from mario_native_vecenv import MarioNativeVecEnv, NativeEvalEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ckpt'); ap.add_argument('--config', default='configs/mario_ppo_native_84.yaml'); ap.add_argument('--level', default='8-4')
    ap.add_argument('--batch', type=int, default=32); ap.add_argument('--rounds', type=int, default=6); ap.add_argument('--max-steps', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--run', default=None, help='run dir to publish into (default: the checkpoint\'s)')
    a = ap.parse_args()
    ck = glob.glob(a.ckpt)[0] if '*' in a.ckpt else a.ckpt
    model, cfg = build(a.config, ck)
    step = int(torch.load(ck, map_location='cpu', weights_only=False).get('iter', 0))
    ec = dict(cfg['env_config']); [ec.pop(k, None) for k in ('name', 'action_type', 'archive_path')]
    # same settings as the trainer's clean door eval: no sticky actions, but the
    # config's reset_noops (random start delay -> varied enemy/Bowser RNG phase)
    ec.update(random_stages=[a.level], sticky_actions=0, explore_eps=0, self_restart_prob=0, n_threads=8, dense_infos=True)
    torch.manual_seed(a.seed); win = None
    for rnd in range(a.rounds):
        env = MarioNativeVecEnv('fw', a.batch, **dict(ec, episode_life=True, seed=a.seed + rnd)); env._raw_steps = True; obs = env.reset()   # hack-free search: the win replays frame-exactly
        starts = []
        for i in range(a.batch):
            env.lib.benv_save(env.env, i, env._sbuf); starts.append(bytes(env._sbuf.raw))
        acts = [[] for _ in range(a.batch)]; alive = np.ones(a.batch, dtype=bool)
        for s in range(a.max_steps):
            with torch.no_grad():
                lg = model({'obs': torch.from_numpy(obs).float(), 'is_train': False})['logits']
            act = torch.distributions.Categorical(logits=lg).sample().numpy()
            obs, r, d, inf = env.step(act)
            for i in range(a.batch):
                if alive[i]: acts[i].append(int(act[i]))
                if alive[i] and d[i]:
                    alive[i] = False
                    if inf[i].get('victory'): win = (i, list(acts[i]), starts[i]); break
            if win or not alive.any(): break
        env.close()
        print('round %d: %s' % (rnd, 'VICTORY found (env %d, %d steps)' % (win[0], len(win[1])) if win else 'no victory in %d episodes' % a.batch), flush=True)
        if win: break
    if not win:
        sys.exit('no winning episode found')
    # replay the winning sequence in a single env for the frames (same door start, deterministic)
    ec_r = dict(ec, episode_life=False, reset_noops=0); ec_r.pop('dense_infos', None)
    env = NativeEvalEnv(**ec_r); v = env.v; v._raw_steps = True; env.reset()
    start = win[2]
    v.lib.benv_load(v.env, 0, start); v._fetch_obs(0); v._post_reset_init([0], v.ram)
    v._ring[0] = (v.obs_u8[0].astype(np.float32) / 255.0)[..., None]; obs = v._obs()[0]
    frames, total, info, life_r, prev_life, pfr = [], 0.0, {}, 0.0, None, []
    for act in win[1]:
        obs, r, done, info = env.step(act); total += r; frames.extend(env.frames4)
        if prev_life is not None and info.get('life') != prev_life: life_r = 0.0
        prev_life = info.get('life'); life_r += r; pfr.extend([life_r] * len(env.frames4))
        if done:
            for _ in range(150):               # the ending: bridge, Bowser falls, the walk to the princess
                obs, r, _d, _i = env.step(0); frames.extend(env.frames4); pfr.extend([life_r] * len(env.frames4))
            break
    env.close()
    run_dir = a.run or os.path.dirname(os.path.dirname(ck))
    path, n = publish(run_dir, step, frames, win[1], start, info.get('max_x_pos', 0), total, info, a.level, pfr)
    print('published %s: %d frames, victory=%s, max x %d' % (path, len(frames), info.get('victory'), info.get('max_x_pos', 0)))


if __name__ == '__main__':
    main()
