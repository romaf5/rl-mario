#!/usr/bin/env python3
"""Play/evaluate a trained Mario PPO agent (headless).

    python play.py runs/<run>/nn/<checkpoint>.pth --games 5
    python play.py runs/<run>/nn/<checkpoint>.pth --level 4-2 --save-video eval.mp4
    python play.py <retro-trained>.pth --backend retro --config configs/mario_ppo_random_stages.yaml

Plays 3-life games (no episode_life) with the SAMPLED policy -- the policy
PPO trains and the observer's eval/*_sampled metrics measure (--argmax for
the greedy one, which has fixed points such as holding right into a step) --
seeded per game (--seed), and prints per-game stats; optionally writes an MP4
of the first game. The network AND the env settings come from --config, so
use the config the checkpoint was trained with.

Backends:
  native (default)  the training emulator (mario_native_vecenv): every
                    checkpoint since the native stack was trained there. The
                    env is the observer's sequential eval: the config's reward
                    set, unpaid cutoff and route, training-style stepping
                    (--raw: hack-free, like the video clips) and the
                    multi-life rule (a stuck life ends by a forced time-up).
  retro             the stable-retro reference chain (create_mario_env), for
                    retro-trained checkpoints.
The backends differ in pixels (grey palette, max-pool order, resize rounding)
and start states, so a checkpoint plays poorly on the other one.

The game starts at --level (default: the config's first trained level, e.g.
4-2 for a single-level run; 'full' = the full game from 1-1) and plays on
through level exits until the lives are gone, the game is won or -- native --
an exit leaves the config's route. "Furthest level" is route-aware: the last
ON-route level reached (a wrong exit or the minus world never counts).
"""

import argparse
import inspect

import numpy as np
import torch
import yaml

from device_support import resolve_device


def level_index(name):
    w, s = str(name).split('-')
    return (int(w) - 1) * 4 + int(s) - 1


def level_name(gp):
    return '%d-%d' % (gp // 4 + 1, gp % 4 + 1)


def route_progress(gp, route_gps):
    """The last ON-route level index at or below gp (as the observer's
    route-aware progress): the env's progress also records the level of a
    wrong exit (4-2 flag -> 4-3), which is a failure at 4-2, not progress."""
    gp = int(gp)
    if not route_gps or gp in route_gps:
        return gp
    below = [g for g in route_gps if g < gp]
    return max(below) if below else gp


def load_model(checkpoint_path, config_path, device='cpu'):
    """Build the rl_games network from the config and load checkpoint weights
    (strict: a config that does not match the checkpoint must fail loudly)."""
    from rl_games.algos_torch import model_builder

    with open(config_path) as f:
        params = yaml.safe_load(f)['params']

    network = model_builder.ModelBuilder().load(params)
    model = network.build({
        'actions_num': 12 if params['config']['env_config'].get(
            'action_type', 'complex') == 'complex' else 7,
        'input_shape': (84, 84, 4),
        'num_seqs': 1,
        'value_size': 1,
        'normalize_value': params['config'].get('normalize_value', True),
        'normalize_input': params['config'].get('normalize_input', False),
    })
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, params['config']


def make_env(cfg, backend, level, raw=False):
    """The eval env for `level` ('W-S', or None = the full game from 1-1)
    built from the checkpoint's config; returns (env, route level indices)."""
    ec = dict(cfg.get('env_config') or {})
    route = list(ec.get('route_levels') or ec.get('random_stages') or [])
    route_gps = {level_index(s) for s in route}
    for k in ('name', 'action_type', 'archive_path', 'video_levels',
              'video_level_steps', 'backend', 'dense_infos'):
        ec.pop(k, None)
    # no noise, no restarts, 3 lives, playing on through level exits
    ec.update(sticky_actions=0.0, self_restart_prob=0.0, reset_noops=0,
              episode_life=False, full_game=True,
              random_stages=[level] if level else None)
    if backend == 'native':
        from mario_native_vecenv import NativeEvalEnv
        # the observer's sequential eval (callbacks._eval_env_config)
        ec.update(explore_eps=0.0, explore_episode_prob=0.0, explorer_envs=0,
                  end_on_stage_exit=False, route_levels=route or None)
        return NativeEvalEnv(raw_steps=raw, **ec), route_gps
    from mario_env import create_mario_env
    known = inspect.signature(create_mario_env).parameters
    ec = {k: v for k, v in ec.items() if k in known}
    return create_mario_env(name='SuperMarioBros-v0', **ec), route_gps


def play_game(env, model, device, gen, route_gps, argmax=False,
              max_steps=20000, frames_out=None):
    """Play one game; returns a dict of per-game stats."""
    obs = env.reset()
    total_reward, steps, max_x = 0.0, 0, 0
    info = {}
    done = False
    start_gp = best_gp = None
    while not done and steps < max_steps:
        obs_t = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            res = model({'obs': obs_t, 'is_train': False, 'prev_actions': None})
        logits = res['logits'].float().cpu()
        if argmax:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            action = int(torch.multinomial(torch.softmax(logits, -1), 1,
                                           generator=gen).item())

        obs, reward, done, info = env.step(action)
        if frames_out is not None:
            if hasattr(env.unwrapped, 'frames4'):
                frames_out.extend(env.unwrapped.frames4)   # native: 60 fps
            else:
                frames_out.append(env.unwrapped.screen.copy())
        total_reward += reward
        steps += 1

        max_x = max(max_x, int(info.get('x_pos', 0)))
        # the env's debounced, monotonic progress (not the raw world byte:
        # the minus world reads as world 37), on-route levels only
        gp = route_progress(info.get('game_progress', 0), route_gps)
        if start_gp is None:
            start_gp = best_gp = gp
        elif gp > best_gp and not info.get('wrong_exit', False):
            best_gp = gp
            print(f"  -> Reached World {level_name(gp)} (step {steps})")
    world = int(info.get('world', 1))
    if info.get('victory'):
        cause = 'victory'
    elif info.get('wrong_exit'):
        cause = ('wrong exit into the minus world' if world > 8 else
                 f"wrong exit into {world}-{info.get('stage', '?')}")
    elif info.get('life') == 255:
        cause = 'game over'
    elif done and info.get('timeout'):
        cause = 'stuck twice at the same spot'
    elif steps >= max_steps:
        cause = 'step cap'
    else:
        cause = 'done'
    return dict(reward=total_reward, steps=steps, max_x=max_x,
                start=start_gp or 0, furthest=best_gp or 0, cause=cause,
                lives=0 if info.get('life', 0) == 255 else info.get('life', 0),
                victory=bool(info.get('victory', False)),
                fps=60 if hasattr(env.unwrapped, 'frames4') else 15)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default='configs/mario_ppo_native_42.yaml',
                        help='Config the checkpoint was trained with')
    parser.add_argument('--backend', choices=('native', 'retro'), default='native')
    parser.add_argument('--level', default=None,
                        help="start level, e.g. 4-2; 'full' = full game from 1-1 "
                             "(default: the config's first trained level)")
    parser.add_argument('--games', type=int, default=5)
    parser.add_argument('--argmax', action='store_true',
                        help='greedy actions instead of sampling the policy')
    parser.add_argument('--seed', type=int, default=0,
                        help='sampling seed (game k uses seed*1000 + k)')
    parser.add_argument('--raw', action='store_true',
                        help='native: hack-free stepping, as the video clips')
    parser.add_argument('--device', default='cpu',
                        help="torch device (cuda:0 falls back to mps/cpu); "
                             "batch-1 inference is fastest on the CPU")
    parser.add_argument('--save-video', type=str, default=None,
                        help='Write an MP4 of the first game to this path')
    parser.add_argument('--max-steps', type=int, default=20000)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg = load_model(args.checkpoint, args.config, device)
    level = args.level
    if level is None:
        rs = (cfg.get('env_config') or {}).get('random_stages')
        level = rs[0] if rs else None
    elif level == 'full':
        level = None
    env, route_gps = make_env(cfg, args.backend, level, raw=args.raw)
    print(f"[play] {args.backend} backend, start {level or 'full game'}, "
          f"{'argmax' if args.argmax else 'sampled'} policy, device {device}")

    results = []
    for game in range(args.games):
        print(f"\n=== Game {game + 1}/{args.games} ===")
        frames = [] if (args.save_video and game == 0) else None
        gen = torch.Generator().manual_seed(args.seed * 1000 + game)
        r = play_game(env, model, device, gen, route_gps, argmax=args.argmax,
                      max_steps=args.max_steps, frames_out=frames)
        results.append(r)
        print(f"  Reward: {r['reward']:.1f}  Steps: {r['steps']}  Max x: {r['max_x']}")
        print(f"  Furthest on-route level: World {level_name(r['furthest'])}"
              f" (from {level_name(r['start'])})  Lives left: {r['lives']}"
              f"  Ended by: {r['cause']}")
        if frames:
            import imageio
            imageio.mimsave(args.save_video, frames, fps=r['fps'],
                            macro_block_size=None)
            print(f"  Video saved to {args.save_video}")

    env.close()
    if len(results) > 1:
        adv = [r['furthest'] > r['start'] for r in results]
        print(f"\n{len(results)} games: advanced past the start level "
              f"{np.mean(adv):.2f}, victory {np.mean([r['victory'] for r in results]):.2f}, "
              f"furthest {level_name(max(r['furthest'] for r in results))}")


if __name__ == '__main__':
    main()
