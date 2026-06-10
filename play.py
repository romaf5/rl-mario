#!/usr/bin/env python3
"""Play/evaluate a trained Mario PPO agent (headless, CPU).

    python play.py runs/<run>/nn/<checkpoint>.pth --games 5
    python play.py runs/<run>/nn/<checkpoint>.pth --save-video eval.mp4

Runs the sequential full game (3 lives, no episode_life) and prints per-game
stats; optionally writes an MP4 of the first game. The network is built from
the config's params, so use the config the checkpoint was trained with.
"""

import argparse

import numpy as np
import torch
import yaml

from mario_env import create_mario_env


def load_model(checkpoint_path, config_path):
    """Build the rl_games network from the config and load checkpoint weights."""
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
    model.eval()
    return model


def play_game(env, model, deterministic=True, max_steps=20000, frames_out=None):
    """Play one full game; returns (total_reward, steps, max_x, max_stage, info)."""
    obs = env.reset()
    total_reward, steps, max_x, max_stage = 0.0, 0, 0, 0
    info = {}
    done = False
    while not done and steps < max_steps:
        obs_t = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0)
        with torch.no_grad():
            res = model({'obs': obs_t, 'is_train': False, 'prev_actions': None})
        if deterministic:
            action = torch.argmax(res['logits'], dim=-1).item()
        else:
            action = torch.distributions.Categorical(
                logits=res['logits']).sample().item()

        obs, reward, done, info = env.step(action)
        if frames_out is not None:
            frames_out.append(env.unwrapped.screen.copy())
        total_reward += reward
        steps += 1

        max_x = max(max_x, int(info.get('x_pos', 0)))
        stage = (info.get('world', 1) - 1) * 4 + (info.get('stage', 1) - 1)
        if stage > max_stage:
            max_stage = stage
            print(f"  -> Reached World {info['world']}-{info['stage']}")
    return total_reward, steps, max_x, max_stage, info


def main():
    parser = argparse.ArgumentParser(description='Play trained Mario agent')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default='configs/mario_ppo_random_stages.yaml',
                        help='Config the checkpoint was trained with')
    parser.add_argument('--games', type=int, default=5)
    parser.add_argument('--stochastic', action='store_true',
                        help='Sample actions instead of argmax')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Write an MP4 of the first game to this path')
    parser.add_argument('--max-steps', type=int, default=20000)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.config)
    env = create_mario_env(name='SuperMarioBros-v0', episode_life=False,
                           stage_bonus=0)

    for game in range(args.games):
        print(f"\n=== Game {game + 1}/{args.games} ===")
        frames = [] if (args.save_video and game == 0) else None
        reward, steps, max_x, max_stage, info = play_game(
            env, model, deterministic=not args.stochastic,
            max_steps=args.max_steps, frames_out=frames)
        print(f"  Reward: {reward:.1f}  Steps: {steps}  Max x: {max_x}")
        print(f"  Furthest stage: World {max_stage // 4 + 1}-{max_stage % 4 + 1}"
              f"  Lives left: {info.get('life', 0)}"
              f"  Flag: {info.get('flag_get', False)}")
        if frames:
            import imageio
            imageio.mimsave(args.save_video, frames, fps=15)
            print(f"  Video saved to {args.save_video}")

    env.close()


if __name__ == '__main__':
    main()
