#!/usr/bin/env python3
"""Play/evaluate a trained Mario PPO agent."""

import argparse
import os
import sys
import time

import yaml
import numpy as np
from rl_games.common import env_configurations
from rl_games.torch_runner import Runner

from mario_env import create_mario_env


def register_mario_env():
    env_configurations.register('mario_custom', {
        'vecenv_type': 'RAY',
        'env_creator': lambda **kwargs: create_mario_env(**kwargs),
    })


def play_with_render(checkpoint, config_path='configs/mario_ppo.yaml',
                     num_games=5, deterministic=True):
    """Play games with rendering using rl_games player."""
    register_mario_env()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override player settings for evaluation
    config['params']['config']['player']['render'] = True
    config['params']['config']['player']['games_num'] = num_games
    config['params']['config']['player']['deterministic'] = deterministic
    # Use full game (no episode_life) for evaluation
    config['params']['config']['env_config']['episode_life'] = False

    runner = Runner()
    runner.load(config)
    runner.reset()
    runner.run({
        'train': False,
        'play': True,
        'checkpoint': checkpoint,
        'sigma': None,
    })


def play_manual(checkpoint_path, config_path='configs/mario_ppo.yaml',
                num_games=3, deterministic=True, render=True, fps=30):
    """Manual play loop with detailed progress tracking."""
    import torch
    from rl_games.algos_torch import players

    register_mario_env()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create a single environment for evaluation (no episode_life)
    env_config = dict(config['params']['config']['env_config'])
    env_config['episode_life'] = False
    env = create_mario_env(**env_config)

    # Load the trained model via rl_games runner
    config['params']['config']['player']['render'] = False
    config['params']['config']['player']['games_num'] = 1
    config['params']['config']['env_config']['episode_life'] = False

    runner = Runner()
    runner.load(config)
    runner.reset()

    player = runner.create_player()
    player.restore(checkpoint_path)
    player.has_batch_dimension = False
    player.init_rnn()

    for game in range(num_games):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_x = 0
        max_stage = 0

        print(f"\n{'='*50}")
        print(f"Game {game + 1}/{num_games}")
        print(f"{'='*50}")

        while not done:
            obs_tensor = player.obs_to_torch(obs)
            action = player.get_action(obs_tensor, is_deterministic=deterministic)
            if hasattr(action, 'item'):
                action = action.item()

            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            x_pos = info.get('x_pos', 0)
            world = info.get('world', 1)
            stage = info.get('stage', 1)
            current_stage = (world - 1) * 4 + (stage - 1)

            if x_pos > max_x:
                max_x = x_pos
            if current_stage > max_stage:
                max_stage = current_stage
                print(f"  -> Reached World {world}-{stage}!")

            if render:
                env.render()
                time.sleep(1.0 / fps)

        lives = info.get('life', 0)
        print(f"  Total Reward: {total_reward:.1f}")
        print(f"  Steps: {steps}")
        print(f"  Max X Position: {max_x}")
        print(f"  Furthest Stage: World {(max_stage // 4) + 1}-{(max_stage % 4) + 1}")
        print(f"  Lives Remaining: {lives}")
        print(f"  Flag Get: {info.get('flag_get', False)}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Play trained Mario agent')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/mario_ppo.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--games', type=int, default=5, help='Number of games to play')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic (non-deterministic) actions')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--fps', type=int, default=30, help='Render FPS')
    parser.add_argument('--mode', choices=['auto', 'manual'], default='auto',
                        help='auto=rl_games player, manual=custom loop with stats')
    args = parser.parse_args()

    if args.mode == 'auto':
        play_with_render(
            args.checkpoint,
            config_path=args.config,
            num_games=args.games,
            deterministic=not args.stochastic,
        )
    else:
        play_manual(
            args.checkpoint,
            config_path=args.config,
            num_games=args.games,
            deterministic=not args.stochastic,
            render=not args.no_render,
            fps=args.fps,
        )


if __name__ == '__main__':
    main()
