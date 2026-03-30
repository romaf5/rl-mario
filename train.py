#!/usr/bin/env python3
"""Train a PPO agent on Super Mario Bros using rl_games."""

import argparse
import os
import sys

import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from mario_env import create_mario_env
from mario_vecenv import register_mario_vecenv
from callbacks import MarioObserver


def register_mario_env():
    """Register custom Mario environment and vecenv with rl_games."""
    register_mario_vecenv()
    env_configurations.register('mario_custom', {
        'vecenv_type': 'MARIO',
        'env_creator': lambda **kwargs: create_mario_env(**kwargs),
    })


def main():
    parser = argparse.ArgumentParser(description='Train Mario PPO agent')
    parser.add_argument('--config', type=str, default='configs/mario_ppo.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom run name (overrides config name)')
    parser.add_argument('--num-actors', type=int, default=None,
                        help='Override number of parallel environments')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override max training epochs')
    parser.add_argument('--video-freq', type=int, default=500,
                        help='Record gameplay video every N epochs (0 to disable)')
    args = parser.parse_args()

    register_mario_env()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.run_name:
        config['params']['config']['name'] = args.run_name
    if args.num_actors:
        config['params']['config']['num_actors'] = args.num_actors
    if args.max_epochs:
        config['params']['config']['max_epochs'] = args.max_epochs

    print("=" * 60)
    print(f"  Training: {config['params']['config']['name']}")
    print(f"  Environment: {config['params']['config']['env_config']['name']}")
    print(f"  Actors: {config['params']['config']['num_actors']}")
    print(f"  Max epochs: {config['params']['config']['max_epochs']}")
    print(f"  Device: {config['params']['config']['device']}")
    print(f"  Video freq: every {args.video_freq} epochs")
    print("=" * 60)

    observer = MarioObserver(video_freq=args.video_freq)
    runner = Runner(algo_observer=observer)
    runner.load(config)
    runner.reset()
    runner.run({
        'train': True,
        'play': False,
        'checkpoint': args.checkpoint,
        'sigma': None,
    })


if __name__ == '__main__':
    main()
