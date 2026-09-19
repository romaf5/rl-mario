#!/usr/bin/env python3
"""Train a PPO agent on Super Mario Bros using rl_games.

    source venv_retro/bin/activate
    python train.py --config configs/mario_ppo_random_stages.yaml

The periodic video recorder builds its eval env in the main process, which is
fine under stable-retro's one-emulator-per-process limit because all training
emulators live in vecenv worker processes.
"""

import argparse
import signal
import sys

import yaml
from rl_games.common import env_configurations
from rl_games.torch_runner import Runner

from mario_env import create_mario_env
from mario_vecenv import register_mario_vecenv
from mario_native_vecenv import register_mario_native_vecenv
from callbacks import MarioObserver
from device_support import resolve_device


def register_mario_env():
    """Register the Mario env config and vecenv type with rl_games.
    Both must be registered before Runner.load()."""
    register_mario_vecenv()
    register_mario_native_vecenv()
    env_configurations.register('mario', {
        'vecenv_type': 'MARIO',
        'env_creator': lambda **kwargs: create_mario_env(**kwargs),
    })
    env_configurations.register('mario_native', {
        'vecenv_type': 'MARIO_NATIVE',
        'env_creator': lambda **kwargs: create_mario_env(**kwargs),
    })


def fresh_archive_conflict(config, checkpoint, resume_archive):
    """Error text if a FRESH run would silently continue an old archive: the
    native env loads whatever file sits at env_config.archive_path, so a
    relaunch of a config (runs normally start from scratch) inherited the
    previous run's cells, wins and tries."""
    import os
    path = (config['params']['config'].get('env_config') or {}).get('archive_path')
    if path and os.path.exists(path) and not checkpoint and not resume_archive:
        return (f'{path} exists: a fresh run would continue that archive. Move it '
                f'next to its run (runs_archive/<run>/), rename archive_path, or '
                f'pass --resume-archive to continue it on purpose')
    return None


def main():
    parser = argparse.ArgumentParser(description='Train Mario PPO agent')
    parser.add_argument('--config', type=str, default='configs/mario_ppo_random_stages.yaml',
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
    parser.add_argument('--eval-episodes', type=int, default=32,
                        help='Sampled-policy door episodes per level at every video epoch')
    parser.add_argument('--eval-level-steps', type=int, default=1500,
                        help='Step cap of a sampled door episode')
    parser.add_argument('--minibatch-size', type=int, default=None,
                        help='Override the PPO minibatch size (must divide '
                             'num_actors * horizon_length). On a Mac smaller '
                             'minibatches keep each GPU command buffer short '
                             'enough for the macOS GPU watchdog')
    parser.add_argument('--device', type=str, default=None,
                        help='Override the config device (cuda:0, mps, cpu). '
                             'A CUDA device on a machine without CUDA falls '
                             'back to mps (Apple Silicon) or cpu')
    parser.add_argument('--resume-archive', action='store_true',
                        help='Allow a fresh run (no --checkpoint) to continue '
                             'the existing archive file at env_config.archive_path')
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
    if args.minibatch_size:
        cc = config['params']['config']
        batch = cc['num_actors'] * cc['horizon_length']
        if batch % args.minibatch_size:
            parser.error(f'--minibatch-size {args.minibatch_size} does not divide '
                         f'num_actors * horizon_length = {batch}')
        cc['minibatch_size'] = args.minibatch_size
    err = fresh_archive_conflict(config, args.checkpoint, args.resume_archive)
    if err:
        parser.error(err)
    if args.device:
        config['params']['config']['device'] = args.device
    config['params']['config']['device'] = resolve_device(
        config['params']['config'].get('device'))

    print("=" * 60)
    print(f"  Training: {config['params']['config']['name']}")
    print(f"  Environment: {config['params']['config']['env_config']['name']}")
    print(f"  Actors: {config['params']['config']['num_actors']}")
    print(f"  Max epochs: {config['params']['config']['max_epochs']}")
    print(f"  Device: {config['params']['config']['device']}")
    print(f"  Video freq: every {args.video_freq} epochs")
    print("=" * 60)

    # Custom keys (popped so rl_games never sees them):
    # - stage_curriculum_freq: curriculum re-weighting of random-stage
    #   sampling toward low-clear-rate stages every N epochs
    # - eval_env_config: overrides for the video/eval env (e.g. start stage)
    curriculum_freq = config['params']['config'].pop('stage_curriculum_freq', 0)
    eval_env_config = config['params']['config'].pop('eval_env_config', None)
    # - shuffle_minibatches / value_norm_clip: rl_games patches (rlg_patches.py)
    import rlg_patches
    rlg_patches.apply(
        shuffle_minibatches=config['params']['config'].pop('shuffle_minibatches', True),
        value_norm_clip=config['params']['config'].pop('value_norm_clip', None))

    observer = MarioObserver(video_freq=args.video_freq,
                             curriculum_freq=curriculum_freq,
                             eval_env_kwargs=eval_env_config,
                             eval_episodes=args.eval_episodes,
                             eval_level_steps=args.eval_level_steps)
    # a plain `kill` (SIGTERM) exits through SystemExit, so atexit handlers run
    # (the native env saves its archive there)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    runner = Runner(algo_observer=observer)
    runner.load(config)
    runner.reset()
    runner.run({
        'train': True,
        'play': False,
        'checkpoint': args.checkpoint,
        'sigma': None,
    })
    # a recording still running at the end (daemon thread inside the native
    # core's threadpool) must finish before the interpreter tears the process
    # down: exiting through it aborted with "terminate called without an
    # active exception"
    vt = getattr(observer, '_video_thread', None)
    if vt is not None and vt.is_alive():
        print('  [Video] waiting for the last recording to finish')
        vt.join()


if __name__ == '__main__':
    main()
