# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL training pipeline for Super Mario Bros using **rl_games** (PPO) with PyTorch on GPU. The goal is to train an agent that completes the full game (worlds 1-1 through 8-4) sequentially with 3 lives -- a speedrunner, not just a forward-runner.

## Commands

```bash
# Activate environment (always do this first)
source venv/bin/activate

# Train on single level (curriculum)
python train.py --config configs/mario_ppo_1_1.yaml

# Train on full game (all 32 stages, 3 lives)
python train.py --config configs/mario_ppo.yaml

# Resume from checkpoint
python train.py --checkpoint runs/<run_dir>/nn/<checkpoint>.pth

# Override training params
python train.py --config configs/mario_ppo.yaml --num-actors 16 --max-epochs 5000 --video-freq 200

# Evaluate trained agent
python play.py runs/<run_dir>/nn/MarioPPO.pth --mode manual --games 5

# TensorBoard (runs dir contains all experiment logs)
tensorboard --logdir runs --bind_all --port 6006
```

## Architecture

The project wraps rl_games' PPO implementation with custom Mario-specific components. Understanding the data flow requires reading across multiple files:

**Environment pipeline** (`mario_env.py` → `mario_vecenv.py`):
- `create_mario_env()` is the single factory that builds the full wrapper chain: `gym_super_mario_bros.make()` → `JoypadSpace` → `EpisodicLifeMarioEnv` → `MarioProgressWrapper` → `MaxAndSkipEnv` → `WarpFrame(84x84 grayscale)` → `ScaledFloatFrame([0,1])` → `FrameStack(4)`. Final observation: `(84, 84, 4)` float32.
- `MarioVecEnv` in `mario_vecenv.py` uses Python multiprocessing (not Ray) because rl_games' `RayVecEnv` cannot see custom env registrations across process boundaries. Each worker imports `create_mario_env` locally. This is intentional -- do not switch to Ray.

**rl_games integration** (`train.py`):
- Two registrations happen at startup: the vecenv type (`MARIO`) and the env config (`mario_custom`). Both must be registered before `Runner.load()`.
- `MarioObserver` (in `callbacks.py`) extends rl_games' `AlgoObserver` interface to log Mario-specific TensorBoard metrics and record gameplay videos. It hooks into `process_infos()` (per-step) and `after_print_stats()` (per-epoch).

**Reward shaping** (`MarioProgressWrapper`):
- Base reward comes from the env (x_pos delta + time penalty + death penalty).
- Adds `stage_bonus` (+500 default) on `flag_get` to heavily incentivize level completion over just running forward.
- Injects `game_progress` (0-31) and `max_x_pos` into the info dict for metric tracking.

**Config structure** (`configs/*.yaml`):
- Follows rl_games' YAML format: `params.{algo, model, network, config}`.
- `config.env_config` kwargs are passed directly to `create_mario_env()`.
- Key relationship: `batch_size = num_actors * horizon_length`, must be divisible by `minibatch_size`.
- `episode_life: True` for training (each life = episode boundary), `False` for evaluation.

## Key Constraints

- **numpy must be < 2.0** -- nes_py has uint8 overflow with numpy 2.x.
- **gym must be 0.25.x** -- gym 0.26+ breaks nes_py's old step API.
- **setuptools must be < 71** -- tensorboard needs `pkg_resources`.
- Action space: COMPLEX_MOVEMENT (12 actions) includes running (B button) and down (pipes), essential for speedrunning.
- The `SuperMarioBros-v0` env automatically chains all 32 stages with 3 lives. Individual levels use `SuperMarioBros-<W>-<S>-v0` format.

## Outputs

- Checkpoints: `runs/<name>_<timestamp>/nn/*.pth`
- TensorBoard: `runs/<name>_<timestamp>/summaries/`
- Custom metrics in TensorBoard: `mario/mean_x_pos`, `mario/best_stage_progress`, `mario/flag_get_rate`
