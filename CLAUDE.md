# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL training pipeline for Super Mario Bros using **rl_games** (PPO) with PyTorch on GPU. The goal is to train an agent that completes the full game (worlds 1-1 through 8-4) sequentially with 3 lives -- a speedrunner, not just a forward-runner.

The environment backend is **stable-retro** (C-core libretro NES emulation, ~4x faster than the old nes_py backend it replaced in June 2026). The SMB integration lives in `retro_integration/SuperMarioBros-Nes-v0/`: extended `data.json` RAM variables plus generated savestates for all 32 levels and full-game mode (`gen_states.py` regenerates them).

## Commands

```bash
# Activate environment (always do this first; the venv is named venv_retro)
source venv_retro/bin/activate

# Train on random stages (current main config: worlds 1-2, uniform per reset)
python train.py --config configs/mario_ppo_random_stages.yaml

# Quick CPU smoke test of the training pipeline (no GPU, ~20s)
python train.py --config configs/mario_ppo_cpu_smoke.yaml --video-freq 0

# Resume from checkpoint (note: runs are normally started from scratch --
# the user prefers clean epoch-0 TensorBoard charts)
python train.py --checkpoint runs/<run_dir>/nn/<checkpoint>.pth

# Override training params
python train.py --config configs/mario_ppo_random_stages.yaml --num-actors 16 --max-epochs 5000 --video-freq 200

# Evaluate a trained agent on the sequential full game (headless, CPU)
python play.py runs/<run_dir>/nn/<checkpoint>.pth --games 5 --save-video eval.mp4

# Re-run env backend validation (obs/reward/RAM semantics, stage states, smoke train)
python validate_env.py

# Play the training env yourself with per-term rewards, rewind and a CSV trace
python tools/play.py --config configs/mario_ppo_native_84.yaml --level 8-4

# TensorBoard (runs dir contains all experiment logs)
tensorboard --logdir runs --bind_all --port 6006
```

## Architecture

The project wraps rl_games' PPO implementation with custom Mario-specific components:

**Environment pipeline** (`mario_env.py` -> `mario_vecenv.py`):
- `create_mario_env()` is the single factory: `RetroMarioEnv` (stable-retro emulator with gym_super_mario_bros-compatible reward/info semantics) -> `EpisodicLife` -> `MarioProgressWrapper` -> `MaxAndSkip` -> `WarpFrame(84x84 grayscale)` -> `ScaledFloatFrame([0,1])` -> `FrameStack(4)`. Final observation: `(84, 84, 4)` float32.
- `random_stages` env_config key (e.g. `['1-1', ..., '2-4']`) makes each reset load a uniformly sampled per-level savestate -- one emulator per process (stable-retro limitation), states swapped on reset. Training uses this so samples concentrate on unmastered levels; evaluation uses the sequential full game.
- `MarioVecEnv` in `mario_vecenv.py` uses Python multiprocessing (not Ray) because rl_games' `RayVecEnv` cannot see custom env registrations across process boundaries. Each worker imports `create_mario_env` locally. This is intentional -- do not switch to Ray.

**rl_games integration** (`train.py`):
- Two registrations happen at startup: the vecenv type (`MARIO`) and the env config (`mario`). Both must be registered before `Runner.load()`.
- `MarioObserver` (in `callbacks.py`) extends rl_games' `AlgoObserver` to log Mario-specific TensorBoard metrics and record gameplay videos. It hooks into `process_infos()` (per-step) and `after_print_stats()` (per-epoch). Video recording runs on a background thread with a CPU deep-copy of the model (never blocks training, never touches the GPU); the eval env comes from the overridable `_make_eval_env()`.

**Reward shaping** (`MarioProgressWrapper`):
- Base reward replicates gym_super_mario_bros: x_pos delta + time penalty + death penalty, clipped to [-15, 15] per frame.
- Adds `stage_bonus` (+500 default) on `flag_get`, `progress_reward` growing bonus (`min(x_delta, 20) * progress_reward * x_pos`), and `idle_penalty` after `idle_threshold` consecutive idle steps.
- Injects `game_progress` (0-31) and `max_x_pos` into the info dict for metric tracking.

**Config structure** (`configs/*.yaml`):
- Follows rl_games' YAML format: `params.{algo, model, network, config}`.
- `config.env_config` kwargs are passed directly to `create_mario_env()`.
- Key relationship: `batch_size = num_actors * horizon_length`, must be divisible by `minibatch_size`.
- `episode_life: True` for training (each life = episode boundary), `False` for evaluation.

## Key Constraints

- The stack runs on **gymnasium** (stable-retro 1.0); the old gym 0.25/nes_py pins no longer apply. `venv/` (nes_py stack) was the pre-June-2026 environment, kept only for replaying old checkpoints.
- **Checkpoints do not transfer between backends**: nes_py-era checkpoints see a shifted pixel distribution on the retro backend (224x240 overscan-cropped screen vs 240x256) and play poorly. Train from scratch on the current backend.
- Action space: COMPLEX_MOVEMENT (12 actions) includes running (B button) and down (pipes), essential for speedrunning.
- The SMB ROM (`retro_integration/SuperMarioBros-Nes-v0/rom.nes`) is gitignored; restore it by copying from `venv/lib/python3.10/site-packages/gym_super_mario_bros/_roms/` and re-running `python -m retro.import` if needed.
- Run names follow `Mario_<Experiment>` (e.g. `Mario_RandomStages`); start training runs from scratch, not from checkpoints (user preference: clean epoch-0 charts).

## Outputs

- Checkpoints: `runs/<name>_<timestamp>/nn/*.pth`
- TensorBoard: `runs/<name>_<timestamp>/summaries/`
- Custom metrics in TensorBoard: `mario/mean_x_pos`, `mario/best_stage_progress`, `mario/flag_get_rate` (primary progress metric under random-stage training)
- Eval videos: `runs/<name>_<timestamp>/videos/epoch_*.mp4` (disk) + animated GIF in TensorBoard Images tab
