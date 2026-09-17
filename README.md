# RL Mario Bros

> **Work in Progress** -- the agent does not complete the full game yet.

RL training pipeline for Super Mario Bros using [rl_games](https://github.com/Denys88/rl_games) (PPO) with PyTorch. The goal is an agent that completes the full game with 3 lives along the warp-zone speedrun route -- 1-1, 1-2 (warp), 4-1, 4-2 (vine warp), 8-1, 8-2, 8-3, 8-4 -- a speedrunner, not just a forward-runner. No level-specific knowledge goes into training.

## Current Status

Run log and decisions: [EXPERIMENTS.md](EXPERIMENTS.md).

- PPO clears 1-1 and 4-1 reliably, 8-1 / 8-2 / 8-3 part of the time, and learned the 1-2 warp (60-90% of sampled episodes)
- First 8-4 victory by the GRPO finisher (`keep/`), fine-tuned from a strong PPO policy
- Current gate: the 4-2 vine warp (`configs/mario_ppo_native_42.yaml`)

## Setup

Apple Silicon Mac or Linux:

```bash
./setup_mac.sh                 # venv_retro + deps, ROM (from the gym-super-mario-bros wheel, SHA-1 checked), native core build
source venv_retro/bin/activate
```

`native/build.sh` rebuilds the C++ core (g++ `-march=native` on Linux, clang++ `-mcpu=native` on macOS). Configs say `device: cuda:0`; on a machine without CUDA training falls back to Apple's MPS backend (or the CPU), and `--device` overrides it.

## Quick Start

```bash
# 4-2 vine-warp run (current experiment)
python train.py --config configs/mario_ppo_native_42.yaml

# warp-route run (all 8 route levels)
python train.py --config configs/mario_ppo_native_routeDisc.yaml

# 15-epoch smoke test
python train.py --config configs/mario_ppo_native_smoke.yaml --video-freq 0

# overrides
python train.py --config configs/mario_ppo_native_42.yaml --num-actors 64 --max-epochs 5000 --device cpu

# render a checkpoint on one level (best of N sampled episodes)
python tools/render_ckpt.py runs/<run_dir>/nn/<checkpoint>.pth --config configs/mario_ppo_native_42.yaml --level 4-2 --episodes 8

# offline checks (seconds, CPU)
python tests/explorer_bench.py; python tests/novelty_bench.py; python tests/reward_bench.py

# TensorBoard
tensorboard --logdir runs --bind_all --port 6006
```

## Architecture

- **Emulator**: native C++ SMB core (`native/`), N games stepped on a threadpool, verified bitwise against stable-retro (`native/difftest.py`, `native/deep_difftest.py`)
- **Algorithm**: PPO (discrete) via rl_games; critic-free GRPO finisher (`grpo/train_grpo.py`)
- **Observations**: 84x84 grayscale (status bar cropped), 4-frame stack
- **Actions**: COMPLEX_MOVEMENT (12 actions) -- includes running and pipes
- **Rewards** (`mario_rewards.py`): positive-only -- first-visit progress per frame, level clear 500 + 100 per extra level gained (the 4-2 vine warp pays 1900); every failure just ends the episode
- **Exploration**: Go-Explore archive of savestate cells -- self-restarts, backward-chaining frontier practice, invisible explorer walks, relative novelty bonus

## Demo

Early training (epoch 2000, full-game curriculum, pre-native stack):

[![Watch Demo](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge)](https://github.com/romaf5/rl-mario/releases/download/v0.1/demo_hd.mp4)


https://github.com/user-attachments/assets/18a551a7-9bb2-48a8-b9aa-7e22a6401b74
