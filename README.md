# RL Mario Bros

> **Work in Progress** -- this project is under active development. Training pipeline is functional but the agent has not yet completed the full game.

RL training pipeline for Super Mario Bros using [rl_games](https://github.com/Denys88/rl_games) (PPO) with PyTorch on GPU. The goal is to train an agent that completes the full game (worlds 1-1 through 8-4) sequentially with 3 lives -- a speedrunner, not just a forward-runner.

## Current Status

- Agent consistently beats worlds 1-1 and 1-2
- Reaches world 1-3 regularly
- Training on full game with natural curriculum (sequential level progression)
- Best mean reward: ~4500 (full game curriculum config)

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train on full game (recommended)
python train.py --config configs/mario_ppo_curriculum.yaml

# Train on single level
python train.py --config configs/mario_ppo_1_1.yaml

# Resume from checkpoint
python train.py --checkpoint runs/<run_dir>/nn/<checkpoint>.pth

# Override params
python train.py --config configs/mario_ppo_curriculum.yaml --num-actors 32 --max-epochs 10000

# Evaluate
python play.py runs/<run_dir>/nn/MarioPPO.pth --mode manual --games 5

# TensorBoard
tensorboard --logdir runs --bind_all --port 6006
```

## Architecture

- **Algorithm**: PPO (discrete) via rl_games
- **Observations**: 84x84 grayscale, 4-frame stack, normalized to [0,1]
- **Actions**: COMPLEX_MOVEMENT (12 actions) -- includes running and pipes
- **Reward shaping**: stage completion bonus (+500), idle penalty after 10 stalled steps
- **Parallelism**: multiprocessing vectorized env (not Ray) for custom env compatibility

## Demo

Latest training (epoch 2000, full-game curriculum):

[![Watch Demo](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge)](https://github.com/romaf5/rl-mario/releases/download/v0.1/demo_hd.mp4)


https://github.com/user-attachments/assets/18a551a7-9bb2-48a8-b9aa-7e22a6401b74


## Key Constraints

- numpy < 2.0 (nes_py uint8 overflow)
- gym == 0.25.x (nes_py old step API)
- setuptools < 71 (tensorboard pkg_resources)

## Configs

| Config | Description |
|--------|-------------|
| `mario_ppo_curriculum.yaml` | Full game, natural curriculum, 32 actors |
| `mario_ppo_1_1.yaml` | Single level 1-1 training |
| `mario_ppo_world1.yaml` | Random stages from world 1 |
| `mario_ppo_world1_lstm.yaml` | World 1 with LSTM memory |
| `mario_ppo.yaml` | Full game baseline |
