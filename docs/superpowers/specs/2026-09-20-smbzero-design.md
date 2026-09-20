# SMBZero design (2026-09-20)

A policy/value network plus AlphaZero-style MCTS on the real game (PAL ROM) that plays
Super Mario Bros 1-1 -> axe live, from any start delay.

## Decisions (with the user)

| | |
|---|---|
| play time | net-guided MCTS with emulator lookahead (savestates), from the true current state |
| budget | real time: one decision per 4 frames = 80 ms wall clock per decision |
| hardware | GPU 1 only (`CUDA_VISIBLE_DEVICES=1`), 32 CPU threads |
| noise | one random delay at the start of 1-1: 0-60 NOOP frames before the first decision -> 61 runs |
| success | must: full game cleared live on all 61 delays; score: mean game time vs search 5:26.4, PAL TAS 4:51.7 |
| input | 84x84 grayscale, last 4 decision frames (max of each step's last 2 frames), status bar cropped |
| ROM | PAL (Europe), 50.007 fps |
| approach | AlphaZero MCTS (PUCT), bootstrapped by the existing search as teacher, then self-play |

## MCTS (C++, `search/src/mcts/`)

- Step = 4 frames holding one of the 12 COMPLEX_MOVEMENT actions. Children are created lazily:
  a simulation descends by PUCT to an edge with no child, emulates that one step (compact
  state of the parent + action), renders the step's 84x84 frame, and asks the net for the new
  node's prior and value (batched across all leaves of a wave, across trees).
- Value = frames to reach the next route level (the segment goal: search/src/route). Terminal:
  goal = 0, dead (route.cpp's rule) = V_DEATH = 4096 frames. The net predicts value / 4096.
- Backup is min (the game is deterministic after the delay): B(leaf) = net value;
  B(n) = 4 + min over created children B(c). Deaths never contaminate a good sibling.
- PUCT at p: q(c) = clamp(1 - (B(c) - B*(p)) / S, 0, 1) with B*(p) the best child, S = 32
  frames; unvisited children q = FPU (0.5); score = q + c_puct P(c) sqrt(N(p)) / (1 + N(c)).
  Batched waves: an edge being created this wave is not selected again; visited paths get a
  virtual visit per pending leaf.
- Commit: the most visited root child (ties: lower B). The chosen subtree becomes the tree.
- Forced moves: if all 12 (action, NOOP) pairs give the same exact state from the root, the
  input does nothing now (transitions, flag, pipes): commit NOOP without searching.
- A forest holds many trees (parallel self-play games) so one net batch serves all.

## Net (`smbzero/net.py`)

IMPALA CNN (16-32-32, residual) on 4x84x84 -> FC 256 -> policy (12 logits), value (sigmoid,
x 4096 frames). Inference fp16 on GPU 1.

## Data and training (`smbzero/`)

- Teacher: the existing explore+beam. Start variants: each level's natural entry state (from
  the verified e2e route, first in-control frame) + 0-60 NOOP frames. The beam from a variant
  uses that level's explore reference replayed from the reference's own start (new
  `ref_start` argument to `ss_optimize`), so it needs no new exploration.
- Samples: (4-frame obs, policy target, value target, mask). Teacher: one-hot action, value =
  frames left on its route. Self-play: root visit distribution, value = frames actually left
  if the segment was finished, else the root's B. Forced moves: value only.
- DAgger: where self-play dies or stalls, the teacher re-solves from a few steps earlier.
- Loop: teacher data -> supervised net -> [self-play games with MCTS -> train] x N, each
  experiment <= 4 h with checkpoints.

## Evaluation (`smbzero/eval.py`)

Full game from the FullGame state (first control in 1-1) after d = 0..60 NOOP frames.
Frame-budget mode (simulations per decision calibrated to 80 ms) for batch evaluation; wall
clock mode for the demo (report p50/p99 decision time). Every run replayed in stable-retro
frame by frame. Time = first control -> axe, delay included.

## Phases and pass criteria

1. Engine: obs renderer = batchenv's pixels; MCTS deterministic, commit = replay, throughput
   (simulations per 80 ms) measured.
2. Teacher dataset (8 levels x variants), supervised net: value error (frames), action agreement.
3. Live play per level entry state + delays: clears.
4. Full game, 61 delays: 61/61 clears (the must).
5. Self-play iterations for speed: mean game time down toward 4:51.7.
