# SMBZero next: value network and world model (plan, 2026-09-21)

## The rule from here on

**At evaluation the agent touches the game only by playing it**: every 4 frames it sees the
screen (84x84) and sends one action. All lookahead happens inside a learned model. No
savestates, no emulator steps for planning, no search route. Training may use the emulator
freely (agent play, teacher labels until stage C, relabelling); play may not.

**Agent play** = the agent playing the game itself with its current net and search, to make
its own training data (AlphaZero calls it self-play; here there is no opponent).

Today's agent breaks this in four places, and all four go:

| today (play time) | why it is a crutch | replaced by |
|---|---|---|
| MCTS steps the real emulator from savestates to look ahead | privileged access to the environment ("cheat code") | a learned dynamics model (stage B) |
| leaf values = progress along the search's route | outside guidance; only works on levels the search solved | a learned value (stage A) |
| survival check: rollouts in the emulator before a commit | same as the first row | learned "dies within k steps" head (stage B) |
| forced-move check in the emulator | same | learned (the model predicts that inputs change nothing) or dropped |

Zero human knowledge stays (no hints; see the memory note). The only human input remains the
level order of the warp route and the 12 COMPLEX_MOVEMENT actions.

## Where we are (2026-09-21, all pushed)

- Engine: C++ MCTS on the native core (`search/src/mcts`), about 1,500 simulations per 80 ms on 32 threads.
- Net: IMPALA CNN, 4x84x84 in, prior over 12 actions + value (value unused: `value_mix = 0`).
- Training: teacher routes (46), DAgger with a local beam teacher, strong-teacher labels at
  stalls, visit counts of 2000-simulation agent play, starts at full-game arrival states.
  Data: `smbzero/data/` (744 MB, 21k episodes, about 1M labelled states; gitignored, local).
- Results: first verified full game 1-1 -> axe in 5:37.0 (agent play, 2000 sims, the search:
  5:26.4). At the live budget (1000 sims): 28/32 level runs; full games 0/8 (8-1 / 8-2).
  Ablation: net prior 28/32, uniform prior 8/32, net alone 0/32 -- the prior matters.
- Latest net: `smbzero/runs/zero8/net.pt` (local).

## Stage A: a value network that can rank states (inside today's MCTS first)

Goal: the net supplies leaf values; the route goes. Do it while dynamics are still exact, so
value errors are measured alone.

1. **Why the current value fails**: it predicts absolute frames to the end of the level, which a
   cropped 84x84 screen does not show (1-1 repeats pipes and hills): it orders nearby states
   at chance (`smbzero/value_test.py`: 54%).
2. **Targets, plenty and free**: every MCTS node already gets a route value; log (stack,
   value) for all created nodes during agent play (millions per hour instead of thousands).
   Later: n-step bootstrapped returns and real outcomes, which need no route.
3. **Make it learnable**: predict what the screen shows.
   - A1 relative value: a head on (root stack, leaf stack) predicting the frames difference
     between them -- all MCTS compares is siblings under one root.
   - A2 longer context: more history frames and the status bar (world number, timer) so the
     net knows which level and how far into it; test cheaply against A1.
4. **Gate**: `value_test` >= 90% on every level; then `value_mix` 0 -> 0.5 -> 1 with the level
   clear rate at 1000 sims holding (baseline 28/32); then the route is removed from play.

## Stage B: a world model (MuZero-style); the emulator leaves play

Goal: the MCTS searches a learned model. The real game is stepped only by committed actions.

1. **Model** (PyTorch, GPU 1):
   - representation h: last 4 frames (+ last actions) -> latent s_0;
   - dynamics g: (s_k, action) -> s_{k+1}, plus heads for the events that matter:
     "segment goal reached", "dead", "inputs do nothing now" (forced), frames elapsed (4 per step);
   - prediction f: policy prior and value (stage A's value moves here).
   - Consistency loss (EfficientZero): g's next latent matches h of the real next frames;
     optional frame-reconstruction head for debugging only.
2. **Data**: everything in `smbzero/data` (teacher routes, agent play, DAgger episodes) unrolled
   K = 5-10 steps; plus fresh agent play. Frames and actions are enough; the emulator provides
   the ground truth during training.
3. **Search**: batched latent MCTS (tree in C++ or Python, all model calls batched on the GPU);
   the committed action = most visited; death head replaces the survival check.
4. **Gates**:
   - B1 model quality on held-out real trajectories: death / goal / forced precision and recall
     >= 95% at 1-10 steps ahead; value ranking along real unrolls (value_test on predicted latents).
   - B2 play: per level at 4 delays with latent MCTS, the environment touched only by committed
     actions: within a few points of the emulator-MCTS baseline (28/32).
   - B3 full game from 1-1 at the live budget, several delays.

### Stage A, results (2026-09-24)

The value predicts W, the frames wasted against perfect play from the tree's root. Held-out
pairs from the agent's own searches:

| | same depth (what q ranks) | whole tree (what b minimises) |
|---|---|---|
| old absolute head | 49.4% | 72.3% |
| relative value, search leaves only | 83.8% | 98.5% |
| relative value, + rollout traps | **91.7%** | 95.4% |

Playing with no route at all (1000 simulations, 4 start delays per level):

| | 1-1 | 1-2 | 4-1 | 4-2 | 8-1 | 8-2 | 8-3 | 8-4 | total |
|---|---|---|---|---|---|---|---|---|---|
| route value | 4/4 | 4/4 | 4/4 | 4/4 | 1/4 | 3/4 | 4/4 | 2/4 | ~26/32 |
| learned, search leaves only | 4/4 | 2/4 | 4/4 | 3/4 | 0/4 | 3/4 | 4/4 | 0/4 | 20/32 |
| learned, + rollout traps | 4/4 | 4/4 | 4/4 | 2/4 | 1/4 | 3/4 | 4/4 | 0/4 | 22/32 |

Adding 120k more rollout pairs on the levels that were behind (8-4, 4-2, 8-1) closed the gap:
the pair test reached 93.6% with no level below 84% (8-4 went 80% -> 95%), and play reached

| | 1-1 | 1-2 | 4-1 | 4-2 | 8-1 | 8-2 | 8-3 | 8-4 | total |
|---|---|---|---|---|---|---|---|---|---|
| learned value, targeted traps | 4/4 | 4/4 | 4/4 | 4/4 | 1/4 | 4/4 | 4/4 | 1/4 | **26/32** |

which is what the route value scores on the same net. **Stage A is passed**: the search plays
as well on a value read off the screen as on progress along a route it was handed, so the
route is no longer needed at play time. It is still used to label training data (valroll),
which stage C removes.

Precision was checked rather than assumed (RL is often sensitive to it): at play time fp16
differs from fp32 by 0.02 frames on average (max 0.24) and flips a sibling comparison 0.05%
of the time, against a value whose own error is 4.2 frames; training the same data and seed
for 10k steps gives 91.1% (bf16, 329 s) against 91.3% (fp32, 600 s) -- inside the noise for
1.8x the time. There is no bootstrapping loop here for small errors to compound in, since
the targets are computed from the emulator. `relvalue --precision fp32` keeps the check cheap.

Two things did not work and are worth not repeating:
- Early fusion (root and leaf as 8 channels through one trunk) is worse than late fusion
  with the embeddings' difference: 74% against 84%.
- Training the value on the search's own backed-up verdicts made it worse (1/16 on the
  levels it had failed, against 5/16): MCTS only revisits shallow nodes, so its verdicts sit
  at median depth 5 with ~18 frames of typical waste, while the value is asked about leaves
  at median depth 25 where typical waste is ~0.

8-4 is the gap left (0/4, stalls not deaths, and the value's weakest level on the pair test).
It is the puzzle level: progress there means bumping a hidden block and taking the right
pipe, which the route's rank reads from the tiles and the net has to see.

### Stage B, first result (2026-09-24, wm1: 6000 trajectories / 250k frames, 30k steps)

Held-out unrolls, average precision (the base rate is in brackets):

| depth | goal | dead | forced | waste error |
|---|---|---|---|---|
| 1 step ahead | 0.18 (0.8%) | 0.50 | **1.00** | 1.3 frames |
| 10 steps ahead | 0.54 (7%) | 0.74 | **1.00** | 1.4 frames |

- Forced is solved: the model knows exactly when the input does nothing, ten steps out.
- Death is real but loose (0.74 against a 7% base rate), not the emulator's certainty.
- Waste, the signal that ranks one line against another, is the weak one: correlation 0.39,
  and on the steps that matter it under-calls badly -- steps that really throw away 8-32
  frames are called 3.5, and the catastrophic ones (mean 158) are called 2.0.
- Latent consistency 0.16: an unrolled latent drifts from the one the real frames give.

What it needs before a latent search is worth writing: far more data and training (250k
frames and 50 minutes is tiny for a MuZero-style model), trajectories that contain many more
catastrophic steps (random play rarely commits the interesting mistakes), a bigger latent,
and a consistency term strong enough to keep the unroll on the rails.

## Stage C: no teacher at all (full MuZero loop)

The teacher (the C++ search) does not extend to other games: it needs savestates (Go-Explore
and the beam reset the emulator thousands of times a second), its progress rank is built from
Mario RAM (area, x/y, camera, tiles), and it covers only levels it has solved. Stage C removes
it completely: no teacher routes, no local / strong-teacher labels, no route values.

- C1 wean: start from the stage-B model; from then on train only on agent play: visit counts, n-step values, event labels from what the real game did,
  MuZero Reanalyze over stored games. Teacher data leaves the replay buffer. Gate: level clears
  and full games hold.
- C2 true zero: train from scratch with no teacher ever -- the real test that the method
  extends. The risk is exploration (the PPO attempts failed on hidden blocks, the vine, the 8-4
  maze), so it needs a general exploration mechanism: novelty in the model's latent space
  (count-based bonuses, or Go-Explore run inside the learned model instead of on emulator
  savestates). Gate: the same level clears without any teacher.
- The one game-specific interface left: the event signals "level finished" and "dead" (level
  number and lives, from RAM) -- like the reward and lives count Atari benchmarks provide.
- Keep the ablation habit: uniform prior / no search / older model, same budget.

## Stage D: live and fast

- Real time: 80 ms of wall clock per decision (latent MCTS is GPU-bound: measure simulations
  per 80 ms); all 61 start delays of 1-1 to the axe, every run replayed in stable-retro.
- Then speed: mean full-game time below the search's 5:26.4, toward the PAL TAS 4:51.7.

## Evaluation protocol (every stage)

- Environment access at eval: committed actions only (assert it: the eval player gets no
  savestate API; count emulator frames = 4 x decisions + the start delay).
- Every eval game saved as a route file and replayed in stable-retro frame by frame.
- Quick check: 8 levels x 4 delays; full check: full game at 8 delays; final: 61 delays and
  wall-clock real time. Ablations beside every headline number.

## Working rules

- GPU 1 only, 32 CPU threads; 4 h per experiment with checkpoints (30 min / 2 h / 4 h),
  judged at the checkpoint; commit and push at checkpoints; `runs/` holds only live runs
  (the rest in `runs_archive/`).
- Launch long jobs detached (`setsid nohup`); never `pkill` a pattern that matches the
  launching shell (use `[.]` brackets, kill and relaunch in separate commands).

## First steps when we resume

1. Log (stack, route value, level) for every MCTS node in agent play -> a value dataset.
2. Train A1 (relative value) and A2 (context) heads offline; run `value_test`; pick one.
3. Start the world-model code (`smbzero/model/`: h, g, f, unroll training on existing data)
   and measure B1 on held-out trajectories before any latent search.
