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

### Stage B and C, 2026-09-25: two bugs, and what the data was missing

Two defects found in review, both of which we had been reading as limits of the approach:

- **The consistency loss compared unrelated samples.** `unroll` returns K latents of shape
  (B, ...); they were joined with `cat(dim=0)` (depth-major) while the target frames are
  built batch-major, so at batch 64 / unroll 6 almost every predicted latent was matched to
  a different sample's frames. Weight 2.0 on a random target -- this is the consistency of
  0.22 we recorded three times and could not explain. Fixed with `stack(dim=1)`; wm3 was
  discarded mid-run because it had trained against it.
- **The latent search ended its run on a settled line.** `_select` returned `None` when the
  best line under PUCT was already terminal, the wave broke, and an empty wave ended the
  search. A settled child now takes its visit, which is what lets PUCT turn to its brothers.
  Measured after the fix: the 1-1 outcome is unchanged (dies at 43 / 46 decisions for death
  cost 0 / 128), so the failure is the model, not the bookkeeping -- but the death-cost
  sweep that came before the fix was not evidence of anything.

What the data was missing, in both stages:

- **The world model had never seen the agent play.** Its trajectories were the route, a held
  input, or noise; a search inside the model only ever visits states the agent's own play
  reaches. `wmdata --modes ...,agent` plays with the net and the learned value.
- **The teacher-free value had no siblings.** `mcdata --branches` was declared and never
  used, so all 120k pairs had one branch per root. A winning branch's W is constant along
  its whole line, so with one branch per root the only learnable signal is alive-vs-dead:
  the ranking metric read 99.6% and the per-level gate collapsed (1-1 1/4, 1-2 0/4, against
  relv3's 4/4 and 4/4). Branches now fan out from a shared root.
- **Two levels were absent.** The spine pool filled with the easy levels and never attempted
  4-2 or 8-4 again, so the value met 6 of 8 levels. It now asks for what is missing, up to
  `--spine-tries` rounds.

**Stage B, 2026-09-25 (wm4: consistency fixed, agent play in the data, heads calibrated).**
Consistency 0.86 against wm3's 0.37; W error 2.3 frames; death P24/R93 raw, P76/R45 and a
Brier of 0.018 once calibrated. The latent search still clears nothing on 1-1 (dies at 151
decisions where wm2 died at 46, or wanders to the cap).

`tools.wmprobe` asks the model the question the search asks -- unroll from one state along
the route, along noise, along a held input, and read the W predicted for each. It clears the
model and indicts the setting:

| probe (route vs random) | truly better | model says so | model right where true |
|---|---|---|---|
| 1-1, 6 steps | 19.5% | 67% | 100% |
| 1-1, 12 steps | 32.8% | 96% | 93% |
| 1-1, 24 steps | 46.9% | 97% | 93% |
| 4-1, 12 steps | 64.1% | 98% | 98% |
| 8-1, 12 steps | 76.6% | 96% | 95% |

Three things follow. The horizon was too short: at 6 steps the route is genuinely better than
noise only one time in five, and a 300-simulation tree over 12 actions reaches about depth 3.
The W magnitudes drift past the training unroll -- at depth 24 the model says the route has
wasted 15.1 frames where the truth is 0.8 -- and the search backs up a plain minimum across
depths, so a good deep node loses to a mediocre shallow one; unroll 12-16 is the fix. And
1-1, the level every stage B test has used, is the least discriminating level in the game,
the same one whose value data had the lowest spread.

**Stage C, second attempt (mc2: 120,754 pairs, 804 roots, 94% of them with true siblings):
15/32**, against 9/32 from the first run and 26/32 from the route-trained relv3. Same
labelling, same training budget; only the collection changed.

| level | mc 9/32 | mc2 15/32 | roots | dead% | W p90 |
|---|---|---|---|---|---|
| 1-1 | 1/4 | 2/4 | 73 | 2.3% | 164 |
| 1-2 | 0/4 | 0/4 | 80 | 25.2% | 512 |
| 4-1 | 4/4 | 4/4 | 145 | 5.6% | 232 |
| 4-2 | 0/4 | 0/4 | 155 | 11.3% | 512 |
| 8-1 | 0/4 | 2/4 | 74 | 34.4% | 512 |
| 8-2 | 1/4 | 4/4 | 138 | 8.8% | 288 |
| 8-3 | 3/4 | 3/4 | 75 | 14.8% | 512 |
| 8-4 | 0/4 | 0/4 | 64 | 57.5% | 512 |

Teacher-free labelling therefore reaches about 58% of the teacher's score. The three levels
that still win nothing -- 1-2, 4-2, 8-4 -- are the branch-heavy ones (warp pipes, the vine,
the maze), where a random prefix rarely leaves a position the agent can recover from; 4-2
has 155 roots and wins none of them.

**The split head (`relvalue --split`): 16/32**, the best teacher-free result so far, and the
two heads are complementary rather than one dominating:

| level | mc 9/32 | mc2 plain 15/32 | mc2 split 16/32 | dead% in the data |
|---|---|---|---|---|
| 1-1 | 1/4 | 2/4 | **4/4** | 2.3% |
| 1-2 | 0/4 | 0/4 | **2/4** | 25.2% |
| 4-1 | 4/4 | 4/4 | 4/4 | 5.6% |
| 4-2 | 0/4 | 0/4 | 0/4 | 11.3% |
| 8-1 | 0/4 | **2/4** | 0/4 | 34.4% |
| 8-2 | 1/4 | **4/4** | 3/4 | 8.8% |
| 8-3 | 3/4 | 3/4 | 3/4 | 14.8% |
| 8-4 | 0/4 | 0/4 | 0/4 | 57.5% |

The split head takes world 1 and loses world 8, in the order of how much death there is in
each level's labels, which is what an over-weighted death probability would do. Picking the
better head per level would give 19/32. Where it wins it is also fast: 1-1 in 33.1 s against
the search's own 31.7 s, 1-2 in 31.7 s against 32.4 s.

**Death charged as a price (`--split --dead-cost 128`): 18/32**, the best teacher-free
result. Tonight's progression is 9 -> 15 -> 16 -> 18, all of it from how the data is
collected and how a death enters the score; the labels and the training budget never changed.

| level | mc 9/32 | plain 15/32 | mixture 16/32 | price 18/32 |
|---|---|---|---|---|
| 1-1 | 1/4 | 2/4 | 4/4 | 4/4, 34.6-36.7 s |
| 1-2 | 0/4 | 0/4 | 2/4 | 2/4, 27.8 s (the search: 32.4 s) |
| 4-1 | 4/4 | 4/4 | 4/4 | 4/4 |
| 4-2 | 0/4 | 0/4 | 0/4 | 0/4 |
| 8-1 | 0/4 | 2/4 | 0/4 | 1/4 |
| 8-2 | 1/4 | 4/4 | 3/4 | 3/4 |
| 8-3 | 3/4 | 3/4 | 3/4 | 4/4 |
| 8-4 | 0/4 | 0/4 | 0/4 | 0/4 |

The price recovers most of what the mixture lost in world 8 without giving up world 1, and
whole-tree ranking rises to 86.7% (8-1 alone from 67% to 80%). What is left is 4-2 and 8-4,
which no variant has ever won: both need something a random prefix essentially never
produces -- the hidden vine, the right way through the maze -- so the spine pool never holds
a line that reaches them, and no amount of branching from the wrong place will make one.

The price itself, swept: **64 -> 16/32, 128 -> 18/32, 256 -> 18/32.** Below 128 the value no
longer fears death enough and loses 1-2 and 8-1; 128 and 256 tie, trading 8-1 (1/4 against
2/4) for 8-3 (4/4 against 3/4).

Note for anyone reading the pair metrics: they pointed the wrong way here. The split head
scores 56.1% on siblings against the plain head's 57.2%, and wins by one level and by a lot
of seconds. Three times tonight the pair metric mispredicted play. Gate, not pairs.

What the pair metrics say, measured against the old absolute head on the same data: whole
tree 83.0% against 58.4%, true siblings 57.2% against 54.9%. The value separates subtrees
well and near-identical siblings barely, which is what play needs -- but it is also why
`relvalue --split` exists: a sixth of the branches die and every one is labelled 512, so the
few frames between two live lines vanish beside them and the W error never falls below ~130.

**Stage B, the depth bound (wm5: unroll 12, calibrated).** The search's lookahead was
measured rather than assumed, after two wrong assertions that it was too shallow. At 1000
simulations the line it believes in is **31 steps long** and the tree reaches 33, against a
model trained to unroll 12 -- and the probe shows W compressed exactly at the good end (8-1,
12 steps: route priced 4.5 where the truth is 0.3, a bad line 23.1 against a true 25.6). A
long imagined line therefore looks cheap precisely where the model has no right to an
opinion. Bounding the tree at the training unroll, on 8-1: dead after 24 decisions unbounded,
**197 at max-depth 12**, 44 at max-depth 8 -- so the optimum is the horizon itself, and the
result reproduces exactly.

Across levels at 1000 simulations, bounded to 12 (wm5): 1-1 dead at 84 / wanders, 4-1 wanders
both delays, 8-1 dead at 197 / wanders until the PAL timer kills it at 120.4 s, 8-2 dead at
89. Zero clears. The bound makes the search safe without making it purposeful, which is what
a compressed good end predicts: among twelve candidate moves whose true W all lie within a
few frames, the model's ordering is noise and the agent random-walks.

The reason for `wmdata --siblings` was that every trajectory gave one action per state, so
the model was never shown the comparison the search makes. **Correction, 2026-09-25 evening:
this experiment was invalid, not negative.** A sibling handed out on a later loop iteration
took the level drawn afresh at the top of that iteration, so its W targets were computed
against another level's route: in world_sib, 98% of the 1310 sibling groups carry mixed level
tags and at least 57% of their trajectories were labelled against the wrong level (the data is
kept in data_archive/world_sib_wrong_level_labels). What follows describes a model trained on
that, and says nothing about siblings. What was written at the time:
wm6, trained on 8000 sibling trajectories on top of everything else, is worse on both
measurements: near-sibling agreement on 8-1 falls from 91% to 82% (flip1) and 92% to 88%
(flip3), and the bounded latent search dies after 84 and 83 decisions where wm5 reached 197.
The compression at the good end is unchanged -- the route still prices at 4.3 against a true
0.3, flip1 at 4.8 against 3.7 -- so showing the model several first actions from one state
does not make it value them apart.

The best stage B configuration is therefore wm5: unroll 12, calibrated, tree bounded to 12.

**The compression was the loss, not the data.** 21-32% of the world-model targets at depth 12
are under one frame, so the model had seen perfect lines tens of thousands of times. But both
W losses were smooth L1 on W/16, quadratic below 16 frames: pricing a perfect line at 4 costs
0.031, a death at 400 instead of 512 costs 6.5. Trained through MuZero's value transform
(wm7, otherwise wm5's recipe exactly), the probe moves the right way: 8-1 twelve steps ahead,
the route priced at 2.1 against a true 0.3 (wm5: 4.5), a random line at 20.2 against 19.6
(wm5: 17.2); on 1-1 the route at 1.3 against 1.5. Near-sibling agreement rises to 93% and 97%.

**And single games cannot rank models.** wm5 alone went 197 decisions on one start and 1500
on another; a game that survives 256 decisions reached 12% of the level. `latent` now scores
each game by how far through the level it got (the route as ruler, evaluation only), over
eight starts. wm7 on 8-1: 0/8, mean 7% -- and six of the eight die at the same spot, 2-4% in,
after about 45 decisions. One hazard, failed every time: a diagnosis, not noise.

**Why it dies there (`tools.deathdiag`).** Replaying that game: at the fatal decision ten of
the twelve moves still survive, every jump among them, and the search walks right into an
enemy, rating that move the cheapest (7 frames against 10-16). The death head along the fatal
path says 0.09 at the step the game ends -- and 0.07 and 0.09 for jumping instead. It cannot
tell the two apart.

**Is it blind to enemies in general (`tools.deathprobe`)?** No. Over 2400 plain lines on five
levels, enemy deaths separate from survivals at AUC 0.89 (pits 0.80): 1-1 0.95, 4-1 0.96,
8-2 0.91, 8-3 0.86, 8-1 0.76. 8-1 is the weak level, and not for lack of data (12% of the
steps, as many deaths as 8-2). Averaging wm5, wm6 and wm7 does not help: on identical states
they score 0.72, 0.66, 0.68 on 8-1 and their average 0.70 -- their misses are the same misses.
(Resampling moves one model by about 0.04; smaller differences are noise.)

**Is the enemy even in the picture (`tools.percept`)?** Suspected from the frames -- sprites
are a few pixels of mid-gray among tree trunks and fence posts. Tested by rendering the same
3000 states of 8-1 five ways from one emulator step and training the same small classifier
on each, to predict an enemy death within six steps of walking right: engine gray 84 0.78,
gray 84 0.81, color 84 0.74, gray full 0.62, color full 0.65. **Refuted.** Neither color nor
resolution helps at this data scale (full resolution trains worse; one seed never learned).
And the telling number: a small classifier on one 84x84 gray frame reaches ~0.8 on 8-1 where
the world model, with four frames, reaches ~0.7. The information is in the input; the world
model does not extract it. The target is how it learns death -- one of several losses on a
48-channel latent unrolled twelve steps -- not what it sees.

Two follow-ups narrowed that. The same test on 8-2 as a control: one-frame classifiers reach
only 0.57-0.72 there, while the world model reaches ~0.88 -- so it does not under-extract in
general; 8-1's failure is specific. And the killer is identified from RAM: **a line of three
Goombas** walking toward Mario, 42/66/90 pixels ahead at decision 41, standing in front of the
trees (the "trunk" pixels in the zoomed frames were the Goombas). The death head gives that
collision 0.05 two steps before contact, against 0.01 for jumping.

**Weighting the death loss 4x (wm8) makes it worse**, on identical states: 8-1 0.68 -> 0.59,
8-2 0.85 -> 0.81, and by distance 1-2 steps 0.94 -> 0.95, 3-5 steps 0.83 -> 0.83, 6-10 steps
0.74 -> 0.67. The loss-competition hypothesis is refuted; overweighting costs the long range.

Across every model the pattern by distance is the same -- 0.94, 0.83, 0.74 -- a deterministic
game, so a perfect model would score 1.0 at every distance: the unroll loses the hazard as it
imagines forward. But these probes mostly measure deaths 6-10 steps out (122 of 174), while
survival is decided in the last 1-3 steps, where a jump still saves it, and that bucket held
five deaths. `deathprobe --near` closes in first and probes only the last three steps.

**Near a hazard the model is good**: enemy deaths within three steps separate at 0.83 (wm5),
0.88 (wm7), 0.90 (wm8), 8-1 included (0.84 / 0.92 / 0.90). Close-in enemy deaths average
p = 0.24 against 0.05 for survivals -- and the Goomba trio sits at 0.05, like a survival. It
is one blind spot, which ends six of eight starts only because it is the first hazard of 8-1.
wm8 separates it best (walking 0.31 against jumping 0.06 four steps on) and still dies there
in play: 0/8, mean 3% of the level.

**The backup is part of it.** latent.py backed up b = min(own estimate, best child), so an
expanded move kept its own first-step guess and nothing found below it could raise it -- the
C++ search backs up 4 + min over children. With `--backup children` the trio is priced right
(walking drops out of the six best moves; everything reaching the trio costs 36-42) -- but in
play it does not help: wm8 3% -> 3%, wm7 7% -> ~1%, often 0% of the level after hundreds of
decisions. The two rules fail in opposite directions: 'self' is myopic and walks in; 'children'
carries every line's far-future death probability (~0.15 on every line twelve steps out,
dead or not, x 512 = ~75 frames), which swamps the few frames of W that reward going forward,
so it retreats. Next: the children rule with a cheaper death (128).

**The sibling re-test, done properly** (wm9: wm7's recipe plus world_sib2, level tags checked
at 0.3% mixed): no gain. 8-1 twelve steps ahead the route at 2.0 against 0.3 (wm7 2.1), one
change 2.6 against 3.7, three changes 3.8 against 7.4 (wm7 4.7); near-sibling agreement 96%
and 96% (wm7 93% and 97%); death AUC 8-1 0.67, overall enemy 0.75 (wm7 0.68, 0.77). A real
negative this time.

**Collision data (world_app, 8000 trajectories, 39% die; wm10)** gives the sharpest death
head yet -- close-in enemy deaths 0.94 overall, 0.95 on 8-1, mean p 0.52 against 0.05 for
survivals -- and it sees the Goomba trio (walking 0.10 two steps out against 0.01 for
jumping). In play it dies earlier: 8-1 mean 3%, nearly every game at 2%.

**Why: a held A does not jump.** The 2% death is a Buzzy Beetle, reached after twenty moves
of holding run + jump. In Super Mario Bros a jump fires only when A is newly pressed; checked
in the emulator from that state, holding run + jump never leaves the ground, while releasing
for one step and pressing again jumps. The model is told nothing about the button before its
four frames, so it assumes every press jumps. Separated cleanly: running into the Beetle with
no A pressed, wm10 sees the collision (0.45 then 0.76 at the steps it dies); holding A, it
predicts 0.01-0.04 -- it believes Mario is in the air. Perception was never the problem.

`--prev` (wm11) adds the previous action to the latent at the root: used (it moves the
latent 4-6% between A held and released) but too weak -- the same Beetle at 0.01 holding A,
8-1 mean 4%. `--edge` (wm12) gives the dynamics the previous action as planes at every
step of the imagined line, so "A newly pressed" is in its input rather than inferred.

**--edge works.** wm12 at the Beetle: holding A now costs 0.22 at the step it dies (wm10 0.04,
wm11 0.01) against 0.10 for releasing first, while running into it with no A reads 0.46 --
the model knows a held A does not jump. Close-in death detection 0.95 overall (mean p 0.62,
survivals 0.07). It still false-alarms after the safe jump (0.42-0.78 at steps 4-6).

**The policy head had never been trained.** wmtrain had no policy loss, so the latent search
explored from random weights; the emulator search clears 28/32 with the net's prior and 8/32
uniform. With the policy net's prior at the root (`latent --prior-net`, deeper nodes uniform),
8-1 over eight starts:

| model | own (untrained) head | net's prior at the root |
|---|---|---|
| wm7 | 7% | 8% (3-22%) |
| wm11 | 4% | 7% (3-8%) |
| wm12 (--edge) | 5% | 8% (3-21%) |

With a trained prior every game passes the Buzzy Beetle (2%) and the Goomba trio (4%), and
nearly all die at the next hazard, **~7%: a Piranha Plant** (enemy 0x0D) rising out of a pipe
into Mario's jump -- a timing hazard, the plant's phase only partly visible in four frames.
Delay 24 reaches 21-22% for every model (the plant's cycle is favourable there). Once the
prior is good, the world models' differences stop showing on 8-1: they all meet the plant.
`wmtrain --distill` teaches the world model's own head that prior (wm13), so the agent is one
model and its deeper nodes get a prior too.

**The distilled head is too blurry to lead** (it agrees with the net's top move 64% of the
time, entropy 1.07 against the net's 0.66), so on its own it plays 8-1 at 4%. As the prior
*below* the root it helps: net at the root, distilled head deeper, 8-1 over eight starts
3 17 7 16 7 7 7 7, mean 9% -- the best there, two games past the Piranha Plant.

**Branching from the agent's own failures (world_fail, wm14)** gives the sharpest death head
yet (P40/R97 twelve steps out) but no gain in play on held-out starts: 6% with its own head,
the same as wm13.

**Stage B's first clear.** wm13, net at the root, distilled head below, 1000 simulations:
**1-1 won from start delay 5 in 35.7 s** (445 decisions), the search planning entirely inside
the world model and the game stepped only by the moves played; replayed in stable-retro, every
frame matched (docs/media/smbzero_latent_1-1.mp4). 4-1 from four starts reaches 34, 34, 34 and
23% of the level (before the trained prior it wandered at ~10%).

**Stage B gate (the stage A and C protocol: 8 levels x 4 starts, 1000 simulations; wm13, net
at the root, distilled head below): 3/32.**

| level | won | mean progress | what stops it |
|---|---|---|---|
| 1-1 | 3/4 (35.3-35.7 s, all three verified in stable-retro) | 77% | one start stalls |
| 1-2 | 0/4 | 15% | |
| 4-1 | 0/4 | 31% | a Piranha Plant (three of four starts, x ~ 1850) |
| 4-2 | 0/4 | 26% | |
| 8-1 | 0/4 | 7% | a Piranha Plant |
| 8-2 | 0/4 | 10% | |
| 8-3 | 0/4 | 14% | |
| 8-4 | 0/4 | 1% | the castle's first hazard |

Against the same 32 games: the emulator-lookahead search with the learned value 26/32, stage
C's teacher-free value 18/32. The first working search without the emulator, not yet a rival.

Piranha Plants are the common wall. Branching from 8-1's deaths (wm14) taught those 8-1
moments -- the fatal jump goes from 0.00 to 0.69 -- but not the concept: on 4-1's plant deaths,
which it never saw, it gives 0.01-0.05. Next: the same thing from every level's deaths (the
gate's lost games), retrain (wm15), and replay all eight levels on starts none of that data
came from.

**Stage C gate, first attempt (relv_mc, 120k teacher-free pairs, 1000 simulations): 9/32**
against relv3's 26/32. The failures are not spread evenly -- a level needs both volume and
spread in the data, and only two had both:

| level | roots | W p90 | dead% | gate |
|---|---|---|---|---|
| 4-1 | 1175 | 108 | 1.7% | 4/4 |
| 8-3 | 723 | 512 | 10.3% | 3/4 |
| 8-2 | 582 | 144 | 4.9% | 1/4 |
| 1-1 | 1172 | 52 | 1.6% | 1/4 |
| 1-2 | 78 | 512 | 25.2% | 0/4 |
| 8-1 | 110 | 512 | 37.0% | 0/4 |
| 4-2, 8-4 | 0 | | | 0/4 |

1-1 is the instructive one: as many roots as 4-1 and still 1/4, because its branches almost
always recover (52 frames at the 90th percentile against 4-1's 108) so nearly every label
is the same number and there is nothing to rank -- three of its four runs are "too long",
the search wandering. Both shortfalls are in how the data was collected, not in dropping
the teacher: mc2 draws pool lines inversely to the roots their level has given, and triples
the random reach for a level whose branches recover more than nine times in ten.

Note on what stage C claims: the labels are teacher-free (realised times from the agent
racing itself), but the agent that generates them plays on relv3, which was trained on
route labels. That is iteration 0 of expert iteration, not a teacher-free agent -- the loop
closes when relv_mc2 replaces relv3 and the data is regenerated.

Also recorded: the value's ranking test pairs two nodes of one tree at equal depth, not two
children of one node -- the shards carry no parent id. Cousins, not brothers; the README
said the stronger thing and now says this one.

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
