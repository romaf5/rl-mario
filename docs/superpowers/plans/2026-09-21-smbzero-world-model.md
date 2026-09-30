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
the search's own 31.7 s, 1-2 in 31.7 s against 23.1 s (see the correction below).

**Death charged as a price (`--split --dead-cost 128`): 18/32**, the best teacher-free
result. Tonight's progression is 9 -> 15 -> 16 -> 18, all of it from how the data is
collected and how a death enters the score; the labels and the training budget never changed.

| level | mc 9/32 | plain 15/32 | mixture 16/32 | price 18/32 |
|---|---|---|---|---|
| 1-1 | 1/4 | 2/4 | 4/4 | 4/4, 34.6-36.7 s |
| 1-2 | 0/4 | 0/4 | 2/4 | 2/4, 27.8 s (the search: 23.1 s) |
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

**Correction (timing):** the search's per-level table in the README (1-2: 32.4 s) includes
1-2's automatic walk into the intro pipe; measured the way SMBZero's games are, from first
control to the next level, the search's 1-2 route is 23.1 s (1-1 is 31.7 s either way). So
SMBZero's 1-2 clears (24.6-31.7 s) are slower than the search, not faster as first written.

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

**Stage C, iteration 1 (the data's player on the teacher-free value, not the teacher's): 17/32**
against 18/32 -- within noise: 1-1 2/4 (was 4/4), 1-2 3/4 (was 2/4), 4-1 4/4, 4-2 0/4, 8-1 1/4,
8-2 3/4, 8-3 4/4, 8-4 0/4. The value's player no longer needs the teacher, and its value ranks
true siblings better (63.8% against 57%). What it cannot do is learn a level it cannot finish:
mc3 holds no 4-2 or 8-4 at all, because no teacher-free player has ever won either. Those two
need first solutions found (the search engine's Go-Explore does this), not better training.
The policy prior is the part still descended from the teacher.

**Closing the policy loop (zeroloop --train policy, 4 h, on the teacher-free value): 12/32, a
regression from 17/32.** 1-2 improves (4/4) but 8-3 falls to 1/4, 8-1 to 0/4, and 8-2's clears
slow from ~48 s to 77-79 s. The loop's own games show why -- 57% won in rounds 1-4, then 36%,
29%, and 21% and 14% in rounds 13-14, the policy loss hardly moving (0.74 -> 0.72): the policy
drifted rather than learned, chasing the visit counts of a search barely stronger than itself.
Plain expert iteration has nothing to stop that; AlphaGo Zero's evaluator does -- a new policy
replaces the old only if it beats it. The loop now keeps every round's checkpoint (this run
kept only the last, so its round-4 peak is lost).

**With an evaluator (`zeroloop --gate 0.15`, zpol2): 15/32** -- the collapse is gone (12 -> 15)
but the original policy still wins (17). The best round was 9 (64% of its own games); after it,
every round trained from that best policy came out worse (36-43%) and was sent back, four in a
row. Training on this search's visits does not just add noise, it pulls the policy down: a
search is only as good as its value, and on some levels the teacher-free value makes a weaker
search than the one the original policy learned from (8-3: 4/4 with the original policy, 1/4
after self-training, in both runs). The same search scores 26/32 on relv3 and 17/32 on the
teacher-free value -- **stage C's bottleneck is the value**, and self-improvement of the policy
waits on it.

**The teacher-free value on both teacher-free datasets (mc2 + mc3, 240k pairs, 50k steps):
20/32**, the best without a teacher -- 1-1 4/4, 1-2 4/4, 4-1 4/4, 4-2 0/4, 8-1 2/4, 8-2 3/4,
8-3 3/4, 8-4 0/4. Against relv3 (route-trained, 26/32) level by level: on the six levels it has
data for, **20 of 24 against the teacher's 21 of 24** (and 8-1 2/4 against 1/4). The whole
remaining gap is 4-2 and 8-4 (0 of 8 against 5 of 8), the two levels no teacher-free player has
ever finished, so no teacher-free data exists for them. What stage C still lacks is not a
better value but first solutions to learn from -- exploration, which the search engine's
Go-Explore already does without a teacher. To be exact about what is teacher-free here: every label; the player that gathered mc3
(its own teacher-free value); not the player that gathered mc2 (relv3), and not the policy prior
(zero8), which still descends from the teacher.

**Iteration 2 (the 20/32 value as the data's player, mc4; value on mc2+3+4, 60k steps): 16/32**,
four fewer -- 8-1 falls to 0/4, 1-1 and 1-2 to 3/4. More data from a slightly different player
did not help. Its spines, at 2000 simulations and 8 rounds per level, finished neither 4-2 nor
8-4 in 120+ attempts each.

**Go-Explore's first ways (tools.firstways, 600 s per start, cells of area / position / camera /
tiles, the level order its only other input):** 4-2 found from 3 of 3 starts (721, 745, 549
decisions -- up the hidden vine to world 8), 8-4 from 3 of 3 (1039, 968, 869 -- through the maze
to the axe). mcdata --seed-lines learns those levels from them: the agent branches from the
first ways as from its own lines, and from a root on one the line's own continuation is kept as
a sibling, so the value sees the hidden way beside the agent's lines that miss it. Measured
against the 20/32 recipe with one change: mc2 + mc3 + the seeded 4-2 / 8-4 data (mc5).

**Result: 15/32, and still 0/4 on both 4-2 and 8-4** -- every game on them wanders to the cap
rather than dying -- while the rest slips (1-1 2/4, 8-2 1/4; the baseline had 20/24 there).
The policy in the gate (zero8) was trained on the search's routes, vine included, and clears
4-2 4/4 with relv3; so the policy knows the way and the teacher-free value, even given 4-2
data, steers it off. Go-Explore's first ways are long and wandering (549-745 decisions on
4-2 against the polished 359), and a value that learns W relative to them may not separate
the right warp pipe, or the right maze pipe, from the wrong ones. Next: where one 4-2 game
actually gets lost.

**Where 4-2 gets lost: the hidden block.** The route bumps it at decision 131 (x=1031) and the
vine spawns; the seeded value's game (897 decisions) never spawns it, wandering at x 1057-1150.
Started from the route's own states before the block (8 games each, 1000 simulations, 80
decisions, tools/vinetest.py), the teacher's value (relv3) bumps it 8/8 from 71 decisions out
and 6/8 from 11; the seeded one (relv_mc235) 0/8, 0/8, 0/8 and 1/8. Scoring 73 twelve-step
lines from 11 decisions out, it rates the lines that spawn the vine 109-141 frames lost and
random wandering 89-114 (the teacher: 5-7 against 13-41) -- it has not learned that the vine
is good. The data says why: branch wins in mc5 were 74%, nearly all from roots after the vine,
because a won branch joins the pool as a line and the pool fills with post-vine lines. Before
the vine there were only the three seed lines per level, and a root within one branch's depth
(60) of the bump is a few percent of a 700-decision line.

**Frontier roots (mcdata --frontier, Go-Explore's backward algorithm).** Per seed line, keep
the earliest root a branch has won from; draw 75% of a seeded level's roots from the 60
decisions before it. The frontier walks back through what the agent can already do and stops
where it cannot (the vine, a maze pipe), so the data piles up there: the seed line's
continuation over the bump beside the agent's branches that miss it. And --cap-by-line: a branch
past (its line's remaining decisions + 128) is HOPELESS whatever it does, so it stops there.
Run: mc6 (4-2, 8-4, 24k pairs), value on mc2 + mc3 + mc6 against the 20/32 recipe.

**Round 1 (mc6, 53k pairs; relv_mc236): the frontier worked, the value did not learn.** On
the seed lines the vine spawns at decisions 281 and 252 (enemy slot 5 -- the first check
looked at slots 0-4 only and misplaced it); the frontiers stopped at 284 and 264, just after
each bump, as designed: nothing the agent played from before a bump ever won. Vine test 3/32
bumps (1, 0, 1, 1 of 8), against 1/32 for relv_mc235 and 27/32 for the teacher. From 11
decisions out, all teacher-free values score every line 115-210 frames lost (the teacher: the
vine lines 7): nearly every branch from before the bump fails and is labelled 512, so the value
learns that the region is bad whatever Mario does, and prefers the vine lines only weakly
(mc236 0.68, mc235 0.83, the teacher 0.93). Its held-out sibling ranking on 4-2 is 53%.
Gate: **19/32** -- 1-1, 1-2, 4-1, 8-3 4/4, 8-2 3/4, 8-1 0/4, 4-2 and 8-4 0/4 (4-2 all too
long): the 20/32 recipe's level within noise, so the frontier data no longer costs the other
levels (the first seeded data did: 15/32), and does not yet win 4-2 or 8-4.

What teaches the contrast is a seed continuation that shows the vine within the search's
horizon (~12) next to the agent's branches from the same root -- roots within ~12 decisions
of the bump, a fifth of a 60-decision window, one continuation against four branches.
Round 2: `--frontier-window 15 --seed-copies 4` (about 16x more of it), the mc236 value as
the player, on mc2 + mc3 + mc6 + mc7.

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

**One turn of learning from its own failures, measured on new starts (12/27/42/57): worse.**
wm15 = wm13 plus 12000 branches from the gate's 22 deaths on six levels (60% die). Its death
head is the sharpest yet (next step P39/R90 against wm13's P15/R73; twelve steps P49/R92), and
in play:

| level | wm13 | wm15 |
|---|---|---|
| 1-1 | 3/4, 87% | 1/4, 53% |
| 1-2 | 0/4, 39% | 0/4, 32% |
| 4-1 | 0/4, 23% | 0/4, 25% |
| 4-2 | 0/4, 26% | 0/4, 28% |
| 8-1 | 0/4, 9% | 0/4, 10% |
| 8-2 | 0/4, 6% | 0/4, 6% |
| 8-3 | 0/4, 19% | 0/4, 15% |
| 8-4 | 0/4, 1% | 0/4, 4% |
| total | 3/32, mean 26% | 1/32, mean 22% |

The loss is 1-1, where wm15 pressed down on the fourth pipe and went into the underground coin
room -- a place no data has ever shown the model -- and stood there until the cap: a
model-based search goes wherever its model is optimistic, including where it has never been.
Everywhere else the two are within noise, and on 8-2 they play move for move the same games.
That is the telling part: with the policy net's prior at the root and values scaled by 32
frames (a move 4 frames better gains 0.125 in q), the prior makes nearly every choice and the
world model's opinions rarely change a move -- which would explain why improving the model has
stopped moving play. Next: the same games with values four times as loud (--scale 8).

**Louder values (--scale 8, wm13, same new starts):** 1-1 1/4 and mean 46% (default 3/4, 87%),
4-1 mean 23% with a wider spread (default 23%). The world model does get a say -- it changes
moves, wins a start the prior lost, loses two it won -- but its say is not better.

**Decision quality (`tools.agree`), which settles it.** At 192 states the latent agent really
visited, three pickers each choose a move, judged by the real game (does the move survive,
then any held input for 20 steps) and by the emulator search's visits:

| | all 192 | danger (43: some move dies) |
|---|---|---|
| survives -- prior / latent / emulator | 81% / 81% / 82% | 14% / 14% / 21% |
| emulator's visit share on the pick -- prior / latent | 0.33 / 0.33 | 0.23 / 0.22 |
| same move as the emulator -- prior / latent | 53% / 49% | 65% / 53% |

**Planning inside the world model chooses no better than the policy alone** -- it survives
exactly as often, and where it departs from the prior it departs from the emulator too. The
probes (close-in death AUC 0.95, the held A, the plant on the states it was shown) measure
whether the model knows things; this measures whether knowing them improves a choice, and it
does not. That is why every model improvement today left play where it was. And the danger
states are mostly lost already (even the emulator saves 21%): the mistakes that matter are made
before the danger is visible, where the model's long-range W is what decides, and that is the
part of it least trained on the agent's own play.

**W from the agent's own value (`wmtrain --value-teacher`, wm16 = wm13 learning W from stage
C's teacher-free value, relv_mc2d, instead of from the route): worse.** Decision quality at
the same 192 states -- survives 80% (prior 81%), in danger states 9% (prior 14%, wm13 14%);
same move as the emulator 32% overall and 16% in danger (prior 53% and 65%). The model's
opinions got louder -- it now overrides the prior far more often -- and wrong more often: it
can only be as good as the value it learns from, and that value orders near-identical
siblings at 57%. wm17 learns W from relv3 instead, the best value there is (26/32 in the
emulator search, 94% on its ranking test): if its choices then beat the prior's, the idea is
right and the teacher was the problem.

**wm17 (W learned from relv3):** decision quality identical to the prior -- survives 81% and
81%, danger states 14% and 14%, emulator visit share 0.33 and 0.33. Three sources of W, one
pattern: the route (wm13) and the best value (wm17) make the latent search agree with its
prior; a noisy value (wm16) makes it worse; none makes it better. The value is not what
stage B lacks. The emulator agent's edge in danger states (21% against 14%) is partly a veto
it has and stage B does not: before playing a move it checks in the real game that it
survives 24 steps, and takes the next candidate if not. Next: that veto, from the world
model's own death head -- strong close in (0.95), which is where a veto works.

**The veto (`latent --safety 12`): no change.** wm13 and wm15 with it: survives 81%, danger
14% -- the prior's numbers exactly. The calibrated death head passes 0.5 on only about half of
the real deaths at this horizon, so the veto seldom fires; and most danger states are lost
already (the emulator itself saves 21%), which leaves little for a veto to rescue.

**Where stage B stands.** Five separate attempts to make planning inside the world model beat
its own prior at a single decision -- a learned W from the route, from the teacher-free value,
from the best value; louder values; a death veto -- and none has: the search agrees with its
prior, or does worse. The probes say the model knows a good deal (close-in deaths 0.95, the
held A, the plant where it was shown); the decisions say that knowledge does not yet change a
choice for the better. The emulator search's edge is exact futures. The principled remaining
route is MuZero's own: train the model's policy and value on the latent search's own visit
counts and returns, so value and dynamics are consistent with each other -- a new phase of
work rather than another knob.

**A bug under every stage B result: the latent search expanded from latents it had not
computed.** A wave picks up to 32 edges and imagines them in one call. A child created earlier
in the same wave had no latent yet and b = 0 -- which reads as perfect -- so the next pick
walked straight into it and expanded it from whatever the latent buffer held (zeros, or a
previous decision's nodes). Measured on wm13 at three 1-1 states, 1000 simulations: **90% of
all expansions** came from such a parent; the tree "reached depth 12" because each wave was one
chain of garbage. Fixed: a node is pending until imagined, pending nodes cannot be picked, and a
wave ends when every way down is blocked -- 0 stale expansions, ~12 picks per wave, the tree
5-7 deep for real, the same ~1.1 s per decision. So every stage B verdict above, the gate's
3/32 and "the search chooses no better than its prior" included, measured a broken search.
Re-measuring from the gate.

**Re-measured (blatent: 32 games at once, same rules, identical visits to the fixed latent.py,
15 ms per decision against 1.1 s -- the gate in 6.5 minutes): 2/32.** 1-1 2/4 (34.2 s, 34.9 s),
the rest 0/4, progress per level where it was (1-1 65%, 4-1 31%, 4-2 26%, 8-1 7%). The bug was
real and not the reason: stage B is weak with a sound search too. To find which part of the
model fails, blatent can take the real game as an oracle for one part at a time: `--real-events`
(deaths and finishes from the game, not the event heads), `--real-value` (W from the stage A
value on the real screens, not the value head), or both -- which also tests, for the first
time, whether the latent search's own rules (PUCT on W, scale 32, fpu 0.5, self backup) can
win with a perfect model.

**With a perfect model the default rules die anyway -- the 'self' backup.** Both oracles (real
events, relv3 on the real screens), 8-2 from delay 20: dead at decision 28, the same place the
model-only search dies. The trace: at 27 the chosen move's subtree (769 visits, 3 deep) held
all twelve next moves, every one a real death -- and the move still cost 1 frame, because
'self' backs up b = min(own estimate, best child): a node's first guess hides whatever is found
below it, so only good news travels up. The same game with `--backup children` (b = best
child, as the C++ search does): alive and at x=1363 after 140 decisions. Earlier, 'children'
lost to 'self' in play -- but under the stale-latent bug, and with the model's ~0.15 far-future
death probability x 512 on every live line making it retreat. Re-measuring stage B with
'children' (300 and 1000 simulations, death cost 512 and 128) and the oracles at 300.

**Factorization, first half (blatent, 32 games, 'children' backup):**

| search | won |
|---|---|
| model only, 300 / 1000 simulations / 1000 at death cost 128 | 3 / 2 / 3 |
| real events (the engine's death rule), model value, 300 | 3 |
| real events (old rule) + relv3 on the real screens, 300 | 4 (1-1 4/4 only) |

Knowing the deaths inside the tree changes nothing. In the real-events games, the last state
from which some held input survives 24 steps came a median 4 decisions before the death
registered (3-7; 7% earlier than 6): the fatal commit is inside the tree's reach for the line
played, but the doomed move's other lines take up to 24 steps to die -- past a tree 5-7 deep --
so the move looks alive unless the value itself knows doom. Stage A has the check that catches
exactly this: before committing, some held input must survive 24 steps in the real game. Duplicate
pruning (the C++ search's other rule) never fires in play: every action writes its own joypad
byte. Next: the same oracles with stage A's veto and with the net's prior below the root.

| search (300 simulations, 'children') | won | how the rest end |
|---|---|---|
| relv3 on the real screens, model events | 4 | mostly dead |
| real events (engine rule) + relv3 | 4 | mostly dead (the same games as the old rule) |
| real events + relv3 + stage A's 24-step veto | 6 | 1 dead, 25 too long |

The veto nearly ends dying (4-1 32% -> 76% of the level, 8-2 13% -> 67%, 8-4 14% -> 32%) and
exposes the next wall: the agent survives and does not get anywhere in time. So stage A's
26/32 needs both the veto and something that drives progress, which the latent search with a
perfect value does not have at 300 simulations. Candidates: the prior below the root (the
model's distilled head against the net on the real screens -- measured next), tree reuse, and
the simulation count.

**And a ruler that was never the same.** The stage A and C gates (tools.levels -> eval) give a
game 2.5x the route's decisions; every stage B gate, latent.py's and blatent's, gave 1.5x. It
mattered little while stage B died; with the veto "too long" is the main ending, so the rest of
the factorization runs at 2.5x. Also different: stage A's 1000 simulations are new visits on
top of the subtree it keeps from the last decision; the latent search starts fresh each time.

| search (300 simulations, 'children', 2.5x cap) | won | how the rest end |
|---|---|---|
| model only | 3 | mostly dead (as at 1.5x) |
| model only, tree reuse | 0 | dead, fast |
| model + stage A's veto | 5 (1-1 3, 8-2 1, 8-3 1) | too long; 8-1 still dies 4/4 |

With the veto, the model does about as well as the perfect value (5 against 6): dying is
fixed by the veto, and the wall is progress. Where it stalls: 4-1 pushes right into a wall at
x=1634 for a thousand decisions; 1-2 holds left+B at x=350 for four hundred. At the 1-2 spot the
only way out is a running jump (x=354 is a wall); relv3 on the real screens does say so -- at
depth 10 the jump lines cost 9 frames, everything else 30 -- but at depth 3-6, where this tree
lives (5-7 deep at 300 simulations), the moves differ by 2-5 frames: 0.08 of q against a prior of
0.76 on left+B. The latent search also floored W at 0, so here every root move tied at exactly
0.0 (`--no-floor` keeps W below 0, as the C++ search does; it separates them, by 2.5 frames,
and the prior still wins). The search needs depth. Tree reuse, the C++ search's way to depth,
makes the model search worse (0/32): the kept subtree was imagined from the previous screen, and
its accumulated visits hold the agent to a future that has already drifted -- MuZero does not
reuse trees either. Next: 1000 simulations with the veto.

**1000 simulations with the veto: 6/32** (1-1 2, 4-1 1, 8-2 1, 8-3 2; 8-3 72% and 4-1 56% of the
level), the wins slow (63-69 s against stage A's ~40). **wm17 re-gated** (W learned from relv3,
judged earlier under the stale-latent bug; model only, 1000 simulations, 2.5x): **4/32**, the
best model-only stage B -- 1-1 3/4 and 8-3 in 41.7 s.

**The factorization, complete (blatent, 32 games, 300 simulations, 'children', 2.5x cap):**

| latent search | won |
|---|---|
| model only (wm13) | 3 |
| model + stage A's veto | 5 (6 at 1000 simulations) |
| real events + relv3 on the real screens + veto | 7 |
| + the net's prior below the root, on the real screens | **11** |
| stage A: the C++ search, same value and prior, 1000 new simulations on a kept tree, veto | 26 |

Re-gated world models (model only, 1000 simulations): wm12 0, wm14 0, wm15 3, wm16 2, wm17 4.
Reading it: the veto turns deaths into slowness (+2-3); the prior below the root is worth +4 --
the model's distilled head, which agrees with the net's top move 64% of the time, is a real
weakness; and even with every part of the model replaced by the real game, 11 of stage A's 26
remain: the rest is the search's budget and shape (stage A's 1000 simulations are added to a
subtree kept from the last decision, which the real game makes exact). The four 8-1 "deaths"
are the level timer, 1500 decisions in -- slowness again.

**The veto, learned (`smbzero/survival.py`).** Stage B may not ask the game, so a net reads the
real screen and the move before it and says, per move, whether some held input then survives
24 steps; trained on emulator labels at 60k states of the latent agents' own games (and short
random branches off them, the last 30 decisions before a death drawn half the time), used by
blatent --veto-net exactly as the real veto is used.

**surv0 (60k states, 15 minutes to label): AUC 0.985, 92.8% of the dying moves caught, 2.3% of
good ones refused -- and in play nothing: wm13 2/32, wm17 3/32, dying as without it.** At the
fatal commits of wm17's games -- the played move dies, another would have lived -- surv0 let
9 of 10 through, several at p = 1.00. Its held-out score came from the easy states: 74% of
states are safe whatever the move, 20% lost whatever the move, and the screen alone says which;
only 6.1% (3685) are mixed, where the move decides, and that is the veto's whole job. Next:
half of every batch from mixed states, scored on held-out mixed states only (surv1).
Also: many 8-1 and 8-2 games were already past saving eight decisions before the end (no held
input survives 24 steps from any move) -- no veto of this kind helps there.

**wm13's and wm17's policy heads agree with the net's top move ~50% of the time at depth 0**
-- where the latent is encoded from the real screen -- and 40-58% down to depth 12. The loss is
the head (one linear layer on the latent, distillation weight 1.0), not imagination; and the
factorization puts the net's prior below the root at +4 of 32.

**surv1 (half of each batch mixed): on held-out mixed states AUC 0.867 (surv0's recipe 0.862),
72% of dying moves caught, 10.6% of good ones refused (15%); train loss 0.002 -- it memorises
the ~3500 mixed states it has. In play: 3/32, as without a veto.** More mixed states:
`survival make --boundary` finds each lost game's last savable decision and samples around it
(23.5% mixed against 6%); 60k of them next (surv2).

**Frontier round 2 (mc7, player relv_mc236, window 15, seed copies 4), 18k pairs in: both vine
lines' frontiers now sit before their bumps** (275 against 281, 250 against 252) -- the agent's
own branches win from before the hidden block. Round 1's frontiers never got there.

**Round 2's value (relv_mc2367: mc2 + mc3 + mc6 + mc7): 20/32, and 4-2 won for the first time
without a teacher** -- 1-1 4/4, 1-2 3/4, 4-1 4/4, **4-2 1/4 (61.5 s)**, 8-1 0/4, 8-2 4/4, 8-3
4/4, 8-4 0/4. Replayed (delay 20, 764 decisions, the same 61.51 s) and **verified in
stable-retro: all 3076 frames, ending in 8-1**. It bumps the hidden block itself at decision 171
(x=1030, where the route does), wanders, climbs into the warp zone at 559 and takes world 8's
pipe at 763 (docs/media/smbzero_teacherfree_4-2.mp4). Vine test 4/32 (from 3). 8-4 still 0/4
(all too long), 8-1 fell from 2/4 to 0/4. Round 3 (mc8, the mc2367 value as the player) runs.

**Round 3 (mc8, player relv_mc2367; value relv_mc23678 on mc2 + 3 + 6 + 7 + 8): 21/32, the best
teacher-free gate** -- 1-1 4/4, 1-2 4/4, 4-1 4/4, 8-1 2/4, 8-2 4/4, 8-3 3/4 -- **but 4-2 back to
0/4** (all too long) and the vine test 0/32; 8-4 still 0/4. With one 4-2 win in four and none in
the next four, the rate is noise-level; next: 4-2 from 16 starts for both values, then round 4.

**4-2 from 16 starts (delays 2-62): round 2's value 1/16 (delay 22, 59.8 s), round 3's 0/16.**
Most games never bump the vine (12/16 and 14/16); about half run on past it to the end of the
level (x 2800-3300), where the ordinary exit leads off the route, and wander there until the cap;
of the games that do bump it (4 and 2), one climbs to world 8. The teacher-free value still
mostly steers away from the vine route the prior knows (relv3 with the same prior: 4/4).
Round 4 (mc9, player relv_mc23678) runs.

**Round 4 (mc9, player relv_mc23678; value relv_mc236789): 21/32** -- 1-1 3/4, 1-2 4/4, 4-1 4/4,
8-1 2/4, 8-2 4/4, 8-3 4/4, 4-2 and 8-4 0/4. **Vine test 5/32, all five from 11 decisions out (5 of
8 there, round 2: 2 of 8), none from 21 or more.** The bump itself is learned; getting to it is
not -- from further away the games still wander before the block or run past it toward the exit
that leads off the route, a consequence hundreds of decisions away. Round 5 (mc10) queued, after
4-2 from 16 starts at 1000 and 2000 simulations.

4-2 from 16 starts with round 4's value at 1000 simulations: 0/16.

**Why the frontier left the vine too early, and the fix.** A frontier moved back on any single
win from a root before it. By round 4 two 4-2 frontiers sat at 50 and 186, far before the bumps
(281, 252) -- one lucky branch each -- so the data stopped piling up on the vine while the value
still bumped it only from close (0/8 from 21 decisions out). The backward algorithm moves the
start back only once the agent succeeds there reliably: `mcdata --frontier-rate 0.5` moves a
frontier back when the last 24 branches from its window win half the time, to the earliest root
won from. Round 5 restarts 4-2's frontiers just past each vine (284, 255, 218) with it.

**Priorities.** Stage C first: it is six levels solid and two short of a full game with no
teacher in its value. Stage B is parked: every knob tried after the bug fixes (backup, reuse,
top-k, simulations, learned veto, a teacher-free W) left it at 0-4/32, and the factorization says
the gap is the search's shape more than the model; the principled next step is MuZero's own --
training the model's policy and value on its own latent-search play, which blatent now makes
fast enough -- once stage C has a full game.

**surv2 (60k boundary states added, half of each batch mixed): held-out mixed AUC 0.981, 93.9%
of dying moves caught -- and 3/32 in play, dying as before; at the fatal-but-avoidable commits
it let 8 of 10 through.** The held-out split was the flaw: states were drawn around one moment of
one game many times over, so a random split put near twins on both sides. On boundary states
from games none of the nets saw (the surv2 gate's own games): AUC 0.63-0.68, 41-45% of the dying
moves caught, ~20% of good ones refused -- all three nets memorised ~450 death situations rather
than learning what kills. A learned veto needs thousands of distinct hazard encounters and an
evaluation on whole held-out games (survival.py now stores game ids and splits by game). Paused:
the CPU goes to stage C first.

**wm18 (wm17's recipe, the policy head with its own features and two layers, distillation weight
4): agreement with the net's top move 46-54% -> 53-62% at every depth (60% at the root); gate
4/32 (1-1 3/4, 4-1 once in 40.0 s -- stage B's first 4-1 without an oracle; 4-1 45% of the level
against wm17's 25%).** Better, and far from the net's own prior that the factorization credits
with +4.

**wm19 (wm18 with W learned from the teacher-free value relv_mc2367 instead of relv3): 0/32**
(1-1 0/4, 24% of the level; everything else dies early). The teacher-free value calls whole
hard regions hopeless whatever Mario does (W 100-200 frames on every line near 4-2's vine), and
a world model that copies it plans worse than one that copies the route-trained value. A
teacher-free stage B needs a sharper teacher-free value first.

**Depth by pruning (blatent --topk: below the root only a node's top-k moves under the prior;
at top 3 the 1000-simulation tree goes from 4.3 to 6.5 deep on average, 12 at most): wm18 0/32 at
top 3 and top 4 (4/32 with all twelve), dying everywhere.** Pruning by a prior that agrees with
the net 55-60% of the time cuts away the moves that save Mario; depth bought this way costs more
than it gives.

## Stage B: MuZero's loop (2026-09-28/29; stage C paused at the user's call -- stage B first)

Everything above trained the world model open-loop on other agents' data, its value copied from
another net. MuZero's loop closes it: the latent agent plays (blatent --record: root noise, moves
drawn from the visit counts), `mzdata` replays its games for the real screens and outcomes, and the
model trains on them (`wmtrain --init --sp-frac 0.5`: the policy on the search's own visit counts,
the dynamics and events on what really happened where the search went), the world data kept for
grounding; `tools.wmcal`; the 32-game gate keeps a model only if it is no worse (`mzloop`).

**MuZero's own value first: frames to go (`--tg`), n-step bootstrapped, so a consequence past the
tree could reach the root (W = tg(leaf) + 4 depth - tg(root)).** wm20 = wm18 + the head, 8k steps on
the route's frames to go: absolute error 66 frames -- and the search on it 0/32 (10% of the level;
the same model on W 2/32). W taken from it was off by 53 frames against the W head's 4: noise,
where siblings differ by a few. wm21 trained the differences too (`--w-tgdiff 4`): 6.4 frames --
but the shared trunk paid (one-step death recall 86% -> 45%, absolute error 427) and the search
went 0/32 again. The frames-to-go value is set aside until it can be learned without that cost.

**Two tracks now.** (1) The loop on the W search (mz1, from wm18, 4/32): policy from visit counts,
dynamics and events from its own games. (2) Is tree reuse the depth the latent search lacks? With
every model part real the latent search made 11/32 against the C++ search's 26; reuse is the one
structural difference left, and with real states it is exact: all oracles + `--reuse` measures its
ceiling. If it is the missing piece, the latent form is reuse with re-imagination -- keep the tree
and its visits, re-imagine every kept latent from the fresh real screen.

**Reuse: +2, not the missing piece.** All oracles + exact reuse (300 simulations): **13/32** against
11 without (the first eight finished were all wins -- the fast ones). Reuse with re-imagination on
the model (wm18, 1000 simulations; the kept tree's shape and visits, every latent, value, event and
prior recomputed from the new screen): **0/32** against 4 without -- the visits a kept subtree
carries hold the agent to its last plan even with fresh futures. Next: the C++ search at the same 300
simulations, to see whether 13 against 26 is the budget.

**mz1, iteration 0** (W search, from wm18): self-play 64 games -- 1 won, 56 dead, 7 at the cap; after
3000 steps on them (half of each batch) with the world data, W error 1.5 frames; gate 4/32, 28% of
the level (wm18: 4/32, 27%) -- kept, a tie.

**The C++ search at the same 300 simulations: 19/32** (1-1 4, 1-2 2, 4-1 4, 4-2 3, 8-1 0, 8-2 1, 8-3 4,
8-4 1) against the all-oracle latent search's 13 with reuse, 11 without -- so a gap remains at equal
budget, widest on 4-2 (3/4 against 0/4, all too long) and 1-2. Rule by rule the two now match (min
backup, q, the veto, the relative shift on reuse, the death rule, the move chosen) but for one: the
latent tree may not grow past 12 (the model's training unroll -- its honesty bound), which the
oracle inherited though the real game needs no such bound, and 4-2's approach is long. Measured
next: the oracle at --max-depth 64, and the model at 24 (past its unroll). mz1 iteration 1: 4/32,
29% (kept).

**Not the depth bound either.** The oracle at --max-depth 64 (reuse, 300 simulations): 12/32 (13 at
12) -- 8-2 4/4 now, 1-2 and 4-2 still too long 4/4, 8-1 dead 4/4. The model at --max-depth 24: 4/32,
the same games as at 12. (Checked that reuse was not starved of nodes by its 3x capacity: a reused
tree gets 287 of 300 new nodes per decision on average, 953 of 1000.) With every part real the
latent search stays 6-8 short of the C++ search at equal budget -- the rest is in mechanics (128
leaves a wave with virtual loss against 32 without, forced decisions committed at once, the timer
rule), none a clear lever -- and the model is far below that ceiling (4/32). The model is where
the wins are.

mz1: iteration 2 -- self-play 4/64, gate 3/32 23%, sent back; iteration 3 -- self-play 0/64, gate
3/32 23%, sent back. Four iterations have not moved the gate (4, 4, 3, 3).

**wm22 (wm18 + a convolutional policy head on the latent, 12k steps): agreement with the net 62-70%
at depths 0-3 (wm18 56-62%), about the same deeper -- and the gate 0/32.** Near-identical models
gate at 0-4 (wm17 4, wm18 4, wm20 2, wm21 0, wm22 0): the latent agent's wins are marginal, flipped by
small differences. The loop is paused after four flat iterations (its best model kept, resumable).

**Imagined frames (`wmdec`).** The factorization's biggest lever is the prior below the root (the
policy net on the real screens: 7 -> 11 of 32), and every attempt to distil that net into a head on
the latent stops at 60-70%. So invert it: the model draws the screen -- a decoder from the latent to
the 84x84 frame, trained on a frozen wm18's unrolled latents against the frames each step really
led to -- and the proven networks judge imagined stacks exactly as they judge real ones: the policy
net for the prior, the value net for W. Stage B's rule holds: nothing past the root is played. The
measure before any search: the policy net's top move and the value net's W on imagined stacks
against the real ones, by depth. (A 2000-step smoke run on a small slice: pixel error 13/255,
agreement 46% at depth 1 and 26% at 12, W 4.7 frames at 1 and 21 at 12 -- early.)

**Imagined frames, measured.** dec0 (decoder on frozen wm18, 20k steps): the policy net keeps its
move on imagined stacks 60/43/34/32% at depths 1/3/6/12, W off by 3.1/5.4/13.1/17.5 frames -- worse
than the latent head past depth 1 -- and at depth 0, the real screen encoded and drawn back, only
59%: the latent never had to keep the sprites (27/255 error on the moving pixels, 6 overall).
wm23 = wm18 trained jointly to draw its frames (Dreamer's reconstruction, moving pixels x5, 20k
steps): 63/64/47/41/38% at depths 0/1/3/6/12, W 2.3 -> 13.1 frames. Better, still below the latent
head (60/56/62/58/53) past depth 1. The search on drawn frames with the real nets: 1/32 (wm18 +
dec0; wm18's own heads 4/32).
wm23 gated: its own heads 3/32 (1-1 3/4, 88% of the level); on its drawn frames with the real
nets 0 of 31 games before it was stopped. Imagined frames are set aside at this fidelity.

**The C++ search without its veto: 24/32 at 300 simulations -- better than with it (19/32)**, and
faster (1-2 22.7 s against 25; 4-2 4/4 against 3/4). The check "some held input survives 24 steps"
is too conservative: it refuses moves the search had right. So the veto is not what makes the
emulator search strong, stage B does not need a learned one -- and every oracle comparison above
that carried it was handicapped. The clean question now: the latent search with every part real
(events, value, prior, reuse, no depth bound) and no veto, against 24.

**Answered: 3/32** (1-1 3/4, everything else dead or too long) -- against the C++ search's 24, and
the same latent search's 12 with the veto. The veto hurts the C++ search and is all that keeps the
latent one alive, so the two handle death differently. They did: the C++ search prices a death at
v_death = 4096, far above any living line; the latent search priced it at HOPELESS = 512 -- while a
living leaf costs up to W (<= 512) + p(dead) x 512. A line the model called 90% fatal cost ~960 and
looked worse than a certain death (512), and with the real game's deaths a dead end tied the worst
living leaf: the min backup walked into deaths. Every latent-search result so far carried this
(model-only and oracle alike). Fixed (`--v-death`, default 4096); re-running the all-oracle search
without the veto and wm18 on its own heads.

**Re-run: identical, game for game** (all-oracle no veto 3/32, wm18 4/32): dead children already
have q = 0, and a node whose children are all dead was too rare to matter. Correct, kept, not the gap.

**Head to head (tools: the C++ player and the all-oracle latent search, fresh trees, 300 simulations,
the states before a lost 8-2 game's fall).** The fall into the pit starts about decision 120 and
Mario is below the screen by 127 (every move then a real death -- the rule is right; the game only
takes the life at 174). Before it the two disagree at 109, 110, 112, 116-121 (at 109 the C++ search
goes right, the latent one waits). Two differences show: the C++ search keeps W below zero (a
child at -6, -3, -2: a line better than expected), the latent one floored it at 0 so good lines
tied at 0-4; and the C++ search spreads its visits (every root child ~10+, 21/152/24/76/...) where
the latent one pours 265 of 300 into one move. **No floor: all-oracle 4/32, wm18 4/32 (the same
games)** -- not the gap. **wm24** (96 channels, six transition blocks, conv policy head; 40k steps;
12-step death calibrated P99/R99): **1/32**. Next: 128 leaves a wave (the C++ player's), the
breadth its virtual loss buys.

**128 a wave alone changed nothing** (identical games): a wave ends where every way down is pending,
~12 picks in. **The head to head was also unfair to the C++ search**: started fresh mid-game it had
no history (its stack a repeated frame). Done fairly -- the C++ forest reset four decisions back and
the played moves committed, so both see the real last four frames -- the two agreed on most moves;
at 8-2's decision 121 the C++ tree proved a move dead that the latent tree, with the same 300
simulations, left alive: its picks had piled into the high-prior brothers and left one unexpanded.
The difference is the C++ search's **virtual loss** -- a simulation in flight counts on every node of
its path as a visit with q = 0 (q x n/(n+pending), sqrt(n+pending+1)), so a wave fans out. With it
(`--vloss --per-wave 128`) the latent search's visits match the C++ search's move for move, and it
proves the same move dead. **All-oracle, no veto, virtual loss: 9/32** (from 4; 1-1 4/4 at 32-36 s,
4-2 once in 30.6 s, 8-2 2/4, 4-1 2/4). wm18 with it: 4/32 (other games; the gate 6x faster). A fair
head to head on a lost 8-3 game now agrees on every move but one -- including both playing a
most-visited move already proven dead (4096). The rest of the 9-against-24 is in how games unfold
(tree reuse, most likely), which a model cannot use. So the factorization again, on the matched
search without reuse: all real / no real prior / no real events / no real value / the model.

**That factorization was void**: virtual loss without reuse makes the tree broad and shallow, and at
1-1's staircase Mario held run + jump for 700 decisions (a jump needs A pressed anew: release, then
press -- two steps the shallow tree never saw); every variant 0/32. Reuse is what gives the C++
search depth -- and for the model reuse fails however it is done: wm18 with virtual loss and reuse
0/32, with reuse re-imagined 0/32 (checked exact: every kept node's value equals a fresh tree's for
the same path, to 0.001). The kept visits lock the search onto whatever the model's errors favoured.

**Doom, not death.** The losses keep having the same shape: the death is sealed decisions before it
registers (a pit fall ~10), and a search must see every continuation die to know -- deep, broad, and
what imagination cannot prove. But doom is a property of a state (Mario airborne over a pit with no
ground in reach), and the survival nets separated all-die from all-live states easily; only the
per-move distinction failed. So move the death event to the doom point: `wmdata --doom` walks back
from each death to the last state from which some move then held input lasts 24 steps and ends the
trajectory at the step into doom, labelled dead. (An oracle version, `blatent --real-doom`, was too
slow to finish a game in 35 minutes.) A/B: 24k trajectories each from the same seeds, deaths at the
doom point (world_doom) or where they register (world_ctrl); a world model on each, same recipe.

**Doom A/B: wm25 (doom labels) 0/32, wm26 (death labels) 1/32.** 7571 deaths were moved to their
doom points; both models calibrate well (twelve steps P98/R98, P99/R97) -- and neither plays: doom
labels did not help, and both fall below wm18 (4/32), whose data included the agent's own play
(world_agent), which this data (route, sticky, random, approach) lacks.

**Where stage B stands (2026-09-29).** The latent search now matches the C++ search decision for
decision (virtual loss; head to head with real history). With every part real it makes 9/32 (with
reuse) against the C++ search's 24; the model alone stays at 0-4/32 through every change to the
model tried here -- a frames-to-go value (0/32), a better policy head (4), a bigger model (1), drawn
frames scored by the real nets (0-1), a learned veto (memorised), doom labels (0), and four loop
iterations on the old search (flat). What is left untried at scale is MuZero's loop on the corrected
search: mz2, from wm18, eight iterations, running.

mz2: iteration 0 gate 3/32, iteration 1 2/32 (self-play 0/64), both sent back -- stopped for RAM.

## Stage B on the RAM (2026-09-29, the user's call)

Every pixel-model failure came back to imagined futures not being exact enough. The RAM is the game's
exact state, so the agent reads the 2 KB RAM at play time instead of the screen (still no emulator
lookahead). Data: `wmdata --ram` (the RAM at every frame) and a `replay` mode (stretches of 1760 games
our agents played), 32k trajectories, 1.5M frames (world_ram).

**The RAM predicted directly (`ramwm`: its 1536 game-state bytes as bits, an MLP with a per-bit skip,
unrolled 12 on its own predictions), held-out trajectories, 40k steps:** changed bits still wrong
19/22/26/35% at depths 1/4/8/12; Mario's x within 8 px 89% at depth 1 and 58% at 12 (median error
0-3 px), y within 8 px 86-67%; enemy flags exact 97-85% -- and deaths, read off the predicted RAM by
the engine's own rule, P0/R0 at every depth: the dying states' bit patterns are too rare for the model
ever to produce one.

**So the RAM as input instead (`--ram`, wm27).** An MLP from the RAM's bits to a latent shaped like the
screen's (48 x 11 x 11); everything downstream -- dynamics, heads, calibration, the search -- as for
wm18, the policy and value teachers still reading the screen, the search's root the real RAM. The
encoder now has the whole state -- positions, speeds, enemy kinds, the held button -- where the
screen's encoder recovered it imperfectly (its policy head agreed with the net 60% even at depth 0).

**wm27 (wm18's recipe, reading the RAM; world_ram, 40k steps): the parts much better -- the gate not.**
Policy head against the net's top move 76% at depth 0 and 70-78% down to depth 12 (wm18 60% -> 53%);
death one step ahead P37/R94 (wm18 P13-19/R82-86), twelve P38/R97, calibrated P98/R97; W 2.0 frames.
Gate **3/32** (1-1 3/4 at 90% of the level, everything else dead), wm18 4. And that is the point: on
this search without reuse the all-oracle agent -- every part real -- made 3-4/32 too. A better model
cannot lift a score a perfect one does not reach; the latent search itself is the ceiling, and only
reuse with virtual loss has lifted it (9/32, perfect parts) against the C++ search's 24.

**Where the latent and C++ searches part, found by playing full games.** The C++ player (300
simulations, no veto) wins 8-3 from delays 5, 20, 50; the all-oracle latent search loses all four.
The two play identical moves to decision 14 (delay 50: 59). Rebuilt there, each with its own reuse,
the C++ root has 2052 -> 2419 -> 2784 visits over three decisions -- its trees hold 32768 nodes --
and the latent root is pinned at 978: with reuse its capacity was 3 x 300 + 66 = 966 nodes, the kept
subtree filled it, and each decision got next to no new simulations. **With 4096 nodes a tree: 14/32**
(1-1 4/4, 4-1 3/4, 8-2 3/4, 1-2 2/4, 8-3 1/4, 8-4 1/4), from 9 -- capacity was a real piece of the
gap to 24. wm27 with virtual loss: 2/32; with re-imagined reuse at 8192 nodes: 0/32 -- reuse still
breaks the model. Since re-imagination recomputes every kept value exactly, what the model's kept
tree carries is its visits: votes cast on the last screen's futures, which the model's values may now
see otherwise. `--reuse-decay` keeps the tree's shape (its depth) and scales the old visits down.

**The learned veto from thousands of hazards (surv3).** `survival make --rollouts 16000`: a level, a
start delay, a stretch of its route, then held inputs at random until Mario dies -- 80k states from
distinct situations, split by situation. Held-out mixed states: AUC 0.72, 57% of dying moves
caught, 24% of good ones refused, train loss to ~0 -- "which move dies 24 steps from now", read off
four 84x84 frames, does not generalise at this scale.

**The world model barely overfits.** On 600 fresh trajectories (seed 77) wm18's W error is the same
5.5 frames as on its training data; death twelve steps ahead P38/R91 (P34/R91 seen), one step
P13/R67 (P16/R84). It is limited by accuracy, not data -- and it is tiny beside MuZero's (48
channels, two transition blocks against 256 and sixteen). wm24: wm18's recipe and data at 96
channels, six transition blocks, the convolutional policy head, from scratch.

**The value head reads only imagined latents (`tools.wmprobe`, now with a 'seen' column).**
wm13 on 1-1, 128 states, 12 steps ahead: scored on the latent it imagines along a line, W
ranks the route against random, held and flipped lines correctly where the real game says the
route is better (97-100%, 87.5% against one flipped action). Scored on the real screen at the
end of the same line, encoded, it says 2-4 frames for everything and agrees 20-37% -- chance.
f was only ever trained on unrolled latents; the consistency loss aligns their projections,
not the latents themselves. So "the emulator's futures, the model's judgement" cannot be asked
by encoding the emulator's leaves (agree --wm-value does that, and would measure only this
mismatch); it needs the imagined latent along each real path.

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
