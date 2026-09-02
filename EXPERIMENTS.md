# Experiment log & decision framework

## Reading a run (in this order)

1. **Health** (first 5 min of any run): fps stable, `losses/entropy`
   decaying slowly (collapse to ~0 = dead exploration), `losses/kl` near
   `kl_threshold`. Broken health voids every other chart.
2. **Task**: `mario/clear/<level>` per start level — the KPI.
   `rewards/iter` is sanity only: reward is shaped, never optimize the chart.
3. **Exploration** (the "curiosity" charts):
   `best_x_pos` = discovery frontier · `mean/max_x_pos` = typical behavior ·
   `loop_rate` = maze-stuckness · `warp/<level>` = shortcut discovery.
4. **Goal**: `eval/game_progress` (sequential from 1-1) + `victory_rate`.
   Everything else is instrumental to these two.

## The 2×2 that picks the next move

|                    | mean_x rising            | mean_x flat |
|--------------------|--------------------------|-------------|
| **best_x rising**  | keep baking              | consolidation problem → practice allocation (self-restart prob/bias) |
| **best_x flat**    | (rare; noise)            | discovery blocked → new *mechanism* (restarts, novelty, entropy), never just more epochs |

## Rules

- One change per run; read at 300–500-epoch checkpoints; compare to the
  best previous run **at the same epoch**, not to its final value.
- Two runs flatlining at the same frontier ⇒ the knob class is exhausted;
  escalate mechanism, don't retune magnitude.
- Never inject level-specific knowledge into training (user constraint);
  tooling/verification outside training is fine.

## Run log (8-4 focus, 2026-08-30)

| run | change vs previous | epochs | outcome |
|---|---|---|---|
| 84_Baseline | (legacy delta-x reward) | 300 | reward-treadmill exploit: eval 82k at x=970 |
| 84_HighWater | high-water x reward | 700 | treadmill dead; **found maze pipe 1**; wall at x=2584 |
| 84_LoopPenalty | +loop −30 / backtrack −0.15 | 626 | looping worthless but wall unchanged; loop_rate 0.97 |
| 84_Novelty | +episodic novelty 0.5 | 251 | wall unchanged — per-episode novelty has no cross-episode push |
| 84_SelfRestart | +restarts from own states 0.3 | 700 | **wall broken at ep 118** (best x 3847); consolidation stalled at 7–16% |
| WarpRoute_Overnight | all 8 route levels, frontier-biased restarts, sticky 0.05 | 20 | shm pipe race froze it at the first curriculum broadcast (fixed) |

## Run log (native stack, overnight 2026-08-31)

| run | change vs previous | epochs | outcome |
|---|---|---|---|
| Native84 v1/v2 | native stack; v2 +shared archive | 900/550 | ~0% past-wall: sticky 0.05 (untested bundle) + per-env archives |
| Native84 v3 | sticky back to 0.1 | 890 | best 2607, ~1% -- still weak: block-granularity novelty |
| Native84 v4 | novelty band 24, bonus 1.0; native eval | 730 | "100% past-wall" -- exposed as a RESTART-START ARTIFACT: from-door reveal rate 2/56, conversions 0/56 |
| Native84 v5 | soft least-practiced restarts; door-only frontier metrics | 400 | door_max ~2570; reveals happen from restarts only |
| Native84 v6-v8 | novelty bonus 3.0; score_reward 0.1; explore_eps 0.05 | ~400 ea | fresh runs kept re-losing archive capital: 0 organic climb chains |
| Native84 v9 | persistent archive (native/archive_84.pkl) + explorer episodes 0.25 | 500 | capital kept, but on-block cells collided with floor cells |
| Native84 v11 | height-aware cell keys (ypix//64 in key) | 657 | **BREAKTHROUGH: archive ratcheted x-cell 18→30** (x≈3840, post-pipe corridor); door_max still ~2588 |
| Native84 v12 | +swim flag in cell key (water≠door collision fix) | 660 | water y-band cells fill; frontier stuck at x-cell 5 |
| Native84 v13 | frontier-recency restarts, cap 512, explorer 300 | 340 | stuck: fstate==0 gate blocked ALL open-water saves |
| Native84 v14 | archive gate accepts swim states | 170 | water cells 6-8 (full zone to wall x1058) |
| Native84 v15 | timer floor t>25, refresh keeps best timer | 640 | wall practiced 2800x, zero side-pipe entries |
| Native84 v16-v17 | idle cap 15/ep; policy-prior macro explorers; doomed-cell prune | 600/620 | still zero entries; policy probe: 64/64 die at wall |
| Native84 v18-v19 | novelty annealing; adaptive refresh by early-death rate | 600/500 | still zero -- ALL blocked by x>4000 guard (below) |
| Native84 v20 | x-guard: debounce 2-step-persistent jumps (was: reject x>4000) | 280 | **corridor x4152-4830 unmasked; cells 32-37 in minutes; FIRST VICTORIES ep 57 (axe grab, victory_rate ~1%)**; victories sporadic (corridor aged out of recency frontier) |
| Native84 v21-v22 | least-uses frontier; post-victory zombie fix (5000-frame skip exhaust latch + nongame guard) | 280/460 | v21 fps 4700->74 (zombie envs) -> fixed; v22 victories 49/150 epochs early but DECAYED (uses != mastery) |
| Native84 v23-v24 | mastery gate <3 wins; then win-band 1-9 + persisted wins | 440/260 | v23 still decayed (band filtered nothing); v24 win-band spread over 40 cells in post-water zones (areas 0/2, 10-49 wins/cell) then STARVED at ep 125 (10-win graduation cap) |
| Native84 v25 | no graduation: band = all winners, least-uses first | 150+ | **59/154 victory epochs, sustained, no decay -- pipeline stable** |
| Native84 v26 | wrap guard + purge 130 quest-2 cells (real decay cause) | 130 | **victories EVERY epoch, rate 2->9% compounding** |
| Native84 v27-v31 | skip-cap 600; HUD crop; renderer fixes; lockstep video | ~500 ea | stable ~5% vr; win band 37 -> 36 -> 35 backward march |
| Native84 v32 | per-cell variant reservoirs (4 states, rotate) | 4000 | Bowser passage stuck ~30% point-blank / 2-4% compound; 34 seeded |
| Native84 v33 | frame-skip 2 (30Hz control), 8000 epochs | running | reward pace ~2x v32 equivalent; watch cell-35 conversion |
| Routev4 | all-fixes route run | 6000 | **WARPS DISCOVERED (1-2 warp 18-25%); eval plays 1-1 -> warp -> 4-1**; clears 49/41/34/12; 4-2 vine warp = next gate |
| NativeRoute v1 | 8 levels, curriculum, full toolbox | 2900 | clears: 1-1 59% 4-1 58% 8-1 52% 1-2 48% 4-2 37%; NO warps; eval undersold ~2x by cross-renderer shift (fixed: native eval adapter) |
| NativeRoute v2 | native eval + full toolbox refresh | running | rewards recovering (3.3k) |
| Native84 v34-v38 | skip-4 revert; idle timeout 150 -> 450; review fixes (debounced ctx, phantom cells, eviction) | ~600 ea | v36 idle 150 killed Bowser patience (conversion collapse); v38 stable ~3% vr, door mean ~2570 = corridor loop point |
| Routev5-v10 | same fixes; stochastic eval videos; off-route = -100 terminal; archive keyed by current level (512 -> 312 cells) | ~2000 ea | v9: clear/1-2 20% but warp/1-2 0% (flag paid +500 and then 1-3 x-reward); world 8 got <1% of restarts (evicted at cap) |
| Native84 v40 / Routev11 | **reward set v2**: highwater rebased on scripted backward transitions (C++ transit flag), loop = -100 + episode end, uniform fail cost 100, no growth/backtrack terms | running | probes: corridor loop -100/done then +12/step; water pipe no penalty, swimming rewarded; death -100 |

## Lessons
- Metrics must separate from-door and from-restart episodes (mario/door_*).
- Frontier-biased restarts push frontiers; BRIDGING needs uniform cell
  coverage -- match restart selection to which problem you have.
- Eval must use the training renderer; argmax eval understates (use
  sampled T~0.7 probes for ground truth).
- When a frontier refuses 5+ mechanisms, suspect the SENSORS, not the
  agent: the x>4000 "glitch guard" silently erased the final corridor
  (rewards, novelty, archive, metrics all blind). Validate coordinate
  frames per zone before shipping guards.
- Probe hygiene: raw benv takes BUTTON BYTES not action indices; a
  wrong probe "verified" an entry rate of 100% that was game-over
  screens. Reuse _ACTION_BYTES everywhere.
- Archive states can be doomed (saved mid enemy contact, low timer,
  0 lives): gate saves on survivability signals, prune/re-roll by
  early-death rate, never downgrade a cell's timer on refresh.
- Reward must be a function of what the agent SEES. Per-life highwater
  x kept paying 0 after a maze loop (and through the whole 8-4 water
  section: same area byte 3), so an identical forward run earned +8 or
  -0.3 depending on invisible history; position-scaled growth paid the
  same run 4x more late in the level. Fix: end the episode on loops,
  rebase on scripted transitions, one flat cost for every failure.
