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
| Native84 v5 | soft least-practiced restarts; door-only frontier metrics | running | watch mario/door_max_x |
| NativeRoute | 8 levels, curriculum, full toolbox | 2400+ | entry levels clear 33-52%; 8-1 ticking; NO warps yet; eval was undersold ~2x by cross-renderer shift (fixed via native eval adapter) |

## Lessons
- Metrics must separate from-door and from-restart episodes (mario/door_*).
- Frontier-biased restarts push frontiers; BRIDGING needs uniform cell
  coverage -- match restart selection to which problem you have.
- Eval must use the training renderer; argmax eval understates (use
  sampled T~0.7 probes for ground truth).
