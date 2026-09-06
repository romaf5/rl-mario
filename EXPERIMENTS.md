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
| Native84 v40 / Routev11 | reward set v2 draft: highwater rebased on any scripted backward transition (C++ transit flag), loop = -100 + episode end, uniform fail cost 100, no growth/backtrack terms | 900 | **FARMED**: a wrong pipe is a scripted transition too -> section-1 pipe x824->312 rebased the highwater; reward 160 -> 5100/episode with door_max_x < 1290 (user spotted it in the video) |
| Routev13 | per-level eval clips (gameplay/level_*), backward-chaining frontier 0.5/k16 for the route | 3250 | warp/1-2 flickered 0-0.5%; off-route exits 4-5% |
| Routev18 | archive never saves states outside the training level set (53 cells of 4-3 had been saved on the 4-2->4-3 confirm step and practised for free); purged, resumed from v17 ep 4500 | running | door max x 5280-5430 = deep 8-1 (longest level), not a glitch |
| Pos84tiles (seed 42) / Pos84tilesB (seed 7) | tuning 3: **archive cell key + level-tile signature** (cell_tiles: true, 1024 cells) on the practice-concentration config; from scratch, two seeds | running (started 2026-09-06 06:05) | why: the human trace shows the corridor maneuver is reveal (block byte 5f->c4 at x2400) -> climb the block (ypix 112) -> jump to the pipe top (ypix 64) -> DOWN at x~2436; the old key merged revealed and unrevealed floor states, so the revealed state was never kept or practised. With the signature each link is one jump. PREDICTION: 30 min (06:35) corridor; 2 h (08:05) archive holds a revealed-block cell (bin 18, sig != unrevealed) with wins; 3 h (09:05) clean-eval mean > 3100; 4 h (10:05) door-side victory in at least one seed |
| Pos84cells | tuning 2: **2D first-visit bonus** (+2 per new (frame, x/64, y/32) cell per life) on top of the practice-concentration config; from scratch, empty archive | running (started 2026-09-06 03:38) | why: both runs' archives show the missing link is bins 14-17 -> the block: a y change at the same x that x-progress never pays for; the bonus pays landing ON the block once per life, never negative, bounded. PREDICTION: 30 min (04:08) corridor; 2 h (05:38) winning cells in bins 14-17; 3 h (06:38) clean-eval mean > 3100; 4 h (07:38) door-side victory. Miss -> actors/threads. RESULT 04:08: near-miss (door max 2400). RESULT 05:38 (2 h): MISS -- only 4 winning cells (Bowser room), no backward chain at all, clean-eval mean 2431 with 44% of door episodes still running at 600 steps (the policy dawdles collecting cells). Stopped; the 2D bonus slows the chain and does not find the block |
| Pos84frontier | tuning 1: **practice concentration** -- self_restart_prob 0.3->0.7, frontier share 0.5->0.8 (56% of episodes practise the failure-weighted frontier vs 15%); everything else = Pos84v1, from scratch, empty archive | running (started 2026-09-06 01:40) | PREDICTION: 30 min (02:10) corridor reached; 2 h (03:40) winning frontier cells in the corridor (x-bin 17-20) and clean-eval mean > 2600; 3 h clean-eval mean > 3100; 4 h (05:40) door-side victory. Miss -> next lever (actors/threads, then noise). RESULT 02:10: PASS (corridor, door mean 1870). RESULT 03:40: MISS -- clean-eval mean 2380 (< 2600), 54 winning cells but the chain stops at the block (bins 18-19), none from bins 14-17; kept to 05:40 as the control for Pos84cells. RESULT 04:40 (3 h): MISS (eval 2508). RESULT 05:40 (4 h): MISS (no victory, eval 2478, 84% corridor-loop cutoffs). Stopped |
| Pos84v1 (pixels; the RAM twin was dropped at 23:50 -- pixels only from here) | **positive-only rewards, from scratch, 8-4 only, 4 h cap** (first-visit progress + clear 500, no penalties, 250-step unpaid cutoff bootstrapped, page reset = 60-step grace, scale_value 0.01) | running (started 2026-09-05 23:35) | PREDICTION written before launch: 30 min -> corridor reached (door max ~2580; confident); 2 h -> door policy past pipe 2 (clean-eval mean > 3100; ~40%); 4 h -> door-side victory (~15%). Miss a checkpoint -> stop and tune the SETUP (curriculum > actors/threads > noise/frontier > novelty), never extend. Judged only at 00:05 / 01:35 / 03:35. RESULT 00:05: PASS (door max 2570, door mean 1800). RESULT 01:35: MISS (clean-eval mean 2460 / max 2567, 72% corridor-loop cutoffs, 28% deaths; frontier 4 cells, no chain into the corridor) -> kept running as the control, tuning started on GPU1. RESULT 03:35 (4 h): MISS -- no door-side victory; clean-eval mean 2453 / max 2566; archive chained wins to bins 0-10, 18-19 (on the block) and 24-37, never from the section-2 floor (bins 14-17) despite ~4k restarts per bin. Stopped |
| (reset 2026-09-05 22:50) | ALL runs stopped, runs/ + archives deleted, tas/ and obsolete retro-era 8-4 configs removed | - | new direction: rewards reimplemented from scratch (positive-only: first-visit progress + clear bonus, terminals pay 0, no loop/penalty machinery), offline reward test bench, replayable eval traces, then 8-4-only experiments capped at 4 h each |
| Native84 v55 / ramv16 / Routev30 | **FULL RESET, epoch 0, empty archives** on the stable fixed code (16-bug sweep + AreaType frame + failure cost 100, loops terminal) | running | user: the resumed policies had been through four reward regimes in one day ("restarted too many times to fry brains of the model"); the from-scratch control had shown the fixed system reaches the corridor in ~150 epochs, so a clean start costs little. Pre-reset archives + checkpoints backed up to scratchpad/backup_0905_prereset |
| Native84 v54 / ramv15 / Routev29 / fresh3 | **AreaType (0x74E) added to the frame**: the 4-2 vine warp (x 1031->75, area byte still 2, AreaType 2->1) was flagged as a same-frame backward LOOP and terminated the episode at the warp entrance -- found in a user play trace of 4-2. Now a legit transition; archive keys migrated (+atype) | running | this is why the 4-2 vine warp was never learned (warp/4-2 stuck at 0 for the whole project); same bug class as 8-4 water but that flips swim (in the frame) while the vine flips AreaType (was not) |
| Native84 v53 / ramv14 / Routev28 / fresh2 | failure cost REVERTED to 100 (loop/death/idle/offroute), loops terminal, all bug fixes kept | running | 6h verdict: cost 30 plateaued 8-4 at the corridor loop point (clean-eval mean 2370-2400, loops 66-88%, door never past pipe 2) on BOTH the resumed run and a from-scratch control -- vs cost 100 pre-churn which reached clean-eval mean 3130 past pipe 2 with loops 9-12%. Terminal penalties are not potential-based, so magnitude changes the optimum; 100 creates the pressure to escape the corridor. Fresh control restarts from scratch at 100 |
| Native84 v52 / ramv13 / Routev27 / fresh (resumed) | loops terminal again (`loop_terminal: always`), cost stays 30 = uniform failure cost | running | non-terminal loops (v50-v51, 90 min): the pixel door policy stopped attempting the hidden block -- clean-eval mean max_x 3150 -> 2554, loops 9% -> 97%; the unpaid re-run is indistinguishable from a first pass and dilutes the corridor's value, and looping twice costs no more than one failed attempt. Severity fix kept, continuation reverted |
| Native84fresh | CONTROL: from scratch (epoch 0, empty archive) on the fully fixed code, same config as v51 | running | question: how fast does a correct system reach the corridor? historical from-scratch runs took ~3000 epochs to the corridor loop point; compare at equal epochs against EXPERIMENTS history and against v51 |
| Native84 v51 / ramv12 / Routev26 | **uniform failure cost 30** (death, idle, loop, off-route) | running | v50 (loop 30 vs death 100) lost the door's block attempts within 20 min (door_max 3740 -> 2580): quitting via the cheap loop beat a 1-3%-success attempt that risks -100 -- the unequal-cost attractor from docs/reward_redesign_proposals.md; equal costs make attempting preferred for any p > 0 |
| Native84 v50 / ramv11 / Routev25 | **loop = setback**: a first backward teleport in a life costs 30 and play continues from where the game put Mario with the highwater KEPT (re-run pays nothing until new ground); a second teleport in the same life ends the episode (cost 30); `loop_terminal: always|repeat|never` keeps the old rule available | running | motivated by 8-4's ledge: the game wraps a Mario standing still past x~1245, which is its loop mechanic but not the agent cycling; treating it as a death (-100, terminal) was the wrong severity |
| bug sweep (4 parallel auditors) | fixed: glitch-world (1-2 warp zone -> world 36) scored/archived as 8-4 (77 route cells, 50k restarts, 0 wins); off-route & loop detection gated on flat kwargs instead of the reward terms; prev_x aliased x_last (phantom transition + lost progress on the FIRST step of every episode); benv_obs shared a process-global buffer (video thread corrupted training observations); early-death prune deleted WINNING cells; cell_tries dropped on reload for never-winning cells; eviction could delete an in-use cell; explore noise leaked past a life loss; clear/<level> counted off-route exits (starved 1-2 in the curriculum); lives averaged the 0xFF sentinel; eval env lost stage/novelty/off-route parity; _play_clip leaked a retro emulator on error; rewind ignored the env RNG and novelty counters | - | all confirmed by probes, all fixed and re-verified; runs restarted from checkpoints |
| Native84 v46 / ramv6 / Routev20 | resumed (ep 12000 / 15000 / 8000); failure weight uses restarts SINCE the credit reset, not lifetime uses | running | corridor after 4h: pipe-2 top 34-42 wins, block cell 15-37, but floor cells 0-2 -- lifetime uses (3.7-4.6k) made every cell's failure rate ~1.0, so weighting was uniform again |
| Native84 v45 / ramv5 / Routev19 | resumed (ep 8000 / 10000 / 4500); credit needs REAL depth (>=4 x-bins or another frame); frontier half of restarts weighted by failure rate (1 - wins/uses); archive wins zeroed | running | corridor probe: from the pipe-2 top RAM presses DOWN 51/64 and passes 32/64 (pixels 2/64); from ON the block RAM passes 14/64; from the floor under the block 0/64 for both -- floor cells got ~1% of restarts and 'won' trivially by reaching the next floor cell |
| Native84 v44 / ramv4 / Routev17 | RESUMED from checkpoints (ep 5500 / 6500 / 2500); exploration noise sticky 0.1->0.05, eps 0.05->0.02; new clean 32-episode door eval (eval/door_*) each video epoch | running | probe: clean policy enters pipe 1 61/64 and reaches the corridor (max_x ~2500) from the door; WITH training noise only 23/64 (pixels) / 16/64 (RAM), 42% die in the opening lava -> door_mean_x ~1400 was mostly the noise; the real wall is the corridor hidden block |
| Native84 v43 / ramv3 / Routev15 | loop = REAL terminal (env reset, no post-loop run: user saw LOOP -100 followed by positive r); eval clips play until over (per-level: until cleared / lives gone) | running | frontier_cells grew 7 -> 82 (pixels) / 6 -> 49 (RAM) / 251 -> 330 (route) in the first ~2h of v42/ramv2/v14 |
| Native84 v42 / ramv2 / Routev14 | **transitive frontier credit**: reaching a deeper cell that already wins counts as a win for the start cell (mario/frontier_cells) | ~2000 | 8-4 archive audit: 0 wins before x4100 over ~200k restarts, 48k wins in the last 2 zones which absorbed ~170k restarts -- victory-only credit pinned the frontier to the Bowser room; water (42k uses) and corridor (1.5k/cell) never chained |
| Native84ramv1 | **RAM features + MLP** (obs_mode ram: 12x13 tile grid, 5 enemies, Mario state incl. absolute x), same env/rewards/archive as v41 | running | hypothesis: pixel policy cannot localize in the repetitive corridor (hidden block has no pixel cue) |
| tas/ | replay HappyLee's #1715 warps TAS frame-exact on the native core, score with training rewards | parked | boot+Start sync; desyncs inside 1-1 (FCEUX 2.1 timing != ours); no other movie for this ROM |
| Native84 v41 / Routev12 | **reward set v2, cycle rule**: frame = (level, area byte, swim); backward jump inside a frame = loop; jump into a cell already visited this life = loop; other jumps = transition (water pipe). Frame stack reset at life/loop boundaries | running | probes: wrong pipe -100/done; instant corridor loop -100/done; water pipe legit, swimming +2..5/step; death -100, fresh stack |

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
