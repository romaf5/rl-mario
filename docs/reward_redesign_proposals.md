# Reward redesign proposals

Draft 2026-09-03. Scope: `_after_step` in `mario_native_vecenv.py`, reward constants in
`configs/mario_ppo_native_84.yaml` / `mario_ppo_native_route.yaml`. Nothing here is implemented.

**Notation.** step = 4 frames · γ = 0.995 (1/(1−γ) = 200 steps) · run speed ≈ 8–10 px/step ·
timer ≈ −1/6 per step · K = idle timeout (450 steps) · D = failure cost · V⁺ = value of the state
just past a hard spot · p = success probability of one attempt at it.

## 1. Audit: reward set v2

| term | value | purpose | what went / can go wrong | hidden state |
|---|---|---|---|---|
| x highwater | `clip(x − hw, 0, 20)`; hw rebased on life / area / swim / level change and legit jumps | progress | regained ground after a backtrack pays 0 (the frame stack spans 16 frames, cannot remember the backtrack); forward jumps paid +20 whatever their size (pipe 2 = 688 px, height check = 565 px) | **hw** (per-life max): medium |
| time | timer tick, mean −1/6 | urgency | none; the 24-frame tick phase only adds variance | phase: low |
| death | −100, terminal | lives matter | camping to the timeout (−40, §1.1) beats dying (−100) and beats attempts with p < 4–9 % | visible |
| idle timeout | −100 after 450 steps without `dx > 0` | anti-camping | the counter resets on **any** rightward dx: left-right oscillation never times out (free camp) | counter: hidden |
| loop | −100, terminal | kill the treadmill | correct; the "visited cell" branch reads a per-life set (only on frame-changing jumps, rare) | jump visible; set hidden |
| off-route | −100, terminal | warp > flag | correct, but undercut by the score leak below | visible |
| stage / victory | +500·Δgp (1-2→4-1 = +5500, 4-2→8-1 = +7500); axe +500 | outcome | fine: once per level, potential-like | visible |
| score | 0.1·Δscore, clip 2000/step: coin +20, stomp +10, power-up +100, flagpole +10..+500, end-of-level countdown up to +200/step | hidden-block reveal (+20) | pays the **off-route 1-2 flag +300..+900 on the steps before the −100**; pays remaining time (HUD cropped: invisible); a coin is worth 120 steps of time → detours | score & timer: **high** |
| novelty | 3/√(1+n), cell (area, x//64, y//24), global n halves every 3000 ticks | exploration | non-stationary by design; 10 y-bands per column pay bouncing; key has no level → cross-level collisions in the route run | global counts: by design |
| clip | `clip(r_x + r_t, −15, 20)` | glitch guard | caps jump pay; redundant since the 2-step debounce | — |

### 1.1 Structural findings (hold for every design)

- **Discount asymmetry.** A deferred terminal is cheaper than an immediate one by γ^k. Camping until the timeout is worth
  `V_camp = −(1/6)(1−γ^K)/(1−γ) − D·γ^K = −29.8 − 0.105·D` (K = 450). Dying now is −D.
  With D = 100: camp −40 · die −100 · zero-gain attempt that dies after 20 steps −90.
  Attempting beats camping only if `p·(V⁺ + 90) ≥ 54`: p ≥ 4 % at the water mouth (V⁺ ≈ 1200), p ≥ 9 % before a level exit (V⁺ ≈ 500). This is problem 3.
- **Rule: D ≤ (1/6)(1−γ^K)/(1−γ) ≈ 30 for K = 450.** Then camp ≈ die ≈ failed attempt, so any p > 0 prefers attempting. The lost future V⁺ (hundreds to thousands) is what makes death expensive; D is only the tie-breaker between ways of ending.
- **Depth-scaled failure costs are the worst attractor.** If a death costs the potential of the life (≈ 1800 at x = 2500 after discounting the replay), camping (−30 − 0.105·1800 = −219) beats dying (−1800) by ~1600. Failure cost must be flat.
- **Idle = "no new highwater for K steps"**, not "no rightward dx". K = 450 stays (v36: 150 broke Bowser patience).
- **Time = constant −1/6 per in-play step.** Same mean, no invisible phase, same for the RAM policy (sees the timer) and the pixel policy (does not).
- **Loops stay terminals.** A gain-first cycle is net positive under discounting even when its potential change is 0: a 1000-px lap over 100 steps is worth `L·[(1−γ^k)/(k(1−γ)) − γ^k]` = +182 (γ = 0.995), +47 (γ = 0.999). "Charge the teleport" alone is a farm.

## 2. Candidate designs

Shared: episode = one life (`episode_life`), 30 % archive restarts unchanged, one flat cost D for every failure and the episode ends, stage / victory outcomes unchanged, no score and no novelty unless listed.

### 2a. Clean minimal (v3-min)

| term | value |
|---|---|
| x | `clip(x − hw, 0, 20)`, hw rebased as in v2 |
| time | −1/6 per step, constant |
| fail | −D, **D = 30**, terminal: death, idle (450 steps without a new hw), loop, off-route |
| outcome | +500·Δgp (route levels), +500 victory |
| removed | score, novelty, timer tick, dx-based idle |

- Terminals / episode: as v2 except D and the idle rule.
- Attractors: `V_camp = −33`, `V_die = −30`, `V_attempt(p = 0, k = 20) = −30`, `V_progress ≥ +7.8/step`. Camping never wins for p > 0, V⁺ ≥ 0; dying never beats a state with positive future; when truly stuck all three endings tie = reset, which is what a per-life episode wants.
- Farms: x pays only new ground within a life, so any cycle earns 0·x − time; every terminal ends the episode → no positive cycle. Nothing accumulates across episodes (return is per episode).
- Hidden state: hw regain (unchanged, medium); the moment the idle terminal fires (low). Everything else visible.
- Hard spots: hidden block — no reveal signal; unchanged reliance on archive practice and RAM localisation (a perception problem, not a reward one). Water mouth — attempts beat camping at any p (was p ≥ 4 %). 1-2 warp vs flag — flag = −30 and nothing else (score leak gone); the ceiling run pays more x than the flag run even before the warp bonus is known. Height check — unchanged, terminal loop.
- Risk: low. Removing score drops the +20 reveal cue for the pixel run.

### 2b. Potential-based (v3-pot)

Φ(s) = 500·gp + x + offset(frame). The offset is set by continuity (jump pays 0) on every legit frame change — the v2 cycle rule still decides legit vs loop. Reward = Φ(s') − Φ(s) per step, plus time and terminals.

| term | value |
|---|---|
| x | signed `Δx` incl. teleports; forward same-frame jumps paid in full (cap 1024); backward same-frame jump = loop → terminal |
| time | −1/6 per step |
| fail | −D, D = 30, terminal: death, idle (no new max Φ for 450 steps), loop, off-route |
| outcome | +500·Δgp is the level part of Φ; victory +500 event |
| clip | remove `[−15, 20]`; keep the 2-step debounce |
| variant (ambitious) | episode = full 3-life game, death charged as lost potential Φ(respawn) − Φ(death) |

- Terminals / episode: per life, as 2a. **3-life variant not viable**: lost-potential deaths are depth-scaled (§1.1, camping wins by ~1600), lives are invisible (HUD cropped, not in the RAM features) so V(door) depends on lives left, and a restart episode would continue from the door after its first death, diluting archive practice.
- Attractors: D rule as 2a. Backtracking is a loss-first cycle (−L now, +L later = −(1−γ^k)·L − time) → never attractive.
- Farms: every cycle has zero potential change; loss-first cycles are negative under discounting; gain-first cycles (teleports) are terminals; a frame change into a cell visited this life is a loop (v2 rule). None.
- Hidden state: **best of the three**: Δx is visible every step, no hw; offsets are touched only on frame changes and pay 0 there.
- Hard spots: hidden block — pipe 2 pays +688 in one step (a strong, visible value cue for "onto the block → DOWN"); height check — the forward teleport pays +565; water mouth and 1-2 as 2a.
- Risk: medium. One-step rewards of +500..+700 next to ±10 steps (normalize_value handles the outcome bonuses already). Walking left is now charged: Bowser dodges and water manoeuvres still net 0, but with per-step sign noise.

### 2c. Outcome-only + intrinsic (v3-sparse)

| term | value |
|---|---|
| outcome | +500·Δgp, +500 victory |
| intrinsic | β/√(1+n) on the first visit **this life** to an archive-resolution cell (level, area, x//128, y//64, swim); n = global visit count, no decay; β = 20 (≈ 8 cells of time) |
| time | −1/6 per step |
| fail | −D, D = 30, terminal as 2a |
| removed | dense x reward, score, novelty grid |

- Episode: per life; the archive and the intrinsic term share one cell key.
- Attractors: D rule holds. Where every reachable cell is depleted (n ≫ 1) and the outcome is out of reach, every action pays −1/6: no attractor but **no gradient**; progress then comes only from restarts plus backward-chained outcomes (how 8-4's last zones were learned; from-door play in a fresh run would crawl).
- Farms: first-visit per life → nothing pays twice; loops terminal; door cells deplete within one epoch (128 envs), so die-and-recollect nets −D.
- Hidden state: n is global and drifts (by design, slower than v2's halving grid); everything else visible. Worst on the principle, best on "nothing dense to exploit".
- Hard spots: hidden block — the on-block cell is rare → pays ≈ β (v2's novelty at coarser resolution); water mouth — the far side is an unvisited zone → β per cell; 1-2 — warp-zone cells are rare → paid, flag −30; height check — post-teleport cells paid once.
- Risk: high (from-door learning, magnitude of β is a new knob).

### 2d. Side by side

| | 2a min | 2b pot | 2c sparse |
|---|---|---|---|
| change vs v2 | small (2 constants, 2 rules, 2 removals) | medium (x-term, clip) | large (x-term gone) |
| attractors | none for p > 0 | none | none, but no gradient when depleted |
| farms | none | none | none |
| hidden state | hw regain (medium) | lowest | global counts (by design) |
| hard spots | as v2 minus camping | jumps paid in full | frontier cells paid |
| risk | low | medium | high |

### 2e. Validation (same probes for every design)

Existing: scratchpad `cycle_test.py` (probes 1–4, currently asserting −100). New probes follow the same pattern (`MarioNativeVecEnv` with `dense_infos`, archive states from `archive_84_test.pkl`).

| # | probe | 2a | 2b | 2c |
|---|---|---|---|---|
| 1 | wrong pipe x824→312 | −30, done | −30, done | −30, done |
| 2 | water pipe x3648→water | legit; swimming +2..5/step | legit, jump pays 0; swimming ±Δx | legit; new water cells pay β |
| 3 | corridor loop x2586→1586 | −30, done, fresh stack | same | same |
| 4 | death | −30, fresh stack, respawn run pays | same | −30; respawn pays only on new cells |
| 5 | new: oscillate ±30 px for 500 steps | done at 450, Σ ≈ −105 (v2: never done) | same | same |
| 6 | new: walk left 100 px, then right 100 px | 0 − time | −100 then +100 | 0 − time |
| 7 | new: DOWN on pipe 2 from the block cell | +20 on the confirm step | +688 | β if the landing cell is new |
| 8 | new: twin runs — one action script from the door vs from an archive state at the same x | per-step rewards identical | identical | identical up to n |
| 9 | TB farm check | any `mario/r/<term>` per-episode mean up > 30 % over 200 epochs while `mario/door_max_x`, `eval/door_max_x_max` and Σ `mario/clear/*` move < 5 % → open the video (`rewards/iter` alone is blind: shorter episodes inflate it) | | |

## 3. Recommendation and migration

**2a now, then A/B the x-term of 2b** (one change per run). Keep 2c's intrinsic as a config-gated term for a frontier that stalls under both.

The decomposed-term interface already exists in progress (`mario_rewards.py`, uncommitted): `Signals` bundle → `Term` objects (`reset(idx, sig)`, `__call__(sig) -> reward`) → `RewardSet` built from a YAML list (`env_config.reward: [{type: ...}, ...]`), with a per-term breakdown. The designs map onto it as follows.

| design | YAML term list (`type: ...`) | still missing in `mario_rewards.py` |
|---|---|---|
| v2 (baseline) | `clip{progress_highwater cap 20, time}`, `fail 100`, `loop 100`, `offroute 100`, `stage 500`, `victory 500`, `score 0.1`, `novelty 3.0` | — (must reproduce today's rewards bit-for-bit) |
| 2a min | `clip{progress_highwater cap 20, time_const}`, `fail 30`, `loop 30`, `offroute 30`, `stage 500`, `victory 500` | `time_const` (−1/6·in_play); idle signal on highwater (`Signals.idle` counts steps without `fwd`, i.e. dx-based) |
| 2b pot | `progress_signed jump_cap 1024`, `time_const`, `fail 30`, `loop 30`, `offroute 30`, `stage 500`, `victory 500` (no `clip`) | `progress_signed`: signed `xd` with forward jumps paid in full (`ProgressDelta` zeroes jumps > 24 px — the v1 treadmill rule, do not reuse); `Signals` needs `fwd_jump` (legit & x > x_last) |
| 2c sparse | `intrinsic beta 20 key archive`, `time_const`, `fail 30`, `loop 30`, `offroute 30`, `stage 500`, `victory 500` | `intrinsic`: first-visit-per-life on the archive cell key with global counts (`Novelty` uses (area, x//64, y//24) and decays); `Signals` needs `cell_new`, `cell_count` |

Steps:

1. Land the refactor with v2 as the default list. Gate: rewards identical on probes 1–4 and on a recorded 2000-step rollout diff (per env, per step).
2. Add `time_const` and the highwater idle signal; switch the two configs to the 2a list. Probes 1–6 and 9; compare at epoch 300 / 500 with v38 (8-4) and v20 (route) at the same epoch.
3. Add `progress_signed` + `fwd_jump`; A/B on the 8-4 config (2b). Probes 6–8.
4. Only if `door_max_x` stalls under both: add `intrinsic` on top of the 2a list (2c).

Conventions to keep while doing it:

- Per-term episode sums into `infos['r_terms']` → TensorBoard `mario/r/<term>` (probe 9 needs this; `RewardSet`'s breakdown is the source).
- `loop`, `off`, `idle_to` are always computed as signals; only a term decides the cost (today `loop_penalty: 0` silently disables detection).
- Terminals are read from signals (`died`, `idle_to`, `loop`, `off`, `game_over`, `victory_new`), never from a term's output, so a zero-cost term cannot change episode structure.
- Archive, frontier credit, explorer episodes, curriculum: untouched; they read signals, not rewards.

## 4. Open questions

1. D = 30 (camp ≈ die ≈ failed attempt) or keep 100 and accept camping below p ≈ 4–9 %? With 30 a stuck agent will sometimes die to reset — right for per-life episodes, costs lives in the 3-life eval.
2. Highwater (regain pays 0, hidden) or signed Δx (backtrack charged, visible, forward jumps paid in full)? Signed is closer to the principle and to a speedrun; is a +688 one-step reward acceptable?
3. Should lives (0–2) become an input (RAM feature or HUD strip) so 3-life episodes are ever viable, or stay per-life for good?
4. Drop score entirely (loses the +20 hidden-block reveal cue in the pixel run) or keep only visible events (coin, stomp, reveal) and drop flagpole / countdown score?
5. Novelty: remove and rely on the archive, or keep 2c's intrinsic term as a stall-only switch?
