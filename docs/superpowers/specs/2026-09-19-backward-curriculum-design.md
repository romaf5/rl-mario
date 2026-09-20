# Backward curriculum on the agent's own route (Go-Explore phase 2)

Status: approved 2026-09-19. Run: `Mario_PPO42l` (4-2, native env, PPO).

## Why

Run k funnel (epoch 2500, 768k steps of the exact training mix):

| start | episodes | reached warp area | warped |
|---|---|---|---|
| door | 446 | 0 | 0 |
| archive x 512-1023 | 302 | 0 | 0 |
| archive x 1024-1279 (vine) | 110 | 16 | 1 |
| archive, inside warp area | 39 | -- | 7 |

The route is found (explorers, restarts at the vine) but never learned backwards; 53% of all steps
loiter at the level end. Salimans & Chen (2018) / Go-Explore phase 2: start episodes on a successful
trajectory, near its end, and move the start back only once the policy succeeds from there.

## Flow

```
explorers / restarts / door ──► archive cells (state variants + action prefix from the door)
                                         │ first real clear with a known prefix
                                         ▼
                            route = prefix + episode actions
                                         │ replay once in a 1-core env (verify it clears)
                                         ▼
                             route states s[0] = door ... s[L-1]
                                         │
          reset: 75% ──► start at s[tau], tau ~ U[tau*, min(tau*+W, L-1)]
                 25% ──► run k's mix (door / archive restart)
                                         │ episode end
                                         ▼
     frontier band [tau*, tau*+D): last N outcomes, success = real clear
     rate >= p  ──►  tau* -= D  (floor 0; tau = 0 is the door state)
```

## Components (all in `mario_native_vecenv.py` unless noted)

| unit | does | notes |
|---|---|---|
| action log | per env, the applied actions of the current episode (`last_action`, after every substitution) | numpy (n, 8192) int8 + length; overflow -> prefix unknown |
| prefix | `start_prefix[i]` (bytes or None) + the episode's logged actions | door: `b''`; restart / walk: the drawn variant's prefix; route start: `route[:tau]` |
| archive entry `[8]` | list of prefixes aligned with `[0]` (state variants) | appended / rotated together; missing on old archives -> None |
| route builder | on the first on-route clear (`good`) in a trained level with a known prefix: replay from the level's door state in a separate 1-core `benv`, save every state, check the level advanced on-route at the last step | failure: warning, no route |
| curriculum | start draw, frontier band stats, tau* update | infos `demo_start`, `demo_tau`, `demo_tau_star`, `demo_len` |
| persistence | `<archive_path>.demo.npz`: level, actions, tau*, stats | written with the archive; loaded with it (`--resume-archive`); a fresh run refuses an existing sidecar too |

Route starts: `start_cell None` (no archive credit), `is_door = (tau == 0)`, `was_restart False`, novelty bonus
as for any non-door episode (none with `cell_bonus_door_only`). Their grounded new cells are archived normally.
Requires deterministic stepping: `reset_noops == 0` and exactly one trained level (`random_stages` of length 1),
both asserted when the curriculum is on (one route per run; per-level routes are future work). Sticky / eps are
recorded as applied actions, so prefixes stay exact.

## Config (`env_config`, defaults = off)

| key | 4-2 run l | meaning |
|---|---|---|
| `demo_start_prob` | 0.75 | share of resets on the route once it exists (0 = feature off) |
| `demo_window` | 32 | W: start draw width in steps from tau* |
| `demo_step` | 16 | D: frontier band width and step-back size; tau* starts at L-1-D |
| `demo_success` | 0.2 | p: frontier success rate to step back |
| `demo_success_n` | 64 | N: frontier episodes per decision |

## Metrics (callbacks.py)

`mario/demo_len`, `mario/demo_tau` (tau*), `mario/demo_frontier_success`, `mario/clear_demo/<lvl>`,
`mario/demo_share` (route starts / episodes). Door metrics keep using `info['door']`.

## Audit (tools/audit_env.py)

Replays the env's route independently from the door and labels an episode `demo` iff its first RAM equals a
route state (tau > 0), `door` iff the door state, else `restart`. New violation kinds: route does not clear
on replay, `demo_tau` outside the window.

## Tests (`tests/demo_bench.py`)

1. prefix fidelity: after a random run with explorers, every archived variant with a prefix replays
   byte-identically from its door (>= 20 cells, restart- and explorer-saved).
2. alignment: variants and prefixes stay paired through reservoir rotation.
3. route creation: the vine route (`tests/data/vine_route_4-2.npy`) played from the door creates a verified
   route of its length.
4. curriculum: synthetic outcomes step tau* back at 20% (13/64), not at 19% (12/64); draws in the window;
   tau = 0 starts are door episodes.
5. persistence: sidecar and entry `[8]` round-trip; an archive without `[8]` loads.
6. `tools/audit_env.py` with a forced route: 0 violations.

All existing benches must stay 0 FAIL.

## Run l predictions (4 h budget)

| time | criterion |
|---|---|
| 30 min | route exists, tau* >= 100 steps back from the warp |
| 2 h | tau* before the vine (main underground) |
| 4 h | tau* = 0 and `mario/clear_door/4-2` > 0 |
| 8 h | `eval/level_clear/4-2` > 0 |

Miss at 2 h: read `mario/demo_frontier_success` at the stuck tau* and replay the route states there before any
change.
