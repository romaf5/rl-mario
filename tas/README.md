# TAS replay (reward audit) — parked

`replay_tas.py` replays an FCEUX `.fm2` movie frame-exactly on the native
core and scores it every 4 frames with the training reward code
(`MarioNativeVecEnv._after_step`). Goal: see what the trained agent would
be paid along a known-optimal any% route.

```
venv_retro/bin/python tas/replay_tas.py --shift -8 --csv out.csv
```

Movie: HappyLee, TASVideos #1715 (04:57.31, FCEUX 2.1, ROM SMB (JU)).

## Status (2026-09-02)

| step | result |
|---|---|
| parse + frame-exact replay | works, 17868 frames in ~5 s |
| power-on + Start (movie frame 41) | accepted; gameplay at frame 189 |
| FCEUX gameplay start | frame 196 (movie's first inputs) |
| our start grid | 189 / 207 / 225 (18-frame steps): 196 unreachable |
| input shift −7 (exact alignment) | dies at frame ~600 in 1-1 |
| input shift −8 | no early death, stalls at x≈562, time-up |
| offset 0-17 × shift −2..2 scan | nothing finishes 1-1 |

Conclusion: the core is bitwise-identical to stable-retro's NES core
(`native/deep_difftest.py`) but not frame-timing-identical to FCEUX 2.1
(lag/latch differences appear within seconds of 1-1). TAS sync would need
emulator-timing work; no other warps movie exists for this ROM (#6622 is
PAL). Alternative for a route audit: replay the agent's own best
trajectories.
