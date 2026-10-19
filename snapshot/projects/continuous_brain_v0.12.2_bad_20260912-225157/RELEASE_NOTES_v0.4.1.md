# Continuous Brain v0.4.1 — Curriculum Correction Release

v0.4.1 is a focused correction to the first real AI2-THOR embodied run. It preserves the v0.4 neural architecture/checkpoint format and fixes the training boundary around early curriculum levels.

## Why this release exists

The first real `FloorPlan1Discovery_physics` L0 run demonstrated that the 3-D adapter, recurrent brain, replay, memory and online learning were operational, but L0 `orient` allowed the five-output policy to choose translation and interaction intents. That made a supposed heading-only task unnecessarily noisy and produced long stagnation plateaus.

v0.4.1 turns curriculum levels into actual action spaces rather than only reward labels.

## Changes

### Curriculum-specific neural action masks

The 5-output policy head is retained for checkpoint compatibility, but only curriculum-valid actions participate in exploration, greedy action selection, TD bootstrap targets, and optional Ollama planning.

- L0: `idle`, `turn_left`, `turn_right`
- L1–L5: `idle`, `advance`, `turn_left`, `turn_right`
- L6–L9: all five intents, including `interact`

The environment also enforces the mask as a hard boundary if an external caller submits an invalid action.

### L0 position lock

L0 records a position anchor when the episode begins. Translation is masked, and any unexpected simulator drift is corrected while preserving current yaw. This isolates orientation learning from navigation.

### Stronger orientation shaping

For `orient` tasks, reward now combines:

- absolute-bearing improvement
- a small bonus for choosing the turn direction that reduces the current bearing
- a penalty for turning the wrong way
- a small penalty for idling while still substantially misaligned
- the existing +1 task-completion reward

Default success tolerance remains 15 degrees and can be adjusted with `--orient-success-degrees`.

### Stagnation watchdog

An episode terminates as `stagnation` when the task metric fails to improve for the configured number of steps. Default:

```text
--stagnation-steps 60
```

L0 progress is angular; later levels use target distance.

### Graduation statistics

The embodied environment now tracks a rolling curriculum window, persisted in checkpoints:

- attempts
- successes
- success rate
- median successful completion steps
- timeout rate
- stagnation rate
- per-episode action counts

Default readiness criterion:

```text
>= 100 recent episodes
>= 95% success
median successful completion <= 6 steps
timeout rate <= 2%
```

`graduation_ready` is shown in the browser state and curriculum dashboard.

### Better diagnostics

Terminal status now includes:

```text
brain=<high-level intent>
thor=<simulator primitive>
dBearing=<angular progress>  # L0
dDist=<distance progress>    # navigation levels
stagnation=N/limit
Lstats=<recent success %>/median steps
```

The browser dashboard now shows allowed actions, masked Q rows, stagnation, and rolling curriculum/graduation statistics.

### WSL browser opening

`--open-browser` now prefers Windows PowerShell/CMD/Explorer via explicit `/mnt/c/Windows/...` paths under WSL. It no longer falls through to noisy Linux `xdg-open` attempts when no Linux browser exists.

### No blind curriculum advancement

`scripts/run_thor_curriculum.sh` now runs the requested level (default L0) and explicitly tells the operator to advance only after the graduation criterion is met. Fixed-step L0→L9 advancement has been removed.

## Compatibility

- Neural model dimensions are unchanged from v0.4.
- Checkpoint `format_version` remains 4.
- Existing v0.4 checkpoints can load.
- For the cleanest L0 experiment, bootstrap a new v0.4.1 embodied branch from the preserved v0.3 checkpoint rather than carrying forward the under-constrained v0.4 L0 policy.

## Validation

- `20 passed`
- 240-step embodied mock smoke run
- 36 orientation goals
- `MoveAhead` calls during L0: **0**
- curriculum statistics persistence: PASS
- greedy and epsilon-random action masking: PASS
- L0 translation hard-boundary and pose lock: PASS
