# Continuous Brain v0.5.2 — Sensor Recovery & Controller Resilience

## Why this release exists

A real Discovery L2 run proved that the first-person sensor stream can freeze upstream of the browser: RGB hash/diff stopped changing across pose changes while trusted depth fell to zero. v0.5.1 detected the fault but only reset the episode; the same broken controller then contaminated curriculum statistics.

## Changes

- Sensor faults are infrastructure faults, not curriculum failures.
- Fault transitions are quarantined from replay, working memory, episodic memory, visual memory, reward totals, learning, and consolidation.
- Scene-reset recovery is attempted first.
- Recovery actively rotates the camera and requires both changed RGB and trusted depth.
- If scene reset fails, the old AI2-THOR controller is stopped and recreated from the original constructor arguments.
- The persistent neural state, replay, episodic DB, mastered snapshots, and spatial map survive controller recreation.
- Controller `step()` exceptions (dead Unity process/socket) enter the same recovery path rather than crashing the learner.
- Repeated controller restarts are rate-limited; an unrecoverable stream causes an atomic save and clean exit.
- RGB and depth watchdogs are independent; fully invalid trusted depth has its own consecutive-event threshold.
- Recovery telemetry: recoveries, soft resets, controller restarts, failures, last method, controller error.
- Bounded forensic log: `sensor_faults.jsonl` (+ one rotated previous file).
- Dashboard exposes RGB/depth/controller fault state and restart counters.
- Checkpoint format advances to 8; older v0.5/v0.4 compatible weights/state still migrate.

## Compatibility

- observation_dim = 6
- action_dim = 5
- L0/L1 mastered neural tensors remain compatible.
- Spatial cognition design is unchanged.
