# Continuous Brain v0.4.2 — Hierarchical Navigation Correction

v0.4.2 responds directly to the first long real L1 AI2-THOR run, where the cumulative goal count did not increase and almost all logged distance deltas were zero.

## Changes

- Separate rolling statistics/counters for every curriculum level.
- v0.4.1 mixed curriculum statistics are preserved only as legacy diagnostics during migration and do not seed v0.4.2 graduation windows.
- Short collision-free L1 teacher paths over `GetReachablePositions` cells.
- Active waypoint goal signal separated from final target.
- Adaptive L1 path difficulty (short first, longer as success improves).
- L1 angular shaping retained from L0.
- Stronger distance-progress signal, successful-forward bonus, blocked/SafetyStop penalty and wrong-way penalty.
- L1 stagnation resets on angular progress, waypoint progress or distance progress.
- Actual physical movement distance reported separately from target-distance delta.
- Curriculum-specific epsilon floor restarts exploration when a new level is unlocked.
- Prior-level rehearsal episodes during L1+ training.
- Replay transitions tagged by skill level and legal action set.
- Stratified replay biases the current skill while retaining previous-skill rehearsal.
- TD target action masking is now per transition rather than based on the currently active episode.
- Per-level goal counts and replay-level counts added to diagnostics.
- Browser monitor displays active waypoint vs final target, configured level vs rehearsal level, path position and epsilon floor.
- Checkpoint format advances to 5; neural tensor shapes remain compatible with v0.4.1.

## Recommended L1 recovery settings

```text
rehearsal_fraction      0.20 initially
nav_initial_path_steps  2
nav_max_path_steps      8
goal_dropout             0.0
teacher-nav              enabled
```

Do not increase goal dropout until L1 becomes reliable under waypoint supervision.
