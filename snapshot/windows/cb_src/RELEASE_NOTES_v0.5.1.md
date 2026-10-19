# Continuous Brain v0.5.1 — Sensor Integrity Correction

v0.5.1 is a defensive patch over the v0.5 spatial-cognition milestone. A real Discovery L2 run showed apparent first-person RGB staleness, later depth collapse to 0.00, increasing blocked moves, and occupancy evidence being washed out.

## Corrections

- Independent RGB freshness watchdog based on sampled-frame hash and mean pixel change when pose/yaw actually changes.
- Camera frame sequence/hash/diff/stale counters exposed in terminal and dashboard.
- Stale RGB ends the affected episode instead of allowing prolonged learning from a frozen visual stream.
- Policy-facing depth sectors use the same trusted-depth philosophy as the spatial mapper; near-zero/self-occluded values are treated as unknown during simulator training.
- Physical/non-native depth safety mode remains fail-closed when depth is invalid.
- Spatial map refuses to integrate stale RGB/depth events.
- Native collision cells become persistent hard-occupied evidence and cannot be erased by later free-depth rays; occupying a cell clears an impossible stale lock.
- Spatial state format 2 migrates v0.5.0 maps by adding an empty hard-collision mask.
- Curriculum statistics include sensor-fault rate; sensor faults prevent mastery.
- Dashboard camera renderer reuses its offscreen canvas and visibly overlays `RGB SENSOR STALE` instead of silently swallowing a frozen/error state.
- Shape-aware checkpoint migration now resets Adam moments whenever model parameter shapes changed, preventing a delayed first-optimizer-step crash.

## Compatibility

- Neural interface remains 6 inputs / 5 actions.
- Checkpoint format remains 7.
- v0.5.0 checkpoints and spatial state load directly.
- v0.4.3-style shape migrations retain compatible neural tensors and safely reset incompatible optimizer/replay state.
