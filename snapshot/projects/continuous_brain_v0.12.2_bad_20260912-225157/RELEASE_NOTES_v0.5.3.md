# Continuous Brain v0.5.3 — Sensor Hysteresis & Recovery-Storm Correction

v0.5.3 closes the final planned v0.5 infrastructure issue exposed by the real Discovery L2 run: depth-only invalidity was being treated as a dead controller, producing repeated scene-reset storms even while RGB and native AI2-THOR physics remained usable.

## Corrections

- Native-collision simulator training no longer treats depth-only invalidity as an infrastructure fault by default.
- Invalid/untrusted depth remains visible as `depth unknown`, is excluded from trusted mapping/policy clearance, and does not poison spatial memory.
- Physical/non-native collision mode remains fail-closed on invalid depth.
- RGB staleness and controller exceptions remain true recoverable infrastructure faults.
- Recovery verification requires multiple independent pose-changing fresh-RGB probes.
- Native simulator recovery verification does not require valid depth; physical/depth-safety mode does.
- Added post-recovery cooldown and a separate rolling soft-reset budget.
- Added per-cause telemetry for RGB faults, depth faults, controller faults, and depth-unknown observations.
- New runtime counters persist in environment state.
- Dashboard distinguishes diagnostic `depth unknown` from an authoritative sensor fault.
- Added `run_l2_spatial_v053.sh` with conservative recovery-hysteresis defaults.
- Added `NEUROMORPHIC_ROADMAP.md`, defining an optional hybrid spiking-neural path without replacing the dense pretrained-compatible cognitive core.

## Project-direction rule

v0.5.3 is intended to be the final dedicated spatial/simulator-infrastructure release unless a future defect prevents experimentation entirely. After validating a useful L2 window, development moves upward to semantic hierarchical skills and foundation-model cognition rather than pursuing perfect AI2-THOR navigation.
