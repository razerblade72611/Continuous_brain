# Continuous Brain v0.4.3 — Navigation Bootstrap Correction

## Why this release

The first real v0.4.2 L1 run showed that the hierarchy/waypoint machinery was active but navigation could still be credited without genuine locomotion and the pre-emptive depth guard could reject valid adjacent-waypoint moves.

## Corrections

- L1 success radius reduced to 0.12 m by default.
- L1 success requires at least one successful physical translation.
- Early L1 paths require at least two grid transitions and initially target exactly two moves when the graph supports it.
- `ADVANCE` is dynamically masked until the active waypoint is aligned within 15° by default.
- Simulator training uses AI2-THOR native movement/collision outcome by default instead of pre-emptive depth `SafetyStop`.
- Optional legacy depth guard retained through `--depth-safety-training`.
- Aligned failed neighbor transitions are recorded, temporarily excluded, and replanned around immediately.
- Repeated edge failures are persisted and excluded from future teacher paths; successful traversal clears stale failures.
- Aligned idling receives a stronger penalty.
- L1 diagnostics now report physical displacement, depth, action success, translation counts, blocked advances, replans and known blocked edges.
- L1 graduation includes a blocked-move-rate constraint.

## Continual-learning / recovery defenses

- bounded replay persists across restarts in `replay.pt`;
- persisted replay reserves a slice for each represented skill to protect old-skill rehearsal;
- incompatible replay signatures are skipped safely when model/sensor dimensions change;
- automatic `skills/L<N>_mastered.pt` snapshots are produced on graduation;
- checkpoint writes are atomic;
- `brain.previous.pt` is retained and automatically used if `brain.pt` is unreadable;
- trusted local checkpoint loading explicitly supports modern PyTorch `weights_only` behavior.

## Compatibility

The neural observation/action dimensions are unchanged from v0.4.2/v0.4.1/v0.3. Compatible neural weights continue to load. v0.4.2 environment statistics migrate; new v0.4.3 fields start from safe defaults.

## Recommended experiment

Preserve all v0.4.1/v0.4.2 navigation states as diagnostics. Build the new canonical branch from the preserved v0.3 neural checkpoint, retrain L0 until the automatic `L0_mastered.pt` snapshot appears, then continue the same state into L1.
