# Continuous Brain v0.10.0 — Dedicated Recurrent Imagination

## Summary

v0.10.0 adds a child-owned recurrent planning cortex as a migration-safe extension of the successful v0.9 parented-development lineage.

## Evidence motivating v0.10

- P3 independent predictive world-model gate: passed before this release.
- v0.9.6 P4a: imagination materially improved outcomes and nearly eliminated wall collisions while the reactive child remained frozen.
- P4b diagnostic variants using the old planning representation: zero detour completions despite strong learned value/return critics.

The release therefore changes the planning representation instead of adding another sidecar critic.

## New

- `PlanningCortex` in the persistent child checkpoint.
- Separate recurrent latent for imagined states.
- Multi-step self-rollout training from contiguous lived episodes.
- Learned spatial/context reconstruction, reward, collision, success, return, action-value and auxiliary bounded goal heads.
- `continuous-parented train-planning` planning-only development command.
- `continuous-parented evaluate-planning` paired reactive-vs-planning qualification command.
- Precommitted P4b completion gate.
- `scripts/check_v100_colab_readiness.py` executable Colab handoff gate.
- v0.9.x migration permits only missing `planning.*` tensors and forces future optimizer refresh when a pre-v0.10 checkpoint enters ordinary whole-brain training.

## Preservation

The planner is not wired into the reactive motor path. Planning-only development freezes all pre-v0.10 parameters. A v0.9.6 checkpoint therefore retains its existing vision, language, motor, value, nucleus and world-model behavior until planning is explicitly invoked.

## Scientific status

Software tests prove migration and isolation properties; they are not evidence that P4b passes. P4b requires the user's trained persistent child to complete the fresh paired detour qualification described in `RUN_v0.10.0.md`.

## Scaling note

Passing v0.10.0 does not authorize an in-place switch from the persistent proof profile to `balanced` or `scale`; those profiles change tensor dimensions. The first Colab phase should train the complete current v0.10 child. A subsequent explicit, tested capacity-expansion migration is required before widening the persistent lineage.
