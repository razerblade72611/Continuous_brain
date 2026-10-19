# Continuous Brain v0.9.5 — Continuation

Use the same persistent child. Do not initialize a new lineage.

```bash
export CHILD="$HOME/projects/continuous_brain_v0.9.0/continuous_brain_project/state/child_v090"
```

The first resume from a v0.9.4 checkpoint should report:

- `world_state_conditioning_migrated: true`
- `optimizer_reinitialized: true`
- `world_dynamics_migrated: false`

This means the new zero-initialized state-conditioning adapters were added while the
learned v0.9.4 residual heads were preserved.

For the world-development phase use `--world-focus` and a high lived-transition
fraction. Do not train on a holdout dataset after it has been designated for evaluation.
