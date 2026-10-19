# Continuous Brain v0.9.5 — Direct State-Conditioned Dynamics

v0.9.5 is a migration-safe world-model refinement for the parented-development line.
It preserves the v0.9.4 child and adds direct current-state conditioning to each
world-model ensemble member.

## Why

The v0.9.4 unseen holdout demonstrated positive action sensitivity and a passing
spatial gate, but structured-state prediction remained below the precommitted 20%
relative-improvement threshold. Inspection showed that residual deltas were predicted
from `workspace + action` while current spatial/context state was only added as the
residual baseline. Physical dynamics therefore had to recover exact state indirectly
through the global workspace.

## Changes

- Each world member now has zero-initialized `state_bus` and `state_context`
  conditioning adapters.
- Existing v0.9.4 predictions are preserved exactly at migration because the new
  adapters initialize to zero.
- No learned v0.9.4 residual heads are reset.
- The optimizer is refreshed once so the new adapter parameters are trainable.
- `--world-focus` adds a developmental specialization phase: causal/spatial losses
  dominate while mature language/vision/motor skills continue receiving smaller
  rehearsal gradients.
- Evaluation reports per-dimension structured-context error, baseline, and relative
  improvement for diagnosis without changing the qualification threshold.
- The world-model gate remains unchanged: spatial prediction must beat no-change by
  at least 2% with positive action advantage; structured state must beat no-change by
  at least 20% with positive action advantage; success accuracy must be >= 0.70.

## Scientific status

v0.9.5 does not declare P3 passed. It exists to test whether direct state conditioning
closes the remaining structured-state gap on an unseen holdout while preserving prior
parented competencies.
