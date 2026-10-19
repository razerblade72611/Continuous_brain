# Continuous Brain v0.10.2 Release Notes

## Main change

Adds a migration-safe child-owned temporal Option / Intention Cortex for learning multi-step behavioral organization from successful lived trajectories.

## Added

- `InfantConfig.option_count` (proof default: 6 anonymous options)
- `InfantConfig.option_dim` (proof default: 64)
- `OptionCortex`
  - option embedding
  - option selector
  - learned termination head
  - option-conditioned primitive action policy
  - option-conditioned action-value head
- `continuous_brain.planning_v102`
- `train-options` CLI
- `evaluate-options` CLI
- deterministic unsupervised option discovery from successful lived future-motion descriptors
- option-only training isolation guard
- option-guided recurrent-imagination evaluator
- v0.10.2 Colab-readiness checker
- `tests/test_v102.py`

## Preserved

v0.10.2 preserves the v0.10.1 recurrent planning dynamics and conservative Bellman Q. During proof option training every non-option tensor is frozen and verified bit-for-bit unchanged.

## Qualification status

P4b is still OPEN until the trained v0.10.2 child is evaluated. Reserved blind seeds `12001–12100` must remain untouched until development seeds justify freezing the option-planning configuration.
