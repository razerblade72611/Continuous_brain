# Continuous Brain v0.10.4 Release Notes

## Purpose

v0.10.3 passed the fixed 100-episode blind P4b hierarchical-planning gate, but the post-P4 retention check showed that the earlier ForagingWorld world-model proof no longer generalized through the current checkpoint. Vision, language, motor, options, and hierarchical planning remained intact. v0.10.4 addresses that localized continual-learning failure.

## Added

- `continuous_brain/world_consolidation_v104.py`
- `collect-world-holdout`
- `seal-world-holdouts`
- `train-world-consolidation`
- `evaluate-world-consolidation`
- equal-weight multi-domain rehearsal
- action-balanced sampling within every world domain
- parent-free raw retention-holdout collection
- image-inclusive transition digests and pre-training holdout seal
- exact non-world tensor/buffer preservation check
- `scripts/check_v104_colab_readiness.py`
- `tests/test_v104.py`

## Fixed training contract

Default consolidation configuration:

- training domains: v0.9.5 non-blind mixed nursery + v0.9.6 detour training nursery;
- updates: 3000;
- batch per domain: 16;
- learning rate: `1e-4`;
- seed: `10410`;
- optimizer scope: `world.*` only;
- equal domain weighting: enabled;
- per-domain action balancing: enabled.

No final holdout metric is used for early stopping or model selection.

## Fixed qualification contract

Before training, collect and seal:

- fresh ForagingWorld holdout: 1200 transitions, seed `24017`, exploration `0.45`;
- fresh DetourWorld holdout: 1200 transitions, seed `24019`, exploration `0.45`.

These holdouts are training-prohibited and contain no external-parent targets.

## P4b status

The frozen v0.10.3 blind result remains a historical PASS. v0.10.4 does not retrain `planning.*` or `options.*`, and the readiness checker requires all non-world tensors to remain exactly equal to the frozen v0.10.3 source checkpoint.

## Colab status

Not ready merely by installing v0.10.4. Readiness requires:

1. frozen v0.10.3 blind P4b result still valid;
2. fresh sealed ForagingWorld world retention PASS;
3. fresh sealed DetourWorld world retention PASS;
4. bootstrap vision/language retention PASS;
5. only `world.*` changed from the frozen v0.10.3 checkpoint.
