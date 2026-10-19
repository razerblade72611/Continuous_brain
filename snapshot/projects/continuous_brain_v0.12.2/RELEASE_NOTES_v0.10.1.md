# Release Notes — Continuous Brain v0.10.1

## Headline

**Conservative Bellman valuation repair for the dedicated planning cortex.**

v0.10.1 follows a decisive v0.10.0 diagnostic: the recurrent planner learned action physics correctly, but its learned values did not rank the necessary UP detour action. Search variants and beam-diversity variants did not fix completion, ruling out simple beam collapse as the primary issue.

## Added

- `continuous_brain/planning_v101.py`
  - offline fitted Bellman Q learning;
  - slowly moving target Q network;
  - balanced action sampling;
  - conservative log-sum-exp penalty;
  - empirical lived-return support clamp;
  - actual-vs-imagined next-latent Q consistency;
  - post-training bit-for-bit non-Q isolation verification.
- `continuous-parented train-planning-q` / `python -m continuous_brain.parented_cli train-planning-q`.
- `scripts/p4b_bellman_q_probe_v101.py` for development-only Q/action ranking diagnostics.
- The earlier v0.10 search and action-semantics probes are included for reproducibility.
- `tests/test_v101.py`.

## Preserved

- Every tensor shape from v0.10.0.
- The already-trained recurrent planning encoder/transition/decoders and outcome heads.
- The entire pre-v0.10 child.
- The frozen P3/P4a lineage and source training-update count.
- The P4b threshold and reserved blind seeds.

## Training isolation

Only these tensors may change in the v0.10.1 training stage:

```
planning.q_head.0.weight
planning.q_head.0.bias
planning.q_head.2.weight
planning.q_head.2.bias
```

The trainer raises an exception if any other model tensor changes.

## Qualification policy

Seeds `9001–9010` remain development-only because they have already influenced architecture decisions. Seeds `12001–12100` remain sealed until the v0.10.1 search rule is frozen and development evaluation shows real detour completion.

Do not lower the P4b gate and do not train on the blind seeds.
