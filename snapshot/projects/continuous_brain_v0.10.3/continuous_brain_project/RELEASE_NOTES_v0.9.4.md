# Continuous Brain v0.9.4 — Dual-Stream Predictive Development

v0.9.4 preserves the parented-development thesis but changes what the child's world model is asked to predict.

## Why this release exists

The real v0.9.3 child became action-sensitive (`world_action_advantage > 0`) but its world prediction remained slightly worse than the no-change baseline. Meanwhile vision/language retention remained strong. The failure exposed a representation problem: the same DINO-aligned global semantic bus was being used for both recognition/language and fine action-conditioned dynamics.

Biological vision separates these jobs. v0.9.4 therefore separates them too.

## New architecture

- **Ventral semantic visual cortex** — the existing child visual cortex is preserved and continues to learn from DINO parent embeddings. It answers primarily “what is present?”
- **Dorsal spatial visual cortex** — a new child-owned raw-pixel cortex starts immature and preserves a denser spatial feature grid. It answers primarily “where is it / how can action change it?”
- **Identity-preserving fusion** — the nucleus receives a learned fusion of both streams. The fusion layer is initialized as exact semantic-stream identity plus zero spatial contribution, so migration does not abruptly change the already-learned nucleus input.
- **Slow EMA target cortices** — child-owned target copies follow the online cortices slowly and receive no gradients. The world model predicts the slow dorsal target instead of chasing a rapidly drifting representation.
- **Structured state dynamics** — the world ensemble predicts both dorsal latent change and the next structured body/world state.
- **Counterfactual action ranking** — the actually executed action must explain the observed transition better than an alternative action.
- **Dedicated dorsal temporal cortex** — the old temporal cortex is preserved; a new temporal organ learns action-conditioned spatial transitions.
- **Motor evidence masking** — imported static language/vision lessons no longer silently supervise the motor cortex as “idle.”
- **World outcome masking** — reward/success world losses are trained only on actual transition evidence.

## Migration

A v0.9.3 child can resume directly.

Preserved:
- ventral visual cortex and all DINO parent heads
- language cortex
- recurrent nucleus/workspace
- old semantic temporal cortex
- motor cortex
- value cortex
- world reward/success heads
- identity, parent registry, weaning state, continuity store

New/initialized:
- dorsal spatial cortex
- slow dorsal target cortex
- spatial-state grounding head
- semantic+dorsal fusion layer (identity on old semantic stream)
- dedicated dorsal temporal cortex
- structured context-dynamics residual heads

Reset deliberately:
- only the failed v0.9.3 future-bus residual outputs, because their target coordinate system changes from ventral semantic to dorsal spatial.

## Qualification

World evidence now requires:
- at least 50 held-out world-valid transitions,
- at least 50 structured next-state targets,
- at least two executed action classes with >=3 held-out examples each.

World-model qualification requires all of:
1. dorsal/spatial prediction beats no-change by at least 2%,
2. correct-action spatial prediction beats wrong-action prediction,
3. structured-state prediction beats no-change by at least 20%,
4. correct-action structured prediction beats wrong-action prediction,
5. success prediction accuracy >= 0.70.

The semantic/language/vision gates remain separate and must continue to pass.
