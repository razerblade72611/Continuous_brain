# Continuous Brain v0.9.3 — Residual Dynamics Hotfix

This hotfix preserves the existing v0.9 child lineage and corrects the first embodied world-model failure observed after 2,000 updates.

## Changes
- World-model future prediction is now residual: `next_bus = current_visual_bus + learned_delta`.
- Residual heads are zero-initialized, so a new/migrated world model begins exactly at the honest no-change baseline.
- On first resume from a pre-v0.9.3 checkpoint, only the failed future-bus residual heads are reset. Other child weights are preserved and optimizer state is safely refreshed.
- World/temporal losses consume detached visual buses so dynamics learning cannot distort the visual cortex.
- Mixed nurseries use stratified sampling (`--world-fraction`, default 0.50) so lived transitions are not diluted by static replay lessons.
- World loss weight increased to 1.0 and temporal loss to 0.5 for embodied proof training.
- World evidence now reports executed-action diversity, not just demonstrator-action diversity.
- If world evidence is eligible, overall qualification now requires the world-model gate to pass.

## Why
The v0.9.2 child retained vision/language well, but on 41 held-out lived transitions its absolute future-bus predictor had error ~0.00301 versus a no-change baseline ~0.000272 and negative action advantage. More training on the same formulation would not be a clean architecture test.
