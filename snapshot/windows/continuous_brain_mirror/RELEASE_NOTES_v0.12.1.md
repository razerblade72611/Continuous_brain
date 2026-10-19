# Continuous Brain v0.12.1 release notes

## Purpose

Unify the strongest current branches without discarding learned state: v0.12 lifelong development remains intact, while the real 14k InfantBrain and competitive-workspace experiment are brought into the same release.

## Changes

- Bundles `state/child_v113_dml_smoke/infant_training.pt` unchanged.
- Integrates competitive broadcast v3 and the cued-detour no-leak experiment.
- Keeps the frozen InfantBrain permanently in eval mode inside the competitive wrapper, protecting BatchNorm buffers as well as parameters.
- Reuses the exact nucleus stream projections for both the frozen recurrent core and competition.
- Replaces DirectML on-device action one-hot scatter with an equality-based implementation while preserving historical CPU/CUDA behavior.
- Replaces DirectML GRU `.chunk(3)` gate splitting with explicit slices; parameter/state-dict schema is unchanged.
- Adds fresh-process DirectML qualification for recurrent backward depth and the real child/competitive graph.
- Keeps full BPTT for the scientific experiment; no hidden-state detach or truncated-BPTT workaround is enabled.
- Leaves the v0.12 lifelong checkpoint schema/build unchanged so existing local individuals can continue.

## What this release does not claim

DirectML on the user's RX 6600 has not been executed in the build environment. The new qualification script is the hardware gate. Competitive broadcast has not yet completed the 4,000-update scientific run. The two model lineages are packaged together but their weights are not merged. No result establishes consciousness or subjective experience.
