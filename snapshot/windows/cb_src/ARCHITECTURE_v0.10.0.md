# Continuous Brain v0.10.0 — Dedicated Planning Cortex

v0.10.0 is the final proof-scale architecture gate before the first intended major whole-brain Colab training phase on the current persistent child's dimensions. Capacity expansion is deliberately separate because switching to the existing `balanced` profile would change tensor dimensions and cannot preserve this checkpoint without an explicit migration.

## Why this change exists

The v0.9.5 world model passed its independent P3 blind gate. v0.9.6 then showed that consulting learned consequences can materially improve behavior: imagination almost eliminated repeated wall collisions and improved detour-world reward by about 96% while the reactive motor policy remained bit-for-bit frozen. However, reactive, one-step, recurrent-workspace, state-value, goal-value, full-return, and action-tail variants all remained at zero detour completion in the diagnostic task.

That evidence points to an architectural limitation rather than a missing critic target. The ordinary recurrent nucleus represents lived perception. Reusing it as though a new hypothetical visual observation had occurred creates internally inconsistent imagined states and rollout drift.

## New organ: `PlanningCortex`

The child now owns a separate recurrent planning system:

1. **Planning encoder** — initializes a planning latent from the slow child-owned dorsal/spatial bus plus the 7-value structured dynamics context.
2. **Action recurrent transition** — a GRUCell advances the planning latent under hypothetical actions. It never claims that a new real image was perceived.
3. **State decoders** — reconstruct predicted dorsal/spatial state and structured physical context.
4. **Outcome heads** — predict immediate lived reward, success and collision.
5. **Value head** — predicts full discounted lived return.
6. **Action-value head** — predicts return conditioned on a candidate action.
7. **Goal head** — predicts a bounded time-to-success-derived goal potential as an auxiliary latent-shaping target; it is not added to the qualification action score.

The lived `CorticalNucleus` and the imagined planning recurrent state are intentionally different state machines.

## Training

`continuous-parented train-planning` trains only `planning.*`. Every pre-v0.10 parameter and BatchNorm buffer is frozen. Training uses complete, contiguous lived episodes and explicitly feeds the planning cortex its **own predicted latent** over multi-step unrolls. Ground-truth next states are targets, never teacher-forced replacements for the predicted planning latent.

The local proof defaults are:

- unroll: 12 steps
- discount: 0.95
- goal discount: 0.97
- planning batch: 32 sequence windows
- planning updates: 4,000

## Migration

A v0.9.x checkpoint has no `planning.*` tensors. `load_checkpoint()` treats only those new tensors as a permitted v0.10 migration. All existing tensors are loaded exactly. The new planner is not consulted by reactive behavior until explicitly selected by the v0.10 planning evaluator.

Planning-only training saves no legacy whole-brain optimizer state; the future integrated optimizer must be intentionally initialized with the new organ included.

## P4b qualification

The final planning gate is deliberately stronger than the earlier reward-only P4a gate. On the untouched final paired seed range, at least one planning horizon must achieve:

- absolute completion >= 20%,
- completion advantage over the unchanged reactive policy > 5 percentage points,
- no more than 2 additional collisions per episode relative to reactive.

Reward improvement alone cannot pass P4b.

## Colab handoff

Major whole-brain Colab training begins only after:

1. P4b blind planning passes,
2. the P3 final blind world-model gate still passes,
3. bootstrap vision/language retention remains strong,
4. all pre-v0.10 tensors are still bit-for-bit equal to the frozen P4a source checkpoint.

`scripts/check_v100_colab_readiness.py` makes those requirements executable.

This architecture is a research system. Passing these engineering gates is not evidence of consciousness, personhood, or AGI.

## Parameter profiles in this release

Fresh v0.10 models instantiate approximately:

- proof: 8,980,579 trainable child parameters, including 428,301 planning parameters,
- balanced: 52,203,628 trainable child parameters, including 1,692,173 planning parameters,
- scale: 181,101,941 trainable child parameters, including 3,042,701 planning parameters.

These are architecture capacities, not interchangeable checkpoint sizes. The persistent proof-profile child must not be loaded as `balanced` or `scale` without a future explicit capacity-expansion migration.
