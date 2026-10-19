# Continuous Brain v0.13.0 — Executive Composition Phase 1

Status: **experimental next-intelligence branch; not canonical yet**

This branch starts only after the reusable working-memory mechanism passed its
pre-registered held-out DEV admission.

## Protected substrate

Frozen source checkpoint:

`state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt`

Expected SHA-256:

`f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c`

That checkpoint contains the admitted:

- real 14k historical brain lineage;
- 64-D gated writer;
- capacity-matched ungated nonlinear memory→workspace reader.

The executive experiment must not update any of those tensors.

## Scientific question

Can the brain itself use an already-written memory differently according to a
goal/instruction that becomes available only later?

This moves the post-hoc NORMAL/INVERT result into the neural brain architecture.

## Executive architecture

The new executive sees:

- current 64-D stored memory;
- late instruction ID.

It does NOT see:

- cue pixels;
- ordinary context vector;
- recurrent hidden state;
- a direct motor connection.

Architecture:

`memory64 -> bias-free 64→128 -> tanh`

`instruction -> embedding128 -> tanh`

elementwise multiplicative composition

`-> zero-init bias-free 128→128`

`-> additive shared-workspace residual`

`-> frozen admitted motor`

Instruction NONE is hard-masked to zero, permanently.

The world model continues to receive the pre-executive admitted workspace.
The executive therefore changes goal-conditioned action selection without
rewriting the learned physical-dynamics representation.

## Four-way counterfactual training unit

Every optimizer update contains exactly:

1. LEFT cue + NORMAL
2. RIGHT cue + NORMAL
3. LEFT cue + INVERT
4. RIGHT cue + INVERT

All four share one structural condition.

Thus neither cue alone nor instruction alone can solve the update.

## Late information rule

The sensory cue disappears first.

Only afterward does the separate executive instruction channel become active.

The instruction is never inserted into image/context or the working-memory
writer, so the stored cue cannot be rewritten to anticipate the instruction.

## Phase-1 gates

1. Audit all 1600 TRAIN conditions:
   - no instruction sensory leak;
   - cue pair becomes identical after cue erasure;
   - instruction arrives strictly after cue and before decision;
   - exact cue XOR instruction mapping;
   - old DEV/TEST IDs rejected.

2. Neural qualification:
   - exact initial behavioral no-op;
   - admitted brain excluded from optimizer;
   - update 1: zero-init output becomes gradient-bearing while upstream
     executive feature/goal maps remain zero-gradient;
   - update 2+: memory and instruction representations become gradient-bearing;
   - admitted brain remains bit-exact;
   - NONE stays an exact no-op even after executive learning.

3. 50-update balanced learning smoke:
   - four members/update;
   - executive-only optimizer;
   - portable checkpoint;
   - closed-loop TRAIN probe is descriptive only;
   - no old DEV/TEST access.

## Why old DEV is not reused

The working-memory DEV split has now been opened and its results are known.

It must therefore NOT become the held-out executive-development split.

If Phase 1 passes, the next production executive experiment will create a fresh
second-generation structural family with new factor values and new frozen
TRAIN/DEV/TEST partitions.

TEST from the old memory family remains sealed as historical final evidence.

## DirectML policy

DirectML is now an execution backend and regression constraint, not the
scientific target.

We will repair backend problems only when they block the architecture.
No further model design will be driven by DirectML quirks.
