# Frozen Writer — Post-Write Instruction Composition Probe v1

## Question

Can the trained 64-D memory be used in a new computation whose meaning is
determined only after the memory has already been written?

## Frozen substrate

Use the winning capacity-ungated update-1600 checkpoint.

The entire original brain is frozen.

During brain rollout there is NO NORMAL/INVERT instruction.

Only after all decision-time memories are collected do we create the new
instruction variable.

Therefore the writer cannot encode, anticipate, or respond to the later
instruction.

## New task

Memory encodes past cue: LEFT or RIGHT.

New instruction:
- NORMAL: respond with remembered cue side.
- INVERT: respond with the opposite side.

Each frozen memory is evaluated under both instructions.

## New readers

### Joint MLP
`[memory64, instruction1] -> 64 tanh -> 2`

This newly initialized reader must learn the late-bound composition.

### Linear concatenation control
`[memory64, instruction1] -> 2`

The cue×instruction operation is XOR-like and not linearly separable in the
ideal binary abstraction. This tests whether nonlinear interaction is required.

### Factorized reader
A new linear `memory64 -> cue2` decoder is trained independently. The later
instruction then preserves or flips that cue prediction at inference.

This is an explicit content-reuse control.

## Evaluation

- deterministic five-fold condition-held-out CV;
- leave-one-corridor-length-level-out;
- leave-one-cue-steps-level-out;
- three independent reader seeds.

Joint-reader interventions:
- memory zeroed;
- instruction forced NORMAL;
- instruction counterfactually flipped;
- LEFT/RIGHT memories swapped within condition.

## Pre-registered TRAIN-side mechanism expectations

Five-fold means:
- joint MLP overall >= 0.95;
- both NORMAL and INVERT correct per same stored memory >= 0.90;
- instruction-flip counterfactual >= 0.95;
- swapped-memory donor-consistency >= 0.95;
- memory-zero <= 0.60;
- forced-NORMAL-instruction <= 0.60;
- factorized reader >= 0.95.

These are mechanism-assay thresholds, not DEV admission thresholds.

## Scope

This does NOT establish environmental writer generalization because the writer
has previously seen the TRAIN family.

It does test whether stored content can be reinterpreted according to
information that arrives after writing, which is a stronger criterion than
simple cue classification.

DEV and TEST remain sealed.
