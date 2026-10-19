# Continuous Brain v0.14.1 — Canonical Executive Convergence v1

## Diagnosis that motivates this run

v0.14.0 failed fresh v3 DEV, but the post-failure diagnostic established:

- fixed-final TRAIN mean rule-pair accuracy: ~0.558
- fixed-final DEV mean rule-pair accuracy: ~0.578
- TRAIN six-way: ~0.428
- DEV six-way: ~0.420
- raw memory64 TRAIN→DEV cue decoding: 1.000
- canonical recall128 TRAIN→DEV cue decoding: 1.000
- recurrent hidden TRAIN→DEV cue decoding: 1.000

Therefore the dominant failure is **executive underlearning/optimization**, not
loss of cue information in memory or canonical recall.

## Architectural rule

This experiment changes **no architecture**.

Protected:
- historical 14k brain;
- admitted 64-D writer;
- admitted nonlinear recall;
- canonical recall content path;
- motor/world model.

Trainable:
- the existing 16,896-parameter canonical rule executive only.

The exact update-1600 executive and Adam moments are resumed.

## Training-only convergence stages

The original epoch 1 was updates 1..1600.

Additional deterministic TRAIN epochs:

- update 1601..3200: lr 2e-4
- update 3201..4800: lr 1e-4
- update 4801..6400: lr 5e-5
- update 6401..8000: lr 2.5e-5

Each epoch:
- reshuffles the same 1600 TRAIN structural conditions;
- uses every condition once;
- retains all six cue×rule counterfactuals per optimizer step.

No v3 DEV metric is consulted during convergence.

## Frozen TRAIN convergence gate

After each full additional epoch, evaluate all 1600 TRAIN conditions at fixed
weights.

Required:

For EACH rule:
- pair full >= 0.95
- coherent canonical donor consistency >= 0.95
- executive-off pair <= 0.05
- canonical-memory-zero pair <= 0.05

Across all six cue×rule combinations:
- six-way full >= 0.90

RULE_NONE:
- exact no-op <= 1e-6

Every structural factor level:
- each rule pair-full >= 0.85
- six-way full >= 0.75

Stop at the first epoch that passes.

If the gate still fails at update 8000, conclude that optimization time alone is
insufficient and redesign the executive transform—not the admitted memory.

## Held-out policy

The already-opened v3 DEV is NOT used to select learning rate, epoch count, or
architecture during this convergence run.

After TRAIN convergence:
- do not claim v3 DEV is fresh again;
- create a new v4 held-out structural family for final executive generalization.

v3 TEST remains sealed.
