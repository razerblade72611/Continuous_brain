# Continuous Brain v0.14.0 — Canonical Recalled-Content Executive

## Why this architecture changes

The v0.13.2 fresh novel-rule executive showed:

- executive OFF -> 0% on every novel rule;
- executive memory ZERO -> 0% on every novel rule;
- executive-only donor swap:
  - invert_lr 100%
  - vertical ~0%
  - vertical_reverse ~0%.

A no-training route-coherence diagnostic then showed that when the admitted
direct recall route D and executive route E are swapped coherently to the same
donor memory, donor-consistent behavior returns strongly:

- DE_DONOR vertical ≈85%
- DE_DONOR vertical_reverse ≈88%.

Therefore the problem was not absence of stored cue information. It was that
two independently memory-reading routes could be forced to disagree.

## Architectural decision

There is now ONE canonical recalled-content stream.

`memory64`
→ frozen admitted nonlinear recall
→ `canonical_recall128`

That same `canonical_recall128` is:

1. added to the historical workspace exactly as in the admitted memory system;
2. consumed by the late-bound executive.

The executive no longer reads `memory64` directly.

Active rule computation:

`canonical_recall128`
×
`rule_embedding128`
→ zero-init `Linear128→128`
→ executive workspace delta.

`RULE_NONE` remains an exact no-op.

## Protected substrate

Frozen:
- historical 14k brain;
- admitted 64-D writer;
- admitted nonlinear memory→workspace recall;
- motor;
- world model.

Trainable:
- rule embedding;
- final executive transform.

No second raw-memory reader exists.

## Fresh third-generation family

2400 new structural conditions:

- TRAIN 0..1599
- DEV 1600..1999
- TEST 2000..2399

Continuous geometry values are disjoint from both first- and second-generation
families.

Rules remain:

- invert_lr
- vertical
- vertical_reverse

This run is the final single-item rule-composition architecture check.

## Fresh DEV admission thresholds

For each rule:

- pair full >= 0.80;
- executive OFF <= 0.15;
- canonical memory ZERO <= 0.15;
- coherent canonical-memory donor consistency >= 0.90.

Across all six cue×rule members:

- six-way full >= 0.70.

RULE_NONE:

- exact no-op <= 1e-6.

Every reported factor level:

- each rule pair-full >= 0.60;
- six-way full >= 0.45.

## After PASS

Do NOT create another single-item rule family.

Freeze this canonical content architecture and move to:

1. multi-item / order-sensitive working memory;
2. operations over first/second/same/different remembered items;
3. goal-conditioned model-based planning using the existing PlanningCortex;
4. metacognitive confidence trained from prediction correctness;
5. long-horizon 3-D embodiment.

Fresh TEST remains sealed.
