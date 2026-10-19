# Continuous Brain v0.13.2 — Fresh Novel-Rule Executive

## Why this exists

Executive Composition v1 learned NORMAL/INVERT extremely well, but failed one
pre-registered causal criterion:

- swapping only the executive's memory under NORMAL did not make behavior
  follow the donor.

The failure is interpretable: NORMAL was already solved by the admitted brain,
so the learned executive became a conditional INVERT override.

We preserve that v1 gate as a formal FAIL. We do not change its threshold.

v0.13.2 removes the inherited-policy shortcut entirely.

## Fresh family

New procedural family:
- 2400 unique structural conditions;
- TRAIN 0..1599;
- DEV 1600..1999;
- TEST 2000..2399.

Continuous geometry values are disjoint from the first-generation task family.
The split is structurally disjoint and non-wrapping.

## Late rules

There is NO NORMAL rule.

Every active rule is novel:

1. `invert_lr`
   - LEFT memory -> RIGHT
   - RIGHT memory -> LEFT

2. `vertical`
   - LEFT memory -> UP
   - RIGHT memory -> DOWN

3. `vertical_reverse`
   - LEFT memory -> DOWN
   - RIGHT memory -> UP

Instruction arrives after cue erasure and is absent from image/context.

Every update contains all six cue×rule counterfactuals.

## Protected brain

Frozen:
- historical 14k substrate;
- DEV-admitted 64-D writer;
- DEV-admitted memory→workspace reader;
- motor/world model.

Trainable:
- fresh 24,960-parameter rule executive only.

## Causal DEV admission

Pre-registered before model DEV evaluation.

For EACH novel rule:
- pair full-action >= 0.80;
- executive-off pair full <= 0.15;
- memory-zero pair full <= 0.15;
- opposite-cue memory donor consistency >= 0.90.

Across all six cue×rule combinations:
- six-way full-action >= 0.70.

NONE rule:
- exact no-op <= 1e-6.

Every reported structural factor level:
- each rule pair-full >= 0.60;
- six-way full >= 0.45.

A PASS demonstrates that the brain can reuse the same admitted memory under
multiple genuinely novel late-bound rules, with both memory and rule identity
required.

Fresh TEST remains sealed.

## After PASS

Do not add another binary-rule variant.

Move to:
1. multi-item/order-sensitive working memory;
2. executive operations over multiple remembered items;
3. goal-conditioned model-based planning;
4. metacognitive confidence;
5. long-horizon 3-D embodiment.
