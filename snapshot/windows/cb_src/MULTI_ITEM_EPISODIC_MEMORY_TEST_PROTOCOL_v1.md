# Continuous Brain v0.23.1 — Frozen Multi-Item Episodic Memory TEST

v0.23.0 passed every preregistered DEV criterion:

- overall accuracy: 1.000
- FIRST / SECOND / SAME / DIFFERENT: each 1.000
- all four queries correct for each pair: 1.000
- FIRST/SECOND attention routing: 1.000
- swapped FIRST/SECOND behavior: 1.000
- swapped SAME/DIFFERENT invariance: 1.000
- unequal-pair order reversal: 1.000
- donor targeted-slot following: 1.000
- donor untouched-slot preservation: 1.000
- zero targeted slot: 0.500
- zero non-targeted slot preservation: 1.000
- frozen single-item remembered-content decoding: 1.000

Historical protected artifacts remained unchanged and the composition TEST was
untouched.

## Purpose

This package performs the one frozen TEST admission promised after the DEV
pass.

No parameter is trained or modified.

TEST condition IDs:
- 1312..1599
- 288 conditions
- paired deterministically into two-event episodes
- all four LEFT/RIGHT event combinations
- all four late queries

The composition TEST is spent after this run regardless of outcome.

## Frozen admission gate

The TEST evaluator uses the exact same thresholds as the DEV gate:

Normal:
- overall >= .95
- each FIRST/SECOND/SAME/DIFFERENT >= .94
- all four queries correct per pair >= .90
- FIRST/SECOND attention routing >= .95

Causal interventions:
- slot-swap FIRST/SECOND >= .95
- slot-swap SAME/DIFFERENT invariance >= .95
- unequal-pair order reversal >= .95
- donor targeted slot >= .95
- donor non-target slot >= .95
- zero targeted slot <= .70
- zero non-target slot >= .95

Preservation:
- frozen single-item content decoding >= .95
- base, admitted memory checkpoint, goal decoder, DEV reader and DEV result
  remain byte-identical during evaluation.

## Scope

A PASS supports a narrow but meaningful claim:

The system has a two-item ordered working-memory mechanism where two distinct
remembered events coexist, late queries select FIRST or SECOND after storage,
SAME/DIFFERENT compare the pair, and direct interventions on slot identity and
content causally control the relevant outputs.

It does not establish general episodic memory, autobiographical memory,
unbounded capacity, broad general intelligence, or consciousness.

## Stop rule

After this run:
- composition TEST 1312..1599 is spent;
- do not tune v0.23 against it.

If PASS:
- freeze the v0.23 reader;
- archive the multi-item milestone;
- move to memory-selected goals and consequence-based planning/replanning.

If FAIL:
- keep the formal FAIL;
- do not tune against TEST;
- move to a new task family if the multi-item mechanism needs redesign.
