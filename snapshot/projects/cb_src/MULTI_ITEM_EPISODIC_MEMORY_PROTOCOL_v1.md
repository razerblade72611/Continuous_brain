# Continuous Brain v0.23.0 — Multi-Item Episodic Working Memory

## Why this milestone starts now

v0.22.2 is archived and the delayed LEFT/RIGHT temporal-skill assay is closed.

The best preserved mechanism now supports:

past cue
-> admitted 64-D working memory
-> canonical recall128
-> remembered goal
-> goal-conditioned temporal skill
-> recurrent within-skill temporal state
-> autonomous action

The next missing capability is not another motor/termination repair. It is the
ability to keep more than one distinct remembered event and select or compare
them later.

## Preservation rule

The admitted single-item route remains exactly intact:

historical brain
-> admitted 64-D writer
-> admitted canonical nonlinear recall128

v0.23 does not resize, retrain, overwrite, or bypass that route.

Instead, canonical recalled content is copied at explicit event boundaries into
a new append-only two-slot episodic working-memory sidecar:

event 1 recall128 -> slot 0
event 2 recall128 -> slot 1

The two slots preserve event identity by order.

This is an additive mechanism. It is not a claim of general autobiographical or
long-term episodic memory.

## Late-query rule

The query is not available while either event is encoded or stored.

Only after both slots contain content is one of four query tokens supplied:

FIRST
    answer the remembered side of event 1

SECOND
    answer the remembered side of event 2

SAME
    answer whether the two remembered event sides match

DIFFERENT
    answer whether the two remembered event sides differ

Binary answer semantics:

FIRST / SECOND:
    0 = LEFT
    1 = RIGHT

SAME:
    0 = no
    1 = yes

DIFFERENT:
    0 = no
    1 = yes

Therefore one stored pair must support several different computations whose
meaning is determined only after writing.

## Learned late-query reader

Input:
- slot0 canonical recall128
- slot1 canonical recall128
- slot validity
- late query token

Architecture:
- shared content encoder;
- learned order/position embeddings;
- learned query embedding;
- query-conditioned attention over the two ordered slots;
- explicit relation features:
  slot0, slot1, absolute difference, elementwise product;
- binary answer head.

FIRST/SECOND receive an auxiliary attention-routing target so the architecture
must learn which ordered slot is being requested rather than merely treating the
two memories as an unordered concatenation.

## Composition split

This experiment intentionally uses only historical single-item TRAIN condition
IDs because the writer itself has already been admitted and is frozen.

New multi-item composition split:

TRAIN:
    condition IDs 0..1023

DEV:
    condition IDs 1024..1311

TEST:
    condition IDs 1312..1599

The splits refer to the new event-pair/order/query composition problem.

Environmental writer generalization is NOT claimed. TEST remains sealed in
v0.23.0.

Each condition is paired deterministically with a different condition. Every
pair is instantiated under all four event-side combinations:

LEFT / LEFT
LEFT / RIGHT
RIGHT / LEFT
RIGHT / RIGHT

and all four late queries.

## Causal interventions

A score alone is insufficient.

### Slot swap
Swap slot0 and slot1 after writing.

Required:
- FIRST follows former SECOND content;
- SECOND follows former FIRST content;
- SAME and DIFFERENT remain invariant.

### Donor replacement
Replace one slot with opposite-side canonical content from another condition.

Required:
- the query targeting the replaced slot follows donor content;
- the query targeting the untouched slot remains correct.

### Targeted slot zero
Zero one slot and clear its validity bit.

Required:
- accuracy for a query targeting that slot falls toward chance;
- the query targeting the intact slot remains accurate.

### Unequal-pair order reversal
For LEFT/RIGHT or RIGHT/LEFT pairs, swapping the slots must reverse the
FIRST/SECOND answer.

These interventions establish content- and order-dependent causality.

## Frozen DEV gate

Normal behavior:
- overall answer accuracy >= 0.95;
- each FIRST/SECOND/SAME/DIFFERENT accuracy >= 0.94;
- all four queries correct for the same pair >= 0.90;
- FIRST/SECOND attention-route accuracy >= 0.95.

Swap:
- swapped FIRST/SECOND accuracy >= 0.95;
- SAME/DIFFERENT swap-invariance accuracy >= 0.95;
- unequal-pair order reversal >= 0.95.

Donor:
- replaced-slot targeted query >= 0.95;
- untouched-slot query >= 0.95 for both slot interventions.

Zero:
- targeted query <= 0.70 after its slot is removed;
- non-targeted query >= 0.95.

Preservation:
- frozen v0.21 single-item goal decoder remains >= 0.95 on DEV recalls;
- historical base checkpoint unchanged;
- admitted writer/recall checkpoint unchanged;
- v0.22.2 milestone untouched;
- composition TEST untouched.

## Meaning of PASS

A PASS establishes an additive two-item ordered working-memory mechanism in
which:
- two distinct past events coexist;
- their order/identity is retained;
- a query arriving only after storage selects FIRST or SECOND;
- relational SAME/DIFFERENT operations use both;
- slot swap, donor replacement, and slot removal causally alter the appropriate
  outputs.

This is a stronger memory capability than single delayed-cue classification.

It does NOT establish:
- general episodic memory;
- autobiographical memory;
- unlimited memory capacity;
- consciousness.

## After PASS

Freeze the two-slot mechanism.

Use the sealed composition TEST once.

If TEST also holds, the next major milestone is to connect selected episodic
content to:
1. explicit goals;
2. the preserved temporal-skill pathway;
3. consequence-based planning/replanning;
4. richer 3-D remember-now/use-later tasks.

## After FAIL

Do not reopen the old single-item memory admission or v0.22 temporal-skill
assays.

Classify the failure as one of:
- slot identity/order representation;
- late-query selection;
- relation comparison;
- content interference.

Repair only that new multi-item layer.
