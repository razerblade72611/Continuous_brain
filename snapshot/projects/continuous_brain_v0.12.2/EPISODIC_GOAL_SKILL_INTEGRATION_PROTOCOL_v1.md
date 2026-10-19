# Continuous Brain v0.24.0 — Episodic Selection → Goal → Temporal Skill

## Starting point

v0.23.1 is a frozen TEST PASS.

The two-slot mechanism supports, on sealed composition TEST:
- FIRST: 1.000
- SECOND: 1.000
- SAME: 1.000
- DIFFERENT: 1.000
- ordered slot routing: 1.000
- slot-swap causal behavior: 1.000
- donor replacement causal behavior: 1.000
- targeted slot removal: chance on the removed item, 1.000 on the intact item

The v0.23 composition TEST is now spent and must never be tuned against.

v0.22.2 remains formally recorded as TEST FAIL because its preregistered
explicit-termination-rate criterion failed.  Its preserved recurrent temporal
controller nevertheless showed strong duration accuracy and is reused here as
the best current temporal execution mechanism without changing that formal
status.

## New question

Can a late query over multiple remembered events causally control autonomous
temporal behavior?

The tested chain is:

two stored canonical recalls
    ↓
late FIRST / SECOND query
    ↓
frozen v0.23 query-conditioned slot attention
    ↓
selected ORIGINAL canonical recall128
    ↓
frozen remembered-goal decoder
    ↓
LEFT / RIGHT explicit goal
    ↓
frozen v0.22 goal-conditioned temporal skill
    ↓
frozen v0.22.2 recurrent temporal state
    ↓
autonomous primitive action

There is no new training.

## Why selected ORIGINAL recall is used

The v0.23 reader contains a private encoded representation for answering
FIRST/SECOND/SAME/DIFFERENT.  v0.24 does not treat that private representation
as a second memory language.

Instead, the reader's frozen attention weights select from the original
canonical recall128 slots.  The already-admitted remembered-goal decoder then
consumes that canonical content.

Thus canonical recalled content remains the shared semantic interface.

## Scope

FIRST and SECOND have an immediate embodied interpretation: retrieve one stored
event and act according to its remembered side.

SAME and DIFFERENT remain preserved relational memory operations.  v0.24 does
not invent an arbitrary motor meaning for them merely to claim integration.

## Evaluation family

No new held-out split is consumed.

The assay reuses historical single-item TRAIN-family conditions only.  It is a
causal integration diagnostic, not a new environmental or statistical
generalization claim.

64 deterministic integration cases are constructed.  Each case contains:
- event 1 condition;
- distinct event 2 condition;
- distinct execution condition;
- distinct donor condition.

Every event pair is instantiated under:
- LEFT / LEFT
- LEFT / RIGHT
- RIGHT / LEFT
- RIGHT / RIGHT

At the execution condition, LEFT and RIGHT worlds are asserted sensor-identical
at the delayed decision.  Therefore a query-dependent action flip cannot be
explained by the current observation.

## Primary controls

### Normal FIRST / SECOND
For every pair:
- FIRST attention must select slot0;
- SECOND attention must select slot1;
- selected canonical recall must decode to that event's side;
- the autonomous goal skill must commit in the selected direction.

### Query flip
For unequal pairs:
- FIRST and SECOND must produce opposite decoded goals;
- their autonomous first lateral commitments must flip.

### Slot swap
Swap slot0 and slot1 after storage.

For unequal pairs:
- FIRST must now decode former SECOND content;
- autonomous commitment must flip accordingly.

### Donor replacement
Replace one slot with an opposite-side canonical recall from a distinct donor
condition.

Required:
- the query targeting the replaced slot follows donor content all the way
  through autonomous commitment;
- the query targeting the untouched slot preserves its original goal/action.

Both slot0 and slot1 replacements are tested.

### Same-pair stability
For LEFT/LEFT and RIGHT/RIGHT pairs:
- FIRST and SECOND must produce the same goal;
- their autonomous commitments must remain the same.

### Current-state baseline
The historical direct-option controller is run from the same hidden-goal
execution states.

The integration path must materially outperform that baseline on correct
lateral commitment.

## Frozen gate

- FIRST/SECOND attention route >= .99
- selected canonical recall -> correct goal >= .99
- autonomous selected-goal commitment >= .95
- both FIRST and SECOND complete causal chain per pair >= .95

Unequal pairs:
- query flip changes goal >= .95
- query flip changes autonomous commitment >= .95
- slot swap changes goal >= .95
- slot swap changes autonomous commitment >= .95

Donor interventions:
- targeted donor goal/action >= .95
- untouched goal/action preserved >= .95 for both slots

Same pairs:
- FIRST/SECOND goal stability >= .99
- FIRST/SECOND action stability >= .99

Baseline:
- historical direct-option hidden-goal commitment <= .70
- memory-selected integration advantage >= .25

Completion rate is reported but is not an admission criterion because this
assay isolates episodic selection → goal → temporal behavior.  The v0.22.2
formal status is not rewritten.

## After PASS

Freeze this integration contract.

The next capability is consequence-based planning/replanning:
- selected episodic content constructs a goal;
- multiple temporal alternatives are imagined;
- predicted consequences influence the high-level choice;
- execution is monitored;
- a mismatch between prediction and reality triggers replanning.

That next stage should use a new task family rather than reopen the closed
LEFT/RIGHT memory assays.

## After FAIL

Do not retrain v0.23 against its spent TEST.

Identify which frozen interface failed:
- episodic slot selection;
- canonical selected-content transfer;
- goal decoding;
- temporal-skill invocation.

Repair only the integration boundary in a new development family.

No consciousness, general episodic memory, or broad intelligence claim follows
from PASS.
