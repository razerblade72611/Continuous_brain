# v0.33.3 — Self-Development Event-Order Diagnostic

## Why this diagnostic exists

v0.33.1 is a formal DEV FAIL.

v0.33.2 tested the hypothesis that subjects crossed the +0.12 autobiographical
self-estimate threshold and later drifted below it after graduation.

Using only cycle-boundary samples, v0.33.2 found:

- ever crossed +0.12: 0.9140625
- graduated to DIRECT: 0.9609375
- final above +0.12: 0.7421875
- graduation without observed crossing: 0.078125
- observed crossing without graduation: 0.03125

Therefore the coarse timing explanation was rejected.

## Remaining measurement issue

The v0.33.1 controller does NOT make decisions at cycle boundaries.

Within each cycle it processes two episodes, A and B, in alternating order.

Before each episode it:
1. reads the current autobiography;
2. computes current reliability for that commitment;
3. compares it with baseline + 0.12;
4. may graduate SCAFFOLD -> DIRECT;
5. executes the episode;
6. appends a new autobiographical record.

v0.33.2 sampled only after both episodes in a cycle.

Therefore it could miss:
- a temporary threshold crossing seen at the actual decision point;
- a threshold crossing created by one episode and erased by the second episode;
- a crossing after the final opportunity to choose that cannot cause graduation.

## v0.33.3 measurement

No training.

Reuse spent DEV subjects:
- 10000..10127

TEST remains sealed:
- 20000..20127

For every subject and every episode, record:

### Before each fragile-commitment decision
- current autobiographical reliability;
- gain relative to bootstrap;
- whether gain >= +0.12;
- strategy before controller update;
- strategy after controller update;
- whether graduation occurred at this exact event.

### After every autobiographical append
- updated fragile reliability;
- gain relative to bootstrap;
- whether +0.12 is crossed.

## Exact causal consistency criterion

The event-order explanation is supported only if:

- predecision threshold-cross rate == graduation rate;
- graduation without a predecision crossing = 0;
- predecision crossing without graduation = 0;
- first threshold-crossing decision is exactly the first graduation decision
  for every subject that graduates.

This is stronger than comparing final or cycle-boundary values.

## Interpretation

If exact alignment holds, v0.33.2 failed because its sampling resolution did not
match the controller's actual causal decision events.

That still does NOT retroactively change v0.33.1 from FAIL to PASS.

It would justify a new DEV repair that replaces the final-state gate with an
event-aligned gate:

`new autobiography causes a >= +0.12 self-estimate increase at the decision
event that changes subsequent behavior`

while retaining the frozen-autobiography counterfactual.

If exact alignment does not hold, do not repair the gate. The developmental
controller or instrumentation has another inconsistency that must be diagnosed.

All admitted v0.31/v0.32 artifacts remain protected and TEST stays sealed.
