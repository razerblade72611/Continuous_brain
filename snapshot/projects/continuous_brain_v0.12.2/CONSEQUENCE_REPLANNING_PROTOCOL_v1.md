# Continuous Brain v0.25.0 — Episodic-Goal Consequence Planning + Replanning

## Starting point

v0.23.1:
- frozen two-item ordered episodic memory TEST PASS.

v0.24.0:
- frozen episodic selection -> canonical recall -> explicit goal -> temporal
  skill integration PASS on a causal TRAIN-family integration assay.

The next missing function is not more memory classification. It is the ability
to compare future consequences, commit to a multi-step plan, detect that reality
violated the prediction, and revise the plan.

## Why this does not reopen the historical Detour planner

The historical PlanningCortex remains intact and untouched.

Earlier v0.20 work established that:
- long open-loop same-option imagination was unreliable;
- same-option short-horizon bootstrapping improved completion;
- fresh validation still failed the collision admission gate.

Therefore v0.25 does not hide a known planner/model weakness inside a new
cognitive test.

It adds a separate small consequence-model sidecar and tests the semantics of
planning/replanning cleanly first.

No claim is made that this new symbolic route model is the historical
PlanningCortex or that it is already a 3-D planner.

## New task family: DynamicRouteGraph v1

The route world is a high-level graph, not another delayed-cue corridor.

Nodes:
- START
- LEFT_MID
- RIGHT_MID
- LEFT_GOAL
- RIGHT_GOAL
- FAIL

Temporal macro-actions:
- APPROACH_LEFT
- APPROACH_RIGHT
- DIRECT
- SWITCH
- CROSS_FINISH

The explicit LEFT/RIGHT goal comes from frozen episodic memory selection.

### OPEN

For LEFT:
START -> APPROACH_LEFT -> LEFT_MID -> DIRECT -> LEFT_GOAL

RIGHT is symmetric.

### KNOWN_BLOCK

The initial route state visibly reports that the direct target lane is blocked.

For LEFT, the cheaper successful plan is:
START -> APPROACH_RIGHT -> RIGHT_MID -> CROSS_FINISH -> LEFT_GOAL

Thus consequence-aware planning must override the naive target-side approach.

### SURPRISE_BLOCK

The initial state is exactly open with respect to blockage.

The learned model therefore predicts:
START -> target-side APPROACH -> unblocked target MID -> DIRECT -> goal

But after the first real macro action, an exogenous blockage appears.

The actual state disagrees with the predicted state.

Correct behavior is:
1. detect prediction error;
2. discard the stale remainder of the plan;
3. replan from the actual blocked state;
4. SWITCH to the alternate lane;
5. CROSS_FINISH back to the remembered goal.

A plan-once controller executes stale DIRECT and fails.

This gives a direct causal test of replanning rather than merely selecting a
different action every step.

## Consequence model

Input:
- symbolic route state;
- currently selected episodic goal;
- candidate macro-action;
- known route-cost features.

Predictions:
- next node;
- next blocked state;
- immediate macro cost;
- terminal goal-success probability;
- action validity.

Training uses only ordinary modeled dynamics.  The surprise transition from
apparently-open START to blocked MID is intentionally absent from training.

The model nevertheless sees blocked MID states during training, so once reality
reveals a blockage it has learned consequences for recovery actions.

## Planner

Short-horizon explicit search:
- enumerate candidate macro-actions;
- predict their next state/cost/success;
- expand for up to four macro steps;
- choose the highest predicted goal-success plan with cost tie-breaking.

The selected plan is cached.

After each real macro:
- compare actual state with the predicted next state;
- if they match, continue the cached plan;
- if they differ, record prediction error, discard the stale remainder and
  replan from the actual state.

This is receding correction at meaningful temporal boundaries, not 100-step
open-loop hallucination.

## Memory integration

For each fresh route DEV condition:
- two canonical episodic recalls are constructed from the already-qualified
  historical TRAIN memory family;
- their sides are opposite;
- FIRST and SECOND are both queried;
- frozen v0.23 attention selects the requested original canonical recall;
- frozen goal decoder produces LEFT or RIGHT;
- the route planner receives only that explicit goal.

Thus changing the late memory query must change the initial route plan from the
same route state.

No new claim about episodic-memory generalization is made here; that already
has its own sealed TEST result.

## Splits

New route family only:

TRAIN:
- 0..2047

DEV:
- 2048..2303

TEST:
- 2304..2559, sealed

v0.25.0 does not touch route TEST.

## Baselines

### Goal-reactive
Always approaches the remembered goal lane and then DIRECTs.
It ignores known consequences and surprise mismatch.

### Plan once
Uses the same learned consequence model and same initial planner, but after
execution begins it never replans.

This is the critical surprise control.

## Frozen DEV gate

Consequence-model fidelity on fresh modeled DEV transitions:
- next-node accuracy >= .995
- next-blocked accuracy >= .99
- cost MAE <= .04
- terminal-success accuracy >= .995
- action-validity accuracy >= .995

Memory:
- selected episodic goal accuracy >= .99

Behavior:
- overall replanning success >= .95
- OPEN replanning success >= .95
- KNOWN_BLOCK replanning success >= .95
- SURPRISE_BLOCK replanning success >= .95

Replanning causality:
- SURPRISE plan-once success <= .10
- prediction mismatch detected in SURPRISE >= .95
- replanning triggered in SURPRISE >= .95
- replanning advantage over plan-once in SURPRISE >= .80

Goal/consequence causality:
- FIRST vs SECOND changes both remembered goal and initial route plan >= .95
- in KNOWN_BLOCK, model-based plan overrides naive target-side approach >= .95

Protected:
- historical InfantBrain unchanged
- admitted memory unchanged
- v0.23 episodic reader unchanged
- v0.24 integration artifacts unchanged
- historical planner/world model untouched
- route TEST untouched

## Meaning of PASS

A PASS establishes, in this new symbolic dynamic-route family:

past episodic content
 -> late memory selection
 -> explicit current goal
 -> learned consequence prediction
 -> multi-macro planning
 -> cached intended future
 -> action
 -> predicted-vs-actual comparison
 -> prediction error
 -> replanning
 -> successful recovery

This is a functional consequence-planning/replanning result.

It does NOT establish:
- robust 3-D planning;
- requalification of the historical Detour planner;
- broad general intelligence;
- consciousness.

## After PASS

Freeze the consequence/replanning contract and use route TEST once.

If TEST holds, the next major step should be richer embodiment:
- use this planning contract in AI2-THOR / Discovery World;
- store object/location episodes;
- select a remembered target;
- plan through rooms/doors;
- replan when a door, route, or object state differs from expectation.

## After FAIL

Do not tune against route TEST.

Use DEV failure to isolate:
- transition/consequence prediction;
- search/ranking;
- mismatch detection;
- cached-plan invalidation;
- memory-goal interface.

Repair only that new planning layer.
