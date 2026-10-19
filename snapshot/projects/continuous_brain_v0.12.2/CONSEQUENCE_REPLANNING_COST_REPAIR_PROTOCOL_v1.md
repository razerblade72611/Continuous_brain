# Continuous Brain v0.25.1 — Cost-Consistent Replanning Repair

## Classification of v0.25.0

v0.25.0 is formally recorded as DEV FAIL.

However, its failed replanning criteria were not an observed failure to recover
from surprise.  The surprise intervention was never reached:

- replanning success: 1.000
- plan-once success: 1.000
- surprise replanning success: 1.000
- surprise plan-once success: 1.000
- surprise replanning trigger: 0.000
- surprise mismatch detection: 0.000
- surprise replanning advantage: 0.000

At the same time:
- consequence-model next-node accuracy: 1.000
- next-blocked accuracy: 1.000
- cost MAE: 0.0283
- success accuracy: 1.000
- validity accuracy: 1.000
- memory-selected goal accuracy: 1.000
- known-block model override: 1.000

The failed result therefore revealed a construction error in plan ranking.

## The bug

v0.25.0 ranked complete plans lexicographically as:

1. predicted success probability
2. cost-adjusted score

For routes that were all predicted successful, tiny neural calibration
differences in success probability could outrank a much lower-cost route.

In the SURPRISE_BLOCK environment this allowed the planner to choose an
unnecessary opposite-lane bypass.  That bypass never touches the hidden
target-lane block, so:
- no exogenous surprise was triggered;
- no predicted-vs-actual mismatch occurred;
- no replanning opportunity existed;
- the plan-once control also succeeded.

This does not test the intended hypothesis.

## Repair

No model is retrained.

The exact v0.25.0 consequence-model checkpoint is reused byte-for-byte.

Only the planner ordering changes.

Corrected ranking:

`predicted utility = 20 * P(goal success) - cumulative macro cost`

Primary ordering:
1. predicted utility
2. predicted success probability
3. shorter plan

Thus probability remains part of the decision, but tiny probability
calibration differences cannot dominate route cost twice.

## New exposure invariant

A causal surprise test must prove that the intervention was actually exercised.

For every SURPRISE_BLOCK case, before the hidden block is revealed, the initial
plan should be the predicted-open target route:

`target-side APPROACH -> DIRECT`

Required rate:
- >= 0.95

Only after this invariant is satisfied can the following count as evidence:
- predicted/actual mismatch;
- plan invalidation;
- replanning;
- recovery;
- plan-once failure.

## Evaluation protocol

Training:
- none.

Development:
- reuses v0.25.0 route DEV 2048..2303.
- no new development window is consumed.

TEST:
- route TEST 2304..2559 remains untouched.

This is a software/experimental-construction repair, not a new scientific
development iteration.

All frozen thresholds from the v0.25 behavioral gate are preserved, plus the
new exposure invariant.

## Required behavioral checks

- memory-selected goal >= .99
- overall replanning success >= .95
- OPEN success >= .95
- KNOWN_BLOCK success >= .95
- SURPRISE_BLOCK success >= .95
- SURPRISE initial target-direct exposure >= .95
- SURPRISE plan-once success <= .10
- SURPRISE replanning trigger >= .95
- SURPRISE mismatch detection >= .95
- FIRST/SECOND changes goal and initial plan >= .95
- KNOWN_BLOCK consequence-aware override >= .95
- SURPRISE replanning advantage over plan-once >= .80

## Preservation

The following remain unchanged:
- historical InfantBrain;
- admitted working memory;
- v0.23 episodic reader;
- remembered-goal decoder;
- v0.25.0 consequence-model checkpoint;
- historical PlanningCortex/world model.

## Stop rule

If PASS:
- v0.25.0 remains historically recorded as formal FAIL;
- v0.25.1 establishes that the repaired experiment now exposes the surprise and
  demonstrates prediction-error-triggered replanning on DEV;
- run one frozen route TEST without further tuning.

If FAIL:
- do not touch route TEST;
- classify the failing layer from the explicit exposure/mismatch/recovery
  metrics;
- do not tune ranking thresholds merely to force a pass.

No consciousness, 3-D planning, or broad intelligence claim follows from PASS.
