# Continuous Brain v0.20.2 — All-Option Success-Goal Development Assay

## Evidence from v0.20.1

The goal itself is not underpowered.

Observed on autonomous direct-option trajectories:
- all-option goal swap rate: 0.2368
- top-2 goal swap rate: 0.1053
- selector-suppressed swap rate: 0.1579
- median goal-swing / historical-margin ratio: 84.5
- predicted-success vs realized-success Pearson: 0.922
- predicted-collision vs realized-collision Pearson: 0.239

The extremely large mean goal-swing ratio is not used because some historical
margins are nearly zero and make the ratio unstable. The median is the useful
summary.

## Architectural change

No learned tensor changes.

1. Every learned temporal option is eligible at planning time.
2. The old selector is no longer a hard top-2 candidate filter.
3. The explicit active goal uses ONLY predicted success.
4. Predicted collision is not used as a steering term because its current
   consequence calibration is weak.
5. Historical imagined reward + terminal option Q remain in the ranking.

Neutral:
`reward + terminal option Q`

Success goal:
`reward + terminal option Q + 0.75 * discounted predicted success`

This is deliberately narrower than v0.20.0.

## Why collision is removed from active steering

Collision avoidance remains an important future objective. But a correlation of
~0.24 between predicted and realized collision consequences is not strong enough
to justify making it a dominant goal signal.

The correct sequence is:
- first establish trustworthy goal-sensitive autonomous planning using the
  well-calibrated success head;
- later retrain/calibrate collision prediction;
- then add safety-sensitive goal control.

## Fresh development window

Seeds 14001-14040 are fresh relative to the v0.20.0/v0.20.1 development window.

No claim is made that they were globally held out from all historical training.

## Frozen development gate

Causal:
- same-state success-vs-neutral option swap rate >= 0.15
- on changed decisions, success-goal plan has >= predicted success than neutral
  in >= 0.80 of cases

Autonomous behavior:
- success completion >= neutral all-option completion
- success completion >= direct learned-option baseline - 0.10
- success collisions <= neutral + 5
- success collisions <= direct learned-option baseline + 10

Software/provenance:
- no training
- protected child/memory/v0.18/v0.19 files unchanged
- v0.19 remains closed

## Promotion rule

PASS:
The next build may connect admitted remembered content to an explicit goal
register because the goal-conditioned planner has shown both causal sensitivity
and acceptable autonomous behavior.

FAIL:
Do not touch memory. Use the failed check to decide whether all-option search,
world-model ranking, or primitive option execution needs repair.

This is a development gate, not a consciousness claim or broad intelligence
claim.
