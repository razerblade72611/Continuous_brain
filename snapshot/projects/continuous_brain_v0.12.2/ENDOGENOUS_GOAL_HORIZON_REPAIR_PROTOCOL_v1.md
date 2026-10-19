# Continuous Brain v0.35.2 — Endogenous Goal Horizon Repair

## Historical record

Preserve:

- v0.34.1: frozen TEST PASS, closed.
- v0.35.0: formal DEV FAIL.
- v0.35.1: read-only completion-dynamics diagnostic.

v0.35.0 failed only:

`self_generated_goal_completes_from_self_estimate`

at:
- 8 target episodes: 0.46875
- required: 0.80

No other v0.35.0 gate failed.

## Diagnostic authorization

v0.35.1 did not train or touch TEST.

It preregistered completion checkpoints:
- 8
- 16
- 24
- 32

Observed:
- completion by 8 = 0.46875
- completion by 16 = 0.90625
- completion by 24 = 1.00
- completion by 32 = 1.00
- never crossing by 32 = 0.00

The first preregistered checkpoint clearing the original >=0.80 completion gate
was 16 episodes.

At 8 episodes:
- discrepancy reduction rate = 1.00;
- mean capability gain = +0.68376.

Among the 68 original 8-episode noncompleters:
- discrepancy reduction rate = 1.00;
- mean capability gain = +0.68001.

Therefore the original eight-episode horizon was too short for the unchanged
frozen autobiographical reliability criterion.

## Repair

Change exactly one scientific parameter:

`MAX_GOAL_EPISODES: 8 -> 16`

Do not change:
- goal generator;
- lower-reliability commitment selection;
- erased-history neutral rule;
- donor-history intervention;
- persistent goal representation;
- interruption semantics;
- target-specific plan;
- frozen autobiographical model;
- frozen recurrent self-trajectory model;
- goal target `baseline + 0.12`;
- original >=0.80 completion gate;
- capability threshold;
- discrepancy threshold;
- persistence thresholds.

No neural training.

## DEV

Reuse already-spent v0.35 DEV:
- 90000..90127

This is a repair confirmation, not a new independent development sample.

TEST remains sealed:
- 100000..100127

## Admission

Preserve all v0.35.0 gates:

- autobiography generates the evaluator-verified fragile developmental target
  >= .95;
- donor autobiography flips the generated target >= .95;
- erased autobiography generates no directional developmental target = 1.00;
- active self-generated goal persists through interruption = 1.00;
- self-generated goal controls target-specific plan = 1.00;
- goal completes from updated autobiographical self-estimate >= .80;
- mean target capability gain >= .35;
- discrepancy-reduction criterion >= .80;
- goal lifecycle persists exactly = 1.00;
- goal context persists exactly = 1.00.

## Integrity interpretation

A PASS does not rewrite v0.35.0.

It establishes that with the first diagnostic-authorized horizon that is long
enough, the unchanged endogenous goal mechanism can:

autobiography
-> author a developmental target
-> retain it through interruption
-> causally control target-specific behavior
-> develop the targeted capability
-> reduce expected-vs-actual discrepancy
-> later recognize its own preregistered autobiographical target as achieved.

## Stop rule

If v0.35.2 DEV PASS:
- freeze the 16-episode horizon and all existing semantics;
- build exactly one frozen TEST on 100000..100127.

If v0.35.2 DEV FAIL:
- keep TEST sealed;
- preserve failure;
- diagnose on DEV only.

A PASS remains a functional endogenous-goal result. It does not establish
subjective desire, sentience, consciousness, qualia, personhood, or an
open-ended semantic goal space.
