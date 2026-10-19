# Continuous Brain v0.22.1 — Goal-Option Temporal Progress Repair

## What v0.22.0 established

The new additive goal-option bank solved the missing-skill problem:

- frozen remembered-goal accuracy: 1.000
- own goal -> correct temporal commitment: 1.000
- both LEFT and RIGHT correct in every paired DEV condition: 1.000
- donor memory -> donor-directed commitment: 1.000
- explicit goal flip -> flipped commitment: 1.000
- historical direct-option baseline commitment: 0.4883
- memory-guided commitment advantage: +0.5117
- historical six options remained bit-exact
- TEST remained untouched

The failure was isolated to temporal termination:

- skill termination rate: 0.8945
- termination within +/-2 teacher lateral steps: 0.5820

Therefore the memory mechanism, remembered-goal decoder, new goal skill policy,
and goal-to-skill causal route are all preserved.

## Why binary termination is replaced

v0.22.0 used a sparse binary target:
- every lateral state -> continue (0)
- one post-lateral state -> stop (1)

That teaches whether termination eventually occurs, but gives weak information
about *how far remains*.

v0.22.1 uses dense temporal progress instead.

For a lateral traverse with N remaining primitive steps, successive training
targets are:

`N, N-1, N-2, ..., 2, 1, 0`

The new head predicts `log1p(remaining_lateral_steps)` from the frozen
goal-option feature vector.

At runtime:
- the frozen goal-option policy still chooses primitive actions;
- after at least one committed skill step, predicted remaining duration <= 0.5
  means the skill has reached its learned stopping boundary;
- control then returns to the frozen historical six-option controller.

The 0.5 threshold is the fixed midpoint between discrete remaining-step targets
0 and 1; it is not tuned on DEV.

## What is trainable

Only a new temporal-progress head:

`frozen goal-option features -> MLP -> predicted remaining lateral steps`

Frozen and verified bit-exact:
- canonical InfantBrain;
- admitted working memory;
- canonical recall;
- v0.21 remembered-goal decoder;
- v0.22 goal-option bank and its already-perfect action policy;
- historical six options;
- PlanningCortex/world model;
- v0.18;
- v0.19.

## Split

Training remains on conditions 0..1023.

Fresh DEV for this repair:
- 1856..1979

This is the last unused portion of the original DEV family before TEST.

TEST:
- 1980..2359
- untouched.

## Fresh DEV gate

The complete causal behavior must remain intact:
- remembered goal accuracy >= .95
- own remembered goal -> correct commitment >= .95
- both LEFT and RIGHT correct in >= .90 paired conditions
- donor goal -> donor-directed commitment >= .95
- explicit goal flip -> flipped commitment >= .95
- commitment advantage over historical baseline >= .25
- historical baseline commitment <= .70

The repaired temporal control must satisfy:
- learned skill termination >= .95
- skill duration within +/-2 teacher lateral steps >= .85
- mean absolute duration error <= 1.50 primitive steps

Correct-goal completion remains descriptive because this milestone isolates the
temporal-skill boundary rather than downstream navigation after handoff.

## Stop rule

This is not another Detour/planner tuning cycle.

If PASS:
- freeze the integrated memory -> goal -> temporal-skill mechanism;
- stop LEFT/RIGHT skill repair;
- move to remembered-goal consequence-based planning/replanning and then
  multi-item memory / richer 3-D embodiment.

If FAIL:
- do not alter admitted memory or the now-proven lateral policy;
- treat temporal-state sufficiency as the remaining issue and move duration
  learning to a recurrent option-state representation rather than tuning a
  threshold or retraining the action policy.

No consciousness or broad intelligence claim follows from PASS.
