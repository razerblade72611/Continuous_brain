# Continuous Brain v0.22.2 — Recurrent Goal-Option Temporal State

## Why v0.22.1 failed

v0.22.1 preserved every already-working causal component:

- remembered-goal accuracy: 1.000
- own goal -> correct commitment: 1.000
- paired LEFT/RIGHT commitment: 1.000
- donor goal -> donor-directed commitment: 1.000
- explicit goal flip -> flipped commitment: 1.000
- historical baseline commitment: 0.4919
- memory-guided completion: 0.3669
- historical baseline completion: 0.0605

But its *static* temporal-progress reader failed:

- skill termination: 0.7056
- mean duration error: 3.44 primitive steps
- duration within +/-2 teacher steps: 0.5605

The v0.22.0 sparse binary termination head had also failed timing.

Two different static readers failing while the action policy remains perfect is
evidence that instantaneous frozen option features are insufficient for robust
within-option temporal state.

## Architectural repair

Do not tune another threshold.

Do not retrain:
- admitted memory;
- canonical recall;
- remembered-goal decoder;
- v0.22 lateral action policy;
- historical six options;
- PlanningCortex/world model.

Add a sidecar recurrent temporal-state controller:

frozen goal-option feature at t
          +
recurrent hidden from t-1
          ->
GRU temporal state
      |            |
      v            v
dense remaining   auxiliary
steps estimate    termination head

The recurrent hidden state is:
- zeroed when a new goal option starts;
- updated once per real primitive skill step;
- discarded when the skill terminates.

Thus option duration is represented as state accumulated *during the option*,
rather than inferred independently from each instantaneous latent.

## Dense temporal supervision

The training sequence for an option with N remaining lateral actions carries:

remaining targets:
N, N-1, ..., 2, 1, 0

termination targets:
0, 0, ..., 0, 0, 1

Loss:
- SmoothL1 on log1p(remaining steps)
- plus weighted auxiliary binary termination loss.

At runtime the dense remaining-step estimate is authoritative.

After at least one committed skill step:

predicted remaining <= 0.5  -> terminate

0.5 is the fixed midpoint between the discrete labels 0 and 1 and is never
tuned on DEV or TEST.

## Data policy

All prior DEV slices 1600..1979 are considered spent.

Training:
- full original TRAIN split 0..1599

Final admission:
- untouched TEST 1980..2359
- all 380 structural conditions
- both LEFT and RIGHT members

TEST is used once for this admission decision.
No parameter, threshold, architecture, or gate may be changed in response to
TEST while preserving a claim of held-out admission.

## Frozen TEST gate

Causal memory/goal/skill path:
- TEST remembered-goal accuracy >= .95
- own remembered goal -> correct lateral commitment >= .95
- both sides correct in >= .90 paired structural conditions
- donor memory -> donor-directed commitment >= .95
- explicit goal flip -> flipped commitment >= .95
- zero memory routes exactly to historical baseline = 1.00

Temporal state:
- skill termination >= .95
- duration within +/-2 teacher lateral steps >= .85
- mean absolute duration error <= 1.50 primitive steps

Behavior:
- commitment advantage over historical direct-option baseline >= .25
- historical baseline commitment <= .70
- correct-goal completion >= .30
- completion advantage over historical baseline >= .20

## PASS

Freeze the entire admitted pathway:

past observation
 -> admitted working memory
 -> canonical recall
 -> remembered goal
 -> goal-diverse temporal option
 -> recurrent within-option temporal state
 -> learned duration/termination
 -> autonomous primitive behavior
 -> handoff to historical skills

Then stop the LEFT/RIGHT assay permanently.

Next:
1. remembered goals influence imagined consequences and replanning;
2. multi-item/order-sensitive memory;
3. richer 3-D embodiment / AI2-THOR;
4. pretrained language/vision integration and continual development.

## FAIL

Do not tune against this TEST.

A failure after two static timing models and this recurrent timing model means
the current frozen planning/skill representation is not sufficient for robust
temporal option-state inference in this family.

Preserve the successful memory->goal->commitment mechanism, close this assay,
and move the next representation work into richer recurrent and embodied state
rather than continuing benchmark repair.

A PASS is evidence for a functional integrated cognitive mechanism only.  It is
not evidence of consciousness, sentience, or broad general intelligence.
