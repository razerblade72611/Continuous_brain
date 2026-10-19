# Continuous Brain v0.30.1 — Joint Self/World DEV Repair

## Source status

v0.30.0 remains formally:

`JOINT_SELF_WORLD_PREDICTION_DEV_GATE: FAIL`

Two gates failed:

1. future-goal prediction:
   - DEV 0.9762
   - required >= 0.98

2. self-consequence ablation:
   - ablated accuracy = 1.00
   - required <= 0.35

The second failure exposed an experimental-control error.

## Counterfactual-control diagnosis

The original v0.30 function removed the numeric future-self utility by setting:

`self_weight = 0`

but the remaining world utility still selected a world-progress dimension using:

`active_goal = current_self.active_goal`

Therefore the supposedly "world-only" control still knew whether this was
Self-A or Self-B.

It was not self-blind.

The 1.00 ablation accuracy cannot be interpreted as evidence that future
self-consequences were unnecessary.

v0.30.1 corrects the assay.

## Corrected self-blind baseline

Both conditions receive:
- the same frozen joint model;
- the same candidate actions;
- the same saturated external world.

Full condition:

`fixed self-blind world utility + predicted future-self commitment utility`

Ablation:

`fixed self-blind world utility ONLY`

The fixed world utility never reads:
- active goal;
- unresolved commitments;
- confidence;
- uncertainty;
- prediction error;
- predicted self state.

Thus when future-self utility is removed, no self-goal label remains hidden in
the scoring function.

Required:
- full future-self-guided choice >= .95;
- true self-blind world-only accuracy <= .35;
- donor self with world fixed flips the selected action >= .95.

## Future-goal repair

The v0.30 joint dynamics are frozen.

They already passed:
- world future prediction;
- continuous self future prediction;
- agency;
- current-self latent reconstruction;
- donor-self action flip;
- self-prediction-error surprise;
- endogenous INVESTIGATE response;
- persistent reload.

Only next-goal accuracy narrowly missed.

v0.30.1 trains a small goal calibration head over FROZEN model outputs:

- joint latent96
- predicted continuous future self6
- original goal logits4

The head never sees the true future state at inference.

The joint model itself is never optimizer-owned.

Required:
- repaired DEV goal accuracy >= .98;
- improvement over the original 0.9762 >= .002;
- TRAIN/DEV accuracy gap <= .01.

## Data use

TRAIN:
- original v0.30 TRAIN 0..19999

DEV:
- original already-spent DEV 40000..44999

TEST:
- 60000..64999 remains sealed

No new DEV window is consumed.

## Preservation

Before and after repair, hash:
- v0.30 source result
- v0.30 frozen joint model
- v0.29 spent TEST result
- v0.28 persistent self artifacts
- canonical historical brain
- admitted working memory

Any mutation invalidates the repair.

## Meaning of PASS

A repair PASS supports two separate conclusions on reused DEV:

1. The frozen joint latent/predicted-self representation contains enough
   information to recover next active goal above the preregistered threshold.

2. Under a corrected self-blind control, predicted future-self consequences are
   causally necessary for the counterfactual A-vs-B choice when external world
   progress is held fixed.

The original v0.30.0 result remains FAIL. v0.30.1 is recorded separately as a
DEV repair.

## Stop rule

If PASS:
- freeze the original v0.30 joint model + v0.30.1 future-goal repair head;
- run one frozen TEST on 60000..64999;
- do not alter the corrected counterfactual assay after seeing TEST.

If FAIL:
- do not touch TEST;
- diagnose goal representation and/or future-self utility within the v0.30 DEV
  family only.
