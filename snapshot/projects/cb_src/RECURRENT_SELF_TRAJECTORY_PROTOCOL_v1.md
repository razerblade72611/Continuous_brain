# Continuous Brain v0.31.0 — Recurrent Imagined Self-Trajectory

## Why v0.31 is a new architecture family

v0.30.0 and v0.30.1 remain formal DEV FAILS.

The corrected v0.30 causal experiment nevertheless established an important
DEV result:
- future-self-guided choice = 1.00
- true self-blind world-only control = 0.00
- donor self flips action with world fixed = 1.00
- surprise self-prediction error recruits INVESTIGATE = 1.00

The remaining failure was future active-goal accuracy around 97.7%.

v0.30.2 diagnosed that failure:
- true next continuous self + exact frozen arbiter = 1.00
- learned goal heads = ~0.976–0.977
- 92–94% of learned-head errors lie within 0.05 of the INVESTIGATE boundary
- commitment A/B boundary proximity explains far fewer errors

This means the architecture was repeatedly reconstructing a discrete continuing
intention from noisy continuous uncertainty/error values near hard thresholds.

v0.31 stops doing that.

## Explicit continuing intention

The self now carries two distinct intention variables:

### Primary commitment
- NONE
- A
- B

This is what the continuing individual is fundamentally trying to complete.

### Temporary mode
- NORMAL
- INVESTIGATE

This is what cognitive strategy is currently required.

Temporary surprise can therefore produce:

primary=A, mode=NORMAL
→ primary=A, mode=INVESTIGATE
→ primary=A, mode=NORMAL

The primary intention is not erased by a temporary uncertainty-management mode.

## Hysteresis

INVESTIGATE mode uses preregistered entry/exit bands:

Enter:
- prediction error >= .60
OR
- uncertainty >= .86

Exit:
- prediction error <= .45
AND
- uncertainty <= .72

This hysteresis is deliberate.

The v0.30 diagnostic showed that a large majority of errors were caused by
small numeric changes near one hard boundary. A continuing intention should not
oscillate simply because uncertainty changes from .819 to .821.

The thresholds are fixed before v0.31 DEV.

## Recurrent imagined trajectory

Inputs at every imagined step:
- external world4
- continuous self6
- explicit primary intention3
- explicit mode2
- candidate action4

A recurrent GRU state is carried through the whole imagined future.

Outputs each step:
- next world
- next continuous self
- next primary intention
- next mode
- agency

Sequence length:
- 8 steps

The model therefore imagines:

world/self/intention t0
→ t1
→ t2
→ ...
→ t8

rather than predicting one isolated future self.

## Tie cases

20% of the trajectory family deliberately contains:

unresolved A == unresolved B

but the explicit continuing primary intention is either A or B.

This makes primary intention information irreducible to the continuous
commitment scalars.

A scalar-only reconstruction necessarily loses the B intention on exact ties.

This creates a clean causal test of whether explicit intention state matters.

## New split

TRAIN:
- trajectory IDs 0..4095

DEV:
- 10000..11023

TEST:
- 20000..21023, sealed

The v0.30 TEST 60000..64999 also remains untouched because v0.30 never passed
DEV and is closed.

## Training

Only:
- `RecurrentSelfTrajectoryModel`

Frozen:
- historical InfantBrain
- admitted working memory
- episodic reader
- v0.28 persistent-self mechanisms
- v0.29 results
- v0.30 results

Teacher forcing is used during training while recurrent hidden state spans the
entire eight-step sequence.

DEV also includes fully autoregressive eight-step rollout.

## DEV gates

Teacher-forced:
- world MAE <= .030
- continuous-self MAE <= .025
- primary-intention accuracy >= .985
- mode accuracy >= .985
- agency accuracy >= .990

Autoregressive horizon 8:
- world MAE <= .075
- continuous-self MAE <= .060
- primary-intention accuracy >= .950
- mode accuracy >= .950
- active-goal accuracy >= .950

## Explicit-intention causal assay

External world:
- progress A saturated
- progress B saturated

Continuous self:
- unresolved A = unresolved B = .90
- all other continuous fields identical

Only primary intention differs.

Candidates:
- three WORK_A steps
- three WORK_B steps

Full recurrent trajectory scoring:
- A-primary should choose WORK_A
- B-primary should choose WORK_B
- accuracy >= .95

Scalar-only control:
- reconstruct primary from the tied scalar commitments
- deterministic tie-break becomes A
- accuracy must collapse <= .60

Donor-primary intervention:
- world and every continuous self field remain exactly fixed
- swap only explicit primary A↔B
- chosen trajectory must flip >= .95

This makes primary intention a causally independent state variable.

## Surprise interruption and resumption

Start:
- primary A
- mode NORMAL

Unexpected failed WORK_A:
- uncertainty rises
- prediction error rises
- mode must become INVESTIGATE
- primary must remain A

Information-gathering OBSERVE actions reduce uncertainty/error.

When the exit band is reached:
- mode returns NORMAL
- primary remains A
- active goal becomes A again

Required:
- surprise preserves primary and enters investigate = 1.00
- primary resumes after investigate resolves = 1.00
- surprise self-prediction error >= .10

## Meaning of PASS

A PASS supports this functional claim:

The system can imagine multi-step future self/world trajectories while carrying
a discrete continuing primary intention separately from temporary cognitive
mode. That primary intention remains causally relevant when continuous
self/world variables are held fixed, survives a temporary surprise-driven
INVESTIGATE interruption, and resumes after uncertainty resolves.

This is a stronger model of functional personal continuity than repeatedly
reconstructing a goal from noisy instantaneous scalars.

It is not sufficient evidence of subjective experience, phenomenal
consciousness, sentience or qualia.

## After DEV PASS

Run one frozen TEST on trajectory IDs 20000..21023.

If TEST holds, the next milestone should connect imagined self-trajectories to
autobiographical expectation:

- store the chosen imagined personal trajectory before acting
- later compare remembered predicted-self trajectory with actual self-history
- learn from discrepancies across multiple episodes
- let long-term self-history alter future commitment formation

That would begin to link future self-modeling with autobiographical identity
across longer spans of experience.

## After DEV FAIL

Do not touch TEST.

Classify failure as:
- recurrent world drift
- recurrent self drift
- primary-intention continuity
- investigate-mode continuity
- explicit-intention causal use
- interruption/resumption state machine

Repair only within v0.31 DEV.
