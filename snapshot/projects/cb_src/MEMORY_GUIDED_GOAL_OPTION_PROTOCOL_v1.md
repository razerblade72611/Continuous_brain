# Continuous Brain v0.21.0 — Memory-Guided Goal Option Control

## Why this is the next milestone

The v0.20 planner-repair phase is closed.

Fresh v0.20.6a validation showed:
- direct learned options: completion 0.63, collisions 17.41
- historical hierarchy: completion 0.49, collisions 18.67
- same-option-bootstrap planner: completion 0.66, collisions 26.24
- completion vs historical: +0.17, paired bootstrap CI95 [0.08, 0.26]
- collisions vs historical: +7.57, CI95 [1.18, 14.60]
- illegal cap handoffs: 0

The continuation-bootstrap planner therefore improved completion but failed the
frozen collision gate. Per the preregistered stop rule, no more Detour planner
tuning is allowed.

The direct learned-option controller becomes the reliable execution backbone.
The repaired planner remains available as an advisory research subsystem.

## New cognitive capability

This milestone asks whether information observed earlier can become an explicit
goal after the cue has disappeared and then causally select a temporally
extended learned option.

Dataflow:

early visual cue
    ->
admitted 64-D working-memory writer
    ->
admitted canonical recall128
    ->
new explicit goal register2
    ->
goal-constrained selection among learned options
    ->
selected option owns primitive control
    ->
real environment transition

This is not primitive-action imitation.

## Trainable component

Only:

`Linear(128 -> 2, bias=False)`

is trained.

It decodes canonical recalled content into an explicit remembered goal:
LEFT or RIGHT.

The zero-bias design gives:

`zero recall -> exactly zero goal logits`

so the zero-memory control has no hidden default goal.

Frozen:
- historical 14k InfantBrain;
- admitted 64-D writer;
- admitted nonlinear recall;
- option selector;
- option embeddings;
- option policy;
- option termination;
- option Q;
- PlanningCortex and world model;
- v0.18;
- closed v0.19.

## Environment and split

Use the existing `VariedCuedDetourWorld`.

The cue is visible early and disappears before the lateral decision.
For every structural condition, LEFT and RIGHT members are identical at the
delayed decision except for remembered history.

Development experiment:
- goal-decoder TRAIN: condition IDs 0..511
- integration DEV: condition IDs 1600..1727
- TEST: untouched

The prefix to the delayed decision uses the environment's cue-independent
teacher action. The script asserts LEFT and RIGHT paired prefixes are identical,
and asserts decision observations, images, contexts and planning roots are
exactly identical.

This means the memory-guided option choice cannot read the cue from the current
decision state.

## Goal-to-option arbitration

At the decision state:

1. frozen option selector computes competence logits for all learned options;
2. each learned option's own policy supplies its current primitive action;
3. remembered goal chooses the desired lateral action LEFT or RIGHT;
4. among options whose own policy action matches that goal, select the option
   with the highest frozen selector logit;
5. if no option matches, fall back to selector top-1.

Thus memory supplies the goal while the existing selector remains a competence
prior.

## Autonomous execution

After the delayed decision:
- the selected learned option supplies primitive actions;
- it retains control until learned termination;
- after termination the ordinary direct learned-option selector takes over;
- no teacher action is used after the delayed decision.

Reported outcomes include:
- first autonomous lateral commitment;
- correct-goal completion;
- collisions;
- duration of the initial temporal option.

## Causal controls

At the exact same paired current state:

- OWN memory;
- memory ZERO;
- opposite-cue DONOR memory;
- explicit GOAL FLIP while memory is held fixed;
- router OFF / frozen selector top-1 baseline.

Required causal behavior:
- own memory decodes the recipient goal;
- donor memory decodes the donor goal;
- donor goal selects an option whose own action matches the donor;
- explicit goal flip changes selected option;
- zero memory is exactly the baseline selector path.

## Frozen development gate

Goal decoding:
- DEV pair accuracy >= 0.95
- LEFT accuracy >= 0.95
- RIGHT accuracy >= 0.95

Option interface:
- a goal-matching learned option exists >= 0.90 of DEV members
- own selected option's first policy action matches remembered goal >= 0.90
- donor memory decodes donor goal >= 0.90
- donor-selected option action matches donor goal >= 0.85
- explicit goal flip changes selected option >= 0.80
- zero memory exactly equals selector baseline = 1.00

Autonomous behavior:
- memory-guided first lateral commitment correct >= 0.80
- selector-only baseline commitment <= 0.70

Full correct-goal completion and collisions are reported but are not the
primary admission criterion for this first integration milestone. The primary
question is whether remembered content causally controls a real temporally
extended option after the cue is absent.

## After PASS

Freeze the memory->goal interface.

Next milestone:
- use the explicit goal register to bias the repaired short-horizon planner as
  an advisory scorer;
- require imagined consequence intervention to change plan selection;
- then move to multi-item/order-sensitive memory and richer 3-D embodiment.

## After FAIL

Do not reopen v0.20 planner tuning.

Use the failed component to distinguish:
- goal decoder failure;
- learned-option lateral coverage failure;
- goal-to-option arbitration failure;
- autonomous option execution failure.

Repair only that integration layer.

No consciousness or general-intelligence claim follows from a pass.
