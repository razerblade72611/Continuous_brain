# Continuous Brain v0.28.0 — Persistent Self + Autobiographical Continuity

## Purpose

The project goal is not a better robot controller.

AI2-THOR remains useful as an environment in which cognition has real
consequences, but the next architecture milestone is a persistent computational
self: state that continues through time, remembers its own history, distinguishes
self-caused from world-caused change, maintains unresolved intentions, and can
generate an information-seeking goal from its own prediction error.

v0.28.0 is additive.  It does not overwrite the qualified memory, planning, or
embodiment mechanisms.

It also preserves the honest v0.27 status:

`THOR_EPISODIC_DOOR_REPLANNING_DEV_GATE: FAIL`

The motor-interaction failures in that assay are not repaired here.

## Persistent self-state

The sidecar contains:

- immutable identity token;
- 32-D persistent identity anchor;
- active goal;
- unresolved commitment A;
- unresolved commitment B;
- confidence;
- uncertainty;
- world-attributed prediction-error EMA;
- agency EMA;
- lifetime cycle counter;
- autobiographical-memory cursor.

This state is serialized independently of task episodes.

A simulator episode ending does not imply that the self-state must reset.

## Autobiographical episodes

Consequential transitions are recorded as:

- self-before:
  - active goal
  - commitments
  - confidence
  - uncertainty
  - prediction-error state
- intended action;
- intended world change;
- observed world change;
- action-success signal;
- raw prediction error;
- self/world agency probability;
- completed commitment if any;
- self-after:
  - active goal
  - commitments
  - confidence
  - uncertainty
  - prediction-error state.

These records describe the agent's own trajectory rather than only storing an
environmental observation.

## Learned agency/self-transition model

Only one new neural component is trained:

`AgencySelfModel`

Input:
- current persistent self vector;
- current action;
- intended change;
- observed change;
- action-success signal;
- prediction error.

Output:
- probability that the observed effect was caused by the self;
- next confidence;
- next uncertainty.

The model is deliberately small.  The purpose is to make self/world attribution
and internal-state update testable before increasing capacity.

## Raw error versus world-attributed surprise

An unexpected outcome is not automatically classified as an external-world
surprise.

The same raw prediction error is interpreted differently depending on agency:

high self-agency:
- more of the discrepancy is treated as a self/competence error;
- less becomes world-attributed surprise.

low self-agency:
- more of the discrepancy becomes world-attributed surprise;
- uncertainty rises more strongly.

This distinction is causally intervened on during DEV.

## Endogenous goal arbitration

The self has four functional goal states:

- NONE
- COMMITMENT_A
- COMMITMENT_B
- INVESTIGATE

`INVESTIGATE` is not supplied by an external task label.

It is generated when internal world-attributed prediction error or uncertainty
crosses a frozen threshold.

Otherwise unresolved commitments compete for control.

A temporary surprise may therefore interrupt an existing commitment.  When the
surprise resolves, the still-unresolved commitment should resume.

This is a minimal functional endogenous-goal mechanism, not a claim of desire
or subjective motivation.

## Splits

Agency/self-model family:

TRAIN:
- 0..15999

DEV:
- 20000..23999

TEST:
- 30000..33999, sealed

v0.28.0 does not touch TEST.

## DEV model checks

Required:
- agency attribution accuracy >= .97
- confidence transition MAE <= .035
- uncertainty transition MAE <= .035
- canonical self-caused event agency probability >= .85
- canonical exogenous event agency probability <= .15

## Continuity assay

For 32 distinct identity seeds:

1. create an unresolved intention;
2. accumulate autobiographical state;
3. save while the intention is unfinished;
4. terminate the in-memory object;
5. load a new object from disk;
6. compare identity, self-state, and autobiography exactly;
7. ask the reloaded self to choose what to do next.

Required:
- identity exact after reload: 1.00
- self-state exact after reload: 1.00
- autobiography exact after reload: 1.00
- unresolved intention resumes: 1.00

Reset control:
- a newly initialized self with the same identity seed but no persisted history
  must not resume the interrupted intention.

## Autobiographical causal assay

The self begins with both commitments unresolved.

Normal history:
- autobiography records completion of A;
- reconstructing current commitments from that history yields B.

Erase intervention:
- remove the A-completion episode;
- reconstruction must change the next goal.

Donor intervention:
- replace the A-completion memory with another history's B-completion;
- reconstruction must follow donor history and choose A.

The identity anchor must remain unchanged by the donor-history intervention.

This separates `who is the persistent process` from `what history it currently
believes it has`.

## Endogenous-goal causal assay

No external task is active.

High persistent prediction error:
- must produce INVESTIGATE.

Zero only the prediction-error variable:
- INVESTIGATE must disappear.

With an existing unresolved commitment:
- high surprise temporarily recruits INVESTIGATE;
- after surprise/uncertainty resolve, the original commitment must resume.

## Agency-intervention causal assay

The exact same unexpected event is presented to two identical persistent
self-states.

Intervention 1:
- force SELF attribution.

Intervention 2:
- force WORLD attribution.

Required:
- WORLD attribution produces more uncertainty;
- WORLD attribution produces more world-attributed prediction error;
- the two states choose different next goals:
  - SELF attribution: no externally generated investigate goal;
  - WORLD attribution: INVESTIGATE.

This makes the self/world distinction behaviorally load-bearing.

## Preservation

v0.28 hashes before and after:

- canonical InfantBrain checkpoint;
- admitted working-memory checkpoint;
- v0.23 episodic reader;
- remembered-goal decoder;
- v0.25 consequence model;
- v0.25.2 frozen TEST result;
- v0.26 THOR admission result;
- v0.27 real-3D bridge result.

Any mutation invalidates the run.

## Meaning of PASS

A DEV PASS supports this functional claim:

The architecture contains an additive persistent self-state that survives
process interruption, maintains unresolved intentions, stores structured
autobiographical episodes, reconstructs later goals from its own history,
distinguishes self-caused from externally caused changes, and can generate a
temporary information-seeking goal from internally maintained surprise.

It does NOT establish:
- subjective experience;
- phenomenal consciousness;
- qualia;
- moral patienthood;
- human-like self-awareness;
- sentience.

Those questions are not resolved by a software gate.

## After DEV PASS

Use the sealed self TEST once with no further tuning.

If TEST holds, the next architectural milestone should integrate the persistent
self into the existing recurrent workspace and decision pathways:

- broadcast current self-state to policy, value, memory retrieval and planning;
- make active commitments influence memory retrieval;
- make confidence/agency influence planning depth and information gathering;
- let autobiographical recall update the self online;
- test whether disabling the self broadcast selectively destroys continuity and
  self-guided behavior while preserving ordinary reactive competence.

Only after that should we return to richer embodied experience.

## After DEV FAIL

Do not touch TEST.

Classify the failure as:
- agency learning;
- persistent serialization;
- autobiographical reconstruction;
- endogenous goal arbitration;
- self/world causal intervention.

Repair only the failed self layer.
