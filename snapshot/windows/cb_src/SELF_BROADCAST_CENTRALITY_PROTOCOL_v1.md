# Continuous Brain v0.29.0 — Persistent Self Broadcast / Causal Centrality

## Purpose

v0.28.1 is a frozen TEST PASS. The persistent self now survives interruption,
maintains unfinished intentions, stores autobiographical history, distinguishes
self-caused from world-caused change, and can generate an internally recruited
INVESTIGATE goal.

The remaining limitation is architectural: that self is still mostly a sidecar.

v0.29.0 asks whether one shared representation of the persistent self can become
causally load-bearing across multiple cognitive consumers while leaving the
historical brain exactly intact when self-broadcast is not recruited.

## Shared self broadcast

Input:
- identity anchor32
- persistent self vector10

One shared 64-D broadcast latent fans out to:
- the actual frozen InfantBrain workspace/policy path
- episodic-retrieval routing
- current-goal routing
- planning / information-gathering mode
- planning-depth modulation
- value/priority modulation
- identity reconstruction

No downstream consumer receives a separate private raw self vector.

## State-dependent recruitment

Broadcast can be enabled/disabled or recruited with a scalar gate.

Disabled/recruitment-zero mode is structurally exact-neutral:
- broadcast = 0
- workspace residual = 0
- retrieval logits = 0
- goal logits = 0
- planning-mode logits = 0
- historical workspace remains bit-exact
- historical motor logits remain bit-exact

This is the key selective-loss control. Turning off self must not simply damage
the whole system.

## Existing brain consumers

The historical workspace width and policy-channel count are inferred at runtime
from the protected InfantBrain.

The workspace residual is added to the true historical workspace and then passed
through the frozen historical motor head.

The retrieval route is passed into the already-qualified v0.23 episodic reader
as FIRST/SECOND selection.

The planning-control outputs are additive control signals for later full runtime
integration; this version does not claim the historical PlanningCortex already
consumes them.

## New broadcast family

TRAIN:
- 0..11999

DEV:
- 20000..22999

TEST:
- 30000..32999, sealed

Goals NONE / A / B / INVESTIGATE are exactly balanced and external reference
workspace is generated independently of the self target.

## Trainable component

Only `SelfBroadcastHub`.

Frozen:
- historical InfantBrain
- admitted working memory
- v0.23 episodic reader
- remembered-goal decoder
- v0.25 consequence model
- v0.28 agency/self model
- v0.28 persistent self seed

## DEV admission

Normal broadcast:
- policy self-target accuracy >= .95
- retrieval accuracy >= .97
- goal accuracy >= .97
- planning-mode accuracy >= .97
- planning-depth MAE <= .05
- value-priority MAE <= .05
- identity reconstruction cosine >= .95

Broadcast disabled:
- workspace residual exactly zero
- combined workspace exactly historical
- policy logits exactly historical
- all categorical self-consumer logits exactly zero
- self-dependent policy accuracy <= .40
- goal accuracy <= .30
- retrieval accuracy <= .35
- INVESTIGATE recall <= .05

Donor-self intervention:
- same external workspace
- swap only self representation
- policy/retrieval/goal/planning must coherently follow donor >= .95

Autobiographical interventions:
- normal completion history controls later broadcast
- donor completion history controls later broadcast
- erased completion history controls reconstructed broadcast
- each >= .95

Existing episodic reader:
- self-selected route must drive frozen v0.23 FIRST/SECOND retrieval >= .95

Persistent reload:
- loading the same persisted self twice must yield bit-exact shared broadcast and
  every consumer output.

## Meaning of PASS

PASS supports the functional claim that the persistent self is no longer merely
stored beside the agent: a single shared representation of identity,
commitments, confidence, uncertainty and surprise causally controls multiple
consumers, including the actual historical workspace/policy path and the
existing episodic reader.

Disabling the self broadcast selectively removes self-guided effects while the
historical policy path remains exactly unchanged.

This is not a test for subjective experience, phenomenal consciousness, qualia,
sentience or moral status.

## After PASS

Use broadcast TEST once.

If TEST holds, next build should implement continuous self/world-model coupling:
- predict both external future and future self-state under candidate actions
- jointly evaluate world consequence and self consequence during planning
- treat self-prediction error as a metacognitive error signal
- update autobiography and self online
- preserve commitments across multiple tasks and interruptions

That is a deeper continuing-agent architecture than adding more isolated task
heads.
