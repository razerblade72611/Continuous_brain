# Continuous Brain v0.29.1 — Self-Broadcast Value-Priority Repair

## Source result

v0.29.0 remains formally:

`SELF_BROADCAST_CENTRALITY_DEV_GATE: FAIL`

Exactly one gate failed:

- value-priority DEV MAE = 0.0661396
- threshold = <= 0.05

The corresponding TRAIN MAE was 0.0656951.

Every other centrality gate passed.

This pattern is important: TRAIN and DEV error are essentially identical.
The failure therefore does not look like DEV overfitting or a failure of the
shared-self causal mechanism.  It is consistent with a readout-capacity
limitation.

## Why the old head is structurally weak

The v0.29 value-priority target is:

`max(unresolved_A, unresolved_B, prediction_error, 0.75 * uncertainty)`

That is a nonlinear max-like function.

v0.29.0 used one bias-free linear projection from the 64-D shared broadcast
followed by a sigmoid.

The same broadcast nevertheless supported:
- policy = 1.000
- retrieval = 1.000
- goal = 1.000
- planning mode = 1.000
- identity cosine = 0.9918
- donor-self coherence = 1.000
- autobiography interventions = 1.000
- existing episodic-reader routing = 1.000

So v0.29.1 does not retrain or replace the shared representation.

## Repair

Freeze the entire v0.29.0 `SelfBroadcastHub`.

Add one new consumer:

`broadcast64 -> Linear(no bias) -> SiLU -> Linear(no bias) -> clamp[0,1]`

Only this priority head is trained.

It receives:
- no raw identity;
- no raw commitments;
- no raw confidence;
- no raw uncertainty;
- no raw prediction-error field.

Its only input is the already-frozen shared 64-D self broadcast.

Therefore a repair PASS means the information needed for nonlinear internal
priority was already present in the shared representation.

## Exact neutral rule

Both repair layers are bias-free.

When shared broadcast is zero:

- first hidden = exactly zero;
- raw priority = exactly zero;
- repaired priority = exactly zero.

So disabled self cannot inject a latent priority signal.

## Data use

TRAIN:
- original broadcast TRAIN 0..11999

DEV:
- original already-spent DEV 20000..22999

No new development window is consumed.

TEST:
- 30000..32999 remains sealed and untouched.

This is a DEV repair, not a fresh-validation claim.

## Preservation

The repair hashes before and after:
- source v0.29 result
- source v0.29 shared hub
- v0.28.1 self TEST
- historical InfantBrain
- admitted working memory
- v0.23 episodic reader
- remembered-goal decoder
- consequence model
- agency/self model
- persistent-self seed

Any mutation invalidates the repair.

## Frozen repair gate

Required:
- every original v0.29 passing gate remains preserved by the unchanged source
  hub;
- repaired DEV value-priority MAE <= 0.05;
- zero broadcast -> exactly zero repaired priority;
- source hub unchanged;
- protected artifacts unchanged;
- TEST untouched.

## Interpretation

If PASS:

The v0.29 shared-self centrality hypothesis survives DEV repair.  The single
failure was the expressiveness of one nonlinear value-priority consumer, not the
shared 64-D self representation.

The original v0.29.0 record remains FAIL.  v0.29.1 is recorded separately as a
DEV repair PASS.

Then freeze:
- source hub
- repaired priority head

and create one frozen v0.29.2 TEST using TEST 30000..32999 exactly once.

If FAIL:

Do not touch TEST.

The 64-D shared broadcast may not expose enough recoverable information for the
priority computation.  The next repair must investigate the broadcast itself
rather than adding further readout capacity.
