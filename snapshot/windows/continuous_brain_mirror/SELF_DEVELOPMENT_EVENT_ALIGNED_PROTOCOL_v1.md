# Continuous Brain v0.33.4 — Event-Aligned Self-Development DEV Repair

## Historical record

Preserve exactly:

- v0.33.0: DEV FAIL
- v0.33.1: DEV FAIL
- v0.33.2: coarse timing explanation rejected
- v0.33.3: exact event-order diagnostic supported

No prior result is relabeled.

## What v0.33.3 established

At the actual controller decision event:

- predecision +0.12 crossing rate = 0.9609375
- graduation rate = 0.9609375
- exact event alignment rate = 1.0
- graduation without predecision crossing = 0.0
- predecision crossing without graduation = 0.0

Therefore the autobiographical self-estimate crossing and the SCAFFOLD -> DIRECT
developmental transition are exactly aligned at the causal decision event.

The final-state reliability gate in v0.33.1 was not aligned with the causal
event. Later behavior creates later autobiography and can change the final
reliability after the developmental transition already occurred.

## Repair scope

v0.33.4 changes no mechanism.

Unchanged:
- frozen v0.31 recurrent trajectory model
- frozen v0.32 autobiographical self-model
- developmental environment
- capability dynamics
- history-relative hysteresis controller
- +0.12 meaningful self-estimate gain threshold
- all original behavioral/capability/discrepancy thresholds

Only the failed gate semantics change.

Old failed gate:

`final autobiographical reliability remains >= baseline + 0.12 in >=80%`

Replacement gate:

`autobiographical reliability reaches >= baseline + 0.12 at the actual
controller decision event in >=80%, with exact crossing/graduation alignment`

Required exact alignment:
- crossing rate == graduation rate
- graduation without crossing == 0
- crossing without graduation == 0
- first crossing decision == first graduation decision for every graduating
  subject

## Why this is not blind threshold tuning

The numerical threshold remains +0.12.

The controller remains unchanged.

No neural model is retrained.

The DEV population is the same already-spent window.

The only change is measuring the preregistered quantity at the event where it
causally affects behavior rather than at an arbitrary later endpoint.

## Additional controls retained

The repair must still pass:

- initial autobiography identifies fragility
- fragile history initially selects SCAFFOLD
- stable history stays DIRECT
- donor autobiography flips strategy
- erased autobiography removes fragile response
- behavior changes next autobiographical discrepancy
- updated self-estimate changes later behavior
- frozen autobiography blocks graduation even while practice continues
- behavior develops hidden capability
- later autobiographical discrepancy is reduced
- full closed-loop criterion
- persistence across save/reload

## DEV / TEST

Reuse DEV:
- 10000..10127

TEST remains sealed:
- 20000..20127

No new development window is consumed.

## Meaning of DEV PASS

A DEV PASS supports this event-aligned functional statement:

1. autobiographical history changes the initial developmental strategy;
2. strategy changes later expected-vs-actual autobiographical evidence;
3. updated autobiography changes the frozen self-estimate;
4. when that self-estimate improves by the already-preregistered +0.12 amount
   at the actual decision event, the behavior transitions from SCAFFOLD to
   DIRECT with exact event alignment;
5. freezing autobiography prevents that transition even while practice and
   hidden capability continue to improve;
6. the changed behavior continues to alter capability and later autobiography.

This is a closed functional self-development loop.

It does not establish subjective experience, sentience, phenomenal
consciousness, qualia, personhood, or moral status.

## Stop rule

If v0.33.4 DEV PASS:
- freeze the exact controller, dynamics, thresholds, and event-aligned gate;
- build one frozen TEST on 20000..20127;
- TEST is spent after that run and may never be tuned against.

If v0.33.4 DEV FAIL:
- preserve the failure;
- keep TEST sealed;
- do not repair against TEST.
