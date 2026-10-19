# Continuous Brain v0.29.2 — Gradient-Safe Value-Priority Repair

## Status carried forward

v0.29.0 remains a formal DEV FAIL.
Its only failed gate was value-priority MAE.

v0.29.1 also remains a formal repair FAIL.

v0.29.1 made the priority error much worse:
- original TRAIN MAE: ~0.0657
- original DEV MAE: ~0.0661
- v0.29.1 TRAIN MAE: ~0.2786
- v0.29.1 DEV MAE: ~0.2787

Every other v0.29.0 self-centrality gate remained PASS.

## Diagnosis of v0.29.1

v0.29.1 used:

`priority = clamp(raw, 0, 1)`

inside the trainable head.

For negative raw predictions the derivative of clamp is zero.
The readout therefore creates dead regions and can strand a large subset of
examples at exactly zero.

The observed low-priority error plateau and prediction mean well below target
mean are consistent with that optimization failure.

This result does NOT support concluding that the frozen shared broadcast lacks
priority information, because the simpler original linear readout already
achieved ~0.066 MAE.

## v0.29.2 repair

The v0.29 shared 64-D broadcast is frozen byte-for-byte.

Only a new nonlinear priority consumer is trained.

Architecture:

`broadcast64 -> Linear128(no bias) -> SiLU -> Linear128(no bias) -> SiLU -> Linear1(no bias)`

Output:

`2 * sigmoid(raw) - 1`

Properties:
- zero broadcast -> raw exactly zero -> priority exactly zero;
- derivative at raw zero = 0.5, so the neutral point remains trainable;
- no raw persistent-self fields are available;
- the repair must decode information already present in the frozen broadcast.

## Data

Uses the same original broadcast family:

TRAIN:
- 0..11999

DEV:
- 20000..22999, already spent and reused for repair

TEST:
- 30000..32999, still sealed

No new DEV evidence is claimed.

## Additional repair sanity gates

The repair must do more than pass the original <= .05 threshold.

It must also:
- improve TRAIN MAE by >= .01 versus the original linear head;
- improve DEV MAE by >= .01 versus the original linear head;
- keep |TRAIN MAE - DEV MAE| <= .02;
- preserve exact zero priority for zero broadcast;
- demonstrate a finite, nonzero input gradient at zero broadcast.

Those checks guard against another optimization artifact.

## Preservation

The following remain unchanged:
- v0.29.0 result
- v0.29.0 shared broadcast hub
- v0.29.1 failed-repair result
- v0.28.1 self TEST
- historical InfantBrain
- admitted working memory
- v0.23 episodic reader
- goal decoder
- consequence model
- agency/self model
- persistent-self seed

## Stop rule

If PASS:
- v0.29.0 remains historical FAIL;
- v0.29.1 remains historical FAIL;
- v0.29.2 is recorded as the successful DEV repair;
- freeze the original shared hub plus v0.29.2 priority head;
- run broadcast TEST 30000..32999 exactly once with no more tuning.

If FAIL:
- do not touch TEST;
- stop adding readout capacity;
- conclude that the current frozen 64-D broadcast may not preserve enough
  recoverable scalar priority information;
- redesign the shared broadcast on a new development family.
