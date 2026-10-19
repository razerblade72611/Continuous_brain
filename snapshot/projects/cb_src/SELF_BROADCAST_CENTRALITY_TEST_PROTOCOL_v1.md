# Continuous Brain v0.29.3 — Frozen Self-Broadcast Centrality TEST

## Authorization

v0.29.0 remains a formal DEV FAIL because its original linear value-priority
consumer missed the frozen threshold.

Every other centrality gate passed.

v0.29.1 remains a formal repair FAIL because its clamped nonlinear head created
dead-gradient regions and degraded priority prediction.

v0.29.2 repaired only that consumer while leaving the 64-D shared self
broadcast frozen.

v0.29.2 DEV result:
- repaired value-priority MAE: 0.02898
- original DEV value-priority MAE: 0.06614
- TRAIN repaired MAE: 0.02835
- zero broadcast -> priority exactly zero
- zero-point input gradient alive
- all source passing checks preserved
- TEST untouched

This authorizes one frozen TEST.

## Frozen components

TEST uses:
- exact v0.29.0 shared SelfBroadcastHub
- exact v0.29.2 gradient-safe priority head
- frozen historical InfantBrain
- frozen v0.23 episodic reader
- frozen v0.28 persistent-self stack

No parameter is trained or modified.

## TEST family

Broadcast TEST:
- 30000..32999
- 3000 examples
- used exactly once

Reference historical workspace uses a new fixed generator seed not used by
TRAIN or DEV.

After this run TEST is spent regardless of outcome.

## Fresh causal interventions

Donor-self:
- fresh example IDs starting at 90000
- same external workspace for A/B donor swaps

Autobiographical history:
- fresh subject identities 720000..720127
- fresh donor identities 730000..730127

Existing episodic reader:
- fresh self examples starting at 100000
- canonical recall condition 37 rather than the DEV condition 29

Persistent reload:
- exact frozen v0.28 persistent-self seed loaded independently twice

## Admission gate

Normal self broadcast:
- policy accuracy >= .95
- retrieval route >= .97
- goal route >= .97
- planning mode >= .97
- planning depth MAE <= .05
- repaired value-priority MAE <= .05
- identity reconstruction cosine >= .95

Priority-specific:
- repaired TEST head improves over original TEST linear head by >= .01
- repaired TEST MAE stays within .025 of repaired DEV MAE

Broadcast disabled:
- workspace residual exactly zero
- combined historical workspace bit-exact
- historical policy logits bit-exact
- repaired priority exactly zero
- categorical self-consumer logits exactly zero
- self-policy accuracy <= .40
- goal accuracy <= .30
- retrieval accuracy <= .35
- INVESTIGATE recall <= .05

Fresh donor-self:
- normal A/B all-consumer coherence >= .95
- donor self makes consumers follow donor >= .95
- broadcast-off erases A/B internal difference >= .95
- repaired priority MAE <= .05

Fresh autobiographical interventions:
- normal completion history controls broadcast >= .95
- donor completion history controls broadcast >= .95
- erased completion history controls broadcast >= .95
- repaired priority MAE <= .05

Existing episodic reader:
- self-selected FIRST/SECOND routes frozen v0.23 reader correctly >= .95

Persistent reload:
- entire shared broadcast path bit-exact across reload
- repaired priority bit-exact across reload

## Meaning of PASS

A PASS supports this functional claim on sealed TEST:

A persistent self representation containing identity, commitments and
metacognitive state generalizes as a shared causal broadcast across the actual
historical workspace/policy path, episodic-memory routing, current-goal routing,
planning control, internal priority and identity representation.

Replacing only the self representation changes those self-dependent consumers.
Changing autobiographical history changes the reconstructed self and propagates
through the same shared broadcast.

Disabling the shared self selectively removes those self-guided effects while
the historical workspace/policy path remains bit-exact.

This remains a functional computational result. It is not sufficient evidence
of subjective experience, phenomenal consciousness, sentience or qualia.

## Stop rule

After this run:
- broadcast TEST 30000..32999 is spent
- do not tune v0.29 against it

If PASS:
- freeze v0.29 as the self-centrality milestone
- next milestone should be continuous self/world-model coupling, where candidate
  actions predict both environmental futures and future internal self-state

If FAIL:
- preserve the formal TEST FAIL
- do not repair against TEST
- carry the architectural lesson into a new development family
