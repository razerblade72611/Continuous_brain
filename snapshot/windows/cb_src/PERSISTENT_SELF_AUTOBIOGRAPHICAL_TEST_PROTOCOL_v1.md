# Continuous Brain v0.28.1 — Frozen Persistent-Self TEST

## Authorization

v0.28.0 passed every frozen DEV criterion.

The result included:
- agency attribution accuracy: 1.000
- confidence-transition MAE: 0.01038
- uncertainty-transition MAE: 0.01831
- canonical self-caused agency probability: 0.999993
- canonical exogenous agency probability: 0.00000486
- identity exact after reload: 1.000
- self-state exact after reload: 1.000
- autobiography exact after reload: 1.000
- unresolved intention resume: 1.000
- reset control resume: 0.000
- autobiographical erase changes goal: 1.000
- donor history changes goal: 1.000
- identity survives donor-history swap: 1.000
- prediction error generates INVESTIGATE: 1.000
- prediction-error ablation removes INVESTIGATE: 1.000
- surprise temporarily interrupts commitment: 1.000
- commitment resumes when surprise resolves: 1.000
- self/world attribution intervention changes goal: 1.000

The self TEST remained untouched.

## Purpose

Run the one frozen TEST admission for the persistent-self/autobiographical
mechanism.

No parameter is trained or modified.

The learned agency/self-transition model is evaluated on held-out self-model
examples:

- TEST IDs 30000..33999

The causal continuity/history assays also use new identity seeds that were not
used in DEV.

## Fresh causal TEST seeds

Continuity:
- 130000..130031

Autobiographical subject histories:
- 140000..140127

Autobiographical donor histories:
- 150000..150127

Endogenous-goal tests:
- 160000+ / 161000+

Agency-intervention tests:
- 170000..170127

These seeds are fixed before TEST execution.

## Frozen gate

The exact v0.28.0 thresholds are retained.

Model:
- agency attribution accuracy >= .97
- confidence transition MAE <= .035
- uncertainty transition MAE <= .035
- canonical self-caused agency probability >= .85
- canonical exogenous agency probability <= .15

Continuity:
- identity exact after reload = 1.00
- self-state exact after reload = 1.00
- autobiography exact after reload = 1.00
- unresolved intention resumes = 1.00
- reset control resume <= .05

Autobiographical causality:
- normal history reconstructs expected goal = 1.00
- erasing completion changes reconstructed goal = 1.00
- donor completion changes reconstructed goal = 1.00
- identity survives donor-history intervention = 1.00

Endogenous goals:
- internal prediction error generates INVESTIGATE = 1.00
- zeroing prediction error removes INVESTIGATE = 1.00
- surprise temporarily interrupts a commitment = 1.00
- unresolved commitment resumes after surprise resolves = 1.00

Self/world causal intervention:
- attribution intervention changes next goal = 1.00
- world attribution raises uncertainty more = 1.00
- world attribution raises world-attributed error more = 1.00

## Preservation

The run hashes and preserves:
- canonical InfantBrain checkpoint;
- admitted working memory;
- v0.23 episodic reader;
- remembered-goal decoder;
- v0.25 consequence model;
- v0.25.2 frozen TEST result;
- v0.26 THOR admission;
- v0.27 formal FAIL result;
- v0.28.0 DEV result;
- frozen agency/self model;
- persistent-self seed.

## Meaning of PASS

A PASS supports the narrow functional claim that the added self system
generalizes to held-out self-model examples and fresh identity/history
interventions while preserving:
- persistent identity;
- autobiographical continuity;
- unresolved intention continuity;
- self/world agency attribution;
- endogenous information-seeking recruitment;
- causal dependence of later goals on autobiographical history.

It is not evidence sufficient to establish subjective experience, sentience,
phenomenal consciousness, or qualia.

## Stop rule

After this run:
- TEST 30000..33999 is spent;
- do not tune v0.28 against it.

If PASS:
- freeze v0.28 as the persistent-self milestone;
- next build should make self-state causally central to the existing recurrent
  workspace, memory retrieval, policy/value and planning consumers;
- compare normal self-broadcast with self-disabled and history-disrupted
  controls while preserving ordinary reactive competence.

If FAIL:
- retain formal TEST FAIL;
- do not tune against TEST;
- redesign the self layer on a new development family.
