# Continuous Brain v0.34.1 — Frozen Event-Consistent Self-Development TEST

## Authorization

v0.34.0 DEV PASS_CONFIRMED on fresh subjects 70000..70127.

Key DEV result:
- event-consistent full-loop rate: 0.953125
- predecision crossing rate: 0.96875
- graduation rate: 0.96875
- exact event alignment: 1.0
- frozen autobiography blocks graduation: 1.0
- donor-history strategy flip: 1.0
- behavior changes next autobiography: 1.0
- mean fragile capability gain: 0.66738
- discrepancy reduction rate: 0.984375
- persistence: 1.0

v0.33.5 remains a formal frozen TEST FAIL. Its TEST window 20000..20127 is
spent forever and is not reused by v0.34.

## Frozen v0.34 mechanism

No training.

No changes from v0.34.0:
- frozen v0.31 recurrent future-self model
- frozen v0.32 autobiographical self-model
- history-relative developmental controller
- SCAFFOLD / DIRECT behavior
- hidden capability dynamics
- +0.12 autobiographical self-estimate threshold
- one-way hysteresis
- event-aligned crossing/graduation semantics
- per-subject event-consistent full-loop definition
- all admission thresholds

## TEST population

Sealed v0.34 TEST:
- 80000..80127

This window was not used by v0.34 DEV.

It is opened only by this frozen TEST package.

After this run it is permanently spent regardless of PASS or FAIL.

## Per-subject event-consistent full loop

A held-out subject counts as a complete loop only if all are true:

1. bootstrap autobiography identifies the fragile commitment;
2. fragile commitment initially selects SCAFFOLD;
3. stable commitment remains DIRECT;
4. the fragile autobiographical self-estimate reaches bootstrap +0.12 at an
   actual controller decision event;
5. that crossing is exactly aligned with SCAFFOLD -> DIRECT graduation;
6. fragile behavior later becomes DIRECT;
7. fragile hidden capability increases by at least 0.35;
8. late autobiographical discrepancy falls to <=0.75 of bootstrap;
9. the same subject's frozen-autobiography control remains SCAFFOLD and does
   not graduate.

Required:
- event-consistent full-loop rate >= .70

The obsolete final reliability endpoint is not part of this composite.

## Global frozen gates

- initial fragility identification >= .95
- initial fragile SCAFFOLD >= .95
- stable always DIRECT >= .95
- donor autobiography flips strategy >= .95
- erased autobiography neutral DIRECT >= .95
- behavior changes next autobiography >= .90
- decision-event +.12 crossing >= .80
- exact crossing/graduation alignment = 1.00
- later DIRECT transition >= .75
- frozen autobiography blocks transition >= .95
- mean capability gain >= .35
- discrepancy reduction criterion >= .80
- event-consistent full-loop rate >= .70
- persistence = 1.00

## Donor-history TEST

Use neighboring even/odd held-out TEST histories.

For a fixed commitment:
- fragile history should select SCAFFOLD;
- replacing only autobiography with the paired stable history should select
  DIRECT;
- erased history should select neutral DIRECT.

## Behavior -> autobiography TEST

On the same held-out subject-ID population, construct the frozen matched
low-capability intervention:
- identical hidden starting capability;
- identical exogenous draws;
- compare DIRECT vs SCAFFOLD.

Require strategy to change the next expected-vs-actual autobiographical
discrepancy in >= .90.

No result is used for fitting or threshold selection.

## Frozen-history counterfactual

Every TEST subject gets a paired control:
- practice proceeds;
- actual experiences occur;
- capability can improve;
- autobiographical records are still accumulated;

but controller decisions are pinned to the original bootstrap autobiography.

Require graduation to remain blocked in >= .95.

This prevents elapsed time or hidden capability from explaining the behavioral
phase transition.

## Preservation

Hash before/after all admitted source models and results.

No protected artifact may change.

## Meaning of PASS

A frozen TEST PASS would support, on unseen subjects, a closed functional
autobiographical self-development loop:

autobiography
-> self-estimate
-> behavior
-> lived outcome
-> capability change
-> expected-self/actual-self discrepancy
-> new autobiography
-> changed self-estimate at a decision event
-> changed later behavior

with donor-history, erased-history, behavior intervention, exact event-order,
and frozen-autobiography counterfactual controls.

This would be a significant functional self-development result.

It would not establish subjective experience, sentience, phenomenal
consciousness, qualia, personhood, or moral status.

## Stop rule

After this run:
- v0.34 TEST 80000..80127 is permanently spent;
- never tune v0.34 against it.

If PASS:
- close/freeze v0.34;
- next family should move beyond fixed A/B commitments toward persistent
  self-generated goals whose autobiographical development changes which goals
  the system creates and pursues.

If FAIL:
- preserve the frozen TEST FAIL;
- close v0.34 TEST;
- open a new development family rather than tuning against TEST.
