# Continuous Brain v0.34.0 — Event-Consistent Autobiographical Self-Development

## Why v0.34 is a new family

v0.33.5 is permanently preserved as a frozen TEST FAIL.

Its only failed admission gate was:

`full_feedback_loop`

Observed:
- 0.6953125 = 89/128
- threshold 0.70

All individual causal and persistence gates passed.

After the TEST, code audit found that the per-subject `full_loop` composite
imported from v0.33.1 still contained the obsolete condition:

`final_fragile_reliability >= initial_fragile_reliability + 0.12`

This remained inside the composite even after v0.33.4 had correctly replaced
the standalone final-state self-estimate gate with the event-aligned causal
criterion.

Thus v0.33.5's single failing composite was internally inconsistent with the
admitted v0.33.4 semantics.

This does NOT allow v0.33.5 to be relabeled or rerun.

The v0.33 TEST subjects 20000..20127 are spent forever.

v0.34 is a new family with a fresh split.

## What remains unchanged

No neural training.

Unchanged from admitted v0.33.4:
- frozen v0.31 recurrent future-self model;
- frozen v0.32 autobiographical self-model;
- history-relative developmental controller;
- SCAFFOLD and DIRECT plans;
- hidden capability dynamics;
- +0.12 meaningful autobiographical self-estimate change;
- one-way hysteresis;
- all top-level admission thresholds.

The purpose of v0.34 is to make the per-subject composite consistent with the
event-aligned causal semantics already justified before the v0.33 frozen TEST.

## Fresh split

v0.34 DEV:
- 70000..70127

v0.34 TEST:
- 80000..80127, sealed

Fresh DEV donor-history probes:
- 71000..71127

Fresh DEV behavior->autobiography probes:
- 72000..72127

The spent v0.33 TEST is never used as v0.34 development data.

## Corrected per-subject full loop

A subject counts as a complete event-consistent self-development loop only if
ALL of the following are true for that same continuing subject:

1. bootstrap autobiography identifies the fragile commitment;
2. fragile commitment initially selects SCAFFOLD;
3. stable commitment remains DIRECT;
4. autobiographical self-estimate reaches bootstrap +0.12 at an actual
   controller decision event;
5. that crossing is exactly aligned with SCAFFOLD -> DIRECT graduation;
6. fragile behavior later becomes DIRECT;
7. fragile hidden capability increases by at least 0.35;
8. late autobiographical discrepancy falls to <=0.75 of bootstrap;
9. the paired frozen-autobiography control for the same subject does NOT
   graduate and stays SCAFFOLD.

The obsolete final-reliability endpoint is not part of the composite.

## Global thresholds

Preserve the v0.33 thresholds:

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

## Strong causal logic

The loop is:

autobiography
-> frozen self-estimate
-> developmental strategy
-> changed experience
-> changed capability
-> expected-vs-actual autobiographical discrepancy
-> updated autobiography
-> changed self-estimate at a decision event
-> changed later strategy

The paired frozen-history control allows:
- practice;
- elapsed cycles;
- actual episodes;
- capability improvement;
- autobiographical record accumulation;

but pins decision-making to the original autobiography.

It must remain scaffolded.

Therefore capability or time alone cannot account for graduation.

## DEV stop rule

If v0.34 DEV PASS:
- freeze the exact v0.34 semantics;
- build one frozen TEST on 80000..80127;
- never tune after TEST.

If v0.34 DEV FAIL:
- keep TEST sealed;
- preserve the failure;
- diagnose only on v0.34 DEV or open a later family.

## Scope

A PASS would support a functional autobiographical self-development loop in the
synthetic cognitive assay.

It would not establish subjective experience, sentience, phenomenal
consciousness, qualia, personhood, or moral status.
