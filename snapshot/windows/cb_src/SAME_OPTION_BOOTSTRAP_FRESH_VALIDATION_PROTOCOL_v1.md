# Continuous Brain v0.20.6 — Fresh Same-Option Bootstrap Validation

v0.20.5 passed every preregistered repair check on the spent 14001-14040
development window:

- completion 0.725
- historical hierarchical completion 0.525
- direct-option completion 0.65
- collisions 29.15
- illegal cap handoffs 0
- same-option bootstrap used 88 times
- completion improvement vs v0.20.4: +0.575

This package performs the single fresh validation promised before leaving the
Detour planner-repair phase.

## Seeds

15001-15100, 100 paired episodes.

These are fresh relative to the v0.20.0-v0.20.5 planner diagnosis/repair
windows. No claim is made that they were globally excluded from all historical
training.

## Frozen validation gate

Same-option bootstrap planner must satisfy:

- illegal cap handoffs == 0
- same-option bootstrap used > 0
- completion >= historical hierarchical completion on the same seeds
- completion >= direct learned-option completion - 0.05
- collisions <= historical hierarchical collisions + 5
- collisions <= direct learned-option collisions + 10

Paired bootstrap 95% intervals are reported descriptively but are not used to
change the preregistered gate after seeing the data.

## Hard stop

After this run, Detour planner tuning stops regardless of outcome.

If PASS:
- freeze the continuation-bootstrap planner;
- begin memory-guided autonomous planning.

If FAIL:
- keep the direct learned-option controller as the reliable execution backbone;
- begin memory-guided autonomous planning anyway;
- retain the planner/world model as an advisory/experimental subsystem until
  later long-horizon retraining.

The purpose is to move the Continuous Brain forward, not continue optimizing a
single synthetic benchmark.
