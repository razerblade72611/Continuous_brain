# Continuous Brain v0.20.6a — Fresh Validation Software Patch

This patch does **not** change the preregistered scientific gate, planner,
checkpoint, seeds, episode count, or model behavior.

The first v0.20.6 execution completed all 100 baseline episodes and all 100
same-option-bootstrap episodes, then crashed before result construction because
the validator looked up:

`baseline["episodes"]["option_direct_diagnostic"]`

The actual per-episode key emitted by
`planning_v103.evaluate_hierarchical_options` is:

`baseline["episodes"]["option_direct"]`

v0.20.6a changes only that lookup.

It also replaces Python's process-randomized `hash(field)` contribution to the
bootstrap RNG seed with a fixed field-to-integer mapping. This changes no gate
threshold and makes the descriptive bootstrap intervals exactly reproducible
across processes.

The frozen validation remains:
- seeds 15001-15100;
- 100 paired episodes;
- CPU;
- no training;
- illegal cap handoffs == 0;
- same-option bootstrap used > 0;
- repaired completion >= historical hierarchical completion;
- repaired completion >= direct option completion - 0.05;
- repaired collisions <= historical hierarchical collisions + 5;
- repaired collisions <= direct option collisions + 10.

No fresh performance numbers were admitted from the failed wrapper run, so this
is a rerun of the same frozen validation after a software-only repair, not a
new tuning cycle.

Detour planner tuning still stops after this validation regardless of outcome.
