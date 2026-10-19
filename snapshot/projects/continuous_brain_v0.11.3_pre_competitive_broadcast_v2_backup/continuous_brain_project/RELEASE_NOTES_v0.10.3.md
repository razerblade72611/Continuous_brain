# Continuous Brain v0.10.3 Release Notes

## Summary

v0.10.3 repairs the integration between the v0.10.2 temporal Option / Intention Cortex and the v0.10 recurrent planning cortex.

No learned tensor is added, removed, resized, or retrained. The v0.10.2 checkpoint remains the same child state.

## Development evidence motivating the change

On development seeds 9001-9010:

- v0.10.2 direct learned options: **5/10 completion**;
- option-policy top-1 proposal with recurrent rollout: **5/10 completion**;
- top-2 proposal primitive search: **0/10 completion**;
- top-3 proposal primitive search: **0/10 completion**.

The old option-planning integration expanded primitive actions directly. The more freedom primitive imagination received, the worse behavior became.

## New behavior

`evaluate-hierarchical-options` implements hierarchical option-level imagination:

- active options own primitive actions;
- high-level planning happens only at learned option boundaries;
- the learned selector proposes candidate option IDs;
- each option is rolled forward as a multi-step macro-action using its own learned action policy;
- recurrent dynamics predict the macro consequences;
- high-level beam search compares option sequences.

## Compatibility

- v0.10.2 trained checkpoints load without migration.
- No new training command is required.
- `evaluate-options` is retained unchanged for legacy v0.10.2 diagnostics.
- New command: `evaluate-hierarchical-options`.

## Primary development configuration

The release predeclares one primary configuration before another development run:

```text
option_depth=2
selector_topk=2
beam_width=8
max_option_steps=12
discount=0.95
min_commit=3
terminate_threshold=0.50
```

## Validation

- v0.9.0 through v0.10.3 lineage regression: 35 passed.
- v0.10.3 unit tests verify option-owned primitive control and option-level search.
- hierarchical evaluator smoke-tested end-to-end with result serialization.

The broader modern/core suite is validated separately in `RELEASE_VALIDATION_v103.md`.
