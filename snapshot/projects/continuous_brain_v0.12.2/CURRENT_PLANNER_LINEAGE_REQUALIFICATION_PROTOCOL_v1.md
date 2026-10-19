# Continuous Brain v0.19.2 — Current Planner Lineage Requalification

## Decision carried forward

v0.19 remains closed. Do not train or repair its timing gate.

Preserve:
- historical 14K child;
- admitted working-memory writer/reader;
- frozen v0.18 intent head.

## Why this run exists

The v0.19.1 integration audit found that the CURRENT canonical child already
contains the trained temporal-option/planning lineage:

- option cortex: `latent-temporal-options-v1`
- planning cortex:
  `recurrent-latent-v1+conservative-bellman-q-v1+temporal-options-v1`
- option updates: 3000
- planning updates: 4000
- Bellman-Q updates: 3000

The v0.11 growth metadata says the relevant interfaces were preserved exactly
during whole-brain expansion:
- option action
- option Q
- option selector
- option termination
- planning latent
- planning step
- target spatial representation.

Therefore we should NOT transplant the old v0.10.3 checkpoint into the current
brain. We should verify the planning capability where it currently lives.

## What is re-run

Two historical, already-spent protocols:

1. smoke: seeds 9001-9010;
2. final historical P4b protocol: seeds 12001-12100.

Each compares:
- reactive policy;
- direct learned options;
- hierarchical option-macro imagination.

The 100-episode stage runs only if smoke passes.

This is NOT a new generalization test because these seed sets were used
historically.

## Fail-closed software rules

This package corrects the class of reporting defects discovered in the v0.19
source review:

- evaluator return data is validated in memory;
- stale output files are deleted before evaluation;
- no decision is made from an old output file;
- evaluator exceptions terminate immediately;
- checkpoint SHA in returned results must equal the current canonical child;
- episode counts and protocol arguments are checked;
- protected file hashes are checked before and after;
- no training is performed.

## After PASS

A PASS establishes that the old hierarchical planning capability survives in
the CURRENT canonical child. The next build can then connect canonical recalled
content and an explicit goal representation to the preserved option/planning
stack without migrating an old brain.

The next actual capability milestone is memory-guided, goal-conditioned,
temporally extended planning with autonomous trajectories and causal
consequence interventions.
