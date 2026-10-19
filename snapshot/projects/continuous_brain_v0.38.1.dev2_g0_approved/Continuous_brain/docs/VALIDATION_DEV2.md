# v0.38.1.dev2 validation and approval record

Status: G0 execution artifact. This is not a G0 result.

Human approval:
- preregistration: `G0-LEGACY-REQUAL-v1`
- approved by: David
- approval date: 2026-09-27
- exact user statement: `I approve G0-LEGACY-REQUAL-v1.`

Scientific constraints retained:
- no historical sealed TEST range may be reopened;
- the seven fresh G0 families are DEV, registered, disjoint, and initially unspent;
- each fresh family is single-use for G0: the local append-only seed ledger records it spent before evaluator launch;
- frozen evaluator thresholds and causal interventions are unchanged;
- only the named seed-family symbol and redirected output/state directory are varied;
- project-local evaluator import closures are hash-pinned and checked before and after execution;
- v0.24 and v0.28 remain blocked.

Build-side validation:
- 181 tests collect;
- 68 G0/prereg/gate/protocol/script/substrate tests pass in the build sandbox;
- G0 runner unit test verifies confirmation lock and spend-before-launch behavior;
- static preflight succeeds for all seven preserved evaluator sources in the recovered mirror/overlay set;
- `continuous_brain_snn` is unchanged from the canonical E0 lineage.

The full 181-test suite and read-only G0 preflight must pass on David's canonical WSL CPU environment before opening any fresh family.
