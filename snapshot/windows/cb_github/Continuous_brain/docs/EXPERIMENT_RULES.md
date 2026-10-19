# Experiment rules as enforced in code

These restate the project's rules (Charter, E2 prereg, Joint Contract) and name the code that enforces each. If a rule here isn't enforced in code, it says so.

| Rule | Enforced by |
| --- | --- |
| A claim that a run passed or failed cites path + SHA-256, or is UNVERIFIED | `ResultRecord.validate`, `RunStore.finalize` (manifest of every file), `RunRegistry` (E0) |
| PASS/FAIL only under an APPROVED/FROZEN preregistration, with an approver, that covers the item | `ResultRecord.validate` (looks the prereg up; `allowed_items`), `PreregistrationRegistry.require` |
| Gates propose verdicts; a verdict becomes PASS/FAIL only once primary artifacts are attached | `gates/common.propose_verdict`, `RunStore.finalize` |
| No PASS/FAIL from a non-scientific context (test fixtures, dev overrides) | `UnifiedRuntime.scientific`, `ResultRecord.scientific_context` |
| Battery items B2–B6 are engineering-only until each is preregistered | `battery/common.py` sets `ENGINEERING_ONLY` + `UNREGISTERED_METRIC` |
| B1 needs the E2 body; no stand-in | `b1_sensorimotor` returns BLOCKED |
| Spent sealed TEST ranges are never reused; new families overlap nothing; ranges compare by range, not name | `SeedRegistry.check_fresh` / `check_usable` / `spend` / `check_seeds`; every gate checks its seeds |
| Battery smoke runs use the ENGINEERING seed family only | `check_engineering_seed` |
| Identical run specs are not silently rerun | `RunStore.create` (content-addressed `run_id`) |
| Evaluators are read-only except through declared interventions | `ReadOnlyRuntimeView`, `ChannelStore.override` token, `InterventionManager.lock` |
| The G2 sign matrix is fixed before data | `InteractionMatrix.freeze` / `check_frozen` |
| Semantic firewall: sensory channels carry physical quantities only | `ChannelSpec` (units whitelist, forbidden name parts), `validate_physical` (E0 anatomy) |
| Transformer / gradient updates never reach the SNN | `SNNSubstrate.train_step` raises; `assert_gradient_firewall` checks every tensor |
| Language is never the clock | the runtime owns ticks; coprocessor calls are synchronous, capped per tick, logged |
| The runtime boots and runs with the coprocessor disabled | `RuntimeConfig.coprocessor_enabled=False` default; `NullCoprocessor` |
| Legacy mechanisms are wrapped by hash, never reimplemented under their names | `LegacyManifest` (VERIFIED only), `verify_entry`, step-contract registry keyed by module SHA-256 |
| No absolute personal paths committed | manifest `check_relpath`; test scans scripts and the unified package |
| Paid cloud work is never launched by code | `runpod_plan.py` prints only; test forbids network use |
| Single exposure for the dense comparator unless a replay budget is declared | `IntervalLedger`, `SingleExposureViolation` |
| Decision rules are separate from experiment config and versioned | `decision_rules.RULES` with implementation SHA-256 |
| Formal failures stay failures | inventory marks `admissible: false`, records `*_passed: false` flags; no code path promotes them |

**Not enforced by code (process only):** David's approval itself; that agent agreement is not evidence; that A1 is pending; that E3–E6 wait for E2 TEST.
