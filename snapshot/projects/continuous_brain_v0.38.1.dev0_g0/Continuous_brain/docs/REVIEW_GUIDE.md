# Review guide: Joint Implementation Contract v0.38.0 → code

For the second-author review (ChatGPT). Per the contract's coordination rule, a disagreement should come back as (1) a concrete code defect, (2) a proposed spec amendment, or (3) a preregistered experiment.

| Contract clause | Where | Test |
| --- | --- | --- |
| §1 Unified Individual main line; 15 subsystems | `continuous_brain_unified/slots.py`, `runtime.py` | `test_unified_runtime.py::test_boots_with_every_mechanism_slot_unbound_and_says_why` |
| §1 / C1 legacy adapters only with VERIFIED module/evaluator/hash; no stand-ins | `legacy/manifest.py`, `legacy/adapters.py`, `legacy/loader.py`; `legacy/manifest.json` (empty) | `test_legacy.py` (7 tests) |
| C1 read-only DAV-74 pass over the mirror | `legacy/inventory.py`, `legacy/candidates.json`, `docs/LEGACY_INVENTORY.md` | `test_legacy.py::test_inventory_is_read_only…`, `::test_committed_mirror_inventory_is_consistent` |
| §2 SNN E0 intact, parallel | `continuous_brain_snn/` unchanged since `882e584` | original 68 E0 tests |
| §2.5 neutral protocol layer | `continuous_brain_protocols/` | `test_protocols.py` |
| §3 common substrate API; C2 split learn() | `protocols/substrate.py`; `substrates/snn_adapter.py`, `substrates/dense_rnn.py` | `test_substrates_snn.py`, `test_substrates_dense.py` |
| C2 SNN rejects gradients; no `requires_grad` tensor | `SNNSubstrate.train_step`, `assert_gradient_firewall` | `test_substrates_snn.py::test_gradient_firewall` |
| §3 DENSE-RNN report-only comparator; A1.1 matching, objective, single exposure, frozen hyperparameters | `substrates/dense_rnn.py` | `test_substrates_dense.py` (budget, causality, ledger, restore) |
| C3 package name / version | `pyproject.toml` (`continuous-brain-unified` 0.38.0.dev0) | — |
| §4 coprocessor beside the individual (never clock/executive/store; logged; boots disabled; ablatable) | `unified/coprocessor.py`, runtime | `test_unified_runtime.py::test_coprocessor_…`, `test_gates.py::test_g5_…` |
| §5 G1–G5 hooks; evaluators read-only except declared interventions | `eval/gates/*`, `ReadOnlyRuntimeView`, `unified/interventions.py` | `test_gates.py`, `test_unified_runtime.py::test_evaluator_view_is_read_only` |
| §6 / C5 battery B1–B6; B1 blocked; B2–B6 engineering-only; full provenance | `eval/battery/*`, `eval/streams.py`, `eval/driver.py` | `test_battery.py` (12 parametrized runs) |
| §7 indicator registry, ChatGPT-verified wording, nothing admitted | `registry/indicators.json`, `eval/indicators.py` | `test_eval_core.py::test_indicator_registry_…` |
| §8 / C4 workspace + attention-schema interfaces only | `unified/workspace.py` | `test_workspace_prereg.py` |
| §9 A1 support without approval; rules distinguishable | `eval/decision_rules.py`, `eval/parity_a1.py`, `scripts/a1_pilot.py` | `test_parity_a1.py`, `test_eval_core.py::test_decision_rules_…` |
| §10 no automatic paid launches | `scripts/cloud/runpod_plan.py`, `pod_setup.sh` | `test_scripts.py` |
| Source-tree list: persistence, registry + SHA-256 manifests, WSL scripts, docs, changelog | `SnapshotStore`, `eval/runs.py`, `eval/seeds.py`, `scripts/local/*`, `docs/*`, `CHANGELOG.md` | `test_eval_core.py`, `test_gates.py::test_g4_…` |

## Places where Claude made a judgement call

These are for the reviewer to accept or push back on.

1. **Canonical slot channels.** Each mechanism slot names at least one channel that any bound component must write (for example `autobiography.record`, `planner.plan`). This lets G2/G3 target functions stably. It is also a constraint on future step contracts.
2. **Encoded prediction target.** The dense objective predicts the *encoded* sensory vector (ON/OFF DoG plus population codes) one frame ahead, rather than raw channels, following A1.1's "same encoders".
3. **Budget matching result.** At full size the matched GRU has hidden size 42 (see `ARCHITECTURE.md`). This is correct under A1.1 as written, but may not be the comparison either agent intended.
4. **SNN efference in open-loop streams.** The replayed efference copy is injected as current into the matching motor populations, with gain 1.5, so the SNN receives the same causal information as the GRU.
5. **Ridge scale floor.** Scales are floored at 1e-3 × the median positive scale; columns constant in training are dropped. This is a numerical guard, not a change of procedure, but it changes probe outputs and should be frozen with the E2 procedure.
6. **Engineering seed family.** `v0.38-ENGINEERING` reserves 900000–909999. Engineering sub-families may nest inside it. No other overlap is allowed.
7. **Synthetic streams** are a two-joint plant with babbling, not MuJoCo. They validate the harness and are never evidence.

## Internal review round (before hand-off)

A separate reviewer agent, which had not seen the code being written, audited the build against the contract. It confirmed 12 defects. All are fixed and each has a regression test:

1. Stand-in components could bind as scientific with hand-made LEGACY_VERIFIED or EXPERIMENTAL_PREREG provenance. Legacy provenance is now accepted only from components that `build_legacy_components` issued (a weak registry). Experimental provenance needs the class to be registered under a preregistration that covers the slot and is still APPROVED.
2. Gates could return PASS/FAIL directly without artifacts, under any approved prereg, on unchecked seeds. Gates now only *propose* a verdict. It becomes the status inside `RunStore.finalize` after artifacts are attached, and only if the preregistration registry approves the prereg for that item (`allowed_items`). All gates check seeds against the registry.
3. A spent range could be reused by giving it the same name. Ranges are now compared by range. `check_usable` and `spend` make a TEST family single-use, and historical ranges are never usable.
4. `ResultRecord.validate` trusted caller flags. It now looks the prereg up in the registry, does not count `spec.json` as a primary artifact, and `finalize` refuses to run twice.
5. A channel read returned the stored array, which could be made writable again. Reads now return views of a read-only base.
6. Intervention payloads were held by reference, stayed mutable after lock, and were missing from the restore identity. They are now copied read-only, specs and payloads are frozen at lock, and a payload digest is part of the identity.
7. The dense `train_step(batch)` accepted any batch. Explicit batches must now match the causal buffer exactly.
8. An explicit `hidden_size` skipped the ±10 % budget check. It is now checked whenever a target is set.
9. The legacy import wrote `__pycache__` under the root and executed in-package files that were never hashed. Bytecode writing is now off during the import, and every file in the transitive in-package import closure must be hashed in the entry.
10. A developer override could run without a log. A log path is now required.
11. The decision-rule hash covered only the wrapper function. It now includes the full source of every module the rule depends on.
12. The SNN `state_hash` omitted the noise RNG and the last spike vector. Both are now included.

Plausible concerns also addressed:
- the view closes over the runtime instead of storing it;
- components get a query-only coprocessor handle;
- coprocessor logging no longer fails on odd payloads;
- a failed restore rolls back;
- the runtime state hash now includes interventions and the coprocessor;
- G4 gives no vacuous verdict;
- the G2 freeze hashes metric source;
- the dense model masks gradients of lesioned weights before clipping;
- A1 compares only runs of equal duration;
- the indicator evidence chain is fixed in code, and cited evidence can be checked against files;
- `RunStore.verify` detects added or missing files;
- the Wilcoxon normal approximation has a tie correction;
- `unified_boot` handles two entries for one slot.

Left open, as spec questions:
- A1's z = 3.57 assumes 140 tests, but with rate and CV tests per population the family is 280 tests (z = 3.75);
- Python cannot seal the evaluator view or the override token against deliberate subversion. These guard against accidents.


## v0.38.1.dev0 second-author changes

This branch starts from `c011d40` and addresses ChatGPT's three confirmed follow-up defects before G0 composition work:

1. engineering sibling seed ranges no longer overlap merely because they share the ENGINEERING purpose; only containment by an explicit reserve (or a declared historical overlap) is tolerated;
2. G1 cannot propose a verdict unless every required mechanism has evaluator and threshold coverage;
3. G2 cannot propose a verdict over a sparse matrix: required cells must be present or explicitly justified `N/A`.

It also adds the read-only G0 evidence plan, named immutable legacy roots, DirectML lock capture, and report-only A1.1b/A1.2b support. See `G0_LEGACY_REQUALIFICATION.md`, `A1_AMENDMENT_SUPPORT.md`, and `VALIDATION_v0.38.1.dev0.md`.
