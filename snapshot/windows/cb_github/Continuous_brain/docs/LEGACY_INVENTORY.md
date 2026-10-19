# Legacy inventory: Windows source mirror (DAV-74 starting point)

Produced by the first revision of `continuous_brain_unified/legacy/inventory.py` (SHA-256 `06f7497ccfc92fcc8fa3d84667fc9f127591d4d504e08348eba3382140193f2c`, recorded as `tool_sha256` in the output). The committed revision adds only path sanitizing, which refuses spec entries that point outside the root, so a re-run gives the same candidate data. It was run read-only on 2026-09-27 against the Windows source mirror (`continuous_brain_mirror`, root label `windows-source-mirror`) using `legacy/candidate_spec.json`. Output: `legacy/candidates.json`, SHA-256 `75df064c4afffaa0d4e60350acb34d09259b45ed560ad62cff7957a3b438ef96`. The tree has 1,121 files: 361 `.py`, 200 `.md`, 162 `.json`, 49 `.zip`, 194 `.log`, and **no `.pt`**.

**Every entry is UNVERIFIED.** These are hashes of a mirror, not the canonical WSL tree. Re-run `bash scripts/local/inventory_legacy.sh` in WSL and compare.

## Primary results found

| Candidate | Result file | SHA-256 (first 16) | Recorded flag |
| --- | --- | --- | --- |
| v0.23 two-item episodic memory | `state/multi_item_episodic_memory_test_v1/result.json` | `d7398710d8d69a32` | `test_admission_passed: true` |
| v0.24 episodic→goal→skill | `state/episodic_goal_skill_integration_v1/result.json` | `fdcbeaab45f59f68` | `integration_gate_passed: true` |
| v0.21–22 goal options (3 parts) | `state/{memory_guided_goal_option,additive_goal_options,recurrent_goal_option_state}_v1/result.json` | `154d67e9…`, `26b4f24b…`, `e2b946cb…` | **dev false, dev false, test false** |
| v0.25.2 consequence replanning | `state/consequence_replanning_test_v1/result.json` | `320e7bae0361bb4f` | `test_admission_passed: true` |
| v0.26 discovery embodiment | `state/thor_discovery_embodiment_admission_v1/result.json` | `2af28fd67360185c` | `admission_passed: true` |
| v0.28 persistent self | `state/persistent_self_autobiographical_test_v1/result.json` | `0e58202326bada3b` | `test_admission_passed: true` |
| v0.31 recurrent self-trajectory | `state/recurrent_self_trajectory_test_v1/result.json` | `b1b79db3a1354008` | `test_admission_passed: true` |
| v0.32.2 autobiographical discrepancy | `state/autobiographical_self_discrepancy_neutral_test_v1/result.json` | `dfc01c20ff7332d6` | `test_admission_passed: true` |
| **v0.34.1 event-consistent self-development** | `state/event_consistent_self_development_test_v1/result.json` | `5ca0aa737f5dec26` | `test_admission_passed: true`, `split.test = [80000, 80127]` |
| v0.35.0 endogenous goal genesis | `state/endogenous_goal_genesis_v1/result.json` | `8ba6357e2ecbd67a` | `development_gate_passed: false` (the preserved v0.35.0 DEV FAIL) |
| v0.29 self-broadcast (formal fail) | `state/self_broadcast_centrality_test_v1/result.json` | `1f394195c26bdc81` | `test_admission_passed: false` |

Full hashes, bundle member lists, module APIs (classes, methods, save/load functions, intra-package imports), and every hash the results themselves recorded are in `legacy/candidates.json`.

## What this means for DAV-74 / DAV-75 / DAV-77

- **DAV-77:** the v0.34.1 primary artifact is likely `state/event_consistent_self_development_test_v1/result.json` + `result_bundle.zip` (`a944a4acc40fc3bdf81ba154556709dee9d17d5c364bf089a25febbad98e70dd`). Confirm on WSL by hash.
- **DAV-75:** checkpoints must come from WSL. The results pin them. For example, v0.24 records `goal_options.pt bcb729e2…`, `goal_decoder.pt d46dfb42…`, `episodic_reader.pt db86e989…` and `recurrent_goal_option_state.pt 2e006c1f…`. Verification should compare WSL files to these recorded hashes.
- **Not in the mirror:** v0.35.2/v0.35.3, v0.36.x (recursive autobiographical curriculum), v0.37.x, and the protected base checkpoint `state/child_v113_dml_smoke/infant_training.pt`.
- **Binding needs step contracts.** A VERIFIED entry binds only with an audited step contract (`register_step_contract`) that says which runtime channels the legacy object reads and writes each tick. None is written yet; writing them is part of the audit, one mechanism at a time.

## Promotion procedure (proposed)

1. In WSL, run `bash scripts/local/inventory_legacy.sh` and `bash scripts/local/verify_protected_checkpoint.sh`.
2. For one mechanism, confirm module, evaluator, primary result and checkpoints against the recorded hashes. Record the audit as a DAV-74 comment with paths and SHA-256.
3. Add the entry to `legacy/manifest.json` with `status: VERIFIED`, `audit_ref` = that comment, and relative paths only.
4. Write its step contract and a G1 evaluator wrapper (re-hosting the original frozen evaluator). Bind it, and run `scripts/unified_boot.py --legacy-root …`.
