# Cross-environment result: David's WSL vs cloud (ENGINEERING_ONLY)

| Item | Detail |
|---|---|
| Run | David's WSL (Ubuntu-24.04, Ryzen 5600X, venv `cb_v0381_g0`), 2026-10-01 |
| Code | dev6 zip SHA-256 `ddb32a4384c9ed860d4275befdacbb46feb9d963f47c366486b9a3e59c0c30a0` (commit e820a61) |
| Seeds | `cloud-dev-claude-eng` slices, 8 seeds per mechanism |

## On WSL itself

- **Tests:** 205/205 pass.
- **Protected checkpoint:** `086932388a9a…d28cfd577f` before and after.
- **E1 PASS for all six mechanisms:**
  - G0 route, full composition and used-hosts composition are byte-identical;
  - return codes are equal;
  - faithfulness checks pass;
  - the legacy tree is unchanged in every run.

## WSL vs cloud

`compare_cross_env.py` reported a FAIL, meaning `result.json` was byte-identical for only v0.36.1. A field-by-field diff of every WSL `result.json` against the cloud G0-route `result.json` for the same seeds shows:

- **Frozen checks:** all 85 (14 + 17 + 10 + 16 + 17 + 11) are **identical**.
- **Pass fields:** all 6 `test_admission_passed` values are **identical**.
- **Values that differ:** 11 float summary values, at relative size 1e-8 to 2.6e-7, which is one to two float32 rounding steps. Examples: `reliability_mae`, `cost_mae`, `self_mae`, `world_mae`, `mean_late_to_bootstrap_discrepancy_ratio`.
- **Which environment is canonical:** WSL. WSL reproduces v0.31's historical value bit-exactly: `mean_model_to_surprise_self_error = 0.25928941718302667`, the same as the original v0.31.1 TEST and the G0 PASS run. The cloud gives `0.25928941706661135`. The difference comes from the CPU kernels: the cloud container runs a `+cu130` torch build on a different CPU, against a `+cpu` build on the 5600X.

## Consequences

1. Cloud development results transfer to David's machine at the level of every frozen check and every pass field, but not byte for byte.
2. Any gate that relies on byte-identity must be run on the WSL machine that holds the scientific record. That includes E1 as G1's engineering gate, and the E2 coupling comparison.
3. This is direct evidence for review amendment S9: the composition manifest must carry an environment fingerprint (torch build, CPU, thread count), and preflight and science must run under the same one.
