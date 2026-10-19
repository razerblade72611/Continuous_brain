# Validation — v0.38.1.dev0

## Canonical baseline evidence

Before this branch was created, David validated the `c011d40` v0.38.0 baseline in a clean WSL CPU environment (`torch 2.14.0+cpu`):

- 166/166 tests passed;
- E0 genesis/stability/telemetry/determinism/continuation/structure passed its infrastructure gate;
- protected child checkpoint SHA-256 matched;
- 14/14 legacy candidate modules were present on the canonical WSL tree.

Those baseline results are not rewritten as v0.38.1 results.

## v0.38.1 implementation validation in the build sandbox

Collected tests: **174** (eight new regressions over the 166-test baseline).

Changed-area groups executed successfully in this build environment:

- eval/G1/G2/G0/A1 group: 33 passed;
- legacy/protocol/script/preregistration group: 27 passed;
- DENSE substrate + unified runtime: 23 passed;
- SNN adapter/firewall: 10 passed.

Total directly reported changed-area successes: **93 tests**. These groups overlap the functional surfaces modified by v0.38.1 and include the new regressions.

Static checks:

- `python -m compileall` succeeds for modified packages/scripts;
- `git diff --check` succeeds;
- `git diff -- continuous_brain_snn` is empty: no E0 source file is modified by v0.38.1.

## Sandbox-only E0 determinism discrepancy

The build sandbox uses Python 3.13.5 and `torch 2.10.0+cpu`. In that environment, unchanged E0's `test_same_seed_is_bit_identical` fails because two separately constructed same-seed `SparseSlots` connectivity objects differ before dynamics begin. This is not observed on David's canonical WSL CPU validation environment, where the same unchanged E0 source passed the complete suite and the E0 determinism gate under `torch 2.14.0+cpu`.

Therefore v0.38.1 does **not** modify E0 to accommodate the build sandbox. Promotion requires rerunning all 174 tests and E0 gate on David's canonical environment. If the discrepancy appears there, it becomes a DAV-32 portability/reproducibility defect and must be investigated before further substrate work.

## Required canonical validation

```bash
python -m pytest -q
python scripts/e0_gate.py --device cpu --out runs/e0_gate_v0381
```

Expected policy: do not claim v0.38.1 engineering PASS until both complete successfully on the canonical machine.
