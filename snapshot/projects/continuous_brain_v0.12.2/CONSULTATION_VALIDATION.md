# Consolidation validation

Build `v120-consultation-consolidated-r1`, 12 September 2026.

This report concerns this consolidation. The preserved `VALIDATION_v0.12.0.md`
contains the original release's historical measurements.

## What moved forward

The consultation produced a reproducible reference result and useful evaluation
requirements. The main agent already contained learned options and planning;
the mock controllers did not add those abilities to it. This build turns the
useful evaluation proposal into integrated code using the actual project APIs.
It adds a direct-options comparator, strict existing-protocol validation,
explicit seed roles, complete decision traces, and separate benchmark/statistical
reporting. It does not claim increased intelligence or subjective experience.

## Preservation

All 268 files from `continuous_brain_v0.12.0_DIRECTML.zip` are retained. Existing
model, environment, learning, device, checkpoint, and storage implementations
are unchanged. The only original source-code edit is the additional evaluator
command in `continuous_brain/parented_cli.py`. The README points to the new guide,
and inventories are regenerated for the complete delivery.

The full v0.11.4 implementation is already part of that source package. The
667,130-parameter local starter remains byte-identical. Its file SHA-256 is:

```text
d8c8ca43c9018c9c2666324abb42b4f3279ac4b44d0860320b2176117785ec01
```

This file hash differs from the model's tensor fingerprint because it hashes
the serialized file, not just its tensors. No previously saved individual was
converted or overwritten. The test individual's state is not bundled.

## Real PyTorch execution

Environment: Python 3.12.14, PyTorch 2.4.1+cpu, NumPy 1.26.4, pytest 8.3.5.
Dependencies were installed into an isolated validation environment.

**44 targeted tests passed, with zero failures/errors/skips, in 44.35 seconds.**
Three warnings concerned the legacy Transformer's nested-tensor configuration.

The suites cover the new evidence contract, original option planner,
v0.11.4 evidence/retention safeguards, and v0.12 state and learning behavior:

```bash
python -m pytest -q tests/test_consultation_integration.py tests/test_v103.py \
  tests/test_v114_integrity.py tests/test_v120_local.py
```

These include real candidate training, unchanged-reference checks, continuation
through checkpoint restoration, source checkpoint protection, failed-evidence
rejection before model loading, seed-range protection, and the distinction
between a positive point estimate and an inconclusive paired interval.

The real-model smoke completed in approximately 3.2 seconds in the final run.
A randomly initialized proof-profile `InfantBrain` with 10,462,416 parameters
executed all three controller paths on both actual randomized environments.
The existing fixed-layout evaluator also executed on this fixture. Its
unqualified report was rejected by the new Python API and by the actual CLI,
without changing the checkpoint file or writing a qualification result.

No PyTorch substitute or no-op optimizer was used. Some unit tests use
synthetic reports or deterministic controller fixtures to test specific logic;
those are not model capability results. The smoke uses real PyTorch and real
project simulators, but untrained weights, and establishes execution only.

The local runtime's CPU preflight passed its matrix, full model
forward/backward/optimizer, demonstration, and lived-replay route checks.

## Included starter: actual learning session

A separate test state ran for 256 lived steps from the supplied trained starter.
It collected three food and five water resources. At step 256 it completed one
candidate learning block of 16 real optimizer updates. The candidate was
admitted by the existing acquisition/retention gates. This is one small
local session using the unchanged learner, not evidence of an intelligence
improvement caused by this consolidation. Its advisory forecaster had only
one example and higher error than the zero-change baseline (0.1554 vs 0.00594).

The full measured summary is `consultation/validation/local_session_256.json`.
After a clean ZIP extraction, the actual shell launcher verified its source
inventory, passed CPU preflight, restored the same identity and learning state,
ran 16 additional steps (256 to 272), and acknowledged the save. The receipt
and full launcher output are in `consultation/validation/extracted_launcher.json`
and `extracted_launcher_stdout.log`. This verifies continuation across processes;
it is not a claim of bit-identical trajectories against an uninterrupted run.

## Reference experiment reproduced

These are outcomes in the separate handwritten mock-grid experiment, not in
the trained Continuous Brain agent:

| Seeds | Reactive resources/episode | Options resources/episode | Paired advantage | 95% interval |
|---|---:|---:|---:|---|
| 3000–3199 | 43.315 | 44.740 | +1.425 | [1.155, 1.690] |
| 5000–5199 | 43.645 | 45.105 | +1.460 | [1.210, 1.710] |

Both directed controllers completed all 200 steps in every episode. The
collection/respawn test now explicitly checks duration reset and goal reselection.
These reproductions use previously reported seed sets; they are not additional
fresh confirmations beyond the original result.

## Remaining limits

- No original trained v0.11 child checkpoint was present in the source releases
  or found by the targeted saved-file search. Therefore no fixed-layout or
  randomized qualification result for that child is claimed. Supply your own
  `infant_training.pt` or frozen endpoint to the driver.
- The end-to-end successful-qualification path has not been exercised using a
  qualified trained child. Its components, real episode paths, report validation,
  and rejection path were tested as described above.
- DirectML/RX 6600 and native Windows launchers were not executable on this CPU
  Linux host. Their existing implementations are preserved; the on-device
  preflight remains necessary. WSL/Linux CPU launcher execution is checked.
- Confidence intervals describe paired episodes of the same authored generator.
  They do not establish transfer to arbitrary environments. Secondary metrics
  and the direct-options comparison are reported separately.
- No nursery data or legacy training weights are manufactured or substituted.
  No new learning is scheduled by the evaluator, and no metacognition or
  consciousness claim follows automatically from a test result.

Machine-readable receipts, raw reference output, session stdout, and preservation hashes are in
`consultation/validation/`. The package inventories cover every included file.
