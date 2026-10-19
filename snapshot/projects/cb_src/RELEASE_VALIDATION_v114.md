# v0.11.4 validation — protected-integration-r2

## Validation performed for this package

Environment: Python 3.12, PyTorch 2.14.0+cpu, pytest 9.1.1, CPU threads limited
to 2 for the test run. No CUDA/DirectML device or user checkpoint was available
in the release-construction runtime.

The final targeted run completed **74 passed, 38 warnings in 80.19 seconds**:

```bash
python -m pytest \
  tests/test_v090.py tests/test_v092.py tests/test_v093.py tests/test_v094.py \
  tests/test_v095.py tests/test_v096.py tests/test_v100.py tests/test_v101.py \
  tests/test_v102.py tests/test_v103.py tests/test_v104.py tests/test_v110.py \
  tests/test_v111.py tests/test_v112.py tests/test_v113.py tests/test_v114.py \
  tests/test_v114_integrity.py -q --disable-warnings
```

Checks include:

- Exact growth/protected migration, identical source tensors and learned parameter counts.
- Exact closed-gate forward and explicit shadow gradient flow.
- Full ten-route training smoke using a synthetic nursery with successful sequences.
- Five updates + saved resume to ten versus uninterrupted ten: all model state
  tensors were bit-identical in the tested environment.
- Frozen legacy tensors unchanged during training; normal planner/selector
  inference remained exact after the closed-gate shadow smoke.
- Periodic durable checkpoint copies matched source hashes; repeated target
  invocation left the checkpoint bytes unchanged.
- Nonempty migration destination, absent prior P4b evidence, over-ceiling updates,
  incomplete previous stage, missing protected state keys, and forbidden training
  inputs were rejected.
- Strict Q-order loss, FP32 gate tolerance, and outer JSON extraction.
- Readiness rejection of stale checkpoint evidence and missing development metadata.
- Colab result reuse only for matching runtime/checkpoint; historical endpoint
  evaluation did not mutate the active checkpoint.
- Archive path traversal rejection.
- Existing v0.9/v0.10 planning/world and v0.11 device compatibility tests.

All project Python modules/scripts and notebook code cells parsed successfully.
CLI help/import checks passed for `protect-growth`, `train-protected-growth`,
and `protected-growth-status`.

The source ZIP is produced with a per-file SHA-256 manifest and verified against
that manifest after packaging. A separate SHA-256 file identifies the ZIP.

## Limits

The notebook was not executed in a live Colab/Google Drive session here. No
5,000-update protected experiment or 100-episode user capability gate was run.
The actual source checkpoint and nursery assets must be provided from the user's
existing verified v0.11.0 bundle.

The synthetic test's small update counts and fabricated evaluator-shaped fixtures
are software tests, not capability evidence; none are supplied as qualification
results. P4b recovery and generalized planning remain experimental outcomes.

The historical high-memory v0.6 suites and unrelated pre-v0.9 components were
not rerun. No inference is made about untested accelerator performance or exact
resume across hardware/PyTorch versions.
