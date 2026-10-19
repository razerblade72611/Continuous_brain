# Run v0.11.1 — DirectML compatibility smoke and Phase A

## 1. Verify backend

```bash
python -m continuous_brain.parented_cli device-info \
  --device dml \
  --compute-test \
  --matrix-size 512
```

Expected fields include:

- `resolved: privateuseone:0`
- `directml: true`
- `compute_test.passed: true`

## 2. Compatibility smoke

Never smoke-test on the immutable exact-expansion branch. Copy it first.
Run 10 routed updates using the same developmental datasets intended for Phase A.
The result must contain:

- `device_backend: privateuseone`
- `legacy_tensors_changed: 0`
- `route_steps_after: 10`

If DirectML raises an unsupported-operator error, stop and preserve the exact
exception. Do not fall back silently to CPU for the real training run.

## 3. Phase A

Only after the 10-update DirectML smoke passes should the 6000-update routed
whole-brain growth run use `--device dml`.
