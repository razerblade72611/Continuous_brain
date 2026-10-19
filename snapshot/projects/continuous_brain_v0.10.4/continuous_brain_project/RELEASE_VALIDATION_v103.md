# Continuous Brain v0.10.3 Release Validation

Validation was performed in the build container.

## Completed checks

### Developmental lineage regression

Command:

```bash
pytest -q tests/test_v090.py tests/test_v092.py tests/test_v093.py tests/test_v094.py \
  tests/test_v095.py tests/test_v096.py tests/test_v100.py tests/test_v101.py \
  tests/test_v102.py tests/test_v103.py
```

Result: **35 passed**.

### Modern/core regression suite

The same modern/core suite used for v0.10.2, excluding the two long-running legacy `test_v061.py` and `test_v062.py` suites, completed its assertions with:

**140 passed**.

The outer build harness did not terminate cleanly after pytest printed its final summary, consistent with the lingering-thread / long-running legacy behavior previously seen in this environment. This is recorded as a harness termination limitation, not an assertion failure.

### v0.10.3-specific invariants

The four new tests verify:

- release version is 0.10.3;
- an option macro rollout uses only the active option's learned primitive policy;
- an active option owns primitive control between learned option boundaries;
- high-level imagination expands learned option candidates rather than unrestricted primitive actions.

### End-to-end smoke

`evaluate_hierarchical_options` was executed against a proof-profile checkpoint carrying valid v0.10.2 option metadata. The evaluator completed an episode, returned the v0.10.3 result schema, and wrote JSON output successfully.

## Weight compatibility

v0.10.3 adds no model tensors and requires no checkpoint migration or retraining. A v0.10.2 checkpoint is consumed unchanged.

## Capability evidence not claimed by release validation

Unit tests and smoke tests are wiring/correctness evidence only. They are not P4b capability proof. P4b remains determined by the fixed behavioral gate on development and then untouched blind seeds.
