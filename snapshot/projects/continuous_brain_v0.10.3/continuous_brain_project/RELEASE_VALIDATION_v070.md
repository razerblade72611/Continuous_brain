# Continuous Brain v0.7.0 — Release Validation

Date: 2026-09-05
Release theme: **First Conversation**

## Scope validated

- v0.6.2 source used as the release base.
- New semantic/communication cortex compiles and executes.
- Byte-level language codec and native answer decoder.
- Fixed 48-value communication context contract.
- Qwen structured-label validation and bounded care targets.
- Frozen-DINO feature-cache training interface.
- Resumable semantic optimizer/checkpoint path.
- Teacher-free held-out qualification path.
- Qualified-bundle digest verification.
- Non-overwriting attachment to the existing individual.
- Existing identity/executive/memory/motor state preservation logic.
- Legacy v0.6 compatibility upgrade now preserves v0.7 semantic attachments.

## Automated validation performed in release container

```text
PYTHONPATH=. pytest -q tests/test_v070.py
5 passed

PYTHONPATH=. pytest -q tests/test_v070.py tests/test_individual.py tests/test_memory.py
36 passed

PYTHONPATH=. pytest -q -k 'not standard'
144 passed, 2 deselected

python -m compileall -q continuous_brain scripts
PASS
```

The two deselected cases are the intentionally expensive full `standard`-profile
serialization/continuation tests.  One was separately started but exceeded the
release container's execution window before pytest returned.  The corresponding
small-profile preservation paths and the broad non-standard regression suite pass.
For the operator machine, run the complete `python -m pytest -q` before attaching a
trained semantic cortex to the persistent individual.

## End-to-end semantic smoke validation

A synthetic 30-frame dataset with cached stand-in DINO vectors was passed through
`continuous_brain.semantic_cli train` and `evaluate`.  The student trained,
checkpointed/resumed, generated teacher-free bytes, wrote a qualification report,
and produced a digest-verified attachable semantic snapshot.  Qualification
thresholds were deliberately relaxed for this plumbing smoke test; this is **not**
evidence that the untrained release already has communication competence.

Observed default student size:

```text
971,390 trainable semantic-cortex parameters
```

## Important boundary

The release archive contains source code, not trained semantic weights.  DINOv3 and
Qwen remain frozen foundation models.  Training creates a new semantic cortex in a
separate lab; only a teacher-free qualified snapshot can be attached to a
continuation of the existing individual.
