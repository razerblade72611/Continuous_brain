# Continuous Brain v0.10.0 — Release Validation

Validation was performed in the artifact build container on 2026-09-07.

## Passed

- Core non-versioned regression group: **90 passed**.
- v0.7.0 / v0.7.1 + vision / visualizer group: **15 passed**.
- v0.9.0 / v0.9.2 / v0.9.3 / v0.9.4 / v0.9.5 / v0.9.6 / v0.10.0 parented-development group: **24 passed**.
- v0.6.1 legacy tests excluding the memory-heavy standard-profile case: **23 passed**.
- v0.6.2 legacy tests excluding the memory-heavy standard-profile case: **22 passed**.

Total completed assertions: **174 passed**.

## Build-container limitation

Two unchanged legacy standard-profile tests could not complete in this constrained build container:

- `tests/test_v061.py::test_upgrade_preserves_all_learned_state_and_original_bytes[True-standard]`
- `tests/test_v062.py::test_attachment_preserves_executive_and_archives_old_motor[standard]`

Running those two together was terminated by the container with exit status 137 (memory limit). They did not report assertion failures before termination. v0.10.0 does not modify the legacy v0.6 executive/motor implementation exercised by those tests. Users with sufficient RAM can run the full suite locally with `pytest -q`.

## v0.10-specific guarantees tested

- A simulated v0.9.6 infant checkpoint migrates by adding only missing `planning.*` tensors.
- Every pre-v0.10 tensor is loaded exactly during that migration.
- The planning cortex has a recurrent state separate from the lived nucleus and responds to hypothetical action identity.
- Planning-only optimization exposes only `planning.*` parameters.
- Planning training leaves every existing child tensor bit-for-bit unchanged.
- The Colab-readiness checker executes end-to-end on a synthetic passing fixture.

These are software/migration guarantees. They do **not** constitute P4b behavioral evidence; the persistent child must still run the diagnostic and untouched final paired detour evaluation in `RUN_v0.10.0.md`.
