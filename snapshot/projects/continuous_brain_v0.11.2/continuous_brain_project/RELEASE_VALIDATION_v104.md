# Continuous Brain v0.10.4 Release Validation

## Release-specific tests

`tests/test_v104.py` verifies:

1. version is 0.10.4;
2. raw world-retention holdouts contain no parent targets and are marked training-prohibited;
3. sealed balanced consolidation trains only `world.*` and preserves every other tensor/buffer exactly;
4. the fresh-holdout evaluator operates without external-parent targets and verifies the seal.

Result in the build environment: **4 passed**.

## Developmental lineage regression

The v0.9.0-through-v0.10.4 lineage suite printed **39 passed** with no assertion failures. The outer harness subsequently reached its execution timeout because of the same lingering-process behavior observed in prior releases; the pytest summary had already completed.

## Modern/core regression

The complete modern/core suite excluding only the two unchanged long-running legacy v0.6 suites (`test_v061.py`, `test_v062.py`) printed **144 passed** with no assertion failures. As with the lineage invocation, the outer harness timed out after pytest had already printed its successful final summary.

## Capability evidence not claimed here

Release validation is software/wiring evidence. It does not establish that the user's persistent checkpoint has recovered multi-domain world retention. That requires the fixed v0.10.4 consolidation run followed by the newly created and pre-training-sealed ForagingWorld and DetourWorld holdouts.

The v0.10.3 100-episode blind P4b result is historical evidence and is not redefined by this release.
