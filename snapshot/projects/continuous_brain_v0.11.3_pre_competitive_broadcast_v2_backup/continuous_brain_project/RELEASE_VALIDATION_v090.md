# Release validation — v0.9.0

Validation performed during build:

- `tests/test_v090.py tests/test_v071.py tests/test_v070.py`: 16 passed.
- Core action/memory/model/spatial/vision/replay/sensor group: 32 passed.
- Embodied/individual/migration/visualizer group: 62 passed.
- Total explicitly completed in these final groups: 110 passed.
- A two-update end-to-end nursery train/evaluate smoke test completed and correctly produced an unqualified result for an essentially untrained child.
- `py_compile` completed for all new v0.9 modules.

A combined full-suite command exceeded the container transport window. A separate run of the historical `test_v061.py` also exceeded the execution window after producing progress dots; it did not return a failing assertion, but it is **not** counted as a pass. This release therefore does not claim a fresh final run of every expensive historical training test.

The new v0.9 lineage is intentionally fresh: old v0.7 child weights are not imported by the parented-development bootstrap. Old datasets may be converted into parent lessons for fast architectural testing.
