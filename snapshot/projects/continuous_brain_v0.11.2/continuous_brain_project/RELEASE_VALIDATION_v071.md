# Continuous Brain v0.7.1 — Release Validation

## New behavior

- seven-way learned dialogue-intent head
- unknown-question/OOD rejection
- persistent identity and goal grounding
- latest-real-experience temporal grounding
- v0.7.0 semantic bootstrap
- dialogue curriculum augmentation
- runtime-compatible temporal nursery memory tail
- local Qwen2-VL compact visual-teacher path and grounding guard
- Transformers 4.x semantic/vision dependency pin

## Validation performed in build environment

- `tests/test_v071.py`: 6 passed
- action/memory/model/replay/visualizer/vision + v0.7/v0.7.1 group: 22 passed
- spatial/sensor-recovery/migration group: 29 passed
- embodied: 24 passed
- individual: 29 passed
- v0.6.1 excluding one standard-profile parameter: 23 passed, 1 deselected
- v0.6.2 excluding one standard-profile parameter and then its SIGTERM test separately: 21 passed, 2 deselected; SIGTERM test 1 passed
- v0.7.1 synthetic semantic train smoke: PASS (10 updates, checkpoint written)
- v0.7.1 synthetic evaluation/bundle smoke: PASS
- Python compile checks for modified modules: PASS

The two standard-profile parameter cases are intentionally not used as release gates in the constrained build environment; the user's real standard-profile lineage remains preserved by copy-on-attach semantics.


Aggregate completed regression cases: **149 passed**. Two standard-profile parameter cases were not completed inside the build-time execution window.
