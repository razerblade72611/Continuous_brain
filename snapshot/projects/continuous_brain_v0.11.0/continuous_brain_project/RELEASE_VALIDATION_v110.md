# v0.11.0 release validation

Software/integrity validation performed for the packaged release:

- v0.11.0-specific tests: **5 passed**.
- developmental regression line v0.9.0 through v0.11.0: **44 passed**.
- remaining modern/core suite excluding the two historically expensive v0.6.1/v0.6.2 suites: **105 passed**.
- combined modern/core + developmental assertions: **149 passed**.
- explicit expansion smoke:
  - source trainable parameters: 9,170,552 in the synthetic proof checkpoint with one tiny parent head;
  - expanded trainable parameters: 53,537,144;
  - new growth parameters: 44,366,592;
  - legacy tensor mismatches: 0;
  - audited maximum absolute output difference: 0.0;
  - exact function preservation: true.
- routed-training smoke:
  - perception, language, embodiment, world, planning and option routes executed;
  - second invocation resumed from route step 6 and continued through route step 10;
  - legacy tensors/buffers changed: 0.
- generalized curriculum collection smoke completed for both randomized-detour and two-barrier environments.
- CLI help/import smoke completed for `expand-capacity`, `train-whole-brain`, `collect-generalized-curriculum`, and `evaluate-generalized-planning`.

The single all-modern pytest invocation exceeded the build container's outer execution window before returning a summary, so it was split into the 44-assertion developmental line and the 105-assertion remaining modern/core batch above. No assertion failure was observed in either completed batch. The legacy v0.6.1/v0.6.2 high-memory suites are not re-characterized by this release.

These are software/integrity validations. They are **not** capability qualification for a trained v0.11.0 child.
