# v0.38.1.dev4 validation note

Dev4 freezes the six successful `G0-LEGACY-REQUAL-v1` results into `registry/g0_requalifications.json` and records their scientific DEV and engineering runtime-preflight ranges as spent in the static registry. The source evidence bundle was `cb_g0_six_scientific_results.zip`, SHA-256 `a191bf3a06160a867c177f6377470879fe6b233a089f768b093250c7aaa277a5`; its 202 manifest entries were independently rehashed with zero mismatches. Every raw evaluator `result.json` hash matches the G0 wrapper record.

The two dev3 unit-test failures seen after running the engineering preflights were stale state assumptions: the tests expected the preflight ranges to be unspent and treated seed 900001 as generic reserve space. Dev4 updates the canonical static history so completed preflight ranges are permanently spent, and uses unallocated engineering-reserve seeds for reserve-behavior tests.

`G1-COMPOSITION-RETENTION-v1` is added only as a PROPOSED preregistration. No G1 range is registered or runnable here.
