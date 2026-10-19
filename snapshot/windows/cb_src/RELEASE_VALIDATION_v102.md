# v0.10.2 Release Validation

Validation performed in the build container.

- v0.9.0 -> v0.10.2 developmental regression line: **31 passed**.
- Modern/core suite excluding the two known long-running legacy v0.6 suites: **136 passed**.
- The monolithic full-suite invocation and `test_v061.py` exceeded the container execution window; this is recorded as an environment timeout, not an assertion failure.
- New v0.10.2 CLI was smoke-tested end-to-end on a synthetic successful multi-phase nursery.
- Synthetic smoke is software validation only; it is not evidence that the user's persistent child passes P4b.
