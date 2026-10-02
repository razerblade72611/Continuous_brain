# i1_dev7_64 — BLOCKED (kept as evidence, not used)

Individuals 906680–906743 (cloud-dev-claude-eng), cloud CPU, ENGINEERING_ONLY.

The run's own `manifest_consistent_across_arms` check failed. LEGACY_REF and NORMAL ran under manifest
`d43aebe7…099ad`; every later arm ran under `230aa4d0…f9bb7`. The author changed the package version
strings (0.38.1.dev6 → dev7 in `continuous_brain_*/__init__.py` and `pyproject.toml`) at 20:41:26 UTC-local
container time, between the NORMAL arm (header 20:40:18) and FIXED_SCAFFOLD (header 20:41:28).

No behaviour changed (only `__version__` strings), but the check cannot know that and is right to block:
the arms of one run must be produced by one code state. The run was repeated unchanged on a fresh slice as
`i1_dev7_64b` (906744–906807). The values in this report were visible to the author and are disclosed, not used.
