# Validation report — v0.38.0.dev0

Validation performed during artifact construction on 2026-09-27.

## Unit suite

- `pytest -q`: **13 passed**.
- Tests cover the dense GRU comparator, split learning API, SNN gradient firewall, engineering-only battery semantics, seed overlap protection, content-addressed provenance, hash-pinned legacy binding, explicit unbound runtime state, integration prediction-matrix completeness, 14-indicator no-admission registry, power-aware parity helper, and shell-only checkpoint integrity.

## Read-only legacy audit

Against the preserved mirror used to generate `registry/legacy_manifest.json`:

- 9 component families reached `ENTRYPOINT_BOUND` after exact file/evaluator SHA-256 verification.
- `recursive_autobiographical_curriculum` is intentionally `UNBOUND` in this manifest because its module is not present in that preserved mirror. The available v0.36.0/v0.36.1 overlay archive hashes are recorded in the manifest note, but the build does not silently merge them into the canonical mirror.
- No component is marked `SCIENTIFICALLY_INTEGRATED`.

## Engineering battery

Dense GRU smoke execution produced:

- B1: `REQUIRES_E2_BODY`
- B2–B6: `UNREGISTERED_METRIC`

No battery item is allowed to emit a scientific PASS/FAIL in this release.

## Packaging

- Source package is installable with setuptools.
- A pure-Python wheel is built under `dist/`.
- The wheel declares `torch>=2.5` and `numpy>=1.26`; a clean environment must provide/install those dependencies.
- Registry JSON files required by the installed CLI are included as package data.

## Known intentional limitations

- E0 source is not reconstructed from prose; only an adapter is supplied for the real external E0 engine.
- E1 and E2 are not implemented in this release.
- The composition shell does not serialize historical mechanism state unless a future audited adapter provides that contract.
- G1–G5 are infrastructure/planning hooks, not passed gates.
