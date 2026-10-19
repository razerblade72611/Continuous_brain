# Release Validation — v0.10.1

Build validation performed in the release container.

## v0.9.0 → v0.10.1 parented-development regression

```text
27 passed
```

Covered:

- v0.9.0
- v0.9.2
- v0.9.3
- v0.9.4
- v0.9.5
- v0.9.6
- v0.10.0
- v0.10.1

## Broad current-suite regression

All test files except the two unchanged legacy v0.6 standard-profile suites were run together:

```text
132 passed
```

The two omitted v0.6 suites are the same high-memory legacy tests documented in `RELEASE_VALIDATION_v100.md`; v0.10.1 does not modify their motor/executive implementation.

## New v0.10.1 checks

- Q-only trainability mask.
- Bellman/CQL loss finite and gradients confined to Q.
- End-to-end tiny nursery training changes `planning.q_head.*` and no other model tensor.
- Metadata records `conservative-bellman-q-v1`.

Synthetic/unit tests are software validation only. They are not P4b behavioral evidence.
