# v0.38.1.dev1 validation status

This development artifact starts from the WSL-validated v0.38.1.dev0 commit `337f2a2d516f610feee678d7f94edb26f798f1ce`.

Canonical WSL evidence for the parent commit:
- `174 passed in 34.60s`
- pytest log SHA-256 `eb36ce2bf8532e9f86cafd31cee31440f1505decdbf54f2677868c8bd0f3a812`
- E0 gate `INFRASTRUCTURE PASS`
- E0 report SHA-256 `1152ec64bc9f30e69552f099fad264fcb4c9ee41cb668decc7e52585160b7797`

Changes in dev1 are limited to the G0 runner/preregistration/seed-plan/evidence-classification layer plus version metadata. `continuous_brain_snn/` remains unchanged.

Build-sandbox validation:
- 178 tests collect after the four new G0-runner tests.
- 31 G0/gate/prereg/protocol/script tests pass.
- The build sandbox cannot complete the full E0-heavy suite within its CPU timeout; the full 178-test run must be repeated in David's canonical WSL CPU environment before dev1 receives an engineering PASS.

No fresh G0 DEV range is registered or opened in dev1. `G0-LEGACY-REQUAL-v1` remains `PROPOSED` and the runner refuses scientific execution until David explicitly approves it and a follow-up artifact registers the frozen fresh families.
