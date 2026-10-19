# G1-COMPOSITION-RETENTION-v2 — PROPOSED

This is a **proposal**, not approval and not an executable scientific gate. It supersedes the unapproved v1 proposal before any G1 family was registered or opened.

G0 established isolated fresh-DEV requalification for six historical mechanisms. G1 asks whether those mechanisms retain their full frozen causal/admission behavior when they coexist in **one composed individual**, rather than six private legacy stacks behind a common shell.

The machine-readable preregistration is `registry/g1_retention_v2.json`.

## Added safeguards versus v1

- One canonical runtime-owned state store is authoritative for overlapping self, autobiography, world, goal and planning domains. Adapter-local authoritative shadow copies are forbidden.
- Every focal condition uses one `UnifiedRuntime` instance with all six adapters in the same scheduler graph. Non-focal mechanisms cannot be disabled, bypassed, detached, mocked, or moved to private runtimes.
- The runtime must emit a composition attestation: runtime id, scheduler graph hash, binding/adapter hashes, state-codec hashes, shared-state identity, base-checkpoint identity, and scheduler eligibility/invocation counts for every mechanism.
- Six disjoint ENGINEERING_ONLY full-composition preflights must complete under the exact proposed composition manifest before any scientific G1 family can open.
- Engineering preflight may validate execution, routing, complete check-key presence and provenance, but must not be used to tune against scientific pass/fail values.
- The successful composition manifest is frozen by SHA-256. A scientific run under a different manifest is an infrastructure failure, not a scientific result.

## Scientific decision rule

A mechanism PASS requires its entire frozen check vector and top-level admission field to remain true while all six mechanisms are genuinely co-resident in the one composed runtime and the frozen composition attestation remains valid. A completed false frozen check is a scientific FAIL. Missing/broken composition, shadow authoritative state, disabled non-focal modules, scheduler-attestation failure, provenance changes, missing checks/results, or process failures are BLOCKED/INFRASTRUCTURE_FAILURE.

No G1 scientific or engineering-preflight family is registered or runnable from this artifact. David approval and a later fail-closed execution artifact are required.
