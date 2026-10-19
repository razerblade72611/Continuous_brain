# Continuous Brain v0.11.3 — DirectML smoke protocol

v0.11.3 is a compatibility-only patch. Do not reuse a branch that experienced a
training exception. Always start the DirectML smoke from the immutable v0.11.0
exact-expansion checkpoint.

## Source checkpoint

Expected SHA-256:

`086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`

## Smoke

Use the same ten routed updates used for v0.11.1/v0.11.2. Success requires the
process to complete, every route to be exercised, and
`legacy_tensors_changed == 0`. A new unsupported DirectML operator is a backend
compatibility finding, not permission to continue to the 6000-update run.

Do not run Phase A until this smoke passes on the actual RX 6600.
