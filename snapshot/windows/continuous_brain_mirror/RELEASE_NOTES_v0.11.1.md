# Continuous Brain v0.11.1 — DirectML backend compatibility

v0.11.1 is a compatibility patch over v0.11.0. It does not change the neural
architecture, growth profile, curriculum, qualification gates, or validated
v0.10.4 → v0.11.0 function-preserving expansion.

## Added

- First-class `--device dml` and `--device dml:N` device resolution.
- Accepted aliases: `directml`, `privateuseone`, and `privateuseone:N`.
- `device-info --device dml --compute-test` hardware/compute diagnostic.
- CPU-first checkpoint deserialization for DirectML/custom PrivateUse1 backends,
  followed by normal model transfer to the resolved device.
- DirectML-safe non-foreach AdamW and gradient clipping in routed whole-brain
  growth training.
- Training output records the resolved device/backend.

## Preserved

- No architecture changes.
- No parameter-count changes.
- No checkpoint migration is required.
- The v0.11.0 exact-expansion checkpoint is directly reusable.
- Legacy weights remain frozen in Phase A exactly as in v0.11.0.

DirectML remains an optional dependency and is imported only when explicitly
requested. CPU/CUDA installations do not require `torch-directml`.
