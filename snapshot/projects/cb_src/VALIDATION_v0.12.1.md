# Continuous Brain v0.12.1 validation

Build preparation date: 12 September 2026.

## Executed in the build environment

- The bundled child checkpoint SHA-256 is `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`.
- The checkpoint reports 14,000 updates, `balanced-v110` growth, and 384/768 vision-parent heads.
- 61 targeted v0.12/v0.11 compatibility tests passed across split runs. The long protected-growth resume test passed separately.
- 13 additional competitive/no-leak/v0.12.1 tests passed.
- The sliced unfused GRU matches native CPU `nn.GRUCell` forward values and gradients, including a five-step temporal hidden-state gradient.
- `CorticalNucleus.project_streams()` / `forward_from_streams()` reproduce the refactored nucleus forward exactly in the CPU test.
- The real 14k checkpoint strict-loads in the v0.12.1 source tree.
- CPU qualification passed: scatter-safe one-hot, a 20-step GRU backward chain, frozen real-child core backward, full competitive motor-only backward, and a complete one-update 0.3-corridor competitive training episode.
- The original competitive preservation checks pass: immediate exact no-op, disabled exact base recovery after trainable perturbation, exact save/restore continuation, frozen/eval base, parameter accounting, and non-zero independent selector gradients once residuals are moved off zero.

## Hardware qualification still required

The build environment has no DirectML/RX 6600 device. `scripts/qualify_directml_v121.sh` must pass on the target machine before any 25-update or 4,000-update DirectML competitive run. This is intentionally a hard gate; no CPU fallback occurs.
