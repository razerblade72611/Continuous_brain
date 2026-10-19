# Continuous Brain v0.12.2 validation

Build preparation date: 12 September 2026.

## Evidence from the target RX 6600 before v0.12.2

v0.12.1 DirectML qualification passed:

- scatter-free one-hot and ordinary matrix backward;
- GRU recurrent backward at 1, 2, 5, 10 and 20 steps;
- the real 14,000-update InfantBrain recurrent core backward.

It failed only at the integrated competitive one-step backward with DirectML/D3D device removal. This is why v0.12.2 targets graph fan-out/resource materialization rather than recurrence or BPTT.

## Executed in the build environment

- Bundled checkpoint SHA-256 remains `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`.
- The checkpoint strict-loads and retains 14,000 updates, `balanced-v110` growth and 384/768 vision parent heads.
- Cued-detour cue-blindness / no-timing-leak suite passes.
- Competitive preservation suite passes: immediate exact no-op, exact disabled fallback after trainable perturbation, exact save/restore continuation, frozen/eval base, parameter accounting and independent-selector gradient checks.
- v0.12.2 train-relevant minimal forward is bit-for-bit identical to the historical full forward on CPU for hidden, motor logits and world workspace.
- CPU qualification passes basic ops, 20-step GRU backward, real child-core backward, competitive one-step backward and a complete one-episode optimizer update.
- 35/35 v0.11.2/v0.11.3/v0.12.0/v0.12.1/v0.12.2 targeted local/backend tests pass.

A broader historical pytest run was attempted but the test harness generated ~29 GB of temporary protected-growth state and exhausted the build container filesystem. Those temp files were removed. No broad-suite pass is claimed from that interrupted run.

## Hardware qualification still required

The build environment does not have the user's DirectML/RX 6600 device. `scripts/qualify_directml_v122.sh` must pass on the target machine. A failure remains a qualification failure, not permission to silently detach hidden state or substitute CPU for the scientific run.
