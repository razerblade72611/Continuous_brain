# Continuous Brain v0.12.2 — DirectML graph hardening

v0.12.2 is a narrow hardening release on top of v0.12.1. It does not change the project goal, the 14,000-update child checkpoint, the competitive-workspace hypothesis, or the v0.12 lifelong runtime.

## Why this release exists

On the target RX 6600 / WSL DirectML stack, v0.12.1 qualified scatter-safe one-hot, recurrent backward chains of 1/2/5/10/20 steps, and the real frozen InfantBrain recurrent core. The first failing gate was the integrated competitive one-step backward, which ended with DXGI device removal. That evidence rules out the GRU cell itself as the primary remaining blocker.

## DirectML identity materialization

The real competitive step reuses tensors at several branch fan-out seams: recurrent hidden state enters both the frozen GRU and selector query; frozen projected streams enter both the recurrent core and selector; and a shared selected broadcast enters recurrent, motor and world residual consumers.

On DirectML only, v0.12.2 places `Tensor.clone()` identity boundaries at those seams. `clone()` is differentiable: values are unchanged and its backward is the identity. It therefore does **not** detach hidden state, truncate BPTT, or change the objective. CPU/CUDA paths return the original tensor and remain unchanged.

## Minimal training forward

The competitive experiment never trains the care/value head and does not use router or metacognition outputs in its loss. v0.12.2 adds `compute_aux=False` for competitive training and qualification so those unrelated frozen branches are not executed during the DirectML training step. The historical full-output behavior remains the default (`compute_aux=True`) for compatibility.

The train-relevant outputs — hidden state, motor logits, and world-model workspace — were verified bit-for-bit identical on CPU between full and minimal paths.

## What remains unchanged

- the bundled child checkpoint bytes and SHA-256;
- full recurrent temporal credit assignment;
- zero-initialized safe growth;
- frozen/eval base and BatchNorm preservation;
- shared / independent / uniform experimental conditions;
- teacher-forced delayed-cue task and world prediction loss;
- v0.12 lifelong individual state format and runtime.

## DirectML qualification

`scripts/qualify_directml_v122.sh` is the hardware gate. Every stage runs in a fresh Python process because a DirectML device-removal error poisons the current process. The release never silently falls back to CPU.
