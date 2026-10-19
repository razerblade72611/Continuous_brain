# Continuous Brain v0.11.3 — DirectML Transformer Compatibility

v0.11.3 is a backend-compatibility patch over v0.11.2. It changes no neural
architecture, parameter shapes, learned values, or checkpoint schema.

## Why

On torch-directml, PyTorch Transformer causal-mask detection can attempt to
create a `-inf` causal comparison mask using a bool/uint8 mask dtype, producing
`RuntimeError: value cannot be converted to type uint8_t without overflow`.
The fused Transformer encoder fast path is also unsupported by some DirectML
builds and may fall back to CPU.

## Repair

- DirectML language decoding uses an explicit floating additive causal mask.
- The mask uses a finite `-1e4` upper triangle and the activation dtype.
- `tgt_is_causal=True` bypasses PyTorch's automatic causal-mask detection.
- MHA/Transformer fast paths are temporarily disabled around DirectML
  Transformer calls, encouraging decomposition into ordinary operators.
- CPU/CUDA retain the historical implementation unchanged.

No checkpoint migration or retraining is required. The immutable v0.11.0 exact
expansion checkpoint remains the appropriate source for a fresh DirectML smoke.
