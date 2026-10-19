# Continuous Brain v0.11.2 — DirectML unfused recurrent compatibility

v0.11.2 is a checkpoint-compatible backend repair over v0.11.1. It changes no
neural dimensions, learned parameters, growth profile, curriculum, or training
gates.

## Why

On torch-directml/PyTorch 2.4.1, `nn.GRUCell` can dispatch
`aten::_thnn_fused_gru_cell`. DirectML attempts CPU fallback for that fused
operator, but the fallback can fail with `NotImplementedError` before routed
whole-brain training completes.

## Repair

- Adds `DMLSafeGRUCell`, a subclass of `nn.GRUCell` with identical parameter
  names and state-dict layout.
- CPU/CUDA continue to use the native PyTorch GRUCell path.
- DirectML/PrivateUse1 uses the standard GRU equations composed from linear,
  sigmoid, tanh, multiply and add operations.
- Current child recurrent sites are covered: nucleus, dynamics temporal cortex,
  and planning transition cell. Shared legacy GRUCell helpers are covered too.
- No checkpoint migration or retraining is required.

## Qualification

Software tests verify native/unfused numerical equivalence on CPU, backward
agreement, state-dict compatibility, and current-child recurrent coverage.
Actual DirectML training qualification still requires the same 10-update smoke
on the user's RX 6600.
