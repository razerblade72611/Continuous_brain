# v0.11.2 release validation

v0.11.2 repairs DirectML recurrent-cell dispatch without changing checkpoint
semantics or neural architecture.

Validated in the build environment:

- v0.11.2-specific recurrent compatibility tests: 5 passed.
- v0.11.0/v0.11.1/v0.11.2 compatibility set: 15 passed.
- v0.9 through v0.11.2 developmental lineage: 54 passed.
- remaining modern/core test batch: 105 passed.
- unfused GRU forward and backward numerically agree with native `nn.GRUCell`.
- `DMLSafeGRUCell` has the same state-dict keys and shapes as `nn.GRUCell`.
- current child recurrent sites (nucleus, dynamics temporal, planning transition)
  use the DirectML-safe cell.

A real DirectML whole-brain training capability claim remains contingent on the
user's RX 6600 completing the 10-update routed smoke. Unit tests cannot substitute
for that hardware result.
