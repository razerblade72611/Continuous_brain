# Continuous Brain v0.11.3 validation

v0.11.3 repairs the DirectML Transformer causal-mask boundary without changing
neural shapes, learned values, or checkpoint keys.

Completed software validation:

- v0.11.0-v0.11.3 compatibility batch: 20 passed.
- v0.9-v0.11.3 developmental lineage: 59 passed.
- remaining modern/core batch: 105 passed.
- DirectML causal mask is floating, finite, upper triangular and parameter-free.
- CPU language path and language state-dict schema remain unchanged.

These tests do not claim DirectML whole-brain training capability. The actual
RX 6600 ten-route-step smoke remains the qualification event.
