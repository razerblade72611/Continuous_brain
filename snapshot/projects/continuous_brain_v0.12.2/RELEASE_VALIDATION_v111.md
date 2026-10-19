# v0.11.1 release validation

v0.11.1 is a device-backend compatibility patch. Validation targets:

1. v0.11.0 architecture and growth tests continue to pass.
2. CPU resolution remains unchanged.
3. DirectML aliases resolve through the optional `torch_directml` module.
4. DirectML checkpoints deserialize on CPU before model transfer.
5. Invalid DirectML indices fail loudly.
6. Routed Phase A uses non-foreach optimizer/clipping operations for broader
   PrivateUse1 compatibility.

A real DirectML training capability claim is intentionally deferred until the
user's RX 6600 completes the prescribed 10-update routed smoke.
