# Continuous Brain v0.12.2 release notes

v0.12.2 responds directly to the RX 6600 DirectML qualification evidence from v0.12.1.

Changes:

- keeps the successful DirectML-safe GRU gate slicing from v0.12.1;
- keeps scatter-free DirectML one-hot;
- adds differentiable DirectML-only clone/materialization boundaries at shared graph fan-out seams;
- adds a minimal competitive training forward that skips unused frozen care/router/metacognition computation while preserving all train-relevant outputs exactly;
- adds v0.12.2 hardware qualification with runtime version reporting;
- retains full BPTT and the exact 14k base checkpoint;
- retains the v0.12 lifelong runtime and state compatibility.

This is a backend-hardening release, not a positive result for the competitive-broadcast scientific hypothesis.
