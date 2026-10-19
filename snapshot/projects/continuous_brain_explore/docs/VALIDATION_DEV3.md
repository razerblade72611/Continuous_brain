# v0.38.1.dev3 validation and recovery note

The first approved G0 execution opened only `g0-v023-dev` (300000–300287). The family was correctly marked spent before launch. The frozen v0.23 evaluator exited before writing `result.json`, so the attempt is an **infrastructure failure with no scientific G0 verdict**.

Captured WSL traceback: loading protected checkpoint SHA-256 `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f` into the accumulated legacy source expected `nucleus.growth.blocks.5..19`, which are absent from the checkpoint. The original v0.12.2 `growth_v110.py` defines `nucleus_blocks = 5`; the accumulated source later changed that to 20 at 2026-09-20 04:46 UTC, after the first-wave historical evaluators had been created/run. The protected checkpoint loads cleanly against the original source.

Dev3 therefore runs G0 with a **historical-core import overlay** rather than pretending the latest accumulated source is the historical runtime. The overlay archive and each substituted source file are SHA-256 pinned in `legacy/historical_core_v0122.json`; mechanism/evaluator files remain pinned by `legacy/g0_plan.json`.

Before any remaining G0 DEV family can open, the same evaluator must complete once on its registered ENGINEERING-only preflight family using the same historical-core overlay. The preflight result exposes only execution/provenance facts, never its metric or pass field. Scientific execution requires that attestation.

`g0-v023-dev` stays spent. Dev3 may engineering-preflight v0.23 to prove the source/checkpoint repair, but a replacement scientific v0.23 family requires a preregistration amendment and David's explicit approval.
