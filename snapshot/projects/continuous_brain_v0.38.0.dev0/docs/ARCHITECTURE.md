# Architecture — v0.38.0.dev0

## Separation of claims

The build distinguishes three states:

1. **SOURCE_VERIFIED** — exact historical source/evaluator files match the committed SHA-256 manifest.
2. **ENTRYPOINT_BOUND** — the verified module imports from the expected root and declared symbols exist.
3. **SCIENTIFICALLY_INTEGRATED** — reserved for a future fresh-family requalification. v0.38.0.dev0 never emits this state.

This prevents fresh code from inheriting an admitted historical name merely because it resembles an older mechanism.

## Packages

- `continuous_brain_substrates`: neutral contracts, GRU comparator, external E0 adapter, feature adapters, provenance and seed registry.
- `continuous_brain_unified`: legacy verification/binding, composition shell, coprocessor contract, intervention schemas, shell checkpointing.
- `continuous_brain_eval`: engineering battery, G1–G5 infrastructure, indicator ledger, optional parity analysis.

The frozen historical `continuous_brain` package remains external.

## Gradient firewall

`Substrate.plasticity_step()` is for local/native adaptation. `Substrate.train_step()` is for gradient training. The external SNN adapter rejects `train_step` and verifies that exposed torch parameters/tensors do not require gradients.

## E0/E1/E2

E0 is treated as an external verified engine because its source was not available in the artifact set used to create this ZIP. E1 and E2 stay behind their own issues/gates and are not implemented here.
