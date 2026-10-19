# Continuous Brain v0.38.0.dev0 — Unified Individual Foundation

This repository is an **engineering integration foundation**, not a claim that a unified cognitive individual has already been scientifically admitted.

It follows the 2026-09-27 joint ChatGPT/Claude implementation contract:

- historical mechanisms are **wrapped, never rewritten under admitted names**;
- every legacy binding is hash-pinned and may be `UNBOUND` if its exact source/evaluator cannot be verified;
- local SNN plasticity and gradient training are separate APIs;
- the package name is `continuous_brain_unified`, so it does not shadow the frozen `continuous_brain` lineage;
- no placeholder GWT/attention-schema mechanism is shipped;
- B1 is blocked until the real E2 body exists;
- B2–B6 synthetic runs are **engineering-only** until their scientific metrics and decision rules are preregistered;
- E1/E2 are outside this build.

## What is real in this package

1. **Hash-pinned legacy inventory and loader.** The included manifest was generated from the preserved `continuous_brain_mirror.zip` available to this build. It verifies source/evaluator bytes before import and refuses silent path shadowing.
2. **Unified composition shell.** It reports `SOURCE_VERIFIED`, `ENTRYPOINT_BOUND`, or `UNBOUND`; it never fabricates a missing mechanism.
3. **Track-neutral substrate contract.** Gradient training and local plasticity are different methods.
4. **GRU comparator.** A real trainable recurrent comparator with causal next-input training support.
5. **External E0 SNN adapter.** It wraps an existing E0 engine if supplied and enforces the gradient firewall. E0 source is not copied or reconstructed here because no verified E0 source archive was available to this build.
6. **Engineering battery harness.** B1 is blocked; B2–B6 exercise APIs and provenance only and always emit `ENGINEERING_ONLY` or `UNREGISTERED_METRIC`.
7. **Integration-gate schemas (G1–G5).** They validate plans, binding availability, persistence plumbing, and coprocessor ablation plumbing without turning infrastructure into scientific evidence.
8. **Provenance.** Content-addressed run manifests, append-only registry, config hashes, artifact hashes, seed-overlap checks.

## Install / test

```bash
cd continuous_brain_v0.38.0.dev0
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

On your existing Continuous Brain venv you may instead do:

```bash
source "$HOME/.venvs/continuous_brain_v110_dml/bin/activate"
pip install -e ".[dev]"
pytest -q
```

## Verify your legacy tree

Point the build at the preserved historical root containing the `continuous_brain/` package:

```bash
export CONTINUOUS_BRAIN_LEGACY_ROOT="$HOME/projects/continuous_brain_v0.12.2"
python scripts/verify_legacy.py
python scripts/show_status.py
```

If your root differs from the source snapshot used to generate this package, the verifier reports mismatches and does not silently substitute code.

## E0 SNN

This ZIP intentionally does **not** reconstruct or copy the E0 SNN from prose. Configure the real E0 package/engine when it is available:

```python
from continuous_brain_substrates.snn_adapter import ExternalSNNAdapter
adapter = ExternalSNNAdapter(real_e0_engine)
```

`adapter.train_step(...)` always raises `GradientFirewallError`.

## Engineering battery

```bash
python scripts/run_engineering_battery.py --substrate dense
```

The output is diagnostic only. It cannot produce a scientific PASS.

## RunPod

`scripts/prepare_runpod.sh` only prints a suggested setup command. It never launches or rents a GPU.

## Scientific boundary

Nothing in this repository establishes subjective experience, consciousness, sentience, personhood, or a consciousness indicator. The indicator registry is an evidence ledger, not a scorecard.
