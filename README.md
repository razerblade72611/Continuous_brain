# Continuous Brain: SNN program, E0 (Genesis)

The deterministic, continuous-time spiking substrate for the E0–E6 program in the *Complete Project Dossier* (23 Sep 2026) and the Linear project. Stage E0 has **no semantic inputs and no pretrained models** (DAV-5).

This package (`continuous_brain_snn`) lives **alongside** the legacy `continuous_brain` (v0.12.2 + overlays); it doesn't replace it. That follows the migration rules: freeze the originals, inventory first, replace only behind tests.

## Status against the Linear issues

| Issue | What | State |
| --- | --- | --- |
| DAV-32 | 1 ms Euler LIF engine: E/I drive kept separate, refractory, homeostatic threshold, seeded noise | Built. Bit-identical per seed on CPU (tested). Cross-backend: statistical harness built; **see the parity finding below** |
| DAV-33 | Traces τ = 5/25/100/500/2000 ms, per-population summaries | Built. Analytic decay, variable-dt partition and steady-state tests pass |
| DAV-34 | Procedural anatomy: retina ON/OFF, proprioception (angle/velocity per joint), motor M+/M−/Mco, cortex 80/20 E/I in spatial columns | Built. Anatomy hash deterministic from seed; semantic firewall validator |
| DAV-35 | Fixed-capacity slots (src/weight/candidate/active/eligibility/utility), dormant candidates | Built. Prune/activate by mask, no shape change; activation that would break occupancy/fan-out caps is refused |
| DAV-36 | Checkpoint + manifest (state, RNG, config/anatomy/connectivity/state hashes, schema, stage, git revision, Linear issue) | Built. Restore → continue reproduces the trajectory bit-exactly; mismatched anatomy is refused |
| DAV-37 | Telemetry schema: JSONL (offline) + optional W&B | Built. Every window record has all `REQUIRED_KEYS` |
| DAV-76 | Run registry: code + config + seed + anatomy + checkpoint + Linear issue + artifact SHA-256 | Partial: registry refuses PASS/FAIL claims without an artifact. Seed-range registry not built |
| DAV-74 / DAV-75 | Audit and migration of the legacy tree | Not started. Needs the WSL tree; the Windows mirror has source code but no checkpoints |

## E0 gate

```bash
# inside WSL, from the repo root
pip install -e ".[dev]"
python -m pytest                                    # 68 tests, ~30 s on CPU
python scripts/e0_gate.py --device cpu --out runs/e0_gate
python scripts/e0_gate.py --device dml --out runs/e0_gate \
    --compare-to runs/e0_gate/<cpu_run_id>/parity_stats.json
```

The gate checks genesis invariants, 10 s stability under synthetic physical input, bit-exact determinism, checkpoint continuation, structural prune/activate, telemetry completeness, and records 60 s of plasticity-off statistics for cross-backend parity. It writes a report, records its SHA-256 in `runs/.../logs/run_registry.jsonl`, and saves a never-overwrite checkpoint. A pass is an **INFRASTRUCTURE PASS**. It makes no cognitive claim.

The first full-size CPU run (10,000 neurons) passed every gated check. Rates under drive: cortex ~9 Hz, motor ~11 Hz, retina and proprioception ~5 Hz. Speed was 3.9 ms per 1 ms step on a 2-core cloud CPU, about 3.9× slower than real time. It hasn't been run on your machine or on DirectML yet, so treat it as UNVERIFIED there.

## Parity finding: DAV-32 tolerances need an amendment

The correlation criterion (KS < 0.1) is well-behaved: KS was 0.005–0.013 between equivalent runs.

The per-population criteria (rate within ±5%, ISI CV within ±0.05, applied to every one of the 140 populations) are underpowered. I simulated 12 pairs of *statistically identical* 60 s runs on the full anatomy, and **4 of 12 failed**. The smallest populations have 10 neurons, and sampling noise alone crosses ±5% in one of 140 comparisons about a third of the time.

The code implements the rule exactly as written. One possible amendment is to scale each population's tolerance to its measured sampling error, e.g. max(5%, 3.5 standard errors of the difference), with a family-wise correction across populations. That changes the acceptance rule, so it's David's decision, and it should be frozen before any cross-backend acceptance run.

## Design notes

- **Determinism.** Anatomy, candidate edges, and weights are generated on CPU from the seed. Spike propagation is a gather (each neuron reads its own K sources), so there are no atomic adds. Noise comes from a seeded CPU generator by default, so every backend gets identical noise; DirectML has no device-side generator.
- **Time.** Absolute time is `steps × dt`, an integer count, so it never drifts.
- **E/I.** `g_e` and `g_i` are integrated and reported separately.
- **Semantic firewall.** Population metadata is restricted to physical keys per region; `validate_physical` rejects anything else, and genesis refuses to build if it fails.
- **Hardware.** The documented local GPU is an RX 6600 8 GB on DirectML. E0–E4 fit locally. A ~27B frozen Qwen (E5) does not, so it has to run on a rented GPU.

## What was parked

An earlier prototype written against the *Hybrid Architecture Spec v0.1* (spike→token transducer, soft-prefix bridge, episodic store, gated sleep) is kept in git history at commit `01bd411`. It conflicted with the dossier: language in from the start, the core learning only during sleep, and a 24 GB CUDA assumption. The parts that fit have moved here (never-overwrite snapshots, the float64-time lesson, the trace math). The bridge and memory ideas belong to E4/E5 and must re-enter only through those stages' gates.
