# Changelog

## 0.38.1.dev0 — 2026-09-27

- Starts from Claude c011d40 canonical v0.38.0 engineering baseline.
- Adds G0 evidence qualification from David's WSL inventory without reopening historical TEST ranges.
- Fixes engineering seed sibling-overlap leakage, G1 evaluator/threshold coverage, and G2 matrix completeness.
- Adds report-only A1.1b DENSE-CORE comparator support and A1.2b z=3.74755 while preserving the original A1 rule id.
- Adds DirectML lock capture rather than guessing a torch/torch-directml version pair.


## 0.38.0.dev0 — 2026-09-27 (Claude's build of the Joint Implementation Contract v0.38.0)

Pre-release marker (C3). No scientific gate has been passed; no historical mechanism is integrated.

### Added
- `continuous_brain_protocols`: `Substrate` (separate `plasticity_step` / `train_step`), `FeatureSource`, `Checkpointable`, `EmbodiedStream` with physical-unit channel specs, `InterventionSpec`, `LesionSpec`, `BudgetReport`, `SeedFamily`, `ProvenanceRecord`, canonical JSON / state digests, status vocabulary.
- `continuous_brain_substrates`: shared sensory encoder; `SNNSubstrate` over the unchanged E0 engine (gradient firewall, exact freeze, lesions, exact restore); `DenseRNNSubstrate` GRU comparator (hidden size solved to ±10 % of SNN active synapses, causal one-frame-ahead objective, TBPTT, single-exposure ledger, lesions, exact restore including optimizer state).
- `continuous_brain_unified`: 15-slot composition runtime; single-writer channel store with write log; DAG scheduler with delayed edges; binding policies; intervention manager; coprocessor interface (null, disabled at boot, logged, budgeted, ablatable); checkpointing through `SnapshotStore`; legacy manifest/verification/loader/step-contract registry; stdlib read-only inventory; workspace and attention-schema interfaces with preregistration-gated registration.
- `continuous_brain_eval`: G1–G5 harness; battery B1–B6; synthetic two-joint streams; track-neutral driver; ridge probes with blocked CV; bootstrap / Wilcoxon / trend statistics; decision rules `DAV-32-v1`, `A1-proposed`, `E2-primary-v1`; A1.2 power-aware parity and its pilots; indicator registry; seed registry; content-addressed runs and result registry.
- `registry/`: seed ranges (historical sealed windows, the flagged v0.31.1/v0.33.5 overlap, the engineering reserve), preregistrations (all DRAFT/PROPOSED), indicators (14 rows, none admitted).
- `legacy/`: empty VERIFIED manifest + JSON schema; candidate spec; inventory of the Windows source mirror (all UNVERIFIED).
- Scripts: `battery_smoke.py`, `unified_boot.py`, `a1_pilot.py`, WSL helpers, RunPod plan/setup (print-only).
- Docs: README, ARCHITECTURE, EXPERIMENT_RULES, LEGACY_INVENTORY, SPECS, REVIEW_GUIDE.
- Tests: 166 (the 68 E0 tests unchanged).

### Fixed before hand-off (independent internal review)
- 12 confirmed defects and 13 lesser concerns found by a separate reviewer agent. See `docs/REVIEW_GUIDE.md` ("Internal review round").

### Unchanged
- `continuous_brain_snn` (E0 Genesis) and `scripts/e0_gate.py`, byte for byte (C6). E1/E2 remain separately gated work.

## 0.1.0.dev0 — 2026-09-23 (commit 882e584)
- E0 Genesis SNN substrate per DAV-5 (see git history). The hybrid prototype remains parked at commit 01bd411.
