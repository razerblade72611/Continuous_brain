# Changelog

## 0.38.1.dev7 — I1: one persistent individual (ENGINEERING_ONLY; preregistration PROPOSED)
- **Integration package.** Adds `continuous_brain_unified/integration/` (hosts, life, route, evaluate, runner). The five G0-requalified self-lineage mechanisms act on one individual. That individual's autobiography, developmental phases, goal lifecycle and curriculum history are blob channels in the canonical store, read by their hosts at every tick, and nothing is kept in Python between ticks. The v0.34.1 developmental controller, not a fixed rule, chooses the strategy for goal pursuit.
- **Arms.** NORMAL, FIXED_SCAFFOLD, LEGACY_REF (the frozen v0.36 driver), store-level ERASED / DONOR / ROLLBACK on the autobiography, PERSIST_SAVE/RESUME across processes, and the validity controls AUTHORITY_SKILLS / _GOAL / _CURRICULUM.
- **Redirects.** All five specs are installed. Driver-side calls to a mechanism's API are blocked; host-side calls run the frozen code and are counted. Model forwards are accounted host-side vs driver-side.
- **Runner.** Three modes: dev (confined to a slice of one registered engineering family, with derived ids checked), preflight (single-use family, values withheld from display and committed by SHA-256, not encrypted) and science (needs APPROVED, `approved_design_sha256 == design_digest`, a manifest equal to the preflight's, and `--confirm-open`). Pins and the historical core are verified before any family is registered. A failure after the spend writes a BLOCKED report. The manifest covers all 84 code files by repo path and is recomputed inside every arm. Approval is one checked act, `scripts/i1_approve.py`, which sets both registry files.
- **Preregistration.** `registry/i1_integration_v1.json`, `I1-INTEGRATED-INDIVIDUAL-v1`, is PROPOSED. Scientific family `i1-v1-dev` 328000–328127 (moved from 320000–320127 before registration, after a conservative derived-id check) and preflight family `i1-v1-preflight-eng-b` 907100–907163; neither is registered. `i1-v1-preflight-eng` (907000–907063) was consumed by mistake in the cloud and is recorded as spent; its sealed report has not been opened.
- **Independent review.** All 13 findings from a fresh reviewer agent (14 table rows, because verification-before-registration is listed separately) are fixed (see `docs/I1_INTEGRATED_INDIVIDUAL_DEV7.md` §6). An independent verification of the first dev7 package found four more issues (the approval procedure, a C2 leak in the preflight view, permissive authority aborts, a derived-id coincidence); all are fixed. `scripts/dev/i1_tamper_check.py` checks 18 properties; every one is detected.
- **Dev result.** `i1_dev7_64c` (906808–906871, cloud CPU, the delivered code): all 12 validity checks hold and C1–C10 meet their frozen thresholds. S3, goal-2 completion with the controller vs fixed SCAFFOLD: 0.224 vs 0.534, difference −0.310, 95% CI [−0.431, −0.190]. Estimation only; engineering data. Four disjoint slices all give a negative difference, ranging from −0.138 to −0.310.
- **Disclosed.** `i1_dev7_64` (906680–906743) is BLOCKED by its own manifest check, because the version strings were bumped mid-run. It was repeated unchanged.
- **Tests.** 221 pass.
- **Not claimed.** Consciousness, sentience, subjective experience, or anything beyond the frozen two-commitment world. No scientific family has been opened.

## 0.38.1.dev6 — store-mediated composition route (ENGINEERING_ONLY)
- **Composition package.** Adds `continuous_brain_unified/composition/` (codec, spec, boundary, route). A frozen legacy evaluator runs unmodified as the runtime's task driver, and every top-level call to one of the six mechanisms' APIs becomes one runtime tick: an encoded request is written to the canonical store, a `MechanismHost` serves it, and the encoded response is read back from the store. Every run writes a composition attestation (scheduler graph hash, per-host and per-API traces, read log, content flows, model-forward accounting, census).
- **Runtime and channels.** Blob channels (`shape=None`); a per-reader read log (counts and hash chain); declared driver channels (`RuntimeConfig.driver`, owner `driver`), with the config hash unchanged when no driver is declared.
- **Shared models.** Frozen models are loaded once per pinned checkpoint, with the global RNG state saved and restored around the load. The evaluator's first load is verified bit-identical to the shared instance. Later loads are reported as private reload probes. Stateless controllers and generators are registered per construction and checked against the default configuration.
- **Dev scripts.** `scripts/dev/` adds the G0-route dev harness, the composition-route runner, E1 (null-composition equivalence) and E1b (per-API store-authority positive control). They run only on the registered engineering family `cloud-dev-claude-eng` (906000–906999), and every use is logged.
- **E1: PASS for all six mechanisms on 32 cloud-dev seeds each.** The G0 route, full composition and used-hosts composition give byte-identical `result.json`; the faithfulness checks pass. Mediation is measured: v0.34.1, v0.35.3 and v0.36.1 are fully store-mediated; v0.31, v0.32.2 and v0.25.2 also make direct model calls in the driver (counted).
- **Review.** An independent reviewer agent's nine confirmed findings are fixed, with regression tests. Tests: 205 pass.
- **Versions.** Package `__version__` strings are aligned to 0.38.1.dev6. The code had said dev3 while `pyproject.toml` said dev5.
- **Scope.** No G1 family is registered or opened, and no scientific claim is made. `docs/COMPOSITION_ROUTE_DEV6.md` explains why E1 makes a retention-style G1 predetermined, and what an integration experiment would need instead.

## 0.38.1.dev4 — freeze six G0 PASSes + propose G1 retention
- Independently verifies the packaged six-result G0 bundle (202 manifest entries, zero mismatches) and freezes the six PASS records and their raw-result hashes in `registry/g0_requalifications.json`.
- Marks the six completed G0 DEV families and seven engineering runtime-preflight families spent in the canonical static seed registry.
- Fixes dev3 tests that assumed those legitimately spent engineering families were still unused.
- Adds `G1-COMPOSITION-RETENTION-v1` as a **PROPOSED**, fail-closed preregistration scoped only to the six G0-cleared mechanisms.
- Freezes each mechanism's full G0 check-key contract and proposed fresh G1 DEV family, but registers/opens no G1 scientific range.
- v0.23 remains blocked pending a separate condition-ID preregistration amendment.

## 0.38.1.dev3 — historical-core G0 recovery
- Records `g0-v023-dev` (300000–300287) as permanently spent after an infrastructure failure that produced no result; it is not a scientific FAIL and is never reusable.
- Adds `INFRASTRUCTURE_FAILURE` to the result vocabulary for evaluator/setup/provenance failures that cannot support a scientific verdict.
- Adds a hash-pinned historical-core overlay from the original `continuous_brain_v0.12.2_unified_workspace.zip` (SHA-256 `a73545ba…aa554`). The protected checkpoint loads with the original 5-block nucleus growth stack; the later accumulated root had been changed to 20 blocks after the first-wave historical tests.
- Adds seven disjoint single-use ENGINEERING-only runtime-preflight families inside the existing v0.38 engineering reserve.
- Scientific G0 execution now requires a completed runtime-preflight attestation with matching evaluator and historical-core identities before a DEV family can be opened.
- Runtime preflight runs the complete frozen evaluator on engineering IDs but intentionally does not surface or interpret its metric/pass field.
- Fixes seed-open/result reports so the family is represented as spent after the irreversible open.
- No replacement v0.23 scientific DEV family is allocated; that requires a preregistration amendment and explicit approval.

## 0.38.1.dev2 — G0 approved execution lock
- Records David's explicit approval of `G0-LEGACY-REQUAL-v1`.
- Registers the seven frozen fresh DEV families exactly as preregistered; all ship unspent.
- Requires explicit `--confirm-open G0-LEGACY-REQUAL-v1` for scientific execution.
- Marks a G0 family spent before evaluator launch, so failed/interrupted attempts cannot be repeated on the same scientific family.
- Adds static evaluator preflight checks before any family is opened.
- No historical sealed TEST range is reopened; E0 remains unchanged.

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

## 0.38.1.dev1 — G0 runner
- Engineering-pass baseline retained.
- Added fail-closed G0 fresh-DEV runner, proposed seed plan, and preregistration draft.
- No G0 scientific seed family is registered or opened in this artifact.

## 0.38.1.dev5
- Preserve the never-approved G1 v1 proposal as superseded.
- Add `G1-COMPOSITION-RETENTION-v2` with canonical shared-state identity and six-way scheduler participation requirements.
- Require a frozen composition manifest and six ENGINEERING_ONLY full-composition preflights before any scientific G1 family can open.
- No G1 range is registered or spent.
