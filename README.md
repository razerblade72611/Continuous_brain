# Continuous Brain v0.38.1.dev3: Historical-Core G0 Recovery

v0.38.1.dev3 starts from the WSL-validated dev2 G0 runner after the first real G0 attempt exposed a source/checkpoint incompatibility in the accumulated legacy tree. The v0.23 scientific DEV family 300000–300287 is permanently recorded as spent with **no scientific result**; the other six approved G0 DEV families remain unspent.

**What this release is:** a recovery/hardening artifact. It adds a hash-pinned pre-growth historical-core overlay from the original v0.12.2 archive, single-use ENGINEERING-only full-runtime preflights for all seven first-wave evaluators, a required preflight attestation before scientific G0 execution, and an explicit `INFRASTRUCTURE_FAILURE` status so setup/provenance failures cannot masquerade as scientific FAIL.

**What it is not:** it passes no scientific gate and claims no integration of historical mechanisms. Every mechanism slot of the runtime is **UNBOUND** until the DAV-74 audit verifies a legacy entry against the canonical WSL tree. Every battery and gate result is `ENGINEERING_ONLY`, `CAPABILITY_ABSENT` or `BLOCKED`.

## Layout

| Package | Role |
| --- | --- |
| `continuous_brain_protocols/` | Track-neutral contracts: `Substrate` (with separate `plasticity_step` / `train_step`, C2), `FeatureSource`, `Checkpointable`, `EmbodiedStream` (physical units only: the semantic firewall at the protocol level), `InterventionSpec`, `BudgetReport`, seed and provenance records, and the status vocabulary |
| `continuous_brain_snn/` | E0 Genesis, **unchanged** from commit `882e584` (C6) |
| `continuous_brain_substrates/` | The shared sensory encoder, the SNN adapter over E0 (H1), and the DENSE-RNN GRU comparator matched to the SNN's active synapses (H2, A1.1, report-only) |
| `continuous_brain_unified/` | The runtime: 15 required slots, a single-writer channel store, the per-tick DAG scheduler, interventions, a coprocessor interface (disabled at boot, null by default), checkpointing, the legacy manifest, loader and read-only inventory, and workspace/attention-schema interfaces (C1, C3, C4) |
| `continuous_brain_eval/` | G1–G5 harness, battery B1–B6, ridge probes with blocked CV, bootstrap statistics, versioned decision rules (`DAV-32-v1`, `A1-proposed`, `E2-primary-v1`), the A1 parity pilot, the indicator registry, the seed registry, content-addressed runs |
| `registry/` | `seed_ranges.json`, `preregistrations.json`, `indicators.json` |
| `legacy/` | `manifest.json` (VERIFIED entries only; empty), `manifest.schema.json`, `candidate_spec.json`, `candidates.json` (read-only inventory of the Windows source mirror, all UNVERIFIED) |
| `scripts/` | `e0_gate.py`, `battery_smoke.py`, `unified_boot.py`, `a1_pilot.py`, `local/*.sh` (WSL), `cloud/runpod_plan.py` + `pod_setup.sh` (explicit invocation only) |
| `docs/` | `ARCHITECTURE.md`, `EXPERIMENT_RULES.md`, `LEGACY_INVENTORY.md`, `SPECS.md`, `REVIEW_GUIDE.md` |

## Run it (inside WSL, Ubuntu)

```bash
cd ~/projects/Continuous_brain                # your WSL clone of the repo
bash scripts/local/setup_wsl.sh              # venv, editable install, full test suite
source .venv/bin/activate

python scripts/unified_boot.py --out runs/unified_boot                       # binding report: 12 unbound, 3 infrastructure
python scripts/battery_smoke.py --size small --out runs/battery_smoke        # B1-B6 on both substrates, engineering-only
python scripts/e0_gate.py --device cpu --out runs/e0_gate                    # the E0 infrastructure gate, as before

bash scripts/local/verify_protected_checkpoint.sh                            # infant_training.pt SHA-256 check
bash scripts/local/inventory_legacy.sh                                       # DAV-74 inventory of the canonical tree
python scripts/unified_boot.py --legacy-root ~/projects/continuous_brain_v0.12.2 --out runs/unified_boot_legacy
```

The legacy root is never committed. Pass it with `--legacy-root`, or set `CONTINUOUS_BRAIN_LEGACY_ROOT`.

Cloud: `python scripts/cloud/runpod_plan.py --job battery --replicates 10 --gpu-hourly <quoted price>` prints commands and a cost estimate. It launches nothing and makes no network calls.

## Findings from this build

1. **The v0.34.1 primary result appears to exist (DAV-77).** The Windows mirror has `state/event_consistent_self_development_test_v1/result.json`. It is SHA-256 `5ca0aa737f5dec2671049b7acd02783b52219a74dce3268737c52716b7585f4d`, records `split.test = [80000, 80127]` and `test_admission_passed: true`, and sits next to `result_bundle.zip` (`a944a4ac…70dd`). This is UNVERIFIED until it is hashed on the canonical tree.
2. **No legacy mechanism can be bound from the mirror.** The mirror has modules, evaluators, protocols and result JSONs for 13 of 14 candidates, but no checkpoint files. The results themselves record the hashes their checkpoints had (for example, the v0.24 integration pins four `.pt` files), so DAV-74 can verify checkpoints on WSL against hashes recorded at test time.
3. **The mirror stops at v0.35.0.** The v0.35.0 result is present (`development_gate_passed: false`, as the dossier says). Nothing for v0.35.2/.3 or v0.36.x (recursive curriculum) was found.
4. **The v0.21–v0.22 components record formal failures in their own result files** (`development_gate_passed: false` twice, `test_admission_passed: false` once). v0.24 nonetheless integrated their checkpoints and passed. DAV-74 must decide which of those components count as admitted.
5. **The legacy line's own manifests stop at v0.12.2.** Only `embodied.py` among the candidate modules matches a legacy-shipped SHA-256. Every later module is unlisted, which is expected but leaves no independent hash to check them against.
6. **A1.2 is exercised, not approved.** `scripts/a1_pilot.py` runs both acceptance pilots (false-fail rate over ≥ 100 identical pairs; sensitivity to a 10 % shift) for `A1-proposed` and `DAV-32-v1` side by side.
7. **Budget matching is dominated by input/output layers.** At full size the SNN has ~375k active synapses. With the shared encoder (2,182 inputs), the matched GRU therefore has hidden size **42** (374,302 parameters, −0.25 %). That is what the A1 matching rule produces. It is flagged in `docs/ARCHITECTURE.md` as a question for David and ChatGPT, not changed.

See `CHANGELOG.md` for everything added, and `docs/REVIEW_GUIDE.md` for a clause-by-clause map from the contract to the code.

## v0.38.1.dev0 — G0 legacy requalification

This development release adds a **read-only G0 evidence gate** before historical mechanisms can enter composition. See `docs/G0_LEGACY_REQUALIFICATION.md`.

On the canonical WSL machine:

```bash
export CONTINUOUS_BRAIN_LEGACY_ROOT="$HOME/projects/continuous_brain_v0.12.2"
python scripts/local/g0_plan.py --legacy-root "$CONTINUOUS_BRAIN_LEGACY_ROOT"
bash scripts/local/refresh_g0_inventory.sh "$CONTINUOUS_BRAIN_LEGACY_ROOT"
```

The first command verifies the immutable evidence plan. The second refreshes result parsing (including the newer endogenous-goal lineage) into `evidence/g0_runtime/`; neither command writes to the historical tree.

A1.1b/A1.2b comparator support is **proposed, not approved science**. See `docs/A1_AMENDMENT_SUPPORT.md`.

## Current scientific status (v0.38.1.dev4)

Six historical mechanisms have completed isolated G0 requalification with PASS on fresh preregistered DEV families; the verified hashes and complete frozen check sets are recorded in `registry/g0_requalifications.json`. This is isolated reproduction evidence only, not composition/integration evidence. `G1-COMPOSITION-RETENTION-v1` is PROPOSED and locked; no G1 seed family is registered in this artifact. v0.23 remains blocked pending a separate condition-ID amendment.

## v0.38.1.dev6 — store-mediated composition route (ENGINEERING_ONLY)

The six G0-requalified mechanisms run through one `UnifiedRuntime`. Each frozen evaluator runs unmodified as the task driver, and every top-level call to a mechanism API is one runtime tick, carried through the canonical store by an exact codec. E1 (null-composition equivalence) passes for all six on 32 engineering seeds each, and E1b (store-authority control) passes. Neither is a scientific result, and no G1 family has been opened.

See `docs/COMPOSITION_ROUTE_DEV6.md` for what is and is not mediated, and why a retention-style G1 is predetermined by E1. WSL reproduction steps are in `evidence/dev6/WSL_COMMANDS.md`.

## v0.38.1.dev7 — I1: one persistent individual (ENGINEERING_ONLY)

The five self-lineage mechanisms act on **one individual** whose state lives in the canonical store. Erasing, replacing, rolling back or restoring that state changes, or preserves, the individual itself. On 64 cloud engineering individuals, every validity check holds and C1–C10 meet their frozen thresholds. One new finding: the developmental controller's graduation to DIRECT lowers second-goal completion compared with fixed SCAFFOLD. The difference is −0.203 (95% CI [−0.339, −0.068]) on the delivered code, and negative on all five disjoint dev slices. The mechanism: DIRECT practice buys throughput but erodes the self-model reliability that the goal is scored on. `I1-INTEGRATED-INDIVIDUAL-v1` is PROPOSED; no scientific family is open.

See `docs/I1_INTEGRATED_INDIVIDUAL_DEV7.md`. The WSL engineering check and preflight steps are in `evidence/i1/dev7/WSL_COMMANDS.md`.
