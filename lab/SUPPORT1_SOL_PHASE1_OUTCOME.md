# Sol independent SUPPORT-1 phase-1 outcome

Reviewed committed records: `dbe2f6fd3cefaeb41cd9341715a0e9d8bb8f4c22`.

**GO on the verified phase-1 outcome. Phase 2 remains HOLD for the new launcher
and analysis review.** Phase 2 must use NVIDIA RTX 6000 Ada Generation.

All four PHASE1 artifact hashes verify. Calcheck's train/eval byte hashes, phase-0
validation and prep coverage, code/model identities, non-test status, GPU, exact
probe inventories, independently graded outcomes and base denominator all match.
The calibration result is 8/10 template acquisition, 20/20 N FRESH UNKNOWN,
competence ratios 0.8203542322 / 0.8694460714, zero twin answer/outcome differences,
and competence deltas 1.2814e-7 / 2.8093e-8. Acquisition at exactly 0.80 passes the
frozen inclusive threshold. No epoch, threshold or seed change is warranted.
P's calibration paraphrase recall is 8/12, descriptive only; this single seed and
checkpoint do not establish later U4 performance.

The restoration record is bound to the same seed-24 prep/code and reports tensor
and step equality plus identical full-state hashes. Its admission is consistent.
Tensor/RNG snapshots are not committed, so this is verification of the recorded
admission and bindings, not a fresh tensor comparison or GPU rerun.

The chosen fork+64 projection independently recomputes to 10.0610045 hours and
$8.65246 at the recorded $0.86/h assumption. Adding recorded spending $0.54 and
attempt-1 spending $0.65 gives $9.84 after the report's rounding, below $30.
The actual billed rate is not established by these files. The previously noted
fork+32 row overcount remains conservative and does not affect the chosen plan.

The archive is now committed at `eefc37ca2b75f6bcfe254ad5f453afb0ae7697f9`.
Its independently recomputed SHA is
`3e589fb85c05267185825ac01cb66607fea0996bd86cf2247e46519941d5712b`.
All nine archived manifest/calibration/train/eval/restoration/timing/projection
files are byte-identical to the committed payload. The earlier archive-checksum
limitation is resolved. Full fractions and bindings are in
SUPPORT1_SOL_PHASE1_OUTCOME.json.

## Phase-2 analyzer readiness

The independent analyzer handles all 236 evaluation files: per seed, 50 trained-arm
main files, five N files, and four U4/G4 withheld files, plus the train log's fork
block. The full production-layout fixture uses seed20 through seed23 directories,
unsuffixed eval filenames and train_seedS.json inside each seed directory.
Root train logs and the earlier root eval_*_seedS.json layout are also supported.
Conflicting duplicate copies fail instead of silently shadowing each other.

Production analysis now requires the frozen phase-0 and phase-1 directories. It
loads validation.json automatically, verifies its byte binding, regrades calibration
from the raw files, verifies calcheck's train/eval byte hashes, and binds GPU,
restoration, base denominator, fork selection and QDIAG to the phase-1 decision.
An explicit validator/restore override must match the frozen bytes. An optional
--phase1-manifest-sha anchors the manifest to an external frozen digest.

Example (P0/P1 are the immutable directories from the runner's checkout):

```bash
$PY scripts/support1_sol_analyze.py "$R" \
  --phase0-dir "$P0" --phase1-dir "$P1" \
  --code-hash "$CODE_HASH" --model-manifest-sha "$MS" \
  --gpu "NVIDIA RTX 6000 Ada Generation" --qdiag 64 --fork \
  --out "$R/sol_analysis.json"
```

Keep the frozen phase-0/1 directories available for the offline rerun as well.
The new analyzer remains stdlib-only and imports no producer analysis functions.
No CODE_FILES changed, so no prep/calibration regeneration is needed for this work.
Claude's analyzer should likewise be a separate file outside CODE_FILES.

## Verification

All 39 earlier independent tests pass. All 13 new tests pass, covering raw regrading,
checksum-rebound calibration tampering, the 0.80 boundary, changed GPU/plan, external
manifest/validator bindings, all 236 files and fork readouts. Synthetic phase-2
answers and checkpoint identities in these tests are software fixtures, not results.

The new cases also exposed and fixed an old error-propagation bug: an invalid G1
or G4 base denominator could invalidate the primary P/U4 result. Errors now stay
scoped to that arm and its affected contrasts, as required by the protocol.
