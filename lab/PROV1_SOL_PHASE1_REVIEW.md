# Sol: PROV-1 review at 24984b1

**HOLD on calcheck, the peer analyzer, and the phase-1 launcher. PASS on the
committed phase-0 prep payloads.** No GPU launch is authorized by this review.
Reviewed `24984b1ae73342cd84e58c7f23ee4be16aa6fa55`.

## Reproduced calibration admission defects

The complete production-shaped synthetic calibration is accepted first: every
probe is present, both outcome fields are consistent, RP acquisition and base
FRESH are perfect, monitors finite, and twins agree. Starting from that control,
the current calcheck still returns `pass: true`, no errors, and exit 0 after:

1. deleting RP/RC checkpoint hashes from the train log and setting all matching
   eval checkpoint identities to null;
2. setting all five base-denominator hashes to null;
3. setting all five base-denominator hashes to a shared non-hash string;
4. replacing both monitor values everywhere with boolean `true`;
5. substituting a foreign model manifest in the train log;
6. substituting test-mode training provenance in that production calibration;
7. substituting a foreign train-log seed.

Fix `scripts/prov1.py:484-538`: require valid non-null 64-hex checkpoint and base
hashes; reject bool monitors; validate complete train schema/experiment/seed,
code/prep/model/test-mode/GPU/request identity. Keep the existing complete
inventory and independent truth/compliance regrade checks. N ratios should be
unit ratios against their own base denominator.

These seven strict-xfail cases are in `tests/test_prov1_phase1_review.py`.
The valid baseline remains an ordinary passing test. The independent calibration
reader also accepts that valid control and rejects the null/hash/bool/model
mutations independently; it does not trust the producer's `pass` flag.

## Phase-1 manifest coverage

The job's PHASE1.sha256 currently covers only calcheck.json, timings, projection,
repro.json and validation.json. It does not bind the calibration train log or
five raw calibration evals; calcheck.json contains no hashes of those inputs.
Add the raw calibration evidence to the phase-1 manifest (and bind the pilot's
source evidence as appropriate), so the independent offline reader can regrade
the actual inputs that justified PASS. A summary flag alone is insufficient.

## Peer analyzer admission and common-pair findings

`tests/test_prov1_peer_admission.py` has a complete synthetic file control that
returns ADVANCE. The current peer loader still returns ADVANCE after changing
only the train-log seed, model manifest, or test-mode provenance. Separately,
its production eval loader accepts a foreign GPU with otherwise correctly
bound production prep/model/request fields.

Bind the full train identity, model, test mode, GPU, checkpoint/state inventory,
exposures and fork parent, as well as prep/code. Bind analysis to the frozen
phase-0 preps and validated calibration. Phase A must use the declared exact-file
admission route: the current optional `--admission` tensor-equality shortcut is
outside that chosen route and should not admit an arbitrary self-asserted flag.

The common K5 contrast has a separate reading defect. It intersects cells after
each arm averages its own eligible variants. If RC is faithful-correct only on
Question and RU only on Statement, there is no shared eligible probe, yet it
reports a numeric RC-RU contrast. Intersect matched eligible variants first,
then perform construction/cell/sleep/seed averaging. Empty common support is
not assessable. The counterexample reports +1 instead of null.

## E: drive guarantee

The output preflight checks only writability and free space. An ordinary
directory standing in for the missing `/mnt/e` mount passes it. If E: is absent,
that directory can be on WSL's root filesystem and therefore backed by C:.
Verify the actual mounted E: filesystem/source and the resolved output path
before creating or writing outputs. The preflight-only counterexample executes
no pod/SSH/launcher operations.

Also remove the mandatory C: dependency at `scripts/cb_prov1.sh:309`: it still
copies the results bundle to `$DL`, defined on C:. A full/unwritable C: makes the
success path fail even after E: preservation verified. The all-E: contract must
hold on the normal success path as well as in preserve().

The phase-1 pilot also unconditionally deletes its checkpoint at line 164 even
when `file_equal` is false. That nonmatching file is not the preserved original
on E:. A local execution of only the pilot fragment confirms deletion after a
mismatch. Retain and preserve it for investigation; do not apply the
"already on E:" assumption to a failed identity check.

## K6 and K5 conventions

**K6: use the same-seed joint rule.** At least three of four seed history-pairs
must each pass both constructions. In the proposed counterexample, seeds 21 and
22 pass both: there are two joint successes, not zero, and the experiment fails
the required three. The independent analyzer now reports each construction and
the joint per-seed result, with the joint count feeding K6.

The earlier independent draft included every supported probe in K5, whereas the
peer uses POS/POS-Q. Align the primary to **POS/POS-Q** and record that population
explicitly in the protocol. The broader supported pool (including update/other
diagnostics) is retained separately as `K5_all_supported_diagnostic`; it does not
determine the primary K5 gate. This resolves a specification ambiguity before
any PROV-1 results exist, not by selecting an observed result.

## Independent CLI and verification

The independent analyzer now supports the launcher's exact interface:

```bash
python scripts/prov1_sol_analyze.py --phase A --dir "$R" \
  --support1-dir "$REF/support1" --out "$R/analysis_sol.json"
```

Use `--phase B` for B. Offline runs can override `--phase0-dir` and `--phase1-dir`;
otherwise those are the committed results directories in the checkout. The
directory loader binds prep/validator/calibration evidence, the historical
SUPPORT-1 archive and reference bytes, GPU, and the eleven exact admissions per
phase-A seed. Missing/malformed evidence yields a structured INVALID JSON and
nonzero exit. Direct `--request` remains supported as a test/draft interface.
Synthetic complete A/B directory tests pass. No producer or peer analyzer is
imported by the independent tools.

Independent phase-0 verification:

- Archive SHA256:
  `5bb11fd26449a14e96d23030748b4103360a87661bdbb2ca1e72f575a95c9cfa`.
- Exactly the five expected prep members; every prep matches PHASE0.sha256.
- The five local tokenizer/config files match the frozen CLS-0B manifest.
- All five preps independently reconstruct: **126,343 checks, zero errors**.
- The independent validation report is exactly equal to Claude's committed run
  of the same tool. `PROV1_SOL_PREP_VALIDATION.json` is refreshed for CODE_HASH
  `c591a45a15663cc3a49b758ce54fb5dd6fcd91e8e706472a1c090f051cb6e5c6`.

Supplied focused suite: **62 passed**. Supplied launcher preservation suite:
**7/7 passed**. Independently deleting either stream-verification line makes
the launcher harness fail. These preservation checks do not cover the missing
mount-source check or the final C: copy.

The new review suite has **14 strict-xfail counterexamples**: seven calibration,
three train-identity, one GPU, one common-K5, one mount-preflight and one
nonmatching-pilot preservation case. With `--runxfail`, the first thirteen fail
in the full counterexample run and the pilot case fails in its focused run;
their valid controls pass. Remove the
markers only after the cases are rejected while their controls remain valid.

The projector's actual-inventory counts agree: phase A has 37,248 regeneration
items and 83,546 eval rows; phase B has 23,680 train items and 148,375 eval rows.
The unaffordable-phase-A case exits 5. No synthetic timing example is an empirical
GPU cost estimate; phase 1 must supply that measurement.

No producer CODE_FILES were edited. Fixing producer calcheck will change CODE_HASH,
so regenerate/revalidate preps and refresh phase 0 before pinning the launcher.
Phases A/B launch paths are not approved by this phase-1 review.
