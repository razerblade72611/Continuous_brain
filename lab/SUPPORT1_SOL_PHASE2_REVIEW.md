# Sol adversarial SUPPORT-1 phase-2 review

**HOLD at the reviewed candidate.** Four peer-analyzer issues below need fixes.
The frozen training/evaluation code, phase-0 preps and phase-1 outcome remain
approved. These analysis fixes can stay outside CODE_FILES; no GPU recalibration
or regenerated prep is needed. Repin phase 2 to the commit containing the fixes.

Reviewed launcher: `487d4cf9a756ecb8513b2c7a23dbe06f4916e7a8`,
sha256 `cd7763fbd40eb7e946e223c945f01d2141d536aa34b431cdef5aae5cebfd03fe`,
PIN `fa9efe1c0ab7ffe68c00a90f82eb89dfe3a752d6`.
Archive source: `eefc37ca2b75f6bcfe254ad5f453afb0ae7697f9`.
The analysis code reviewed is scripts/support1_analyze.py at that PIN; Sol's
independent implementation was frozen at 232d569 before this cross-review.

## Required analyzer fixes

1. **Base-denominator identity is not compared.** load_eval only checks that each
   identity looks like a hash. twins does not require equality. On a fully bound,
   otherwise-valid four-seed synthetic matrix, changing U4's merged base hash, or
   changing both U4 twins to a foreign reference, still yields
   SUPPORT1_FUNCTIONAL_AND_GROUNDED with no invalid arms. The frozen integrity rule
   requires the same fp32 base denominator. Compare the twins and their N reference
   (and the calibrated reference if available). Keep invalidation scoped: bad U4
   invalidates the primary claim; bad G4 disables mechanism; bad G1 disables its
   comparisons while preserving P/U4 assessment. Six tests cover split twins and
   shifted references for U4, G1 and G4.
2. **Reading guards are weaker than D1/D4.** guard() omits FIRST acquisition from
   D1 and the <=1.05 per-seed competence ceiling from D4. Reproducers show primary
   D1=false or D4=false while readings.U4.guard remains "ok". The FIRST case keeps
   identical POS/FIRST prompts consistent and gives P the same new-cohort weakness;
   POS minimum/relative requirements still pass. The competence case has one seed
   at 1.06 and three at 1.00, pooled 1.015: mean passes, per-seed guard fails. Reuse
   the complete arm-specific criteria instead of maintaining a reduced copy.
3. **Paired construction contrast is row-weighted.** SEL-Q can sample both
   wordings of a cell, so averaging rows double-weights some cells. The contract
   says equal cell weight. A valid synthetic case gives peer 0.3333333333 versus
   independent cell-weighted 0.2545454545; with ACQ=1 this flips the >=0.30
   CONSTRUCTION_GAP interpretation. Group Question rows by rendered cell, average
   within each cell, compare to its Statement twin, then average cells and seeds.
4. **Invalid P can crash instead of returning INVALID.** Once P is invalid, every
   reading becomes {assessable:false}; dose/grounding still index missing fields.
   One inconsistent P outcome produces KeyError: SEL_Q_unknown. Return the invalid
   result with dependent comparisons NOT ASSESSABLE, preserving the report.

The legitimate-zero/missing-value correction is correct. Both primary analyzers
agree on D1-D4, functional and mechanism on the valid full matrix. The issues above
are counterexamples beyond that baseline, not disagreements on experimental data.

Reproduce as ordinary failures:

```bash
python -m pytest -q tests/test_support1_peer_review.py --runxfail --tb=short
```

The valid baseline passes and ten counterexamples fail. Strict xfail marks label
the known issues until fixed; remove them when the corresponding checks pass.
No experimental answers, prep labels or frozen thresholds were changed.

A reporting gap also remains: the fork result reports FORK-H Statement outcomes
but omits the withheld FORK-H-Q reading. Both constructions are evaluated; expose
both per-seed observed/withheld rates and denominators, as the independent output
does. This requires no additional evaluations.

## Launcher and preflight

- Launcher bytes and PIN-only diff from fa9efe1 verify. Outer, BOOT and JOB syntax
  all pass bash -n. CODE_HASH remains
  `38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae`.
- Both frozen manifests, validation-copy/prep coverage, model identity, nonempty
  equal GPU names, calcheck result and chosen fork+64 plan are consistent. The
  card is fixed to NVIDIA RTX 6000 Ada Generation, matching phase 1.
- Training passes the correct validator, calibration result and seed-24 prep.
  The four-seed loops produce 236 evaluations and 108 LoRA checkpoints. File names
  match both analyzers, including root/per-seed byte-identical prep/train copies.
- Sol's invocation passes the frozen phase-0/1 directories and the selected plan.
  Its nonzero analysis status is printed and does not discard the acquired data.
- The phase-1/phase-2 boot, cleanup and exactly-once regression groups pass.
  The 18-hour job cap and preserve-before-remove contract remain present.

The extracted JOB is also exercised locally with model/GPU commands stubbed:
normal matrix, Sol analysis failure, and wrong GPU. No RunPod, SSH, keys or actual
model/GPU execution is used. This checks command arguments, file counts, early
GPU STOP and result-bundle preservation. The transport tests cover actual cleanup
control flow with mocks.

If Claude's analyzer raises a runtime error, the current job bundles via fin 1
before invoking Sol. Running both analyzers unconditionally would improve
forensics; the raw data are still preserved, so this is not a data-acquisition
blocker. A job exit of zero after Sol fails is execution completion, not a valid
scientific verdict; both offline reports still require review.

## Archive and capacity

The previously unavailable archive now independently hashes to
`3e589fb85c05267185825ac01cb66607fea0996bd86cf2247e46519941d5712b`.
All nine archived payload files (manifest, calibration/train/evals, restoration,
timings, projection) are byte-identical to their committed copies. Phase-1 GO is
unchanged; the archive-checksum limitation is resolved.

108 x roughly 175 MB is about 19 GB transferred. preserve() keeps both the tar and
extracted checkpoints locally, so allow roughly 38 GB temporary destination space,
plus the smaller result/log archives. The recorded $9.84 cumulative projection is
conditional on the recorded $0.86/h accounting rate.

## Tests and scope

All 81 existing fast tests pass after restoring the CPU test dependency. One of
Sol's old tests assumed the archive was absent; it now checks the actual presence
and verified checksum. The separate peer review has one passing valid baseline
and ten expected failures, confirmed as actual failures with --runxfail.

This is a software/pinned-artifact review. Phase 2 stays HOLD until the peer fixes
and the newly pinned files are reviewed; David decides and launches.
