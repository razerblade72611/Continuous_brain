# Sol phase-1 launcher review

Reviewed producer/launcher: `4cabb09e027eb4b92f852244fcfe214df4abb114`.
Launcher: `scripts/cb_support1_p1.sh`.
Unpinned launcher sha256:
`6087e8ed36affa73b5cb43866ad92fc2f4225d6fbc25b5a8cdcd4e3a4b88c14e`.

**GO for the reviewed launcher body and calibration invocation.** The calibration
HOLD is released. The current PIN is deliberately `__PIN__`; final launch approval
requires the resolved PIN and PHASE0 manifest. David decides and launches.
No RunPod, SSH, keys or David-machine access was used for this review.

## Job and artifact bindings

- Boot checks out the exact PIN, installs the recorded environment, checks the full
  model-file manifest against CLS-0B, and runs the producer/ABST fast tests.
- The job checks all PHASE0 file hashes, CODE_HASH and MODEL_MANIFEST before
  calibration training. The independent CPU PASS covers all five current preps.
- Calibration training is seed 24, **P only, sleep 1**, writing to `$R/cal`.
  Its train log is `$R/cal/train_seed24.json`.
- The merged/unmerged evaluations use `$R/cal/cortex_P_s1.pt`; N is the base model.
  All three use seed-24 prep, the recorded model identity, full 64-row QDIAG and
  observed history. `calcheck --train $R/cal/train_seed24.json` therefore binds to
  the correct run, checkpoint and prep. Its report records the train/eval byte
  hashes. Protocol failure exits 4 and cannot continue to restoration/projection.
- Restoration tests seed-24 U4 state equality. An equality failure records the fork
  as unadmitted; the projection can select a main-only plan. Runtime failure stops
  the job and preserves its evidence.
- Timing denominators count P sleep 1 plus U4 sleeps 1–3 and two sleep-4 executions,
  and all rows in the three calibration evaluations. Projection uses those measured
  wall times and actual development item/probe counts.
- Plan priority matches the declared order: fork+64, no-fork+64, fork+32,
  no-fork+32, subject to restoration admission and remaining budget. No affordable
  plan exits 5. The test suite exercises that STOP and the fallback decisions.
- PHASE1 hashes the calibration, restoration, timings and projection reports, and
  records PIN, CODE_HASH, MODEL_MANIFEST and actual GPU for phase 2.

USD_PER_HR is copied into the job before upload; its value must be the selected
card's actual rate. Boot allowances and row/item scaling remain projections;
phase-1 measurements replace the phase-0 estimates. The known fork+32 row-count
excess is conservative and does not bypass the cost STOP.

## Transport and failure handling

Outer script, embedded BOOT and embedded JOB all pass `bash -n`. The three
retargeted launcher regression groups pass: boot polling; evidence cleanup and
preservation; job upload/start/poll. A syntactically valid mutation replacing the
atomic mkdir condition with true is caught: an existing start marker must not
restart the job. The harness reports a job failure without `command not found`.

Upload hashes are checked before activation. The atomic start marker prevents a
second start after a lost acknowledgement. Missing logs and unreachable polls are
bounded by their existing deadlines. Cleanup distinguishes evidence, confirmed
absence and inspection failure; evidence is downloaded and checksum-verified
before removal, otherwise the pod is stopped and kept.

## Verification

- 39 independent tests pass, with no xfail/XPASS markers. This includes all former
  calibration corruption cases, the new boolean-ratio case, and the budget gate.
- All 13 producer tests pass, including the slow CPU end-to-end run and gate
  attacks; all 13 ABST fast tests pass.
- The independent reader accepts the actual tiny calibration merged/unmerged/N
  inventories (210 rows each) and withheld/32-row inventory (289 rows). Twin
  answers, outcomes and competence ratios match exactly; restoration is admitted.
- All five production preps pass 346,082 independent checks, zero errors. Their
  candidate hashes match, and independent regeneration is byte-identical.

## Final pinned review

The remaining check is mechanical, against the concrete published files:

1. PIN is a resolved 40-hex commit containing the frozen PHASE0 manifest and these
   validated prep bytes. The five CANDIDATE hashes must match the prep entries.
2. CODE_HASH is
   `38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae`
   and MODEL_MANIFEST is
   `69d069331c87531e7aa054e70085cef25752b3e6e3614b03e851a2e997a4b85b`.
3. After replacing only the launcher PIN line with `PIN=__PIN__`, its sha256 equals
   the reviewed launcher hash above. Any other code change is reviewed on its merits.
4. Preserve the independent report alongside the frozen phase-0 record; its
   guard-compatible output is `lab/SUPPORT1_SOL_PHASE0_VALIDATION.json` on Sol's
   branch. Including its checksum in the phase-0 evidence is recommended.

This approves preparation for phase 1, not phase 2 or an experimental finding.
