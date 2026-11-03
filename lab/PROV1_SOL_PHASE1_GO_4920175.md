# Sol: phase-1 GO at 4920175

**GO for pinning the reviewed phase-1 producer/launcher body at
`49201753bcfaafd4353080b8ed5f369e76b99fa8`.** David decides and launches.
Phases A/B remain for their later reviews; this does not approve their production
analysis/admission paths.

## Verified closure and tests

All fourteen original counterexamples now pass as ordinary tests with their
valid controls preserved. The affected PROV-1 suite has **90 passed, zero
xfails**, including actual CPU training/evaluation, cache reuse/recovery,
calibration and both synthetic directory-analysis paths. The independent
calibration reader's additional suite has **10 passed**.

The calcheck fixes enforce non-null checkpoint identities, valid shared base
hashes, numeric non-boolean monitors, complete train identity and one recorded
GPU. Base unit ratios use the declared absolute tolerance 1e-9. Input hashes are
recorded. The independent reader now uses that same tolerance and verifies all
six stored input hashes against the files it independently regrades. A forged
input digest is rejected.

The phase-1 manifest now covers thirteen files, including the raw calibration
train log and five evals, timing log and pilot train log, plus PIN/code/model/GPU.
The phase-1 pilot retains a nonmatching checkpoint; all remaining checkpoints
are streamed and verified. A failed Downloads copy occurs after removal and
does not change success.

The supplied preservation harness passes **7/7**. Deleting either verification
line makes it fail. Additional local, mocked controls show:

- a mounted-E: identity passes;
- the wrong source drive or wrong mount target refuses;
- an output symlink escaping E: refuses;
- failure of the post-removal C: copy still returns the successful job status.

No external infrastructure, real SSH, keys, GPU or David's devices were accessed.

## Refrozen phase 0

- Archive SHA256:
  `42c7021399c170a11e9ce7abfc7fa89dce0aa25a187ea69a800c04a41db3fe17`.
- Exactly five prep members, all payload hashes equal PHASE0.sha256.
- All five local tokenizer/config files match the frozen CLS-0B manifest.
- Independent reconstruction: **PASS, 126,343 checks, zero errors**.
- The complete independent report exactly equals the committed run by Claude.
  `PROV1_SOL_PREP_VALIDATION.json` is refreshed for the current prep hashes.
- PROV-1 CODE_HASH:
  `6793a356b1d1c0e55d1589146cce3ec363b812e6556598f5b5217ac12a6217ab`.
- SUPPORT-1 CODE_HASH remains
  `38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae`.

The reviewed launcher is intentionally UNPINNED. Its current SHA256 is
`85e375264113a98c9ded1daf0c22c3e833ba049e83ec4568e172dd80e779874f`.
Record the final PIN and pinned-file SHA. A PIN-line-only update does not reopen
the reviewed phase-1 job body.

## Deferred A/B analyzer admission findings

The original analyzer counterexamples are fixed. Two additional admission
followups are recorded in `tests/test_prov1_later_admission.py`, explicitly
outside the phase-1 job and outside producer CODE_FILES:

1. The peer loader validates main full-state hashes but does not validate each
   withheld branch's full_state_hash. Removing RC's withheld hash from the valid
   synthetic control still yields ADVANCE.
2. bind_frozen accepts a bare calibration pass flag with current code/prep hash
   even when no PHASE1.sha256 or raw calibration exists and the model field is
   foreign. Before A/B approval, verify the committed phase-1 manifest and raw
   inputs (or equivalently verify the input hashes while independently regrading
   them), together with the full model/GPU identity.

Both followup counterexamples reproduce with --runxfail. They do not block
collecting phase-1 calibration/timing evidence. They should be resolved with
positive admission controls before the A/B analyzer is approved. Their fixes
are outside CODE_FILES and do not require repeating calibration.

The calibration reader adjustment and review artifacts here change no producer
CODE_FILES and leave the refrozen prep/code identities valid.
