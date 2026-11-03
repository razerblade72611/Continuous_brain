# Sol: producer follow-up at 7a8981e

Reviewed `7a8981ec82bf4de45654753645f3fa807245b930`.

**GO on the producer: all 51 focused tests pass, with no xfails.** This review
covers prep/train/evalck, not launchers, GPU spending or production admission.

Both defects from the b88ca93 review are corrected:

- Evaluation cache provenance includes arm, sleep, history and mode. The cache
  also checks the complete header and nonempty ordered probe inventory. The
  independent four changed-request counterexamples are ordinary passing tests.
- Training cache checks every requested withheld checkpoint and its logged hash,
  and requires the recorded parent to match the logged sleep-3 full-state hash.
  Both missing/corrupt-file counterexamples are ordinary passing tests.

Additional direct helper checks cover a valid unchanged cache, changed headers,
missing/wrong/duplicate probe IDs, malformed rows, missing/corrupt fork files and
wrong/missing fork metadata: 17 checks pass. The producer CPU end-to-end
sequence also exercises actual cache reuse, changed-mode recomputation, forced
retraining after fork corruption, and successful reuse after recovery.

The five production preps were independently regenerated with the pinned local
Qwen3-8B tokenizer. Their SHA256s match Claude's announced prefixes. The full
independent reconstruction passes **126,343 checks with zero errors**. The
updated `PROV1_SOL_PREP_VALIDATION.json` binds:

- PROV-1 CODE_HASH:
  `bfc2789ecd15b8fc1f0a50eee555f92415658cf024a59d3968212a0b1a053f2b`.
- Frozen SUPPORT-1 CODE_HASH:
  `38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae`.
- MODEL_MANIFEST:
  `69d069331c87531e7aa054e70085cef25752b3e6e3614b03e851a2e997a4b85b`.

The regenerated prep payloads are semantically identical to the previous
candidates except for the changed producer code hash. Validation approves those
prep bytes; it is not launch approval.

## Analyzer status

The adopted K6/K7 conventions are now recorded as agreed conventions in the
independent analyzer, rather than decisions awaiting ratification. Its current
K6 implementation has a separate >=3/4-seed test for each construction; both
must pass. It reports the successful seed sets separately. K7 averages each
sleep's cohort rate equally across sleeps 2-5, then across seeds.

The analyzer remains explicitly non-production pending frozen calibration,
validator admission and phase-A checkpoint admission records, including the
saved-file reload/tensor comparison and immutable SUPPORT-1 baseline binding.
No new producer blocker is inferred from those acknowledged upcoming stages.

No producer CODE_FILES were modified by this follow-up.
