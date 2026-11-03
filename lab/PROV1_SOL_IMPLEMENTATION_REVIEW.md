# Sol: PROV-1 implementation review at b88ca93

**HOLD for two reproduced resume/cache defects. Prep reconstruction passes.**
Reviewed `b88ca93ff56d875c6c7ce0812a2ab818cc1982f9`; no producer or other
CODE_FILES were edited. SUPPORT-1 CODE_HASH remains
`38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae`;
PROV-1 CODE_HASH is
`c082864035787abf1b106e73a68aad3e2278c656e03957ab6cf414567a33f35a`.

## Required producer fixes

1. **Evaluation cache identity is incomplete** (`scripts/prov1.py:380`).
   Cache provenance does not include arm, sleep, history, or merge mode. Reusing
   the output path while changing any of those returns the old evaluation with
   exit success. Four independent counterexamples reproduce this. An additional
   replay with the actual CPU end-to-end output requests unmerged and returns
   the existing merged evaluation (819 rows), without executing a forward pass.
   Include the full request identity and verify the cached header and exact
   requested probe inventory before reuse; recompute or refuse a mismatch.

2. **Training cache does not verify required fork artifacts**
   (`scripts/prov1.py:311`). With `--fork --sleeps 4`, delete or corrupt
   `cortex_RC_s4_withheld.pt` after a completed run: training still reports
   "up to date" because only main checkpoints are verified. Check every requested
   withheld checkpoint's existence/hash and the corresponding completed fork-log
   inventory and parent identity before skipping.

`tests/test_prov1_sol_review.py` contains six strict-xfail reproducers, to turn
into ordinary tests when fixed. With `--runxfail`, all six fail at b88ca93; these
are observed defects, not speculative risks. No launcher, external machine,
network service or GPU is exercised by the reproducers.

## Accepted implementation choices

- Exclude update cells as well as all four A REV cells from counterfactual
  training. This avoids direct counterfactual supervision of the UPD reading.
- Accept the explicit cross-history input-token tolerance of 1%, while item and
  loss-token counts remain exact and RU/RC budgets match exactly within each
  history. This is not a claim of exact token matching between histories.
- Independent reconstruction checks all branch records, including changes to
  non-target H questions caused by withholding the target fact.

## Independent verification

- Producer tests: **12 passed**, including the actual tiny-model CPU
  prep/train/all-arms/fork/evaluation plumbing test.
- Five production preps regenerated using the pinned local Qwen3-8B tokenizer,
  model manifest `69d069331c87531e7aa054e70085cef25752b3e6e3614b03e851a2e997a4b85b`.
- Independent prep validator: **PASS, 126,343 checks, zero errors**, seeds 20–24.
  It reconstructs all schedules, counterfactual cycler draws, shared interleaves,
  exact token IDs/loss tails/background chunks, per-history records, corruption
  conditions and truth/compliance targets. Frozen SUPPORT-1 preps are hash-bound
  historical inputs; no producer modules are imported by the independent tools.
- Independent tests: **33 passed, six expected failures** (the defects above).
  Tests include prep mutations, synthetic complete phase-A and four-seed phase-B
  matrices, grading/inventory/provenance/monitor/denominator attacks, cell-weighted
  K5, and an empty required H denominator. Synthetic perfect scores are plumbing
  evidence, not model capability evidence.
- The independent evaluation reader also accepts the real tiny-model RC merged,
  RC unmerged and N sleep-2 files (819 rows each), and the phase-A P file
  (630 rows), with independent truth/compliance grading and no errors. The actual
  merged/unmerged pair matches.

The full producer fast-suite claim of 247 was not independently rerun here.
The focused producer and independent suites above were run.

## Independent tools and remaining interface decisions

`scripts/prov1_sol_validate.py` is ready for the current prep contract. Its report
`PROV1_SOL_PREP_VALIDATION.json` binds the five regenerated prep hashes. It is
prep approval only. Any producer fix changes CODE_HASH and requires regenerated
preps and a fresh report before freezing phase 0.

```bash
python scripts/prov1_sol_validate.py \
  --prep /path/to/prov1_prep_seed{20,21,22,23,24}.json \
  --support1-phase0 results/support1/phase0 \
  --tokenizer /path/to/pinned/tokenizer \
  --model-manifest-sha 69d069331c87531e7aa054e70085cef25752b3e6e3614b03e851a2e997a4b85b \
  --out /path/to/validation.json
```

`scripts/prov1_sol_analyze.py` is an **independent analyzer draft**, not a
production gate. It reads both phases and calculates per-seed/pooled fractions,
K1–K7, the paired K5 intervention, common-eligible RU/RC results, P-GATE and fork
readouts. Its explicit request JSON specifies prep/train/evaluation/reference
paths. It records all input hashes. Production CLI execution returns nonzero
and every report says `production_ready: false` while these contracts are pending:

1. **K6 construction scope:** recommend requiring both FORK-H (Statement) and
   FORK-H-Q (Question) to meet the three-of-four-seeds criterion, with each
   construction reported separately. Current text does not choose explicitly.
2. **K7 sleep weighting:** recommend equal weight to sleeps 2–5 within seed,
   then equal weight to seeds. K5 separately retains §9's cell×sleep weighting.
3. **Production bindings:** freeze the new calibration/validation artifacts and
   phase-A regeneration/admission report schema. A matching live full-state hash
   alone must not admit a saved file: bind the reloaded-file tensor check too.
   The analyzer draft currently admits only exact checkpoint hashes.

The choices above are exposed in `draft_choices`, not silently presented as
frozen protocol. Production baseline loading must also bind the historical
P evaluations to the approved SUPPORT-1 result bundle; the draft request API
does not yet supply that immutable archive binding.

No RunPod, SSH, David's devices, keys or launchers were accessed or operated.
