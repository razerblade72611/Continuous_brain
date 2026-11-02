# Sol's independent SUPPORT-1 tools and review

Reviewed producer and regenerated candidate preps:
`4cabb09e027eb4b92f852244fcfe214df4abb114`. The A1 prep contract remains
`981f0e77c0c5197b30482d7870af3a5b91cdad85`.

**Current status:** the committed phase-1 outcome at
`dbe2f6fd3cefaeb41cd9341715a0e9d8bb8f4c22` is independently GO. Phase 2 remains
HOLD for its launcher and analysis review. See `SUPPORT1_SOL_PHASE1_OUTCOME.md`
and `.json` for raw-answer regrading, provenance, restoration and cost evidence.
The tarball is not committed; its compressed SHA has not been recomputed here.

## A1 decision and production prep verification

A fixed fake identity per A entity controls the negative-training repetition
structure more closely than a new fake identity per presentation. Keep the
implemented algorithm: traverse **sorted(A)**, choose the **first unused** name
from shuffled sorted(T_swap) matching all 18 framed question signatures, and use
that one-to-one mapping in both G doses at every sleep. Make sorted(A) and unused
selection explicit in the amendment text. I also accept ATTRS9 split order and
the 100,000 retry cap. Keep that retry algorithm for this run; the cap changes
none of the five accepted splits.

Qualify the claim that experience is the *only* remaining difference: lexical
identity still differs. The controlled contrast concerns whether a negative's
name has **positive stream facts**, with identity repetition and length matched.
The fake identities themselves become familiar through negative supervision.
GTQ is G's acquisition diagnostic, not evidence of a support rule. Its rows are
distinct delivered (cell, wording) pairs, not independent people; the experiment's
unit remains the seed.

I independently downloaded the five tokenizer/config files at revision
`b968826d9c46dd6066d109eabc6255188de91218`, verified each against the archived
CLS-0B manifest, and checked the derived full model-manifest identity against all
five preps. Model weights were not downloaded or re-evaluated. All candidate file
hashes match CANDIDATE.sha256. The amended independent validator reports:

**PASS, 346,082 checks, zero errors, production test_mode=false, seeds 20–24.**

| Seed | Accepted retry | Maximum main-schedule tokens | Distinct swap identities | Sleep-5 GTQ pool / sample |
|---|---:|---:|---:|---:|
| 20 | 212 | 80 | 8 | 58 / 16 |
| 21 | 195 | 79 | 8 | 66 / 16 |
| 22 | 402 | 80 | 8 | 62 / 16 |
| 23 | 939 | 79 | 8 | 62 / 16 |
| 24 | 65 | 80 | 8 | 60 / 16 |

The GTQ pool contains 20 distinct pairs already at sleep 1, so all five sleeps in
all seeds retain the full 16-row GTQ allocation. I regenerated all five production
preps with the verified tokenizer: every output is byte-identical to its candidate.

The complete guard-compatible output is
`lab/SUPPORT1_SOL_PHASE0_VALIDATION.json`. The identity checks, regeneration
hashes and test evidence are recorded in `lab/SUPPORT1_SOL_VALIDATION.json`.
The phase-0 full-plan projection is 20.86 hours / $17.94 at the stated $0.86/h,
including the protocol's phase-1 allowance. This uses the inherited rates, not
measured SUPPORT-1 throughput; phase 1 must replace the estimates.

## Calibration follow-up: resolved at 4cabb09

The original reduced-inventory/provenance attack and all five subsequent
integrity attacks are rejected with exit 4 and errors. The five expected-failure
markers have been removed. A boolean monitor case is also rejected. The gate
requires valid non-null checkpoint/base-denominator hashes, positive finite
non-boolean ratios, and TF32 explicitly false, in addition to its complete
inventory, identity, provenance, independent grading and twin checks.

A complete, correctly bound oracle-perfect synthetic control passes before the
mutations. These are recorded-artifact validation tests; the actual CPU tiny
end-to-end test exercises the train/eval/calibration plumbing as well.

The independent producer rerun reproduces the new candidate bytes for all five
seeds. Their semantic contents are identical to the previously approved A1 preps;
only code_hash changed. The new report binds CODE_HASH
`38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae`
and MODEL_MANIFEST
`69d069331c87531e7aa054e70085cef25752b3e6e3614b03e851a2e997a4b85b`.
Any later CODE_FILES edit requires the normal regeneration/revalidation step.

The phase-1 transport groups pass, and disabling the atomic start marker makes
the job regression fail as intended. A new independent projection test covers the
full plan, restoration failure, the no-fork/32-row fallback, and exit 5 when no
plan fits the budget.

A conservative projection discrepancy remains nonblocking: the producer's
fork+32 plan counts all withheld QDIAG rows. Its estimate is higher than the
actual fallback inventory. The independent phase-0 projection uses the subset.

## Independence and coverage

`scripts/support1_sol_spec.py` transcribes the protocol's templates, normaliser,
oracle, CODE_FILES hashing and inherited training frame. Statement stems come
from the previously frozen independent ABST-0 file-format inventory. Neither it
nor the analyzer imports producer modules, torch or transformers. The validator
loads transformers only when a local tokenizer is requested. The producer CLI is
used solely to generate fixtures and to reproduce the calibration-gate bug.

The validator independently reconstructs strata, first acceptable split, T
partition, persistent cyclers, replay deduplication and relabelling, REV quotas,
base-block occurrences, A1 entity swaps with all 18 signatures, interleave, positive identities/positions, QA token
IDs and loss tails, background chunks, token/loss budgets, fork replacements,
per-history probe gold and inventories, QDIAG sampling/eligibility/pairing, and
normalisation vectors. It validates structural stream invariants and its canonical
hash; it does not independently regenerate ABST-0 phonotactics or the stream RNG.
The frozen stream construction and code identity remain additional provenance.

A test-mode prep can receive **TEST_ONLY**, never PASS. The A1 tiny seed-20
fixture passes all applicable checks. Missing tokenizer, external code hash or external model
manifest prevents production PASS. `--require-phase0` additionally requires seeds
20–24 and a complete development-cost projection. The phase-0 projection uses the
protocol's ABST-0 rates; measured SUPPORT-1 phase-1 throughput must replace them.

The stdlib analyzer joins on probe IDs and recomputes every gold and outcome. It
checks train provenance and exposure counters, exact evaluation inventories,
checkpoint identity, GPU identity, normalisation, both competence monitors,
shared base denominators and merged/unmerged twins. Invalid low-dose arms disable
their contrasts without invalidating U4. Invalid P/U4 invalidates the primary
verdict; invalid G4 disables mechanism assessment. Failed restoration records the
secondary fork as unrun without invalidating the main experiment.

Outputs contain per-seed counts/rates, equal-seed pooled rates with all component
fractions, descriptive micro counts, updates, D1–D4, mechanism, GR, construction
pairs, dose, G's own acquisition, and reversal. Construction contrasts weight
cells equally. Reversal acquisition requires 3/4 cells in 3/4 seeds, and success
checks both sleeps 4 and 5. Fork summaries use **seed branch pairs**, separately
for Statement and Question. Missing denominators remain null. Inclusive threshold
comparisons allow only 1e-12 absolute binary-representation error.

The tools cross-check recorded checkpoint/full-state/base-loss identities. They
do not re-execute tensor/RNG equality or model forward passes during analysis;
those require restoration/calibration artifacts. Production prep/model validation
requires the external pinned tokenizer and model-manifest identity.

## Usage

Run from a checkout containing the producer code that generated the preps:

```bash
python scripts/support1_sol_validate.py \
  --prep results/support1/phase0/prep_seed{20,21,22,23,24}.json \
  --tokenizer /path/to/pinned/tokenizer \
  --model-manifest-sha "$MODEL_SHA" --require-phase0 \
  --out results/support1/phase0/sol_validator.json
```

The report retains the agreed guard shape:
`{"verdict":"PASS","prep_sha256":[...]}`, with validator identity, contract,
code/model bindings, test-mode status, per-prep checks/errors and cost projection.

For production analysis, supply `--phase0-dir` and `--phase1-dir`. Phase 0
contains all five preps, validation.json and PHASE0.sha256. Phase 1 contains the
complete committed calibration inputs/reports, restoration, timings, projection
and PHASE1.sha256. Keep both frozen directories for the offline rerun.

Phase-2 input may use `seed20/` through `seed23/`, each containing train_seedS.json
and the protocol's unsuffixed evaluation filenames. Root train logs and root
`eval_ARM_sL_MODE_seedS.json` files remain supported. There are 55 main evaluation
files plus four withheld files per seed for the chosen plan: **236 total**.
Conflicting duplicate copies fail. The output records every evaluated file hash.

The analyzer verifies the frozen validation bytes, regrades calibration from raw
answers, checks calcheck train/eval hashes, and binds phase-2 code, model, GPU and
base denominator to phase 1. The requested QDIAG/fork must match the recorded plan.
Validator and restoration overrides must be byte-identical to the frozen records.
The phase-1 binding and analyzer-source fingerprint are included in the output.

```bash
python scripts/support1_sol_analyze.py /path/to/phase2 \
  --phase0-dir /path/to/phase0 --phase1-dir /path/to/phase1 \
  --code-hash "$CODE_HASH" --model-manifest-sha "$MODEL_SHA" \
  --gpu "NVIDIA RTX 6000 Ada Generation" --qdiag 64 --fork \
  --out /path/to/sol_analysis.json
```

For an independent phase-1 check alone:

```bash
python scripts/support1_sol_analyze.py /path/to/phase1 --phase1-only \
  --phase0-dir /path/to/phase0 \
  --code-hash "$CODE_HASH" --model-manifest-sha "$MODEL_SHA" \
  --out /path/to/sol_phase1_outcome.json
```

`--phase1-bundle` optionally supplies the archive for that check. Omission is
reported explicitly; an explicitly supplied missing/mismatching archive fails.
`--phase0-pin` and `--phase1-manifest-sha` optionally anchor external frozen
identities. The recorded rate and prior spending default to $0.86/h and $0.65;
use the actual accounting inputs when they differ.

Omit `--fork` when the recorded phase-1 plan excludes it. Restoration and
validation reports are loaded from the frozen directories automatically.
`--test-mode` permits plumbing tests and explicitly marks the output as test data;
it does not approve a production prep or scientific finding.

## Verification

See `lab/SUPPORT1_SOL_VALIDATION.json` for the recorded checks. The independent
suite uses oracle-perfect synthetic answers to exercise the whole four-seed
analysis matrix and attacks counts, stale replay, fork leakage, forged gold,
probe omission, pairing, delivery eligibility, token IDs, QDIAG selection,
provenance, nonfinite ratios, denominator identity and invalid-arm propagation.
These synthetic answers have no experimental interpretation.

All 13 producer tests pass at 4cabb09, including the actual slow tiny-model CPU
end-to-end test and the strengthened calibration attacks. The
independent reader accepted its P merged/unmerged and N files (210 rows each),
and its withheld/32-row file (289 rows), with zero twin answer/outcome/ppl
mismatches and restoration admitted. The CPU model's calibration scores are
plumbing only.
