# Sol's independent SUPPORT-1 tools and review

Reviewed producer: `014bb209803ae193f0e5feefe3aa1d2622809c55`.
Contract: protocol v2.1, cycler clarification, and implementation clarifications at
`320dc1fd6eb08fae3d2720585af9a5a394158075`.

**Disposition: HOLD before production phase 0 / paid phase 1.** The new branch is
ready for Claude's cross-review. The replay clarification is accepted. Real,
pinned-tokenizer preps have not yet been supplied or approved.

## Findings requiring resolution

1. **Split ordering differs from the literal contract.** §2 specifies
   `atts = sorted(PM attributes of e)`, then shuffle. `split()` instead sorts by
   ATTRS9. On seed 20 the literal algorithm exhausts all 1,000 retries; the
   implementation accepts retry 212. This is a real disagreement before outcomes,
   not an analyzer adjustment to results. Resolve it explicitly by amendment or
   code change, including the STOP policy, before freezing production preps.
2. **Swap shuffle input order differs.** §2.1 says names are sorted before any
   shuffle. `build_schedules()` shuffles T_swap in the partition's existing order.
   The validator uses `sorted(T_swap)` first and rejects all 106 seed-20 swap
   choices in the tiny fixture. Assignment is per base-block occurrence, without
   replacement globally; repeated source/wording pairs can have different names.
   G1 and G4 must share that occurrence mapping. No cached source-pair mapping is
   required or assumed by these tools. Resolve the shuffle-input rule explicitly.
3. **Calibration is not fail-closed on inventory/provenance.** `calcheck` accepts
   two identical files containing only one POS template row and one FRESH row,
   with no provenance, as a passing calibration. It also trusts stored outcomes.
   The reproducer is `test_producer_calcheck_must_reject_reduced_inventory`:

   ```bash
   python -m pytest -q tests/test_support1_sol.py --runxfail -k producer_calcheck
   ```

   It fails at the reviewed producer commit, demonstrating acceptance of the bad
   input. Require the exact prep inventory, seed 24 / sleep 1 / P merged and
   unmerged / N base / observed identities, prep/code/model/checkpoint and
   test-mode bindings, independently graded outcomes, and both finite positive
   competence monitors against the same base denominator. Bind the passing result
   consumed by development training to the current code, model and calibration
   prep, rather than only `pass=true, seed=24`.

The launcher boot, cleanup and exactly-once regression harness passes all three
groups. Its unresolved PIN prevents use now. Before resolving PIN, require the
independent **production** PASS covering all five preps and their projection;
file checksums alone do not establish that CPU validation passed. The actual
chosen card's hourly rate must be supplied to the cost calculation.

A smaller, conservative projection discrepancy: the producer's fork+32 plan counts
all withheld QDIAG rows, whereas evaluation uses the fallback subset. This
inflates that plan's estimate; apply the same fallback selection to withheld rows.

## Independence and coverage

`scripts/support1_sol_spec.py` transcribes the protocol's templates, normaliser,
oracle, CODE_FILES hashing and inherited training frame. Statement stems come
from the previously frozen independent ABST-0 file-format inventory. Neither it
nor the analyzer imports producer modules, torch or transformers. The validator
loads transformers only when a local tokenizer is requested. The producer CLI is
used solely to generate fixtures and to reproduce the calibration-gate bug.

The validator independently reconstructs strata, first acceptable split, T
partition, persistent cyclers, replay deduplication and relabelling, REV quotas,
base-block occurrences, swaps, interleave, positive identities/positions, QA token
IDs and loss tails, background chunks, token/loss budgets, fork replacements,
per-history probe gold and inventories, QDIAG sampling/eligibility/pairing, and
normalisation vectors. It validates structural stream invariants and its canonical
hash; it does not independently regenerate ABST-0 phonotactics or the stream RNG.
The frozen stream construction and code identity remain additional provenance.

A test-mode prep can receive **TEST_ONLY**, never PASS. Under the currently
unresolved ordering contract the tiny seed-20 prep receives FAIL, with exactly the
two findings above. Missing tokenizer, external code hash or external model
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

For analysis, place `prep_seedS.json` and `train_seedS.json` in the input root.
Evaluations may be root files `eval_ARM_sL_MODE_seedS.json` (omit MODE for N), or
`seedS/eval_ARM_sL_MODE.json`. For a one-seed test directory, unsuffixed evaluation
filenames are also accepted. Copying artifacts preserves their original bytes.
Select QDIAG and fork from the decision logged **before** production evaluation;
the analyzer never infers a reduced inventory from missing rows.

```bash
python scripts/support1_sol_analyze.py /path/to/results \
  --code-hash "$CODE_HASH" --model-manifest-sha "$MODEL_SHA" \
  --gpu "$RECORDED_GPU" --validator-report /path/to/sol_validator.json \
  --qdiag 64 --fork --restore-report /path/to/restoretest.json \
  --out /path/to/sol_analysis.json
```

Omit `--fork --restore-report ...` when the fork was not admitted/selected.
`--test-mode` permits plumbing tests and explicitly marks the output as test data;
it does not approve a production prep or scientific finding.

## Verification

See `lab/SUPPORT1_SOL_VALIDATION.json` for the recorded checks. The independent
suite uses oracle-perfect synthetic answers to exercise the whole four-seed
analysis matrix and attacks counts, stale replay, fork leakage, forged gold,
probe omission, pairing, delivery eligibility, token IDs, QDIAG selection,
provenance, nonfinite ratios, denominator identity and invalid-arm propagation.
These synthetic answers have no experimental interpretation.

The producer's actual slow tiny-model CPU end-to-end test also passed. The
independent reader accepted its P merged/unmerged and N files (210 rows each),
and its withheld/32-row file (289 rows), with zero twin answer/outcome/ppl
mismatches and restoration admitted. The CPU model's calibration scores are
plumbing only.
