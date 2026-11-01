# Independent ABST-0 analyzer (Sol)

`scripts/abst0_sol_analyze.py` implements the ABST-0 protocol using Python's standard library. It does not import or execute any experiment module. The requested original tiny end-to-end test is used separately as a black-box fixture generator.

Basis: protocol text at `d6ed50eb355fa4ca454cec593e03815e2c39bcc7`, JSON artifact formats, and the already committed phase-1 hashes. This does not change the launch at `45dc22b`, its PIN `0cf967e`, any frozen label, or any file in CODE_HASH.

## Run on the downloaded phase-2 directory

The input directory contains `seed12` through `seed15`. Each seed directory contains:

- `prep_seedS.json`, `train_seedS.json`, `exposure_seedS.json`;
- `eval_N_sL.json` for L=1..5;
- `eval_ARM_sL_MODE.json` for ARM=P,U,G, L=1..5, MODE=merged,unmerged.

No model weights, GPU, torch, network access or credentials are needed to analyze those JSON files.

```bash
python3 scripts/abst0_sol_analyze.py \
  --dir /path/to/phase2/abst0 \
  --out /path/to/sol_analysis.json
```

To compare with the original analyzer's saved report after the independent computation:

```bash
python3 scripts/abst0_sol_analyze.py \
  --dir /path/to/phase2/abst0 \
  --out /path/to/sol_analysis.json \
  --compare-production /path/to/phase2/abst0/analysis.json
```

Exit codes: 0 for an assessable completed analysis, including NO_ADVANCE; 2 for INVALID; 3 for a comparison discrepancy when otherwise valid. Always read the JSON verdict. A missing, malformed, duplicate-key or nonfinite input fails closed. A schema or reference-report read failure also yields INVALID.

Production defaults require exactly seeds 12–15, the frozen code/model identifiers, the exact committed prep hashes, E=4, the committed calibration-result hash, and each seed's committed Claude audit-label hash recorded in training provenance. `--test-mode --seeds 12` permits the tiny fixture and an empty model-manifest identifier; its output is explicitly marked test mode. The documented tiny-tokenizer truncation exception is allowed only in test mode, with equal delivered budgets still required across arms.

## Output and calculations

- `per_seed[seed][arm][sleep]`: primary and template metrics, full outcome tables by category/kind, A/H/exposed-A/unexposed-A transition cohorts, update outcomes and both competence ratios.
- Every rate has `numerator`, `denominator`, and `value`. Zero denominators produce null, not zero.
- `pooled[arm][sleep]`: equal-weight means over seeds, with every contributing seed fraction preserved. `descriptive_micro` is separately labelled and never used for a criterion. A missing seed denominator makes the primary pooled rate missing.
- Outcome tables jointly report correct, substantive-incorrect, UNKNOWN, empty, answer rate, and selective accuracy. Answered means correct plus substantive-incorrect. Selective accuracy is correct divided by answered.
- Positive primary measures use only entailed paraphrase probes; templates are separately reported. Negatives use their emitted template probes. Familiar A and H transition cohorts must each satisfy D3; A-exposed and A-unexposed cohorts are also reported.
- `integrity[seed:arm:sleep]`: probe identity/count/order, exact answer and stored outcome agreement, absolute competence-ratio deltas, and validation errors. `ok` is the twin check; `valid` also requires all provenance/inventory/gold checks. N gets a base-only validation entry.
- D1–D4 include an explicit list of each requirement, observed value, threshold and pass/fail. Competence requires both monitors, pooled seed mean <=1.02 and every seed <=1.05 at every sleep.
- Mechanism contrasts report G minus U HNM confabulation at every sleep, with per-seed differences. Sleep 5 requires a pooled reduction >=0.30 and strict improvement in at least three seeds.
- Invalid U or P gives INVALID. Invalid G leaves functional success assessable and makes the mechanism NOT_ASSESSABLE. The nested and top-level verdicts always match.
- `input_sha256` records every input file actually consumed, including invalid inputs that could be read. `schema_sha256` identifies the format schema used.

Answers are independently normalized to lowercase ASCII letters/digits with punctuation and spacing removed. Empty normalized answers are empty; normalized `unknown` is UNKNOWN. Gold is reconstructed from acquired stream facts, never from evaluation `value`, `old`, `new`, `outcome` or `measures`. The stored gold and outcome fields must nevertheless agree with that reconstruction. Thus two twins carrying the same forged gold still fail.

Update rows contain the current value plus old/new fields; before the change is acquired, `new` is null. Updates have separate old/new/other/UNKNOWN/empty descriptive counts and never enter D1 or retention.

## Independence and validation boundary

The implementation was written without reading or importing `abst0.py`'s measures, criteria, integrity, or decision implementation. Earlier session work reviewed other parts of the experiment; this is not a claim that this session had never encountered the repository. The original test imports/runs the pipeline as requested; the independent analyzer itself imports standard-library modules only.

`abst0_sol_probe_schema.json` is a fixed lexical format inventory extracted from the generated tiny fixture's N evaluation rows, with entity names replaced by `{e}`. All nine attributes and their emitted template/entailed-paraphrase kinds are represented. No probe-generator or analysis source was copied or imported. The emitted reference format is checked at runtime against this static schema.

The analyzer independently checks positive, retention, first-appearance, familiar/unfamiliar transition, fresh-name and update membership from the prep stream. For HNM/ANM it independently checks the eligible cells, exclusion of known/planned cells, the min(40, eligible) count, uniqueness, wording/kinds, and exact agreement with N's ordered inventory. **It does not regenerate the particular seeded HNM/ANM subset:** the protocol text does not state that sampler's RNG key/order. A coherently substituted eligible subset across N and every twin is outside this check's coverage. The frozen production validator remains a separate check of its exact seeded inventory.

The exposure report's bytes must match the train log and its A-transition cell inventory must be complete. Its `exposed` flags remain the experiment's recorded verifier-inferred proxy; this analyzer does not rerun the verifier. Checkpoint hash references must match train/evaluation provenance. Actual checkpoint bytes and model weights are not inputs, so this JSON analysis does not certify their contents or the underlying numerical execution.

The optional comparison adapter reads a saved production report only after the independent result is complete. The current production report contains pooled rates and integrity, not per-seed count/denominator tables. The adapter explicitly reports this limitation; the independent per-seed numerators/denominators are emitted but are not falsely counted as reference comparisons.

## Validation performed before real phase-2 results

1. `tests/test_abst0_sol_analyze.py`: hand-defined streams and hand-counted rates; unequal seed denominators; complete functional/mechanism success; fewer than three improving seeds; H-cohort failure hidden by aggregate TPOST success; pooled competence failure despite acceptable individual maxima; missing monitors; missing/altered checkpoint references; missing/shared-omitted probes; nonfinite ratios; inconsistent answers/outcomes; altered wording; duplicate JSON keys; shared forged gold; missing exposure; production rejection of test fixtures; and absence of experiment imports.
2. Original `tests/test_abst0.py::test_end_to_end_tiny`: passed on local CPU, 203.16 s. Environment: Python 3.12, torch 2.15.0+cpu, Transformers 5.21.0, tokenizers 0.23.3. An initial attempt with Transformers 4.57.6 failed on tokenizer `token_type_ids`; it was replaced by the experiment's recorded Transformers/tokenizers versions before the successful run. This is format/plumbing validation, not the production GPU environment.
3. The original test leaves G sleep-5 unmerged competence deliberately corrupted by +0.001. The independent analyzer detected it and returned NO_ADVANCE_MECHANISM_NOT_ASSESSABLE. That temporary evaluation was then regenerated from its tiny checkpoint (rather than trusting the evaluator's resume shortcut), giving a clean fixture with all 20 base/twin validation entries valid and no errors.
4. Clean independent result: ABST0_NO_ADVANCE; D1=false, D2=false, D3=false, D4=true. Compared with the original saved report: 460 pooled scalar metrics, 912 outcome rates, 105 integrity fields, five functional/gate fields, mechanism, invalid arms and verdict all match: **1,485 checks, zero differences**.

The initial calculation implementation and format schema were hashed before inspecting the saved production analysis report. Later additions were the report-comparison adapter, a missing general UPD outcome-table presentation, and stricter frozen-artifact provenance checks. No scalar formula or decision threshold was adjusted to match the production report.

Fast tests (no model dependencies):

```bash
python3 -m pytest -q tests/test_abst0_sol_analyze.py
```

Original tiny fixture generation (requires the model-test dependencies above):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
python3 -m pytest -q tests/test_abst0.py::test_end_to_end_tiny \
  --basetemp=/tmp/abst0_sol_tiny
```

The directory is `/tmp/abst0_sol_tiny/test_end_to_end_tiny0/evals`. Its final intentionally corrupted G file should yield a mechanism-not-assessable result. To obtain the clean fixture, move that temporary `eval_G_s5_unmerged.json` aside and rerun `abst0.py evalck` with that fixture's `prep2.json`, model `m`, checkpoint `run/cortex_G_s5.pt`, sleep 5, unmerged mode, and `--test-mode`; then compare to the fixture's original `an.json`.

No real phase-2 outcomes were used to implement or tune this analyzer.
