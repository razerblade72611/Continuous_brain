# MERGE-ENG: merged-vs-unmerged equivalence on the CONS-0A run-2 checkpoints (engineering; 2026-11-01)

**Read-only engineering investigation (Sol GO).** No training. It does **not** revalidate CONS-0A run 2, which stays
INVALID.

## Setup
- **Checkpoints:** the 24 run-2 cortex checkpoints (4 seeds × {QA, V shared sleep 1; I-QA, S-QA, I-V, S-V sleep 5}),
  sha256-verified locally and on the pod.
- **Platform:** NVIDIA A40 with run 2's exact pip freeze. Model and tokenizer equal the CLS-0B manifest.
- **Code:** scripts/merge_eng.py (pinned 48764b2) and scripts/cb_merge_eng.sh.
- **Output:** results/merge_eng/merge_eng.json.
- **Harness validation: perfect.** On all 24 checkpoints, configuration R reproduces run 2's in-process answers exactly
  (1.0), and M16 reproduces run 2's fresh-merged answers exactly (1.0).

## Configurations
| Name | What it computes |
|---|---|
| R | bf16 base + fp32 LoRA (CONS-0A in-process) |
| M16 | production bf16 merge, two roundings (CONS-0A fresh) |
| M16b | bf16 merge with a single rounding |
| C32u / C32m | targeted linears computed in fp32 inside the bf16 model, unmerged / merged, identical casting |
| F32u / F32m | whole model in fp32, unmerged / merged |

## Results
Answer agreement, min / median over checkpoints. "Pass" means the CONS-0A per-seed rule (≥ 0.97 answers, ppl ≤ 0.01).

| Pair | QA: agreement | QA: grade agreement (min) | QA: pass | V: agreement | V: max ppl diff | V: pass |
|---|---|---|---|---|---|---|
| R ~ M16 | 0.898 / 0.956 | 0.988 | 3/12 | 0.982 / 1.000 | 0.034 | 10/12 |
| R ~ M16b | 0.898 / 0.957 | 0.988 | 3/12 | 0.982 / 1.000 | 0.038 | 9/12 |
| C32u ~ C32m | 0.920 / 0.948 | 0.983 | 3/12 | 0.988 / 1.000 | 0.013 | 10/12 |
| **F32u ~ F32m** | **1.000 / 1.000** | **1.000** | **12/12** | **1.000 / 1.000** | **0.0000** | **12/12** |
| R ~ C32u (no merge at all) | 0.920 / 0.945 | 0.988 | 2/12 | 0.982 / 1.000 | 0.044 | 8/12 |

- **Full fp32:** first-token KL between merged and unmerged is ≤ 5e-7. In bf16 it is up to 0.09.
- **Divergences in bf16** occur where the top-2 log-prob margin is 0.125–1.0 nats, in steps of 1/16 to 1/8. That is the
  resolution of bf16 logits, so these are quantised near-ties.

## Conclusions (engineering)
1. **Merging is not the problem; bf16 is.**
   - In full fp32, merged and unmerged models are equivalent on every checkpoint.
   - In bf16, a single-rounding merge is no better than the production merge.
   - Two unmerged bf16-family paths (R vs C32u) disagree as much as merged vs unmerged do.
   - For outputs with many near-ties (confabulated synthetic names), 0.97 exact-string agreement is unattainable in
     bf16, whatever the merge.
2. **Grades are far more stable than strings.** Correctness agreement is ≥ 0.983 in every bf16 pair.
3. **Run 2's descriptive findings are precision-robust.** Pooled, fp32 merged (F32m) vs run 2's bf16:
   - I-QA life-1 at sleep 5: 0.958 vs 0.958 (per seed 0.83 / 1 / 1 / 1).
   - S-QA life-1 at sleep 5: 0.00 vs 0.00 (0 / 0 / 0 / 0).
   - Near-miss and unfamiliar confabulation: 1.00 in both.
   - Competence: within 0.004.
4. **Recommendation for future protocols** (Sol to rule). Fix the primary direct representation before training as
   **fp32 evaluation of the merged model.** Exact equivalence then holds, and the merge check becomes a strict
   integrity check at tolerance 0 that it can actually pass. On an A40 this costs about 4–5 min per fifth-sleep
   checkpoint. Training can stay in bf16 with an fp32 cortex.
