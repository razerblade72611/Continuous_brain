# CONS-0A run 2 (Amendment 2, post-run-1-amended development rerun): report, 2026-10-31

**Frozen verdict: `CONS0A_INVALID`.** Neither matched pair passed merge validity, so no arm advances. **Even if valid,
nothing would advance:** I-QA passes A1, A2, A3 and A5 but **fails A4** (near-miss confabulation 1.00 at every sleep).
Under the protocol, that sends the work to a separate abstention protocol, not to CONS-0B.

## Run
- **Code and platform:**
  - code 271b39e;
  - portable launcher f0d51a0;
  - NVIDIA A40 in EU-SE-1;
  - torch 2.14.1+cu130, transformers 5.21.0, Python 3.12.3 (full freeze in ENV.txt).
- **Replays:** CB_REPLAY=run1, using run 1's A40 replays (pinned, 52/52 exact; provenance checked).
- **Boot checks:** the 12 tests passed on the pod; GPU feasibility OK; model and tokenizer equal the CLS-0B manifest.
- **Calibration (QA family, seed 6):** E = 4 (template 0.80, competence in band). N's near-miss confabulation was 0.0.
- **QA material:** 115–126 accepted pairs per seed.
- **Bundle:** sha256 efe22ab2…e2f9. All 24 cortex checkpoints are sha256-verified on David's PC
  (~/cb_results/cons0a/ckpt).

## Why INVALID (merge validity, Amendment 2.3; thresholds unchanged)
- **QA pair:**
  - Per-seed answer agreement between in-process and fresh-merged evaluation was 0.90–0.96, below 0.97, in 12 of 16
    checks.
  - Pooled template accuracy differed by 0.025–0.05 (one or two net flips on 40 items), above 0.02.
  - **Anatomy of the disagreements:**
    - most are on near-miss and unfamiliar items, where both models confabulate and differ only in which invented
      string they produce;
    - the rest are same-grade differences on positive items;
    - only 0–2 per evaluation change correctness.
  - Typical case: near-tie continuations of long synthetic names, e.g. "Landveith" vs "Landveiththeir".
- **V pair:** answers agree 100%, but fresh vs in-process ppl differs by 0.016 and 0.034 (S-V sleep 5, seeds 4 and
  5), above 0.01.
- **Likely cause:** bf16 rounding when the fp32 cortex is merged into the bf16 base. This is not established; as Sol
  ruled, a cause is not a reason to declare equivalence. Merge precision needs its own engineering investigation.

## Measures (descriptive; fresh-merged at sleeps 1 and 5, in-process otherwise; 4 seeds pooled; exact match)
| | Sleep 1 | Sleep 2 | Sleep 3 | Sleep 4 | Sleep 5 |
|---|---|---|---|---|---|
| I-QA, new paraphrase | 0.67 | 0.89 | 0.88 | 0.87 | 0.96 |
| I-QA, life-1 paraphrase | 0.67 | 0.88 | 0.94 | 0.94 | **0.96** (0.83 / 1 / 1 / 1) |
| S-QA, new paraphrase | 0.67 | 0.79 | 0.86 | 0.75 | 0.91 |
| S-QA, life-1 paraphrase | 0.67 | 0.02 | 0.04 | 0.00 | **0.00** (0 / 0 / 0 / 0) |
| QA arms, near-miss and unfamiliar confabulation | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| V arms, any positive measure | ~0 | ~0 | ~0 | ~0 | ~0 |
| N | 0 positives, 0 confabulation | | | | |

- **Competence (fresh ppl ratio, mean / max):**
  - I-QA: sleep 1 0.86 / 0.88; sleep 5 0.89 / 0.93 (in band).
  - S-QA: sleep 5 1.13 / 1.20 (out of band).
  - I-V: sleep 5 1.40 / 1.49 eval, 1.82 / 1.91 modern.
  - The I-minus-S contrast (descriptive) is negative for both families from sleep 3 on.
- **Frozen criteria computed by the code** (descriptive: the run is INVALID on merge validity, so this is not "only A4
  failed"):
  - I-QA: A1, A2, A3 and A5 pass; A4 fails.
  - I-V: only A4 passes; it acquires nothing, so it has nothing to confabulate with.

## Reading (development; not a confirmation; same streams as run 1)
1. **Direct acquisition from replayed own experience.**
   - Each life's new facts are recalled at 0.67–0.96 (exact match, audited paraphrases, QA distillation scaffolding).
   - Those figures are for different facts at each sleep, not a rising curve on one cohort (Sol).
   - The longitudinal observation is life-1 recall: 0.67 at sleep 1, then 0.88–0.96 and holding under I-QA.
2. **The retention trajectories are consistent with interleaved replay preventing forgetting,** relative to equally
   budgeted background-prose rehearsal. This is a development hypothesis; the frozen experiment was INVALID.
   - Life-1 recall at sleep 5: I-QA 0.96 vs S-QA 0.00, in 4/4 seeds.
   - Sequential consolidation loses it within one sleep.
   - The comparison is interleaved historical replay vs equally budgeted background-prose rehearsal.
3. **No support boundary.**
   - Every QA-trained model answers every near-miss and every unfamiliar-name question, while the base model abstains
     on all of them.
   - QA-only training has taught an answer-producing policy with no policy for recognising unsupported questions.
   - Misbinding may contribute, but run 2 does not isolate it (Sol).
4. **Verbatim replay alone yields no frame-accessible knowledge.**

## Sol's review (2026-10-31)
- **MERGE-ENG: GO,** as a read-only engineering investigation on the 24 saved checkpoints. No training, and no
  revalidation of CONS-0A. It compares four configurations:
  - unmerged bf16 base + fp32 LoRA (reference);
  - bf16 merge;
  - fp32-merged target projections, with explicit, identical input/output casting;
  - full fp32 unmerged vs merged.
  For each it records raw and correct-answer agreement, competence, pre-generation logit divergence and top-2 margins
  at the first diverging token.
  - Future protocols choose ONE primary direct representation before training: a fresh model with only the cortical
    LoRA active, or a validated merge. Either counts as learned knowledge, as long as no retrieval or context supplies
    the answer.
- **ABST-0: GO for proposal drafting; training on HOLD** until the protocol is frozen.
  - Base mechanism: I-QA.
  - Arms: positive-only replay vs matched-budget experience-derived UNKNOWN supervision, plus a generic-UNKNOWN
    formatting baseline. Positive replay exposures stay equal.
  - "UNKNOWN" means not supported by the checkpoint's own experiences. Negatives come only from an audited index of
    the episodes acquired so far, never from the world generator or evaluator labels.
  - Four dimensions are judged jointly:
    1. known facts are still answered;
    2. unsupported questions are rejected, on held-out entity × attribute cells, fresh unfamiliar names and
       held-out negative constructions;
    3. **UNKNOWN → KNOWN transitions** (deliberate in the streams): a fact unknown at life 2 is learned at life 4;
    4. persistence and competence.
  - A further contrast guards against an "unfamiliar name ⇒ UNKNOWN" shortcut: fresh entities that DO have taught
    facts vs fresh entities with no support.
- **CONS-0B:** HOLD.
- **No GPU campaign to chase a valid CONS-0A.**

## Status
- No CONS-0B.
- Next, before anything is designed, Sol's review of:
  - an abstention protocol (the protocol's A4 route);
  - a separately documented merge-precision investigation (the 0.97 / 0.01 checks were not attainable under bf16
    merging here).
