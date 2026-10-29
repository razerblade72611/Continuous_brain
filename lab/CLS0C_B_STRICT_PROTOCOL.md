# CLS-0C Phase B-strict: one bounded strict-entailment revision (frozen 2026-10-29, Claude + Sol)

## Why
In Phase B, the frozen B1_K1 passed every gate but committed an answer on 89 of 110 NOT ENTAILED paraphrases (80.9%).
It judges topical or plausible support, not strict entailment. Exactly one development revision is authorised before
the independent sealed test (Sol's ruling, option B).

## The single change: B1_STRICT
- **What changes:** only the VERIFY support criterion. The question line of the frozen B1 VERIFY prompt, in the three
  generic examples and in the scored item, is replaced by the frozen strict question:

  `Taken literally, does the memory above establish what completes this statement, without assuming anything the
  memory does not say? Answer yes or no.`

  The examples, their yes/no answers, the memory format and the statement format are unchanged. No example from the
  development failures is added.
- **Unchanged:** the Qwen3-8B reader, K1 top-5 candidates, the yes/no first-token log-probability rule (support iff
  yes > no), EXTRACT_VALUE, the temporal rule, the evidence contract, grading, the data and the chronology.
- **Exactly one prompt:** no alternatives are tried, no threshold is adjusted, and there are no further development runs.

## Promotion rule (deterministic, development seeds 2–5, life 5 pooled)
B1_STRICT is selected for the sealed test iff ALL of the following hold:

| Measure | Requirement |
|---|---|
| NOT ENTAILED commitments (any answer given) | ≤ 30% (≤ 33/110) |
| Template accuracy | ≥ 0.99 |
| ENTAILED paraphrase accuracy | ≥ 0.97 |
| Near-miss unsupported answers | ≤ 0.05 |
| Unseen-person unsupported answers | 0 |
| Temporal | 12/12 |
| Provenance | ≥ 0.95 |

- **Otherwise:** the original B1_K1 goes to the sealed test unchanged.
- **Reporting:** the four NOT ENTAILED forms (allergy p0, allergy p1, favorite object p0, pet species p1) are reported
  individually, so an aggregate cannot hide a systematic failure.
- **Sealed gate unchanged:** unsupported implication commitments ≤ 10% on the sealed test, whichever variant is
  selected. A 30% development rate is not a claim that strict entailment is solved.
- **Control:** if B1_STRICT is selected, the original B1_K1 runs in the sealed test as a descriptive control only.

## Run
- **Script:** `scripts/cls0c_bs.py` reuses the Phase B code (`scripts/cls0c_b.py`) with the strict VERIFY.
- **Scope:** only B1 is evaluated. The original B1_K1 numbers are taken from the committed Phase B summary, which is
  recorded as an input hash.
- **Pod:** one A40, the same launcher pattern and the same preflight (2.5 h budget).
