# CLS-0C sealed test: report

**Verdict: FAIL.** B1_K1 does not meet the sealed bar. It passes 8 of 11 adequately adjudicated primary gates and
fails three, each under its frozen criterion:

| Failed primary gate | Result | Threshold |
|---|---|---|
| Synthetic implication commitments | 24/150 = 0.16 | ≤ 0.10 |
| Narrative supported recall | 54/91 = 0.593 | ≥ 0.90 |
| Narrative provenance among answered positives | 99/111 = 0.892 | ≥ 0.95 |

- The recall failure survives both post-hoc checks below (a lenient span measure, and excluding W07).
- The provenance failure is a failure of the preregistered evidence-reference contract. 8 of the 12 wrong-episode
  citations contain the gold value, so not every provenance error is shown to be a factual hallucination.
- The two narrative categories declared insufficiently adjudicated before the run (implication, temporal) are reported
  descriptively only.

## What ran
- **One sealed run.**
  - Commit 0d8ac64; sealed.json sha256 `b993ae9c…54bd4` (the bundle's copy is byte-identical to the repo's).
  - The harness is byte-identical to the 2b240fc freeze (checked in launcher preflight). Model and embedder files
    match the CLS-0B manifests (checked in preflight and again by the harness).
  - A40. RUN_STARTED 2026-10-30T03:41:54 (pod time). The run took 420 s, after a bridge write of 6,669 s.
  - Bundle sha256 `110902d5…926fd` (results_cls0c_sealed.tar.gz).
- **Primary:** B1_K1 (K1 top-5 → frozen VERIFY → temporal rule → EXTRACT_VALUE → verbatim evidence contract).
  B0 is descriptive only.
- **Interpretation:** results/cls0c_sealed/VALIDITY_AMENDMENT.md, fixed before the run. The raw harness `passed=false`
  is an engineering output; the verdict below applies the amendment.

## Primary gates (B1_K1)

| Gate | Synthetic | Narrative | Threshold |
|---|---|---|---|
| Template accuracy | **0.99** pass (n 200) | — | ≥ 0.95 |
| SUPPORTED paraphrase | **0.93** pass (n 200) | — | ≥ 0.90 |
| SUPPORTED narrative | — | **0.593 FAIL** (54/91) | ≥ 0.90 |
| Near-miss answered | **0.05** pass (n 80) | **0.00** pass (n 50) | ≤ 0.10 |
| Unfamiliar answered | **0.00** pass (n 80) | **0.00** pass (n 50) | ≤ 0.10 |
| IMPLICATION commitments | **0.16 FAIL** (n 150) | *descriptive* (see below) | ≤ 0.10 |
| Temporal | **1.00** pass (n 12) | *descriptive* (see below) | ≥ 0.90 |
| Provenance (answered positives) | **1.00** pass | **0.892 FAIL** | ≥ 0.95 |

**Preregistered prediction:** "B1_K1 fails implication_commit in both formats."
- **Synthetic:** confirmed (0.16 > 0.10). That is far below the 89/110 committed in development, but it still fails.
- **Narrative:** not evaluable as a primary result (insufficiently adjudicated).

## Descriptive: narrative IMPLICATION and TEMPORAL (cohorts frozen in COHORTS.json)

| Cohort | n | B1_K1 |
|---|---|---|
| IMPLICATION, independently agreed | 42 | commit **0.143** (above 0.10) |
| IMPLICATION, boundary-disputed (author IMPL / Grok B NEAR-MISS) | 41 | commit 0.000 |
| IMPLICATION, author intention (all retained) | 83 | commit 0.072 |
| TEMPORAL, independently agreed | 23 | correct 0.652 |
| TEMPORAL, boundary-disputed | 16 | correct 0.688 |
| TEMPORAL, author intention | 39 | correct 0.667 |

The amendment mattered. Pooling the boundary-disputed items, which behave like near-misses (0 commitments), would
have produced a nominal narrative implication "pass" (0.072). The agreed cohort commits at 0.143.

## Descriptive: where narrative recall fails (scripts/cls0c_sealed_descriptive.py → DESCRIPTIVE.json)

| Narrative SUPPORTED (91) | n |
|---|---|
| Correct | 54 |
| Abstained | 14 |
| Right episode, span boundary wrong (e.g. "sauce that filled the pantry shelves" for "sauce"; "stall" for "cake stall") | 13 |
| Wrong episode | 8 |
| Right episode, other words | 2 |

**Mechanisms**, from each row's candidates and VERIFY margins (supported + temporal together, n = 130):

| Outcome | Cause | n |
|---|---|---|
| Abstention (19) | gold episode retrieved, but VERIFY rejected every candidate | 11 |
| | gold episode not in K1 top-5 | 3 |
| | gold episode retrieved and verified, but extraction or the evidence contract failed | 5 |
| Wrong episode (12) | gold episode also VERIFY-accepted; the most-recent selection chose another | 9 |
| | VERIFY rejected the gold episode and accepted another | 2 |
| | gold episode not retrieved | 1 |

- **Wrong episodes come from the interaction between permissive episode verification and the frozen most-recent
  rule.** 9 of the 12 cite a later life than the gold evidence, and in 9 of the 12 the gold episode was also verified.
  For unqualified stems the frozen rule then prefers the latest verified episode, which sends questions about a
  specific past event to a later episode mentioning the same person or value. Examples: "the person who directed the
  winter play" was answered from "Nia Vale is now the primary director"; "the person hired … to help with the cafe"
  was answered from a later mention.
- **Post-hoc, lenient span** (right episode and the gold value contained in the answer): SUPPORTED 0.714, TEMPORAL
  0.744. **Still far below 0.90**, so the failure is not an exact-match artefact.
- **One world is much worse:** W07 scores 1/8.
  - Its stems end with a trailing space, which validate did not reject. That is a possible authoring artefact.
  - Excluding W07 post hoc: 0.639. Still a fail.
- **Per world, near-miss and unfamiliar are perfect** (0 answers in all 10 worlds). Abstention on absent facts and
  people is solid.
- **Natural-question challenge** (descriptive, out-of-distribution, no gate): the 91 SUPPORTED items asked as ordinary
  questions score 54/91 = 0.593.

## Bridge (adapters, seed 101)
- 52/52 episodes written, none budget-censored; 52/52 reconstructions exact.
- B1_K1 on reconstructions vs raw text: answer agreement **1.00**, provenance agreement **1.00**.
- Bridge scores: template 1.00, paraphrase 0.94, implication commits 0.13, temporal 1.00.

In this tested bridge (52 synthetic seed-101 episodes), adapter storage introduced no corruption: every downstream answer
and citation matched raw text. This does not show that neural storage is lossless for arbitrary narratives. **On this
workload the failures are in addressing and reading, not storage.**

## B0 (descriptive)
On synthetic, B0 commits on 55% of implication items and answers 35% of near-misses, against B1_K1's 16% and 5%.
VERIFY is doing real work. B0 has no narrative rows (no narrative calibration bank), as designed.

## What CLS-0C establishes
1. **On template-like memories (the synthetic format):**
   - B1_K1 generalises to fresh seeds and new paraphrases on recall, abstention, temporal and provenance.
   - The adapter bridge reproduced raw-text behaviour exactly on its 52 episodes.
   - It still fails the synthetic implication gate (0.16).
2. **It is not a general addressing solution.** On naturalistic narrative memories, recall drops to 0.593 (0.714 even
   with a lenient span) and provenance to 0.892.
   - Narrative implication has no primary verdict (insufficiently adjudicated). Descriptively, the agreed cohort
     commits at 0.143 and the author-intention pool at 0.072.
3. **Measured mechanisms of the narrative failure:**
   - VERIFY rejecting retrieved gold episodes (11 abstentions);
   - VERIFY accepting competing episodes, combined with most-recent selection (9 wrong episodes);
   - EXTRACT_VALUE span length (13 supported items);
   - a few retrieval misses (4).
4. **VERIFY plus selection does improve negative rejection over B0** (synthetic near-miss 0.05 vs 0.35; implication
   0.16 vs 0.55). The failure does not erase that.

Reliable episodic storage and reliable autobiographical reading are distinct capabilities. This test supports the
first on its bounded bridge workload and exposes the limits of the second.

## Decision
CLS-0C closes as **FAIL** on its stated bar, as preregistered. Per the agreement before the run, there are no
further B1 prompt experiments under CLS-0C, and the sealed data is not reused for tuning. The sealed material is kept
as an evaluation artefact, not a tuning resource. The diagnosis goes to consolidation as input, and any redesigned
reader needs genuinely new held-out evidence and a new sealed test.

Consolidation should take independently verified evidence, or controlled retrieval, as its input. It should test
whether experience produces lasting changes in the learner's competence, not whether a frozen reader can retrieve
it. Sol approved closure after the five wording corrections in this version.
