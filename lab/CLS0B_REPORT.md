# CLS-0B: one-adapter-per-episode memory over five lives (recorded 2026-10-29)

Reviewed by Sol. Development evidence: seeds 2–5, five lives, 52 episodes per seed. This is not a sealed test.

## Validity
- **Gates:** all passed: complete, not_test, design, config, one_commit (203653c), model_files, prefix, seeds_unique.
- **Consistency:** the external-embedder manifest, model manifest and config hash are identical across seeds 2–5.
- **Provenance:**
  - Final bundle SHA-256 `63a40ccb…`, archived in `results/cls0b_final/` with `FILES.sha256`.
  - Seed-2 files are byte-identical to the pilot (`294c9f…` json, `3e46b8…` bank).
  - The frozen analyzer was run unchanged.
- **Operations (not science):**
  - **Seed 3 took three invocations.** The pilot preflight stopped under two-worker contention, and one pod
    continuation was interrupted. Seed 3 was then resumed from its bank; recorded write outcomes are terminal.
  - **Seed-3 write time.** Its `write_seconds` (351 s) covers only the last invocation; its total write time is
    `throughput.seconds` = 5,143 s.
  - **Two continuation-orchestrator defects were fixed before the final run.** Both fixes are covered by
    `tests/test_cls0b_continue_resume.sh`.
    - The backup was selected by modification time (Sol).
    - `cp -p` is refused on the network volume; that run failed closed before the worker started.

## Verdict (frozen rules, K1, life 5, pooled)
**"2 addressing becomes the priority."**
- Measured: route 0.99, iRAG 0.99, raw 0.99, paraphrase iRAG 0.9125, coverage 1.0.
- False activation was 0.406 against a limit of 0.15.
- No qualifiers fired.
- Note: neural episode storage reproduces the usable information of text storage.

## Findings (life 5, seeds 2/3/4/5 → pooled)

### Storage
- **Exact reconstruction:** 52/52 episodes in every seed (208/208), with 0 prefix failures and no failed writes.
- **Masks:** 78 → 94 → 113.
- **Retention and binding:** life-1 retention was 1.0 at every checkpoint in every seed, and binding was 4/4
  everywhere.
- **References:** none 1–2/50, oracle 50/50.

### Recall (K1)
| Measure | Seeds 2/3/4/5 | Pooled |
|---|---|---|
| Template route | 48/50/50/50 | 198/200 |
| Paraphrase iRAG | 90/95/96/84 | 365/400 |
| Direct, no context | — | 6/200 |
| Temporal | 3/2/2/3 | 10/12 |

- iRAG equals raw on every count.
- Both temporal misses are the unqualified latest-state probe (seeds 3 and 4); the life-1 and life-3 items are
  always correct (Sol).
- Recall is adapter-based episode reconstruction followed by in-context reading. The facts did not become direct
  model knowledge.

### Inappropriate retrieval (K1)
- **Unseen people:** 1/1/3/2 → 7/80 (8.75%). This matches the ≤10% calibration, which used unseen people only.
- **Near-miss** (a familiar person asked about an attribute never observed for them): 13/17/15/13 → 58/80 (72.5%).

### Key ablations (pooled)
| Key | Route | Near-miss | Reading |
|---|---|---|---|
| K3 TF-IDF | 200/200 | 80/80 | Exact matching is not relevance. |
| K2 internal L24 | 48/200 | 6/80 | Low only because it barely routes. |
| K4 permuted | 5/200 | 58/80 | Identical to K1 by construction (threshold on the max score); its route shows the key–episode pairing matters. |

### Cost
- **Writing:** ≈ 124 s per episode on an A40 (6,207 / 5,143 / 7,048 / 7,437 s per seed).
- **Evaluation:** ≈ 20 min per seed.

## Interpretation (Claude + Sol)
- **Three problems to keep separate:**
  - **Preservation:** strong positive evidence within the tested conditions.
  - **Relevance:** the system cannot tell a *related* memory from *supporting evidence*.
  - **Integration:** untested, and not achieved (direct answering 6/200).
- **Leading hypothesis (not yet proven):**
  - Each episode is one fact, and its text repeats the entity name four times.
  - A single paragraph-level embedding is therefore dominated by the entity.
  - The threshold was calibrated only on unknown people, so a familiar person with an unobserved attribute clears it.

## Decision
- **CLS-0B status:** frozen as a successful storage experiment with a failed abstention gate.
- **Next experiment:** CLS-0C, evidence-gated addressing. It reuses the four existing banks and writes no new adapters
  for development.
  - **Phase A:** a read-only diagnostic, logging the top-5 candidates for every cue (entity, attribute, similarity,
    time) and the score distributions.
  - **Phase B:** compare three remedies:
    - hard-negative calibration;
    - a candidate-plus-evidence verifier;
    - a structured index built from experienced text only.
  - **Phase C:** a sealed test.
- **Phase A must answer three questions:**
  1. Of the 58 activated near-misses, how many retrieve the same entity with a different attribute?
  2. How much do the positive and near-miss similarity distributions overlap, pooled and per seed?
  3. Does any single K1 threshold give ≥95% positive recall with ≤10% near-miss activation?

  It also runs entity-swapped and attribute-swapped probes. If no threshold meets question 3, global-threshold tuning
  is eliminated as the primary remedy.
- **Answer contract for CLS-0C:**
  - **SUPPORTED:** a value, plus the supporting episode ID and an evidence span.
  - **UNSUPPORTED:** ABSTAIN.

  Supported candidates come only from retrieved experiences — never from gold answers, hidden labels or the 50-value
  pool. Unsupported-answer rate is scored separately from activation.
- **Sealed substrate (agreed):**
  - The sealed addressing test uses raw episodic text.
  - One prespecified sealed seed runs on both raw text and a new adapter bank, on identical questions, as the
    substrate bridge.
- **Stopping rule (agreed):**
  - CLS-0C is time-boxed: Phase A, then Phase B on development data, then one sealed campaign, then a return to
    consolidation.
  - A structured index may be adopted if it also passes the narrative and held-out paraphrase requirements.
  - If no method passes, CLS-0C closes with unreliable abstention recorded as an open limitation.
