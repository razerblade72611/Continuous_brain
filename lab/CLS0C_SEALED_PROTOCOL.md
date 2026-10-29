# CLS-0C sealed test: independent evidence-generalisation test (frozen 2026-10-29, Claude + Sol)

## What is tested
**B1_K1, frozen** (Phase B; the strict revision was not promoted):
1. K1 top-5 candidates;
2. the frozen Qwen3-8B yes/no verification;
3. the temporal rule;
4. value extraction;
5. the verbatim-evidence contract.

**B0** (frozen K1 routing plus extraction) runs only as a descriptive reference, on the synthetic part only, because
narratives have no calibration negatives.

## Independence boundary
The harness, scorer, sampling rule, prompts, models, gates and this protocol are committed **before any sealed item is
authored**: `scripts/cls0c_sealed.py`, tested only on public fixtures in `tests/test_cls0c_sealed.py`.

Claude operates both Grok sessions and will see the items, but nothing executable may change after this commit. Any
material post-exposure change invalidates the sealed claim.

## Material
### Synthetic part
- **Streams:** fresh seeds 101–104 from the frozen generator (`cls0b.build_episodes`, condition S, no tokenizer
  matching), evaluated at life 5.
- **Per seed:**
  - 50 template probes;
  - near-miss and unseen negatives from the frozen `build_cues` (20 + 20);
  - 3 temporal items.
- **Paraphrases (Grok-authored, adjudicated templates):** 200 SUPPORTED and 150 IMPLICATION questions.
  - Split across seeds as 50/50/50/50 and 38/38/37/37.
  - Each seed's (fact, template) pairs are ordered by a fixed hash and the first n are taken, never by performance.

### Narrative part
- 10 worlds written by Grok A to `lab/CLS0C_SEALED_AUTHOR_SPEC.md`.
- Per world: 10 SUPPORTED, 10 IMPLICATION, 5 NEAR-MISS, 5 UNFAMILIAR and 4 TEMPORAL items, minus any exclusions made
  before sealing.
- Each world is one individual's memory. A TEMPORAL item is asked at its stated checkpoint; every other item at life 5.

### Descriptive challenge
Each SUPPORTED narrative item's natural-language question is run through the frozen statement interface and reported
separately, without a gate.

## Procedure
1. Freeze (this commit).
2. **Grok A authors** the dataset. Raw submissions are kept.
3. **`validate`:** errors are returned to Grok A for correction **before** adjudication. Development-wording duplicates
   are rejected automatically; near-duplicates go to adjudication.
4. **`blind`:** produces the package for Grok B, a fresh session. It has opaque, shuffled IDs, no author labels,
   answers or explanations, and templates not split by kind.
5. **Grok B adjudicates** with the labels SUPPORTED / TEMPORAL / IMPLICATION / NEAR-MISS / UNFAMILIAR / AMBIGUOUS,
   giving answers and evidence refs.
6. **`reconcile`:** disputes go to David as a short list. Unresolved or ambiguous items are excluded, and the exclusions
   are recorded.
7. **`seal`:** writes `sealed.json` and `SEALED.sha256`, committed. No content-driven change after this point.
8. **Run once.** The pod runs `bridge-write` (seed 101, CLS-0B storage config, resumable), then `run` with the bridge
   reconstruction. A `RUN_STARTED` marker forbids a second run.

## Gates (B1_K1)
Each gate applies **separately** to every format in which the category exists. There is no pooling or averaging across
formats or categories.

| Gate | Threshold |
|---|---|
| Template accuracy (synthetic) | ≥ 0.95 |
| SUPPORTED paraphrase accuracy (synthetic) | ≥ 0.90 |
| SUPPORTED narrative accuracy | ≥ 0.90 |
| Near-miss unsupported answers | ≤ 0.10 per format |
| Unfamiliar unsupported answers | ≤ 0.10 per format |
| IMPLICATION commitments (any answer given) | ≤ 0.10 per format |
| Temporal accuracy | ≥ 0.90 per format |
| Provenance among answered positives | ≥ 0.95 per format |

- **Correctness:** an answer is correct only with the normalised gold value, an episode among the gold evidence
  episodes, and a span verbatim in that episode.
- **Abstaining** on a SUPPORTED or TEMPORAL item counts as a miss.

## Preregistered prediction
B1_K1 fails the IMPLICATION gate in both formats. On development data it committed on 89/110 not-entailed paraphrases,
and on 87/110 under the strict wording.

## Adapter bridge (seed 101, synthetic facts only)
- The adapters are written with the CLS-0B storage configuration. The same B1_K1 then runs on the identical seed-101
  questions using the reconstructed text.
- Reported: reconstruction exactness, answer agreement and provenance agreement with the raw-text run, plus the bridge's
  own scores.
- A failed bridge limits any claim about neural storage, but does not change the raw-text verdict.

## Interpretation and what follows (Sol's rule)
- **Pass:** adopt B1_K1 as the evidence-aware controller for this workload and return to consolidation.
- **Fail only the IMPLICATION gate (predicted):** conclude "evidence-aware for missing or unrelated facts, NOT for
  strict entailment".
  - Do not use B1 as an unrestricted factual authority.
  - Consolidation research uses verified evidence or controlled retrieval.
- **Generalisation failure:** document the limitation and return to consolidation with controlled retrieval.
- **Integrity failure** (leakage, chronology, provenance): pause dependent claims until it is repaired.
- **In every case:** no further retrieval optimisation.
