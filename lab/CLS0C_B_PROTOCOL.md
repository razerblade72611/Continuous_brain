# CLS-0C Phase B: evidence-aware addressing, development comparison (frozen 2026-10-29, Claude + Sol)

## Purpose
Phase A showed that K1 usually finds the right episode, but its similarity score cannot tell *related* from
*supporting*:
- 54 of 58 false activations retrieve the same person.
- No single threshold keeps reworded-question recall.

Phase B compares two ways of making the final support decision explicit. A frozen K1 baseline uses the same answer
contract. The best method goes to one sealed test; after that comes consolidation.

## Data and restrictions
- **Development only:** CLS-0B seeds 2–5 (committed `results/cls0b_final`). Episode texts are sha-checked.
- **No training:** no new adapters and no memory writes.
- **Substrate:** raw experienced text, because reconstruction was exact 208/208 and iRAG equalled raw on every count.
- **Cues:** the CLS-0B cues, rebuilt with the frozen `build_cues` and checked against the recorded `n_cues`.
  - Facts: template + 2 paraphrases.
  - Negatives: 20 unseen people + 20 near-misses. Near-misses are counted only once the person has been met.
  - Temporal: life-1, life-3 and latest items.
- **Paraphrase labels:** `lab/CLS0C_PARAPHRASE_AUDIT.md` (14 entailed / 2 borderline / 4 not entailed).
- **Chronology:** at checkpoint L, every method sees only episodes experienced by life L, including B2's index entries.
  Life 5 is primary; all lives are recorded.
- **Prohibited inputs:** no method sees the hidden `entity`/`attr`/`value` fields, the evaluator's 50-value pools, the
  list of ten attributes, or any gold label. Only the evaluator uses them.
- **Reader and embedder:**
  - **Reader:** the same frozen Qwen3-8B as CLS-0B, bf16, greedy, plain-text prompts, no chat template.
  - **Embedder:** the same Qwen3-Embedding-0.6B. Both manifests must equal those CLS-0B recorded.

## Memory presentation (all methods)
`Memory (from life {t}): {text}`. Episode time is experience metadata the individual legitimately has.

## Methods
All prompts are the constants in `scripts/cls0c_b.py` and are frozen.

### B0: frozen K1 baseline, under the answer contract
- Route exactly as CLS-0B (K1, threshold from the calibration negatives, top-1).
- If routed, the reader extracts the value from that memory. There is no verification; this is how the CLS-0B system
  commits.

### B1: candidates, then verification
- **Candidates:** K1 top-5 within the bank, with no threshold.
- **Candidate-set rule (frozen):** if pooled life-5 K1 recall@5 on ENTAILED paraphrases is below 0.98, the selectable
  B1 uses the K1 top-5 ∪ K3 top-5 union. Both variants are always run and reported. The rule uses retrieval recall
  only, never verifier outcomes.
- **Verification:** for each candidate, the `VERIFY` prompt compares the log-probabilities of " yes" and " no" on the
  first answer token. The candidate is supported iff yes > no. No threshold is tuned, and these are not treated as
  calibrated probabilities.
- **Selection:**
  1. Temporal rule (below) among the supported candidates.
  2. Otherwise, the largest yes − no margin.
- **Answer:** the reader extracts the value from the chosen memory (`EXTRACT_VALUE`). If no candidate is supported, or
  extraction fails the evidence check, B1 abstains.

### B2: open-vocabulary structured index
- **Index:** the reader extracts `subject | relation | value` once per episode from the experienced text
  (`EXTRACT_RECORD`). The vocabulary is open; no attribute list is given.
- **Query parse:** the reader parses each cue into `subject | relation` (`PARSE_QUERY`).
- **Match:** normalised subject equality AND relation match.
  - **B2:** exact normalised relation string.
  - **B2e:** cosine ≥ 0.85 between entity-free relation-string embeddings (embedder, no instruction). τ = 0.85 is fixed
    a priori, and a τ sweep is reported descriptively only.
- **Then:** the temporal rule, then the record value. No match means abstain.
- **Closed-schema reference (descriptive upper bound, never selectable):** B2c gives the reader the ten attribute names
  as the relation vocabulary.

### Temporal rule (B1, B2, B2e)
- A cue that begins with "At life N," selects the supported episode experienced at life N.
- An unqualified cue selects the most recent supported episode available at the checkpoint.
- "At life N," is the probe format of this workload. Narrative time expressions are a sealed-test concern.

## Answer contract and scoring (evaluator side)
- **SUPPORTED:** a value, an episode ID and an evidence span.
  - The span is the sentence of that episode's text that contains the value.
  - The value must occur verbatim (case-insensitive) in the episode, otherwise the method abstains (fail closed).
- **ABSTAIN:** no adequately supporting experience.
- **A positive is correct iff all three hold:**
  - the normalised value equals the stored value (lowercase, punctuation stripped, leading articles removed);
  - the episode states that value;
  - the span is verbatim in that episode.
  - Temporal positives additionally require the episode of the requested time (latest = the most recent at the
    checkpoint).
- **Abstaining on an entailed positive is a miss.**
- **Negatives:** any committed answer is an *unsupported answer*. B1's candidate-level retrieval of irrelevant episodes
  is reported descriptively and is not a failure.

## Selection on development data (life 5, pooled; frozen)
- **A method PASSES if all hold:**
  - template accuracy ≥ 0.95;
  - ENTAILED-paraphrase accuracy ≥ 0.90;
  - near-miss unsupported-answer rate ≤ 0.10;
  - unseen-person unsupported-answer rate ≤ 0.10;
  - correct provenance ≥ 0.95 among answered positives.
- **Selectable methods:** B1 (per the candidate-set rule), B2 and B2e.
- **If B1 and some B2 variant both pass:** B1 is selected, unless that B2 variant uses ≤ 0.5× B1's reader seconds or
  has ≤ 0.5× B1's total errors (misses plus unsupported answers, pooled life 5). If several B2 variants qualify, the
  one with fewer total errors is chosen.
- **If only one passes:** it is selected.
- **If none passes:** Phase B reports the best by total errors, and Claude + Sol decide between one bounded revision
  and closing CLS-0C with the limitation recorded. There is no open-ended tuning.
- **Afterwards:** the selected method is frozen for the sealed test. That test uses new streams, raw text, narrative
  items, and one prespecified adapter-bridge seed.

## Cost control
- **Preflight:** the real reader latency is timed per call type, and unique calls are counted after caching across
  lives. If the projected time exceeds 2.5 h, the run stops before the main work (exit 4).
- **Pod:** one A40, with a 4 h cap in the launcher.
