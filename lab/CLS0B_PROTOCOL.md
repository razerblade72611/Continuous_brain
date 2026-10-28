# CLS-0B protocol (frozen before data, 2026-10-28)

Designed by Claude with Sol's requirements, after the CLS-0 resource gate. That gate (b1eaab5) passed: on Qwen3-8B,
3/3 episodes reconstructed exactly at 16.4 GB peak, and the masks needed were 78–94.

The code is authoritative:
- `scripts/cls0b.py` — the runner;
- `scripts/cls0b_analyze.py` — the decision rules;
- `being/cls/episodic.py` and `being/cls/keys.py` — the mechanism;
- tests: `tests/test_cls0b.py`, `tests/test_cls_episodic.py`. The runner was also run end to end on a tiny CPU model
  in test mode.

Development evidence only: seeds 2–5 were used throughout BIND-I1, DECAY-1 and WRITE-1.

## Revisions after Sol's review of aefc0fb (pre-data)
- **K4** is permuted only among memories experienced by each checkpoint, so no future keys are used.
- **Temporal questions are chronological.** Before life 3, only the life-1 state is asked about, and it counts as the
  latest. From life 3, the life-1, life-3 and latest questions are all asked.
- **Raw vs iRAG is no longer a direction.** Exact reconstruction makes raw and iRAG contexts identical, so equality is
  expected when storage works. It now produces a substrate/cost note only. "Raw > iRAG + 0.05" sets the direction
  "investigate reconstruction fidelity".
- **Direct-answer accuracy** is measured end to end, with declined retrievals counted as wrong. Accuracy conditional on
  retrieving the right episode is reported separately.
- **Statuses are separate** for data collection, analysis execution and scientific validity: COMPLETE,
  COLLECTED_INVALID, ANALYSIS_FAILED, FAILED and PREFLIGHT_STOP.
- **Bank and model integrity:**
  - the bank digest covers the episode metadata, the keys and the full storage config;
  - a full model manifest (SHA-256 of all weight and config files, plus the recorded HF revision) is kept, and all
    seeds must match it.
- **Recorded write outcomes are terminal on resume:** failed or censored writes are never retried. A bank saved
  without its log record is recovered from the bank.
- **The scoring cache is keyed on the whole candidate pool.** The 50-value pools are documented as evaluator-only.
- **Calibration threshold fixed:** exactly floor(10% × 40) = 4 calibration negatives may activate (previously 3).
- **Inference preflight:** before any training, each worker times a representative 50-way score and a 100-token
  reconstruction, and records free VRAM with both workers resident. If the projected evaluation time exceeds the budget,
  it stops (PREFLIGHT_STOP).
- **Second operational review (Sol):** the preflight times the real scorer (backbone + lm_head + log-softmax +
  gather). The call count is a conservative upper bound from the evaluation loops (reference prompts once; routed
  readouts at every checkpoint, ignoring the cache), and the evaluation budget is 5,400 s.
- **Resume:** `resume` mode continues an interrupted run. Pod setup never moves `results/cls0b` aside, and a resume
  is refused if the model manifest or storage config differs from the run that wrote the bank.

## Question
Does an independently written, keyed episodic memory preserve and retrieve one individual's experiences across five
lives? Retrieval must work without life labels, decline when nothing relevant is stored, and handle same-attribute
competition.

## Fixed elements
- **Model:** Qwen3-8B, bf16, eager attention; the model's file hashes are recorded.
- **Stream tokenizer:** the original Qwen3-0.6B tokenizer, so the BIND-I1 streams are reproduced exactly. A design
  check against the committed BIND-I1 results is required.
- **Storage:** published mechanism. One masked rank-2 adapter per episode on the layer-10 down_proj, trained from the
  base with no other episode active.
  - **Deviation (disclosed):** the initial mask is 78, measured as the smallest successful size, instead of the
    published 45.
  - **Unchanged:** growth ×1.2, 10 attempts, AdamW lr 0.05, 1,000 epochs per attempt, and exact greedy
    reconstruction as the storage criterion.
  - **Cap:** 1,200 s wall-clock per episode. A capped attempt is recorded as budget-censored, not as a storage failure.
  - **Seeding:** each adapter's initialisation and mask are seeded by (seed, episode id, mask size).
- **Episodes:**
  - **Count:** 52 per seed — all 50 observed facts of BIND-I1 condition S (lives 1–5), plus 2 temporal-diagnostic
    episodes. The temporal pair is one entity and one attribute, with different values at lives 1 and 3.
  - **Text:** the four observed phrasings, in order.
  - **Records:** each has its time, source and a SHA-256 of its text.
  - **Order and persistence:** written chronologically, each exactly once. The bank is saved after every episode, and
    a restarted run resumes from it after a digest check.
- **Retrieval:** these are conditions applied to the same bank, not separate training arms.

  | Key | Method |
  |---|---|
  | K1 | Qwen3-Embedding-0.6B (last-token pooling, L2 norm, instruction prefix on queries) |
  | K2 | Internal: layer-24 MLP input, mean-pooled |
  | K3 | Lexical TF-IDF over experienced text only |
  | K4 | K1 keys permuted across episodes (negative control) |

- **Null decision:** per method and checkpoint, the threshold lets at most 10% of 40 calibration negatives activate a
  memory. Calibration negatives are unobserved people only, and the calibration never sees an answer.
- **Cues:**
  - **Facts:** for every stored fact, the template probe plus two held-out paraphrases.
  - **Test negatives:** 20 unseen people, plus 20 near-misses (a familiar person asked about an attribute never
    observed for them). Near-misses are only counted once that person has been met.
  - **Temporal:** "At life 1/3, …" and the unqualified probe, retrieved top-2.
- **Answer test:** 50-way forced choice. The candidates are the true value plus the other 49 values the individual
  experienced, scored by summed log-probability. Context lines read "Memory from life t: …".
- **Readouts:**
  - **iRAG** (primary): the context is the episode reconstructed from the routed adapter.
  - **raw**: the original experienced text of the routed episode.
  - **direct**: K1 only — the routed adapter is active, with no context.
  - **References:** none (no memory) and oracle (the correct episode's original text).
- **Checkpoints:** after each life, using only episodes experienced by then.

## Decision
K1 at life 5, pooled over seeds. Statements are evaluated in order; the first that fires sets the direction.

| # | Condition | Direction |
|---|---|---|
| 1 | Coverage < 0.95 | Storage failures |
| 2 | Template routing < 0.90, or false activation > 0.15 | Addressing becomes the priority |
| 3 | iRAG template accuracy < 0.80 | Reconstruction-to-reasoning transfer becomes the priority |
| 4 | Raw-text accuracy ≥ iRAG − 0.02 | Prefer direct episodic storage; neural consolidation becomes a separate track |
| 5 | None of the above | Continue toward online consolidation |

**Qualifiers:**
- Paraphrase accuracy lower than template accuracy by more than 0.20.
- The temporal life-1 and life-3 questions not both correct in ≥ 3/4 seeds.
- K1 not beating K4 by ≥ 0.20 in routing accuracy.

**Also reported:**
- per-seed curves;
- retention by age;
- failure split: none retrieved / wrong episode / right episode, wrong answer;
- binding pairs;
- cost: masks, parameters per token, seconds per episode, epochs per second.

## Execution
- **Pilot:** seeds 2–3 first, one worker per seed in parallel on one A40, with an 8 h cap (about $5 at A40 rates).
  - This is operational evidence only.
  - The two workers' measured epochs per second, compared with the single-worker benchmark, decide the worker count
    for seeds 4–5.
- **Seeds 4–5:** run only after an operational review.
- **Spending:** within David's standing GPU limit.
