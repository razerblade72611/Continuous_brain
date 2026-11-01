# ABST-0 proposal: learning a support boundary during consolidation (DRAFT for Sol's review; 2026-11-01)

Development protocol. **Training is on HOLD** until Sol approves a frozen protocol and its implementation.

## Background
- **What CONS-0A run 2 showed (descriptively, INVALID; precision-robust under fp32 per MERGE-ENG):**
  - an interleaved QA-distilled cortex (I-QA) acquires each life's facts (0.67–0.96);
  - it retains life-1 facts (0.96 vs 0.00 for sequential consolidation);
  - but it answers 100% of near-miss questions (experienced entity, unobserved attribute) and 100% of questions
    about never-seen people. The base model abstains on all of them.
- **Sol's framing:** positive-only QA training taught an answer-producing policy without a policy for recognising
  unsupported questions.

## Question
Can a cortical learner acquire an **evidence-sensitive answer/abstain decision from its own experienced history**,
while preserving correct recall, retention, and the ability to learn newly encountered facts? The last includes a
fact that was UNKNOWN becoming KNOWN.

**Definition.** "UNKNOWN" means *not supported by the experiences this checkpoint has acquired*. It does not mean
false, and it does not mean the world lacks the value.

## Design

### Streams (new; deliberate structure, not incidental BIND-I1 collisions)
- **Seeds:** 4 development seeds (12–15) plus calibration seed 16. Never used before.
- **Generator:** lifegen's vocabulary, name generator and TRAIN phrasings. Each fact is one episode of its 4
  phrasings, as in CLS-0B.
- **Size:** 5 lives of about 10 facts.
- **Entities:**
  - **Core entities** are introduced in lives 1–2, with one or two attributes.
  - **New entities** (2–3 per life) arrive in every life with taught facts. This is the guard against an "unfamiliar
    name ⇒ UNKNOWN" shortcut, and is measured as first-appearance recall.
- **Planned cells, per seed:**
  - **Transition cells (8):** an entity introduced in life 1–2, with attribute a* first taught in **life 4**. These
    should be UNKNOWN after sleeps 1–3 and KNOWN after sleeps 4–5.
  - **Update cells (4, descriptive):** a value taught in life 2 and changed in life 4. These test that abstention
    training does not replace temporal resolution.
- **Entity split:** core entities are split, seeded, into **A** (abstention-training entities) and **H** (held out
  from all negative training). Transition cells are divided evenly between A and H.
- **Name pools:**
  - fresh unfamiliar names for **evaluation** come from a pool disjoint from all stream names;
  - fresh names used for **training** negatives come from a third, disjoint pool.

### Replay source (OPEN QUESTION 1)
- **Proposed:** replay from each episode's verbatim stream text.
- **Why:** CLS-0B's store reconstructed 208/208 episodes exactly on the A40, so exact reconstruction equals the
  text. The rerun attempt showed reconstruction is hardware-sensitive (H100 42/52). Building and verifying new banks
  for new streams would add A40-only CLS-0B training and verification to every seed.
- **Disclosed deviation:** ABST-0 tests the *cortex's* support boundary given verified experiences. A store-replay
  version is required before any sealed claim.
- **Alternative:** build episodic banks for the new streams with the CLS-0B pipeline on an A40, and fail closed
  unless all reconstructions are exact.

### Arms (all built on I-QA: interleaved QA distillation, as in CONS-0A Amendment 2)
| Arm | Positive slot (identical across arms) | Third slot (equal items, tokens and loss tokens) |
|---|---|---|
| **P** (positive-only) | new QA items × E, plus the interleaved replay slot | background chunks, token-length and loss-tail matched (as CONS-0A's S slot) |
| **U** (experience-grounded UNKNOWN) | same | UNKNOWN items about **A-entities**, derived from the checkpoint's own experiences, plus training-pool fresh names |
| **G** (generic UNKNOWN) | same | the same number of UNKNOWN items, about **training-pool fresh names only** (teaches the format and a familiarity rule, not evidence sensitivity) |
| N | base model | — |

- Positive exposures are identical across P, U and G by construction. Total budget is equal and asserted fail-closed.
- The third slot has size ⌈0.5 × E × |new items|⌉ per sleep (OPEN QUESTION 3: the ratio).
- The arms diverge from sleep 1, because the third slot differs there. They share the cortex initialisation, the QA
  material, the schedule seed and the positive slot's order.

### Experience-grounded negatives (U); a frozen procedure, like QA distillation
At each sleep L, for each A-entity e mentioned in the episodes acquired up to L:
1. **Memory.** The memory is the concatenation of acquired episode texts that contain e's name. This is a
   deterministic, audited string index of the learner's own episodes. It never uses the generator's fact table,
   attribute labels, future episodes or evaluator labels.
2. **Generation.** The frozen base model is prompted with "Memory: … Write three questions about {e} that this memory
   does NOT answer", greedy decoding.
3. **Acceptance** requires all of the following:
   - the question mentions e;
   - the frozen entailment verifier (a separate call, its own few-shot prompt) judges the memory **does not** answer
     it (no > yes);
   - the leak filter passes: token Jaccard < 0.8 with every evaluation stem;
   - at most 3 per entity per sleep.
4. **Target.** The exact frame answer UNKNOWN, with the newline terminator. Loss falls on "UNKNOWN\n".

Further rules:
- **Fresh-name negatives:** questions about training-pool names, from the same generator with no memory, with target
  UNKNOWN. These are used by U and G in the same proportion of U's slot.
- **Regenerated every sleep.** Negatives are regenerated from the memory up to L. So once a transition cell is taught
  (life 4), the verifier sees the new memory and questions about it stop being accepted. Earlier UNKNOWN items about
  that cell are **not** replayed after it becomes supported: the interleaved negative replay is drawn only from the
  current sleep's accepted set.
- **Logged and hashed:** every candidate, rejection and acceptance. For **reporting only**, a post-hoc audit labels
  which accepted negatives touched a later transition cell (using the future episode, after the run).

### Evaluation (the fp32-merged primary representation; Sol's contract in lab/MERGE_ENG_REPORT.md)
- **Every decision-bearing evaluation:**
  - fresh process;
  - an actual merged model, adapter computation removed;
  - full fp32, with autocast and TF32 disabled;
  - frozen decoding (CONS-0A's frame, greedy, ≤ 8 tokens, first line, exact UNKNOWN).
- This covers calibration, every sleep, N and competence.
- Each checkpoint is checked against fp32 unmerged: zero answer and outcome-class mismatches, |Δ ppl ratio| ≤ 1e-4.
- **Outcome classes per probe:** correct, incorrect substantive answer, UNKNOWN, empty. Positives and negatives are
  reported separately.
- **Probes at checkpoint L:**
  - **Positives:**
    - audited held-out paraphrases of every acquired fact (primary);
    - template stems (acquisition only);
    - first-appearance facts of new entities;
    - life-1 retention.
  - **Negatives:**
    - **H near-misses** (H-entity × attribute not observed by L; primary);
    - A near-misses (descriptive; trained distribution);
    - evaluation-pool fresh names;
    - **transition cells before teaching** (L ≤ 3).
  - **Transitions after teaching** (L ≥ 4): recall of the newly taught value.
  - **Updates (descriptive):** which value is answered.
  - **Competence:** ppl on competence_eval and competence_modern, relative to the fp32 base.
- Every evaluation wording is held out from training. Only evaluation stems are used for the leak filters.

### Calibration
- Seed 16, arm P at sleep 1, fp32-primary evaluation.
- E is the smallest of {4, 8, 16} with exact template acquisition ≥ 0.80 and competence in the band. **STOP if none
  passes.**
- One E for all arms.

## Advancement criteria (development; U must meet all; pooled over 4 seeds, with ≥ 3/4 seeds where marked)
| # | Dimension | Requirement |
|---|---|---|
| D1 | Known facts answered | At every sleep: U's paraphrase recall over all acquired facts ≥ 0.50, and ≥ P − 0.10. First-appearance recall of new entities ≥ 0.50. |
| D2 | Unsupported rejected | At every sleep: U's H near-miss confabulation ≤ 0.10, and evaluation-pool fresh-name confabulation ≤ 0.10. **Evidence sensitivity:** U's H near-miss confabulation ≤ G's − 0.30, pooled and in ≥ 3/4 seeds. |
| D3 | UNKNOWN → KNOWN | Transition cells: UNKNOWN rate ≥ 0.80 before teaching (sleeps 1–3). Recall ≥ 0.50 after teaching (sleeps 4–5), and ≥ P's − 0.15. A-cells and H-cells are reported separately; both must pass. |
| D4 | Persistence | Life-1 recall at sleep 5 ≥ P − 0.10. The H near-miss criterion holds at sleep 5. Competence is in the Gate v2 band at every sleep. |

- **Reported jointly, never one at a time:** answer rate, selective accuracy (correct ÷ answered), and the full
  outcome-class table.
- **If U meets D2 but not D1 or D3:** "abstention at the cost of acquisition". Not an advance.
- **If U and G both meet D2:** the boundary is a familiarity rule, not evidence-sensitive. Not an advance.

## Prediction (Claude, before data)
- U reduces H near-miss confabulation substantially relative to P and G. G reduces fresh-name confabulation but much
  less H near-miss confabulation.
- The main risk is D3 for A-cells: a cell trained UNKNOWN may resist being learned at life 4.

## Open questions for Sol
1. **Replay source:** verbatim stream text (disclosed) vs new episodic banks.
2. **The D2 threshold:** 0.10 (CONS-0A's A4) vs a looser development threshold.
3. **The size and composition of the third slot,** and whether P's background slot is the right null (vs P having
   no third slot and unequal budget).
4. **Whether the A/H entity split is sufficient held-out structure,** or whether attributes should also be held out
   (H-attributes never appearing in any training negative).
5. **Anything in the negative-generation procedure** that leaks evaluator knowledge.

## Cost estimate
- **Per seed:** 3 arms × 5 sleeps of training, plus 16 fp32 fresh evaluations at about 4–5 min each, plus QA and
  negative generation.
- **Total:** about 6–8 h on an A40, roughly $4–5, plus calibration.
