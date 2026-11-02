# Next design: Claude's independent draft (ABST-1, "coverage")

Written before reading Sol's draft. Its sha256 is committed before the exchange.

## 1. Primary goal, and what ABST-0 did for it

- **Goal.** A persistent agent that consolidates its own experience into a cortex during sleep, and answers from
  that cortex in a way that is sensitive to evidence: it answers what it has experienced, says UNKNOWN for what it
  has not, and updates when a gap is later filled.
- **What ABST-0 established.**
  - Consolidating QA-style "unknown" negatives does not break recall: retention is unchanged (RET 0.90 vs 0.92).
  - But it did not produce a support boundary. Near-miss confabulation was 0.64–0.88.
  - The grounded and name-swapped arms are indistinguishable. Whatever the model learned was not tied to its own
    experience.
- **How the structure helped.** The two-phase design, frozen code, blind dual audit, fp32 twin checks and two
  independent analyzers make this negative result trustworthy. We know it is not label noise (the audited bad mass
  was about 10%), not a merge artefact, and not an analysis bug.

## 2. Post-hoc evidence from the ABST-0 data (descriptive, not a criterion)

Script: abst0_posthoc.py (to be committed with this draft). All seeds, sleeps 1–5, merged evaluations.

- **Most negatives missed the tested attributes.** Only 193 of 482 A-negative exposures (40%) target one of the 9
  schema attributes that the HNM/ANM/FRESH probes test. The rest are off-schema questions such as favorite color,
  siblings, allergist or job title. The T negatives are similar: 148 of 482.
- **U's UNKNOWN rate tracks how specific the training was, not the entity:**

  | Near-miss probe | U | G |
  |---|---|---|
  | Exact (entity, attribute) cell trained as a negative | 0.63 | 0.49 (same questions, swapped name) |
  | Same attribute trained, for other entities (HNM / ANM) | 0.23 / 0.34 | 0.35 / 0.27 |
  | Attribute never trained as a negative (HNM / ANM) | 0.19 / 0.26 | 0.24 / 0.25 |

- **Reading.** The model learned three things:
  - a generic prior of about 0.2 UNKNOWN;
  - a weak association between an attribute and UNKNOWN;
  - weak memorization of individual trained cells, which is the only place grounding shows (+0.14 for U over G).

  It did not learn the rule "no consolidated fact for (entity, attribute) → UNKNOWN". The U ≈ G result follows,
  because G keeps the attribute and only changes the name.

## 3. Hypotheses and what would separate them

- **H-COV (curriculum).** The rule is learnable, but ABST-0's negatives covered too little of the absent-cell space
  and were mostly off-schema.
  - *Predicts:* with systematic in-schema coverage, held-out absent cells (untrained attributes of trained entities,
    and every cell of entities with no negatives) move toward UNKNOWN well beyond the item-memorization level.
- **H-MEM (parametric limit).** A LoRA cortex consolidated from experience can memorize particular "unknown" cells,
  but cannot represent a general absence rule.
  - *Predicts:* trained cells reach a high UNKNOWN rate while held-out cells stay near the generic prior, even with
    full in-schema coverage and a high dose.
- **H-DOSE (strength only).** The rule is learned, but too weakly at E=4 with about 3 exposures per cell.
  - *Predicts:* everything scales up together with dose, trained and held-out alike.
- **What distinguishes them.** The held-out versus trained-cell gap, as dose and coverage change. H-COV gives a small
  gap that closes as coverage grows; H-MEM gives a gap that stays large at any dose; H-DOSE gives a gap shrinking in
  proportion, with both rising.

## 4. Proposed experiment: ABST-1 (coverage)

- **Negatives are generated from the schema, not by an LLM.**
  - For an A entity at sleep L, the absent cells are the schema attributes it has no fact for, excluding planned
    future facts unless the cell is in the reversal set below.
  - Each negative is that attribute's QA template paired with the answer UNKNOWN, in the same QA format as the
    positives. Positives already transfer from QA training to Statement probes (POS ≈ 0.8).
  - Labels are correct by construction, so no LLM generation and no audit are needed; this removes ABST-0's 8.7%
    bad-label rate as a confound.
- **Coverage split, fixed per seed before training.**
  - For each A entity, a random half of its absent attributes are TRAINED; the other half are HELD-OUT-WITHIN.
  - H entities get no negatives at all: HELD-OUT-ACROSS.
  - FRESH names are as in ABST-0.
- **Reversal set.**
  - Among trained cells, include up to 2 per seed that are planned transitions, learned at sleep 4.
  - This directly tests "explicit UNKNOWN, then later learn the fact", which ABST-0 could not assess (0 exposed
    cells).
- **Arms, all on the same I-QA base.**
  - P: positives only, with the background third slot as in ABST-0.
  - C1: coverage negatives at ABST-0's per-sleep negative budget.
  - C3: the same cells at 3× exposure, replacing background (the dose arm).
  - G1: C1's exact questions name-swapped onto T names (the grounding control).
  - N: base.
- **Scale.** Seeds 12–15; sleeps 1–4, since effects and retention were visible by sleep 3 in ABST-0 and sleep 4
  carries the reversal. fp32 merged evaluation as primary, with twin integrity as in ABST-0.
- **Primary measures, at sleep 4, pooled with per-seed values.**
  - UNKNOWN rate on TRAINED, HELD-OUT-WITHIN, HELD-OUT-ACROSS and FRESH.
  - The generalization ratio GR = (held-out-within − P's level) ÷ (trained − P's level).
  - POS, RET, FIRST and competence, with D1/D4-style guards.
  - Reversal: TPOST correct on reversal cells.
- **Predeclared readings.**
  - **GENERALIZES:** C1 or C3 has held-out-within UNKNOWN ≥ 0.70 and held-out-across ≥ 0.50, with confabulation
    falling correspondingly, while the D1/D4 guards hold. Then consolidation can learn support with the right
    curriculum: move to a curriculum design (CONS-0B) built on it.
  - **MEMORIZES:** trained ≥ 0.80 in C3, held-out-within ≤ (P level + 0.20), and GR ≤ 0.25 in both C arms. Then
    stop trying to train abstention into the cortex; the next design adds an explicit support mechanism (episodic
    store lookup, or a learned absence verifier gating the answer), evaluated against this same probe set.
  - **WEAK / DOSE:** trained cells don't reach 0.80 even in C3. Then the training signal is too weak to judge, and a
    single follow-up (dose or LR) is allowed before we decide.
  - **GROUNDING:** C1 − G1 on held-out-within ≥ 0.20 means a grounded rule; otherwise any rule is attribute-level,
    not experience-level.
- **Cost.** 4 arms (P, C1, C3, G1) plus N, 4 seeds, 4 sleeps. That is about 0.9× ABST-0 phase 2's evaluation
  count (3 arms × 5 sleeps), so about 7 h on one L40S, roughly $7–9. No audit phase is needed, so phase 1 shrinks to
  a calibration check.

## 5. Alternatives considered and rejected

- **Rerun ABST-0 with a higher dose of the same LLM negatives.** 60% of the dose is off-schema, so it would mostly
  strengthen item memorization; it cannot separate H-COV from H-MEM.
- **Go straight to an explicit support mechanism.** Possibly where we end up, but premature: ABST-0 never gave the
  cortex a fair curriculum. ABST-1 is the cheapest experiment that decides between the two.
- **Train negatives in the Statement probe format.** That would conflate learning a rule with matching the probe
  format. Positives already show that QA → Statement transfer works, so keep QA.
- **A reversal-only experiment.** Folded into ABST-1 via the reversal set at no extra cost.

## 6. What would change my mind

- Held-out-within moving clearly while held-out-across does not: the rule is entity-local. That favors per-entity
  coverage curricula, and grounding should then show in C1 − G1.
- C3 harming retention (RET below P − 0.10): dose is capped by interference, which itself favors the explicit
  mechanism.
- **Known risk: a surface cue.** If the negatives use schema templates while the positives' QA is LLM-worded, the
  model could learn "templated wording → UNKNOWN".
  - Requirement: every arm's positives also include the same attribute QA templates, answered with the value, so
    wording cannot separate positives from negatives.
  - Check: P's and C1's POS template-probe correctness must not drop relative to their paraphrase probes.
  - I have not yet verified how ABST-0's positive QA is worded. If Sol finds this requirement insufficient, the
    design changes.
