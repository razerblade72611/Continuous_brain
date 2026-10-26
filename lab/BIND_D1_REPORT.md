# BIND-D1: rival attribution and paired-entity diagnostic (recorded 2026-10-26)

Agreed by Claude and Sol. Descriptive development evidence: two arms, two seeds, five lives. This is not a sealed test.

## Validity
- **Reproduction:** exact against the archived A0 jobs for both arms (L kl1.0 lr1e-4, S layer21) and both seeds (0, 1).
  Facts identical; 0 generation and 0 forced-choice differences; maximum |Δ margin_best| = |Δ logprob| = 0.0; memory
  hashes identical at checkpoints 1, 2, 3, 5. The instrumentation did not alter training.
- **Measurement checks:** tokenizer-prefix failures: 0. Identical-value pairs excluded: 2 (seed 0), 0 (seed 1).
- **Provenance:** code being/v3-bind @ b796346; backbone `c75424fb…`; bundle SHA-256 `52cad98b…` (`results/dev3/BUNDLES.sha256`).
  Results in `results/dev3/bind_d1/`; frozen analysis in `results/dev3/bind_d1_analysis.json`
  (`scripts/bind_d1_analyze.py`, written before the data).

## Findings — L (LoRA, KL 1.0, lr 1e-4)
- **Acquisition is entity-conditioned at first.** At life 1, shared-attribute facts are 6/6 and 5/5, and unique 4/4
  and 5/5. At life 5, shared facts are 0/6 and 0/5.
- **Rivals are values taught to other entities.** The strongest rival had been taught to another entity under the same
  attribute in 8/10 → 7–9/10 → 9/10 facts at lives 2/3/5. Taught values are a small share of the 49 distractors, so
  the expected count under a uniform draw is about 0.3 → 0.5 → 0.9. That uniform-draw figure is a prevalence reference,
  not a calibrated null. Winners are almost always the most recently taught value (age 0–1 lives).
- **Taught values gain much more probability.** At life 5, compared with wiped, taught distractors gain +6 to +31 nats
  (mostly +12 to +31); untaught distractors move between +0.04 and −11.
- **Discrimination survives but A answers wrong.** Paired contrasts for the primary cohort (A in life 1, B later):
  - D_A = l(V|A) − l(W|A) is negative at every post-exposure checkpoint (−0.1 to −10.7);
  - D_B = l(W|B) − l(V|B) is +17.6 to +33.2;
  - C = D_A + D_B is +10.8 to +26.2.

  The model still discriminates which entity W belongs to, yet A's own query prefers B's newer value.
- **Context-independent shift exceeds the entity-specific part** (exploratory, algebraic). Taking
  G = (D_B − D_A)/2 as the context-independent preference for W and S = C/2 as the entity-specific part:
  - seed 0: G is +21.9 / +16.7 / +11.2 and S is +11.2 / +9.7 / +6.4 at lives 2 / 3 / 5;
  - seed 1: G is +13.1 / +18.0 / +14.3 and S is +13.1 / +9.1 / +5.4.

  These are functional components of a 2×2 preference pattern, not identified internal pathways.

## Findings — S21
- **Same qualitative pattern, much weaker.** Taught rivals: 2–3 → 6–7 → 7/10. Taught distractors +10 to +24 nats,
  untaught +0.5 to +5.
- **Small contrasts.** Primary D_A −3.2 to +5.3, D_B −0.5 to +7, C +3.8 to +5.8.
- **Weak binding from the start.** Life-1 shared facts: 2/6 and 1/5. S21 shows learned value-level interference without
  robust evidence that it ever formed comparably strong entity–value bindings. It is therefore not a counterexample to
  the L finding.

## Interpretation (leading functional hypothesis, not an established mechanism)
LoRA learns the original entity–value associations correctly. Later same-attribute learning makes those same queries
prefer values belonging to other entities, even while measurable entity discrimination survives. This is consistent with
a recency-weighted attribute–value preference overwhelming a weaker, decaying entity-specific association. Whether the
damaging update is attribute-specific, value-specific, or a broader consequence of continual optimisation is NOT yet
established.

## Decision (Claude, after Sol's review)
- **Next: BIND-I1,** a controlled three-condition interference intervention on L. It branches from an identical life-1
  state (weights, optimiser and RNG copied), with matched steps, matched sequences and no target rehearsal. Conditions:
  - **S:** B learns W for A's attribute;
  - **X:** B learns the same W under an unrelated attribute;
  - **U:** B learns an unrelated value under an unrelated attribute.

  Fresh dev seeds 2–5, five lives. Primary outcome: the within-seed effect E(S−X) on ΔD_A. Interpretations are frozen
  before the data.
- **Entity-keyed architecture options** (keyed associative update, contrastive negative evidence, update
  orthogonalisation) are drafted in parallel; implementation is deferred until BIND-I1 is analysed.
- **A2 Phases 2/3** remain suspended.
