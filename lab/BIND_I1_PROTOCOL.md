# BIND-I1 protocol (frozen before data, 2026-10-26)

Designed by Claude after BIND-D1. Sol reviewed b7c87a8 and required pre-data revisions, which were accepted in full
(c99f066). Sol then audited c99f066 and found four measurement/validity defects, all fixed before data:
- **V5 used loss ratios:** now the perplexity ratio exp(L − L_base), per monitor, failing closed. This was Claude's bug.
- **Interpretation 3 ignored reverse-direction differences:** now both directions are checked.
- **V1 now pins the run identity:** exact seed set, approved config and locked backbone.
- **Tokenizer-prefix preflight:** runs before any GPU work. This file records the protocol. The frozen code is authoritative: `scripts/bind_i1.py` (runner) and
`scripts/bind_i1_analyze.py` (decision rules). Tests: `tests/test_bind_i1.py` (unit) and `tests/smoke_bind_i1_stubbed.py`
(stubbed end to end).

## Question
Does teaching a new entity B a value W for the **same attribute** as an earlier, entity-specific memory A→V damage A's
preference for V more than teaching the same W under an unrelated attribute, or teaching unrelated material?

## Design
- **Arm and data:** L (LoRA, lr 1e-4, KL 1.0, night passes 4). Dev seeds 2–5. Five lives of 10 facts, each fact in 4
  training phrasings.
- **Life 1:** 4 targets (A_i, T_i, V_i), 4 counter-entity facts (C_i, T_i, Z_i) and 2 fillers. Each target attribute
  therefore carries two entity-specific values.
- **Life 2:** four treatment facts plus six fillers. The treatment facts depend on the condition:

  | Condition | Treatment fact |
  |---|---|
  | **S** | (B_i, T_i, W_i) |
  | **X** | (B_i, X_i, W_i) |
  | **U** | (B_i, X_i, U_i), with U_i token-count-matched to W_i |

- **Lives 3–5:** fillers only.
- **Filler and stream controls:** fillers never use a target attribute and are identical across conditions. A, C, V
  and Z never reappear after life 1.
- **Branching:** life 1 is trained once. The following are snapshotted after it and restored for each condition:
  - LoRA weights;
  - AdamW state;
  - Python RNG;
  - torch RNG.

  The following are asserted equal across conditions:
  - per-life step counts;
  - the background/teacher text sequence (hashed per life);
  - the RNG end state;
  - the backbone hash.
- **Probes:** taken on the wiped model, at checkpoint 1 (shared) and at checkpoints 2, 3 and 5 for each condition.
  - **Candidate pool:** one fixed 50-candidate pool per target, [V, Z, W, U, 46 never-taught neutrals].
  - **A:** queried under T_i.
  - **C:** queried under T_i.
  - **B:** queried under T_i.
  - **Treatment facts:** every condition's treatment fact is scored under its taught attribute.
  - **Competence:** measured at every checkpoint.

## Decision rules
- **Primary measure:** dDA = D_A(5) − D_A(1), where D_A = l(V|A) − l(W|A). It is averaged per seed over the four
  targets. Effects are means of per-seed differences.
- **Thresholds:**
  - **Robust-worse:** mean ≤ −2 nats and negative in ≥ 3 of 4 seeds.
  - **Small:** |mean| < 2.
- **Validity gates:**
  - **V1, V2 — integrity:** failure makes the run INVALID.
  - **V3 — target acquisition:** ≥ 3 of 4 in ≥ 3 of 4 seeds, otherwise INVALID.
  - **V3b — bidirectional initial binding:** failure is qualified as "INITIAL BINDING NOT ESTABLISHED".
  - **V4 — treatment uptake** in every condition at checkpoint 2: failure is INCONCLUSIVE.
  - **V5 — competence band:** perplexity ratio exp(L − L_base) for eval and modern separately, at every checkpoint.
    Requires mean ≤ 1.02 and max ≤ 1.05, fails closed, and is flagged on the verdict.
- **Verdicts:**

  | # | Condition | Verdict |
  |---|---|---|
  | 1 | S robust-worse than X and U | Same-attribute interference supported |
  | 2 | S and X robust-worse than U, S−X small | Value exposure compatible; attribute specificity unresolved |
  | 3 | No robust difference in either direction for any pair | No differential effect. Called general drift only if U itself declines robustly |
  | 4 | Anything else | Mixed or unclassified |

- **Frozen secondary measures:**
  - the D_A^VZ trajectory;
  - life-5 patterns per target (survives / intrusion-with-binding / deterioration / other);
  - wiped and checkpoint-1 scores of W and U;
  - training tokens.

Four seeds and five lives make this development evidence, not a sealed test. The thresholds are not calibrated on D1.
