# CLS-0C sealed test: validity interpretation (post-adjudication, pre-evaluation amendment)

**Status:** recorded after adjudication and sealing, before any sealed evaluation. It is NOT part of the pre-authoring
freeze (2b240fc), and it does not change B1_K1, the scorer, the gold labels or any numeric threshold.
**Approved by:** Sol (review) and David (tie-break authority), 2026-10-30.

**Sealed dataset:** results/cls0c_sealed/sealed.json, sha256
`b993ae9c1674f9faa91f1ff2c3281b4119402954fe6c655044ae65a018c54bd4`.

## Dispute resolution (class rules only; no item-level judgment)
Resolved by FINAL_RULES.json → final_labels.json (scripts/cls0c_final_labels.py). The frozen `seal` supports exclusion
only, so every kept item keeps its author label.

**Excluded:**

| Author → Grok B | Count | Reason |
|---|---|---|
| IMPLICATION → SUPPORTED | 14 | the text states the answer |
| SUPPORTED, span-boundary disagreement | 7 | gold ambiguous under the exact-match grader |
| → AMBIGUOUS | 5 | |
| TEMPORAL → NEAR-MISS | 1 | |
| Disputed templates | 4 | |
| Agreed templates with a form defect | 2 | the value slot is not at the end: "{e} turns first to for advice", "{e} reaches for this first when leaving home" |

**Kept, descriptive only (gold-safe):**

| Author → Grok B | Count | Why gold-safe |
|---|---|---|
| IMPLICATION → NEAR-MISS | 41 | both agree abstention is correct |
| TEMPORAL ↔ SUPPORTED | 17 | identical gold answer and evidence |

## Sufficiency is judged on independently AGREED category membership

| Narrative category | Retained | Agreed | Floor | Status |
|---|---|---|---|---|
| SUPPORTED | 91 | 90 | 80 | primary |
| NEAR-MISS | 50 | 50 | 40 | primary |
| UNFAMILIAR | 50 | 50 | 40 | primary |
| IMPLICATION | 83 | 42 | 80 | **INSUFFICIENTLY ADJUDICATED**: descriptive only |
| TEMPORAL | 39 | 23 | 32 | **INSUFFICIENTLY ADJUDICATED**: descriptive only |

- Templates: SUPPORTED 38/40 (floor 32), IMPLICATION 26/30 (floor 24). Every attribute keeps ≥ 1 template of each kind.
- The synthetic gates are unaffected and remain primary. This includes synthetic implication and synthetic temporal.
- Narrative provenance is primary. The 17 TEMPORAL↔SUPPORTED items count there as valid answered positives (their
  gold is agreed).

## Reporting rule
1. `gate_report()` evaluates the author's retained labels. Its raw `passed` boolean is preserved unchanged as an
   engineering output. It is **not** the scientific verdict.
2. **Scientific verdict:**
   - The narrative IMPLICATION and narrative TEMPORAL numbers are reported descriptively, whatever their values.
   - Every other gate keeps its original threshold.
   - The campaign cannot receive an unqualified PASS while these two categories are insufficient.
   - A failure in any adequately adjudicated gate remains a failure.
3. The preregistered prediction "B1_K1 fails the implication gate in both formats" is testable only in the synthetic
   format. The narrative half is descriptive.
4. **Descriptive cohorts:** membership is frozen in COHORTS.json. Run-row keys are `N|world|iid`.
   - agreed-only (IMPLICATION 42, TEMPORAL 23);
   - boundary-disputed (41, 16, plus the 1 SUPPORTED→TEMPORAL item);
   - author-intention (83, 39), reported as a sensitivity analysis.
   - Per-world retained-vs-agreed counts are reported alongside the pooled numbers.
5. No secondary analysis can override the primary verdict.
