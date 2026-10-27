# DECAY-1: what causes the rapid loss of life-1 memories? (recorded 2026-10-27)

Reviewed by Sol. Development evidence: seeds 2–5, one continuation life. This is a mechanistic follow-up, not a
sealed test.

## Validity
- **Gates:** all passed — V1, V2, V3, V3b, S, U (uptake 4/4 in A, B and E in every seed), V5.
  - **R:** reproduction exact in every seed for checkpoint 1 vs BIND-I1 and for arm A vs BIND-I1 condition U (diff
    0.0, 0 mismatches, equal hashes).
  - **T:** anchor KL 0.30–0.41.
- **Provenance:** bundle SHA-256 `1d930c0a…`; re-running the analysis reproduces the archived output. Code is the
  frozen commit b6c39ee, run at 5134507.

## Verdict (frozen rules)

| Contrast | dV mean | Per seed | Result |
|---|---|---|---|
| A−B | +2.05 | +7.4 / −2.1 / +0.1 / +2.8 | KL **protects** during new-fact learning (robust_pos) |
| C−D | +3.56 | +4.2 / +2.9 / +2.0 / +5.2 | KL **protects** without new facts (robust_pos; dB also robust_pos) |
| A−C | −15.93 | −12.2 / −18.3 / −17.1 / −16.2 | new-fact learning adds forgetting with KL (dB −9.97, robust) |
| B−D | −14.42 | −15.4 / −13.3 / −15.2 / −13.8 | new-fact learning causes forgetting without KL (dB −8.64, robust) |
| E−A | +0.33 | −1.5 / +3.1 / +1.6 / −1.9 | unresolved: no conclusion about previous-self consolidation |

Interaction (A−B)−(C−D): dV −1.5, dB −1.3 (small).

## Findings
- **My hypothesis (frozen-backbone KL causes the erasure) is contradicted.** KL is modestly protective.
- **Writing new facts into the shared adapter is the dominant cause.** One life of 10 unrelated new facts costs l(V|A)
  −22 to −37 nats, destroys forced choice (0–2/4) and destroys binding.
- **The newly taught value becomes a generic answer.** U was taught to B under an unrelated attribute, yet it gains
  +21 to +50 nats under A's query.
- **Background-only training fades V and Z about equally.** Losses are −6 to −22 nats, binding mostly survives, and
  forced choice stays 3–4/4 in seeds 3–5. This is slower and non-swapping.
- **Gradient diagnostic at checkpoint 1:**
  - The new-fact CE gradient is positively aligned with ∇l(V|A) in all 4 seeds (cos +0.10 to +0.28), so descent
    directly lowers the old answer.
  - Background-CE and KL gradients are roughly orthogonal to it.
  - Norms differ in normalisation and are not used as effect sizes (Sol).

## Decision (Claude + Sol)
- **Shared-parameter write interference** is the immediate failure mechanism.
- **Next is WRITE-1:** parameter isolation, meaning a frozen life-1 adapter plus a fresh life-2 adapter, combined
  readout, no routing. It is run against the historical baseline and an optimizer-reset control.
- **Pre-registered predictions:** both expect the combined readout to stay vulnerable, through the new adapter
  learning generic answer preferences.
- **If combined P fails** while old-only stays exact and new facts are learned, that is strong evidence for selectively
  addressable memory over permanently summed adapters.
