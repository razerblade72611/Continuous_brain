# BIND-I1: controlled S/X/U interference intervention (recorded 2026-10-27)

Reviewed by Sol. Development evidence: dev seeds 2–5, five lives. This is not a sealed test.

## Validity
- **Run:** code at a0daf89, with the experiment frozen at 506aa7a. All 4 workers exited with code 0.
- **Provenance:** bundle SHA-256 `c67e4de2…` (`results/dev3/BUNDLES.sha256`), verified on David's PC and here.
  Re-running the frozen analyzer on the committed results reproduces the archived analysis exactly.
- **Preflight:** tokenizer prefix check, 0 failures in 1,200 pairs per seed.
- **Gates:** all passed.
  - V1: identity.
  - V2: identical schedules.
  - V3: acquisition 3/4, 4/4, 4/4, 4/4.
  - V3b: bidirectional binding 3/4, 4/4, 4/4, 4/4.
  - V4: treatment uptake 4/4 in every seed and condition.
  - V5: competence perplexity ratios 0.67–0.74. These are below 1, as in every L run. That indicates domain
    adaptation; no monitor sentence appears in the background pool.

## Frozen verdict
**2 — value-exposure mechanism compatible; attribute specificity unresolved.**

| Contrast | Mean (nats) | Per seed | Result |
|---|---|---|---|
| E(S−X) | +0.66 | +3.8 / +4.8 / −6.8 / +0.9 | small (not shown equivalent) |
| E(S−U) | −9.92 | −16.6 / +2.6 / −16.5 / −9.1 | robust-worse |
| E(X−U) | −10.59 | −20.4 / −2.2 / −9.7 / −10.1 | robust-worse |

## The larger, condition-independent finding (descriptive)
- **True-answer loss:** l(V|A) falls 32–46 nats by life 5 in every condition, including U, where W is never taught.
- **Forced choice at life 5:** 0–1/4 in every condition (U: 1/16 across seeds).
- **Binding collapses everywhere:** the V-vs-Z preference ends near zero or negative in all conditions.
- **The loss is fast:** most of it appears after one subsequent life. In U for seed 2, D_A goes +42.5 → +15.1.

## Interpretation (Claude + Sol)
- **Value exposure adds a measurable effect.** Teaching another entity a value under any attribute adds about 10 nats
  of damage to A's query compared with unrelated material. This is a development-stage differential effect, not an
  exact entity-independent write strength.
- **The dominant problem is generic.** Continued optimisation makes old answers inaccessible even without competing
  values. This is a loss of functional accessibility; the information is not shown to be erased.
- **Entity-keyed binding fixes would address only the smaller component.**

## Decision
Next is **DECAY-1** (`lab/DECAY1_PROTOCOL.md`): five one-life arms from the same life-1 state (2×2 of data × KL to
the frozen backbone, plus a previous-self teacher) and a gradient diagnostic. These committed results are its frozen
reference. Architecture changes stay deferred.
