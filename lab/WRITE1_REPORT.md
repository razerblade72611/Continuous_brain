# WRITE-1: parameter isolation vs. continued writing (recorded 2026-10-27)

Reviewed by Sol. Development evidence: seeds 2–5, one continuation life. This is not a sealed test.

## Validity
- **Gates:** all passed — V1, V2, V3, V3b, V5.
  - **R:** three exact reproductions in every seed — checkpoint 1 vs BIND-I1, A vs BIND-I1 condition U, and A vs
    DECAY-1 arm A.
  - **I:** P's zero-init check was exact, and the old-only readout equals checkpoint 1 exactly in every seed.
- **Trainable parameters:** equal across arms (2.29M).
- **Provenance:** bundle SHA-256 `e88def12…`; re-running the analysis reproduces the archived output. Code is the
  frozen commit fed5d34, run at f72fd2e.
- **Log note:** a torch UserWarning appears from `float()` on a requires-grad tensor in the zero-init assert. It is
  harmless.

## Verdict (frozen rules)
**P: "frozen old knowledge is disrupted by combining adapters: readout interference demonstrated."**
A, R and P all fail preservation, and all learn the new facts (uptake 4/4 in every seed).

| Contrast | dV mean | dB mean | Robust? |
|---|---|---|---|
| P−A | +1.60 | +2.42 | no |
| R−A | +1.54 | +0.92 | no |
| P−R | +0.06 | +1.50 | no |

## Findings
- **Isolating the old parameters bought almost nothing.** The life-1 adapter was bit-identical, yet the combined
  readout lost the old answers almost exactly as much as overwriting did.
  - dV: P −29.6 / −25.7 / −21.4 / −19.3 vs A −29.9 / −25.1 / −25.1 / −22.3.
  - Value intrusion dU: P +34.7 / +23.3 / +46.8 / +26.2 vs A +35.5 / +21.5 / +49.3 / +26.5.
- **The new adapter is not an independent memory.** New-only uptake is 0–1/4: the adapter was trained conditionally
  on the old one.
- **The old knowledge is intact.** The failure is in which stored update is applied to which query.
- **Predictions:** Claude predicted that P would beat A modestly; that was wrong in degree, since P−A was not robust.
  Sol predicted the old adapter would be protected internally while the combined output stayed vulnerable; that was
  right.
- **Refinement (Sol):** this shows overwriting is not *necessary* for catastrophic loss of access, not that it never
  contributes.

## Decision (Claude + Sol): architectural pivot
- **Stop scaling the summed-LoRA design.** Permanently summed, always-on parameter writes fail one-life preservation.
- **New leading hypothesis: a complementary learning system.** A stable general learner, a fast self-acquired episodic
  memory with selective (addressed) retrieval, and slower consolidation into weights. Memory must also support
  correction and temporal change.
- **"No diary" is revised.** Durable storage of the individual's own experiences is allowed. External ground-truth
  ledgers and privileged evaluation information are not. Provenance enforces the distinction.
- **Closest published work:** Pan, Hahami, Siegelmann & Sompolinsky (2026), *A complementary learning system for
  continual episodic memory in large language models*, bioRxiv 10.64898/2026.08.24.746712. It uses per-episode masked
  LoRA, embedding-keyed winner-take-all routing, internal RAG and offline consolidation. The code is public but carries
  no licence, so the plan is a clean reimplementation from the paper, with attribution.
- **READ-0 dropped.** Its answer was largely determined by WRITE-1.
- **Next: CLS-0.**
  - (A) a faithful reimplementation validated on the published setting;
  - (B) the same mechanism on our five-life streams, with:
    - one adapter per experienced fact/event;
    - addressing arms: Qwen3 embedding, own backbone, lexical baseline from observed text, permuted keys;
    - a null-memory threshold;
    - addressing, reconstruction and use measured separately;
    - a raw episodic-text retrieval baseline;
    - filtered and unfiltered accuracy;
    - a temporal-change diagnostic;
  - (C) a 0.6B continuity check.

  A single-episode 8B resource benchmark gates any 8B GPU spend.
