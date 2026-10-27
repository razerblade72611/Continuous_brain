# DECAY-1 protocol (frozen before data, 2026-10-27)

Designed by Claude and Sol after BIND-I1. The frozen code is authoritative:
- `scripts/decay1.py` (runner) and `scripts/decay1_analyze.py` (decision rules);
- tests: `tests/test_decay1.py` (unit) and `tests/e2e_tiny_decay1.py` (the real runners on a tiny CPU backbone).

This is a mechanistic follow-up on the BIND-I1 development streams (seeds 2–5), not independent confirmation.

## Why
BIND-I1 found a robust value-exposure effect of about 10 nats. Its larger finding was condition-independent:
- l(V|A) fell 32–46 nats in every condition, including U, where no competing value was taught;
- life-5 forced choice was 0–1/4 everywhere;
- V-vs-Z binding collapsed.

Most of the loss had already appeared after one subsequent life. DECAY-1 asks which part of the training step causes
that loss.

## Design
- **Life 1:** identical to BIND-I1 (same stream, RNG and loop).
- **Branching:** the checkpoint-1 state is restored for each arm:
  - LoRA weights;
  - AdamW state (inherited in every arm);
  - Python and torch RNGs.
- **Continuation:** one life per arm.

| Arm | Continuation data | KL teacher |
|---|---|---|
| A | U-type new facts (= BIND-I1 condition U, life 2) | frozen backbone, 1.0 |
| B | same new facts | off |
| C | 40 background sentences, token-length-matched to the 40 fact texts (separate RNG) | frozen backbone, 1.0 |
| D | same background sentences | off |
| E | same new facts | end-of-life-1 model (previous self), 1.0 |

- **Identical schedule across arms:** step count and positions, and the background-rehearsal and teacher draws, are the
  same in every arm. KL-off arms draw the teacher texts and ignore them. This is asserted by a per-arm schedule hash and
  the RNG end state.
- **E's teacher:** the student with its adapters swapped for detached clones of the checkpoint-1 adapters, inside a
  no-grad forward pass. Teacher logits on a fixed sentence are hashed before and after the life.
- **Reproduction checks:**
  - checkpoint 1 must reproduce BIND-I1 checkpoint 1;
  - arm A must reproduce BIND-I1 condition U at life 2.

  Both compare against the committed BIND-I1 results with a tolerance of 1e-3 and require equal memory hashes. On the
  tiny CPU test backbone, both reproduce with a difference of exactly 0.
- **Gradient diagnostic at checkpoint 1:** no optimiser step, and the memory hash is checked afterwards. It compares
  the gradients of the following against the gradients of mean l(V|A) and mean [l(V|A) − l(Z|A)], reporting norms,
  cosines and dot products:
  - new-fact cross-entropy;
  - background cross-entropy;
  - frozen KL;
  - the AdamW update direction.
- **Saved states:** end-of-life-2 branch states are saved on the volume (not bundled; checksums are bundled) for a
  separately approved extension to life 5.

## Decision rules
- **Co-primary measures** per seed, averaged over the 4 targets, from checkpoint 1 to checkpoint 2:
  - dV = Δ l(V|A);
  - dB = Δ [l(V|A) − l(Z|A)].
- **Contrasts:** A−B, C−D, A−C, B−D and E−A, paired by seed.
  - **robust_neg:** mean ≤ −2 with ≥ 3 of 4 seeds negative.
  - **robust_pos:** mean ≥ +2 with ≥ 3 of 4 seeds positive.
  - Otherwise unresolved, never "equivalent".
- **Statements** (for dV and dB separately):
  - A−B neg → KL adds forgetting during new-fact learning;
  - C−D neg → KL adds forgetting without new facts;
  - B−D neg → new-fact learning causes forgetting without KL;
  - A−C neg → new-fact learning adds forgetting with KL;
  - C and D both mean dV ≤ −10 → background CE / inherited optimiser dynamics implicated;
  - E−A pos → teacher choice matters;
  - E−A unresolved → no general conclusion about previous-self consolidation.
- **Gates:**

  | Gate | Requirement | On failure |
  |---|---|---|
  | V1 | Identity, including the locked backbone and a clean prefix check | INVALID |
  | V2 | Identical schedules across arms | INVALID |
  | V3 | Life-1 acquisition | INVALID |
  | V3b | Binding | Qualifier |
  | R | Reproduction | Qualifier; contrasts stay internally valid |
  | T | E teacher logits unchanged | E−A contrast invalid |
  | V5 | Competence | Qualifier |
