# CONS-0: consolidation from the episodic store into weights (proposal for Sol's review; not yet frozen)

Drafted by Claude on 2026-10-31, after CLS-0C closed (lab/CLS0C_SEALED_REPORT.md).

## Question
Can offline replay from the individual's own episodic store turn experienced episodes into **direct** knowledge? Direct
means answerable with no adapter active, an empty context and a fresh process. The knowledge must also:
1. be **used**: held-out paraphrase, and composition across lives;
2. be **retained** across later sleeps;
3. leave **abstention** intact: no new confabulation on facts never experienced;
4. leave **general competence** intact.

## Why this, now (David's four questions)
- **Primary goal.** A persistent individual whose experience changes what it can do. CLS-0B and CLS-0C established
  the fast half of a complementary learning system: lossless episode storage, and addressing that works on
  template-like memories. Integration was never achieved: direct answering scored 6/200. Consolidation is the missing
  half, and it is the one that matches David's growth criterion: learning vs frozen, retention, transfer.
- **How the current structure helped.**
  - The episodic store reconstructs every episode exactly (CLS-0B 208/208; CLS-0C bridge 52/52). That gives
    consolidation a **verified, self-acquired replay source** without depending on the B1 reader that failed on
    narratives. This is Sol's condition: "independently verified evidence or controlled retrieval."
  - DECAY-1 and WRITE-1 diagnosed why naive sequential writes destroy old knowledge (shared-parameter gradient
    alignment; combined-readout interference). This experiment tests the textbook CLS remedy for that exact failure:
    interleaved replay.
  - The harness, attestation, sealing and launcher machinery carries over.
- **Best choice.** It tests the defining claim of the architecture we adopted (Pan et al. 2026: "cortical
  consolidation … fine-tuning the base weights through batch replay"). Their abstract reports no retention or
  forgetting control, so the result is informative either way and not a reproduction of a published number.
- **What would change direction.**
  - If interleaved replay does not beat sequential consolidation on life-1 retention, or if consolidated facts don't
    transfer beyond the trained wording: weight consolidation is not delivering integration at this scale. We would
    then keep knowledge in the episodic store and put the effort into the reader.
  - If abstention collapses (direct confabulation): consolidation needs a gate before anything else.

## Design (development on existing data, then one sealed test)
- **Substrate.** Qwen3-8B, as in CLS-0B/0C. Development reuses the four CLS-0B episode banks (seeds 2–5, 5 lives,
  52 episodes each) already on the volume, so **no new episode writing is needed for development.**
- **Cortex.** A single slow "cortical" LoRA over all attention and MLP projections (rank fixed on dev).
  - **Disclosed deviation:** Pan et al. fine-tune base weights. A full fine-tune of 8B is out of budget, and a LoRA
    also makes the consolidated state separable, hashable and reversible.
  - The episodic adapters are never active during consolidated evaluation.
- **Sleep schedule.** One sleep after each life (5 sleeps).
- **Replay source.** Episodes are reconstructed from their own adapters (exactness verified, as a measurement only)
  and turned into training text: the episode verbatim, plus self-generated cue/answer pairs taken from the episode
  text. No gold labels, value pools or evaluator questions ever enter training.
- **Arms** (matched optimizer steps per sleep and matched trainable parameters):

  | Arm | What it trains on each sleep |
  |---|---|
  | **N** | Nothing: no consolidation (floor; expected ≈ the CLS-0B direct rate) |
  | **S** | Sequential: new episodes only (the DECAY-1 regime; expected catastrophic forgetting) |
  | **I** | Interleaved: new episodes plus replay of all earlier episodes from the store (the CLS remedy) |
  | **I+KL** | I, plus a frozen-teacher KL on background text (the Baseten recipe for protecting competence) |
  | **J** | Joint: all five lives trained at once after life 5 (a non-continual ceiling, reference only) |

- **Probes.** Every probe runs with all adapters off and an empty context, in a fresh process, after every sleep:
  - template and held-out paraphrase recall per life (forgetting curves for lives 1–5);
  - forced choice within the attribute pool (evaluator-only) plus the specific margin (the Gate v3 amendment A
    measures);
  - **two-hop composition** across lives, with a never-written-hop negative control;
  - **temporal state:** latest vs earlier value for changed attributes;
  - **abstention:** near-miss and unfamiliar questions asked directly, measuring the confabulation rate;
  - **competence:** the existing held-out perplexity sets (build_competence_sets.py) plus the ARC-Easy subset, under
    the one-sided rule.
- **Primary outcome (sealed).**
  - Life-1 direct recall (paraphrase, forced choice) after sleep 5 in I vs S, as a paired lower 95% bound above 0.
  - Plus I vs N on held-out paraphrase and composition, with competence inside the band and direct near-miss
    confabulation ≤ 0.10.
- **Prediction (Claude, pre-registered).**
  - S forgets life 1 by sleep 3; I retains most of it.
  - Composition stays weak (two-hop is the hard part).
  - Direct near-miss confabulation rises in every consolidating arm. This is the CLS-0C relevance failure moving
    into the weights.
- **Sealed test.** Fresh seeds with newly written banks (≈2 h per seed on an A40, as in CLS-0B). The questions are
  authored and adjudicated as in CLS-0C, but planned for agreement from the start: the label definitions are tightened
  using the CLS-0C boundary disputes.
- **Cost estimate.**
  - Dev: about 5 arms × 4 seeds × 5 sleeps of LoRA training (minutes each) plus fresh-process probes. ≈ 3–5 A40 hours,
    ≈ $2–3.
  - Sealed: ≈ $5–8. Total well within the remaining budget.

## Open questions for Sol
1. Cortical LoRA vs base-weight fine-tune: is the deviation acceptable, and at what rank?
2. Replay text: is it acceptable to verbalise episodes into self-generated cue/answer pairs (self-supervised, from the
   episode text only)? Or should replay be the verbatim episode only? The latter is purer, but may not produce
   question-answer access.
3. Is 5 lives enough to show retention? Or should dev also chain the four seeds' banks into a 20-life stream (no new
   writes needed)?
4. Should abstention be a primary gate or a co-primary safety gate?
5. Anything that makes this a reproduction rather than a contribution?
