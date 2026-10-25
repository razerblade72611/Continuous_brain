# Gate v3: frozen design (R3), debate between Claude and Grok

- **Status:** DESIGN FROZEN FOR BUILD, 2026-10-25. This is not a preregistration. `PREREG_v3.md` comes after the dev runs and is reviewed by Grok (standing in for Sol). The sealed run needs David's go.
- **Record:** Linear DAV-105, plus a live grok.com chat (model: Expert) that David signed Claude into on 2026-10-24.
- **Sealed proposals:** Claude's R1 was sealed (SHA-256 `26690de2c14e05737c95f4b358f1c8995407d62e957370834dccd0b6daf311f2`) before Grok's R1 and revealed after it. Both R1s are posted verbatim on DAV-105.

## Question

After many sequential lives, are early memories still **reachable** from the weights alone, with an empty prompt and in a fresh process? Being *stored* is not enough.

## Why (David's four questions)

- **Primary goal.** A persistent individual whose memory lives in its weights. Gate v2 showed that one interfering life halves recall, and reachability after many writes is the open problem in the field (Baseten, arXiv 2607.11020: retention plateaus at 25–28% by 100 writes while about 69% of the log-prob signal remains).
- **How the current structure helped.** v2's gate, seal, resume and attestation machinery carries over unchanged.
- **Best choice.** It targets an open problem, against the published state-of-the-art comparator, with a result that is informative in either direction.
- **What would change direction.** If on dev L ≥ G and S ≥ T, we drop memory-as-a-layer and build on L or S.

## Design

1. **Backbone.** Qwen3-0.6B, frozen, SHA-256 hash checked at the start and end of every seed. The memory insertion layer is chosen on dev from {7, 14, 21} of 28 and frozen for sealing. If 0.6B is too slow on CPU, dev moves to a rented GPU (with David's approval and within budget); we do **not** switch to a smaller backbone.
2. **Lives.** 100 sequential lives × 10 facts = 1,000 facts.
   - Answers are multi-token. Each fact is told in 4 training phrasings, and the test phrasings are held out.
   - 2–3 facts per life are bridge facts that reuse an entity from an earlier life.
   - No diary: nothing from a past life is ever re-read.
3. **Generator** (spec by Grok, clarified by Claude).
   - **Entities.** 200 seeded synthetic proper names. **Values** come from a closed pool of 500 invented or rare words.
   - **Attributes.** 10: city, occupation, favorite object, sister's name, pet species, childhood fear, workplace, hobby, birth month, allergy.
   - **Uniqueness.** Each (entity, attribute) pair is unique per stream. A value never answers two questions about the same entity.
   - **Subject.** The subject is "David"; the other entities are his world.
   - **Composition items.** Two-hop, across different lives, with both facts required. Negative control: one hop never written.
   - **Prior floor.** Wiped and stranger must score ≤5% on both exact match and forced choice on dev. Otherwise the pool is regenerated, and the regeneration is logged.
4. **Arms**, all on the same frozen backbone:
   - **T:** Titans memory in the residual stream (v2 scaled).
   - **S:** sparse product-key slot memory with Meta's slot-selection rule (arXiv 2510.15103).
   - **L:** LoRA with varied data and a frozen-teacher KL penalty (the Baseten recipe), as the state-of-the-art comparator.
   - **G:** self-generated sleep replay on whichever of T or S wins on dev, run both gated and ungated. Gated means a template-structure filter plus a self-consistency gate: rehearse only if the same answer comes back for ≥3 of 4 self-generated cues.
   - **CMS:** a multi-timescale ablation arm, only if compute allows.
   - **Controls:** wiped and stranger.
   - **Ceiling reference only:** the facts in the prompt.
5. **Matched budget** (accepted by both).
   - Same life text and phrasings.
   - Same awake gradient-step count and sleep step count. G's replay steps *replace* background-rehearsal steps.
   - Trainable parameters within 2× (reported).
   - ≤8 dev configs per arm, chosen by the same rule: best dev life-1 recall at life 100 within the competence band.
6. **Checkpoints.** After lives 1, 10, 25, 50 and 100. Probe the facts of lives 1, 10 and 50.
7. **Probes.**
   - Template recall.
   - Held-out paraphrase recall.
   - Composition, with the negative control.
   - "Use": free-form generation scored by string match, reported only.
   - Scoring is greedy exact multi-token match, plus the log-prob gain (stored vs. reachable).
8. **Competence.** v2's held-out perplexity sets, plus a fixed 200-item ARC-Easy subset, at every checkpoint, under the one-sided v2 rule.
   - ARC-Easy is only distributed as the AI2 S3 zip, which Claude's workspace can't reach (Grok's G4 found no GitHub-raw copy), so David downloads it. Grok gives the license as CC BY-SA 4.0 (not yet checked by Claude).
   - If the ARC set is delayed, competence falls back to the perplexity sets alone; the pass rule doesn't depend on ARC.
9. **Pass rule** (10 sealed seeds, paired).
   - **Primary:** life-1 recall at life 100, with lower 95% bounds above 0 for both G − L and G − (T or S without replay).
   - **Secondary:** paraphrase and composition margins over wiped above 0.
   - **Integrity, as in v2:** backbone hash, fresh-process identity, competence.
   - **Reported, never used in updates:** the oracle false-write rate (confabulations G rehearsed).

## Confabulation controls

- **Rejected as a mechanism:** ground-truth spot-checks that steer replay. They would be a diary by the back door. They are kept as measurement only.
- **Accepted:** the template filter, the self-consistency gate, and the gated vs. ungated ablation.

## Related work, checked by Claude

- **Titans** (Behrouz et al. 2025).
- **Nested Learning / HOPE** (Google, Nov 2025).
- **Baseten**, arXiv 2607.11020.
- **Meta**, arXiv 2510.15103.
- **Marek et al.**, arXiv 2605.26097 (self-generated replay nearly eliminates forgetting).
- **Behrouz et al.**, arXiv 2606.03979 ("Language Models Need Sleep": Knowledge Seeding + Dreaming). This is the closest to G, so G is **not novel as a mechanism**; the open test is its effect on reachability.
- **OpenBMB Self-Replay Fine-Tuning** (blog, 2026-10-12).
- **LAMOL** (2019).

## Work split

- **Claude (build):**
  - the life/entity generator and probe harness;
  - the S slot layer;
  - the T, L and G arms;
  - the Qwen3-0.6B throughput benchmark;
  - dev runs, then `PREREG_v3.md`.
- **Grok (review / assist):**
  - **G4:** an ARC-Easy source reachable from GitHub;
  - **G5:** a leak audit of the generator's pilot output;
  - **G6:** review of the S code and budget accounting;
  - a drift check against items 1–9;
  - the PREREG review.
- **David:**
  - download Qwen3-0.6B (Hugging Face is blocked in Claude's workspace);
  - approve any GPU spend;
  - give the go for the sealed run.
