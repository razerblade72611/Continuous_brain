# Gate v2 preregistration (DRAFT, not yet locked)

**Status:** DRAFT r2, 2026-10-19, Claude (builder). Revised after Sol's two reviews on DAV-104.

Not locked. The sealed run is impossible until this file records the final `configs/v2.json` SHA-256 (the runner checks it). The config is chosen on dev seeds 0–9 only. Sol reviews the final revision, and David separately approves running the sealed seeds.

## Question

After a life, does a frozen backbone with a Titans memory, which only that memory's weights moved, answer questions about the life? The test uses an empty prompt and a fresh session. A wiped copy should fail, and ordinary competence should hold.

## Fixed design

The design is in `being/gate_v2.py`, at the commit that locks this file.

- **Backbone:** distilgpt2, frozen. Its hash is checked before and after the run.
- **Memory:** a Titans MLP inserted after block `layer`, born with a zero output layer.
- **Surprise:** the gradient of the backbone's own next-token loss on each life sentence, with respect to the memory weights only.
- **Life:** 30 facts of the form "David's <attribute> is <value>.", with values drawn from an 80-word pool and attributes sampled per seed.
- **Sleep:** replay passes. Sleep is rejected if the dev-set perplexity ratio exceeds `sleep_band`.
- **Interference:** 30 facts about another person, after sleep, awake only.
- **Arms:**
  - lived;
  - wiped (same birth, never lived, which makes it the backbone itself);
  - stranger (same questions, different answers, scored on the lived answers).
- **Checkpoints:** A, after sleep; B, after interference.
- **Fresh session at A:** the weights file is reloaded into a new Being in the same process, and also into a separate Python process that loads only the backbone, the config and the weights, and sees only the questions.

## Scores per probe

The probe is the question prefix alone.

- `forced`: the true value ranks first among the 80-word pool. This is **forced-choice recall**; chance is 1/80.
- `full`: the true value is the top-1 next token over the **whole vocabulary**, which is unrestricted next-token recall.
- `generated`: a greedy 3-token continuation contains the value word. This is reported but not gated.

## Pass rule

The sealed seeds are 1000–1019 (n = 20). Every check must hold.

1. A − wiped, forced: one-sided 95% lower bound > 0.
2. A − stranger, forced: lower bound > 0.
3. B − wiped, forced: lower bound > 0 (retention after interference).
4. A − wiped, full vocabulary: lower bound > 0.
5. Recall floors at A: mean forced ≥ 50% of facts, and mean full-vocabulary ≥ 30% of facts.
6. Competence at **both** A and B, on **both** sealed confirmation sets (Gutenberg and Brown), with the memory active: mean perplexity ratio ≤ 1.02, and no seed above 1.05 (one-sided, decided by David; see below).
7. Fresh session: recall from the reloaded weights equals recall in the live run, exactly, both in-process and out-of-process. The weights hash matches.
8. The backbone hash is unchanged.

## Claim boundary

- **A pass shows:** facts from a scripted life are recoverable from the memory weights alone. Recovery works with an empty prompt and in a fresh process, both as forced choice and as unrestricted top-1 next token, with competence held and the frozen backbone untouched.
- **A pass doesn't show:** that it knows David, that it can converse, that it generalises beyond the trained question form, or anything about sentience. Paraphrase probes are reported, not gated.
- **Failures are kept.** Every dev and sealed run's JSON is committed, including failures and interrupted attempts.

## Competence sets

- **Dev monitors** (seen during development, never used for the gate): `data/competence_eval.txt` (Gutenberg) and `data/competence_modern.txt` (Brown).
- **Sealed-only confirmation sets** (built by `scripts/build_competence_sets.py`, disjoint from every other set, never printed or measured before the sealed run):
  - `data/confirm_gutenberg.txt`, 200 sentences from chesterton-ball and edgeworth-parents, sha256 `2f2b5ee5…59c0`;
  - `data/confirm_modern.txt`, Brown items 200–399, sha256 `7d2f6e3d…bc6`.

## Open before lock (needs Sol and David)

1. **Rehearsal** (never-scored background sentences interleaved during sleep, and optionally while awake) was added after the first dev results.
2. **Competence rule: DECIDED by David, 2026-10-19 ~20:50 CT.** One-sided: mean perplexity ratio ≤ 1.02 and no seed above 1.05, on both sealed confirmation sets, at both A and B. Improvement isn't counted as harm.
   - This replaces the two-sided rule (0.98–1.02) that Sol reviewed. The change was proposed after seeing a dev result: seed 0 with rehearsal gave a Gutenberg dev ratio of about 0.91. It's recorded here for that reason.

## Sealed config

To be filled in at lock time: `configs/v2.json` SHA-256 = `<pending dev tuning>`.
