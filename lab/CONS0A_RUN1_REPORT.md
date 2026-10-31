# CONS-0A run 1: report (2026-10-31; development data)

**Frozen verdict: `CONS0A_INVALID`, no arm advances.** Run 1 is **not** a valid test of the hypothesis: two implementation
defects masked the primary measure. Post-hoc diagnostics (below; never a verdict) show a strong interleaving effect that
a corrected rerun must confirm under exact scoring.

## Run
- **Code and launcher:** code 10484d6 (Amendment 1), launcher 04a4806, A40, about 2 h 50 min.
- **Bundle:** sha256 `bded6eae…f8e8f`, archived in results/cons0a_run1/.
- **Preflight:** model manifest OK, stream design OK. prep gave 52/52 exact for all four seeds, with no identity errors.
- **Calibration (seed 6):** no E reached template acquisition 0.80 (all 0.0), so E = 16 by rule.
  - N near-miss confabulation was 0.0, so no contract revision was needed.
  - The projection was 16,695 s, inside the limit.
- **QA:** 121–128 accepted pairs per seed. Rejections were mostly leakage (35–45), then the per-episode cap, then span.
  Entailment rejected only 0–2.

## Defects found in the raw answers
1. **The QA targets had no end-of-answer terminator.** The training sequence ended at the answer, so the model never
   learned to stop.
   - Answers begin with the correct value and run on, e.g. "Kouskquisvun**ex**", "Romquouth**lysskournourn**".
   - Exact-match scoring therefore gave 0, and the run-on tails differ between merged and unmerged models, which
     caused all 16 QA merge-check failures (agreement 0.75–0.90).
2. **The V arms never leave UNKNOWN in the test format.** Training on raw episode sentences does not carry over to the
   question-and-answer frame. The training loss also only fell to about 1.4 nats per token, so the text was not
   memorised.
   - Calibration, measured in the frame, therefore could not pass, which forced E = 16.
   - E = 16 harmed competence: modern ppl ratio 1.76 at calibration, and A5 failed for I-V.
3. **Merge tolerance granularity (minor).** 0.02 is below a single item on 40-item measures (0.025). The V-arm merge
   failures are one-item flips, plus three small ppl differences (0.011–0.021).
4. **Disclosed confound (minor).** data/background.txt comes from the same sources as the competence sets, so the S
   arms' background slot can lower perplexity. A5 for I arms is unaffected, because their slot holds replay, not
   background.

## Post-hoc diagnostics (scripts/cons0a_posthoc.py → results/cons0a_run1/POSTHOC.json; 4 seeds pooled)
"Value-prefix" counts an answer that **begins** with the gold value. It is lenient by construction and is used only
because of defect 1.

| Arm | Sleep | New facts, template | New facts, audited paraphrase | Life-1 paraphrase | Near-miss answered | log p(V) vs N | V − W margin |
|---|---|---|---|---|---|---|---|
| I-QA | 1 | 40/40 | 56/56 | 56/56 | 160/160 | +57.8 | +53.2 |
| I-QA | 3 | 38/40 | 51/53 | **56/56** | 160/160 | +57.9 | +10.0 |
| I-QA | 5 | 37/40 | 44/45 | **56/56** | 160/160 | +57.8 | +12.6 |
| S-QA | 3 | 37/40 | 51/53 | **0/56** | 160/160 | +38.4 | −9.0 |
| S-QA | 5 | 36/40 | 42/45 | **0/56** | 160/160 | +29.1 | −6.6 |
| I-V | 5 | 0/40 | 0/45 | 1/56 | — | +46.4 | +3.7 |
| S-V | 5 | 1/40 | 0/45 | 0/56 | — | +33.4 | −1.7 |
| N | any | 0 | 0 | 0 | 0 | 0 | −2.7 |

**Readings (post-hoc and descriptive).**
- **QA distillation produces near-complete direct acquisition,** including held-out audited paraphrases.
- **Interleaved replay preserves life-1 knowledge through 5 sleeps (56/56); sequential consolidation loses it after one
  more sleep (4/56, then 0/56).**
  - The competitor value W overtakes A's value V only in S (negative V − W margin), which is the DECAY-1 failure.
  - Replay prevents it.
- **V arms show the same retention ordering in log-probabilities but no generatable access.**
- **Abstention collapses in QA arms:** 160/160 near-misses are answered, usually with a value the individual really
  experienced, but for the wrong person or attribute. This matches the pre-registered prediction and is the expected
  A4 failure.

## Status
- No progression to CONS-0B.
- The corrections (Amendment 2) are reviewed with Sol before any rerun.
- These post-hoc numbers are hypotheses for the rerun, not results.
