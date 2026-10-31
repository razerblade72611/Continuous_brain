# CONS-0A protocol: consolidation development mechanism test (frozen 2026-10-31, before any CONS-0A data)

Claude (build) and Sol (review). This supersedes the draft lab/CONS0_PROPOSAL.md. Sol approved the design after two
rounds and gave GO after the corrections included here. Development only: this is not a sealed test. CONS-0B (fresh
sealed streams, composition, eight-stream sign test) is planned separately and is funded only if 0A advances.

**Code is authoritative:** scripts/cons0a.py (stages `prep`, `qa`, `calibrate`, `run`, `fresh`, `analyze`) and
tests/test_cons0a.py.

## Question
Can experiences reconstructed from the individual's own episodic store produce **direct** factual knowledge? And does
interleaved replay preserve that knowledge better than sequential consolidation, under matched budgets, without new
confabulation and within the competence band?

## Definitions
- **Direct.**
  - The episodic adapters are never installed in the training or evaluation process.
  - No retrieval and no episode context.
  - Knowledge lives in a single **cortical LoRA**: rank 16, α 32, on q/k/v/o/gate/up/down in every layer; fp32
    parameters; frozen base.
  - The primary endpoints (sleeps 1 and 5) are evaluated in a **fresh process** with the LoRA **merged** into the base
    weights.
  - **Merge equivalence:** at least 97% identical generated answers vs the in-process unmerged evaluation, with every
    pooled primary metric within ±0.02. Otherwise the run is INVALID for that arm.
- **Replay source.** Episodes reconstructed from the committed CLS-0B banks (results/cls0b_final/seed{2..5}_bank.pt,
  hashes checked against FILES.sha256). Reconstruction exactness against the stream is recorded as a measurement
  only; training always uses the reconstruction.
- **Direct-answer contract** (identical for every arm and probe). A fixed frame with two shots, one known and one
  UNKNOWN:
  - greedy decoding, ≤ 8 new tokens, first line;
  - **correct** iff normalised answer == normalised value;
  - **UNKNOWN** iff the normalised answer is exactly "unknown". "UNKNOWN: Leston" or "UNKNOWN because …" is an
    answer, so on a negative item it counts as a confabulation;
  - **confabulation** = any other non-empty answer on a negative item;
  - **no confidence thresholds.**
- **Positive probes** (facts experienced by the checkpoint):
  - **Template stem:** the first lifegen.TRAIN template ending in {v}, truncated before it. It measures **acquisition
    only**, because training text contains it.
  - **Audited paraphrases:** the ENTAILED held-out paraphrases from the frozen CLS-0C audit (`AUDIT` in
    scripts/cls0c_b.py). They give the **primary** generalisation measure.
- **Negatives** (labels follow the checkpoint):
  - **Near-miss:** an entity experienced by the checkpoint × an attribute never observed for it by then. Up to 40 per
    checkpoint, seeded.
  - **Unfamiliar:** 20 fresh names, fixed per seed.
  - Errors on both are reported against the base N model.
- **Descriptive diagnostics.**
  - **Value intrusion (BIND-I1 targets):** teacher-forced log p of V (A's value), W (the condition-S competitor taught
    to B on the same attribute in life 2) and Z (the counter value), under A's template stem.
  - **Temporal pair:** which of the life-1 and life-3 values is answered after sleeps 3–5.
  - **Forced choice / margins:** none (direct generation is primary, per Sol).
- **Competence:** perplexity on the dev monitor sets (competence_eval, competence_modern) relative to base, using the
  Gate v2 one-sided rule: mean ratio ≤ 1.02 on each set, and no seed above 1.05.

## Arms (per seed; one sleep after each of the 5 lives)
| Arm | Trains on |
|---|---|
| **N** | nothing (base) |
| **S-V** | new life's reconstructed episodes, verbatim; extra slot = background |
| **I-V** | new life's episodes, verbatim; extra slot = replay of earlier episodes |
| **S-QA** | new life's verified QA pairs; extra slot = background |
| **I-QA** | new life's verified QA pairs; extra slot = replay of earlier QA pairs |

- **Budget matching** (within each format family). Every sleep trains E passes over the new items plus an **extra
  slot** with the same item count.
  - **I:** the slot is filled with earlier items, cycled in seeded shuffled order. Exposure per earlier life is logged.
  - **S:** the slot is filled with data/background.txt chunks of **identical token length** (passed as token ids,
    not re-tokenised), with loss on the same number of final tokens as the matched I item.
  - Item count, update count, total tokens and loss-bearing tokens are therefore equal between S and I. At sleep 1
    there are no earlier items, so both use background and are identical.
  - **Shared branch:** S and I share one sleep-1 model and optimizer state (snapshot), then branch.
  - **Optimizer:** AdamW lr 2e-4, betas (0.9, 0.999), weight decay 0, batch 8, seeded shuffle. Optimizer state
    persists across sleeps.
- **E (passes)** is chosen once, before 0A, on **calibration seed 6** (raw stream text, V format, sleep 1 only).
  - E is the smallest of {4, 8, 16} giving template acquisition ≥ 0.80 with competence in band; otherwise E = 16.
  - Seed 6 is never used again. The calibration also reports N's near-miss confabulation under the contract.
  - **Contract revision:** if N confabulates on more than 0.10 of calibration near-misses, the contract is revised
    once, before 0A, and logged.
- **QA family: evidence-grounded QA distillation** (disclosed scaffolding).
  - **Generation:** the frozen base model sees only the reconstructed episode text (never the structured fact, gold,
    pools or evaluation wording) and proposes up to 3 question | answer lines, greedy.
  - **Acceptance** requires all of the following:
    1. the answer is 1–4 words and a verbatim whole-word span of the episode;
    2. the question does not contain the answer;
    3. a separate fixed **entailment verifier** (frozen base, its own few-shot yes/no prompt, a call independent of
       generation) says yes > no. Disclosed: it is the same model, used as an independent fixed verifier;
    4. **leakage:** the question's token Jaccard with every evaluation stem (template or paraphrase of any attribute,
       entity replaced) is < 0.8.
  - At most 3 accepted pairs per episode.
  - **Logged and hashed:** candidates, rejections with reasons, and acceptance by episode and attribute.
  - QA material is built before any training. All QA arms use the same frozen set.
  - QA training examples use the direct-answer frame, with loss on the answer tokens only.
- **No negative or UNKNOWN training** in any arm (Sol).

## Checkpoints and measures
- **After each sleep L (1–5), for every arm** (N evaluated once per L):
  - recall on template and paraphrase probes for every experienced life;
  - near-miss and unfamiliar confabulation;
  - value intrusion;
  - the temporal pair;
  - competence.
- **Fresh merged evaluation** after sleeps 1 and 5 for every trained arm.

## CONS-0A advancement criteria (Sol)
A single interleaving arm, either I-V or I-QA and the same one throughout, advances only if it meets **all** of the
following. Pooled means over seeds 2–5, with primary endpoints from the fresh merged evaluation where one exists.

| # | Criterion | Requirement |
|---|---|---|
| A1 | Immediate acquisition | After each sleep L, audited-paraphrase accuracy on life-L facts ≥ 0.50 pooled |
| A2 | Gain over N | A1's measure exceeds N by ≥ 0.30 pooled at each sleep, and is positive vs N in ≥ 3/4 seeds at each sleep |
| A3 | Retention | Life-1 audited-paraphrase recall after sleep 5 exceeds its matched S arm, pooled and in ≥ 3/4 seeds |
| A4 | Abstention | Direct near-miss confabulation ≤ 0.10 pooled after every sleep |
| A5 | Competence | Within the band after every sleep |

- These are development advancement thresholds, not significance tests.
- **If both arms qualify:** I-V is preferred (no scaffolding).
- **The qualifying family becomes 0B's single frozen contrast.**
- **If none qualifies:** stop before 0B and report the actual trajectories. If S acquires and I does not, say so; do
  not call it "no acquisition".
- **If A4 alone fails:** abstention becomes its own protocol before 0B. Nothing is adjusted in 0A.

## Prediction (Claude, before data)
- QA arms acquire paraphrase knowledge better than V arms.
- S-V and S-QA lose most life-1 recall by sleep 3; I retains it.
- Near-miss confabulation rises in every trained arm, most in QA, and may fail A4.
- W intrudes on A's query more under S than under I.

## Integrity and operations
- **Pinned commit; model manifest equal to CLS-0B; bank hashes checked;** results bundled with sha256.
- **Preflight:** the calibration run doubles as the A40 memory and throughput check. `project()` turns its timings
  into a conservative campaign projection, and a projection over 6 h stops the run (exit 4) before any 0A seed.
- **Seed failures:** a seed that fails operationally is rerun from scratch, never partially. All stage outputs are
  hashed.
- **0A is development:** no outcome-driven edits after data. Any change is a logged protocol amendment with Sol's
  review.

## Amendment 1: pre-data implementation corrections (Sol's review of dc5d6e7; no CONS-0A data existed)
Sol put the GPU command on HOLD. The design was unchanged; the implementation was corrected. The original freeze
(dc5d6e7) stays in the history.

1. **Stream identity (would have invalidated the experiment).**
   - dc5d6e7 rebuilt the streams without CLS-0B's token matching. Only **14/52** episode texts per seed then equal the
     committed banks' texts (checked against the banks).
   - Streams are now built with the CLS-0B stream tokenizer (`--stream-tokenizer`, default
     /workspace/Continuous_brain/models/qwen3-0.6b). For dev seeds the design must equal results/dev3/bind_i1.
   - `prep` **fails closed** unless the bank holds exactly the stream's 52 episodes, every bank text equals the stream
     text, and all 52 reconstructions are exact.
   - `run` and `fresh` fail closed unless the replay matches the stream (ids, texts, values).
2. **Exact UNKNOWN** (see the contract above). Regression tests cover the malformed forms.
3. **Fresh-process competence.**
   - Merge equivalence now also requires identical probe identities and every competence ratio within **0.01**.
   - A5 at sleeps 1 and 5 uses the fresh-process competence.
   - A missing or failed fresh check makes the run INVALID.
4. **QA supervision boundary.** QA training sequences are built as prompt ids (tokenised exactly as at evaluation)
   followed by answer ids, so the loss falls exactly on the answer tokens. This is asserted in the tests.
5. **Fail-closed budget.** Before every sleep, the (items, tokens, loss tokens) of I and S must be equal, or the run
   stops. Every training item carries exact token ids.

## Amendment 2: post-run-1 corrections (Sol's review, 2026-10-31; disclosed as after run 1)
Run 1 (lab/CONS0A_RUN1_REPORT.md) is **INVALID** and stays so. This amendment was written **after** run-1 data
existed. The rerun is a **post-run-1-amended development experiment** on the same four streams. It supports **no
confirmatory claim**; CONS-0B remains the independent sealed confirmation. Sol: HOLD on the first draft, then GO
after the changes below. Thresholds A1–A5, the arms, budgets, probes and the contract are unchanged.

1. **QA terminator.**
   - QA targets are prompt ids + ids(" " + answer) + ids("\n"), with loss on the answer **and** the newline.
   - Run-1 targets ended at the answer, so trained models ran on past the value (e.g. "Kouskquisvunex").
   - Budget matching follows automatically, and the S/I budget assertion is unchanged.
   - Tests:
     - the training tail decodes to " answer\n";
     - a tiny model trained on such a target **generates** the answer followed by the newline;
     - with the terminator removed, the same test fails (reproduces the run-on).
2. **QA-family calibration, no default.**
   - Seed 6, sleep 1. The frozen QA procedure (`qa_build`) is applied to seed 6's raw stream text, because seed 6 has
     no bank.
   - Calibration from synthetic raw episodes is **not** evidence about episodic-store consolidation.
   - E is the smallest of {4, 8, 16} with **exact** template acquisition ≥ 0.80 and competence in the band.
   - **If none passes, STOP** (calibration exit 4). The run-1 rule "otherwise E = 16" is removed: in run 1 it selected
     a regime that damaged competence (modern ppl ratio 1.76).
   - One E applies to both families. V is **not** independently calibrated, because it has no frame-accessible
     measure.
3. **Merge validity: strict, with matched-pair invalidation.**
   - **Per seed:**
     - identical probes;
     - ≥ 97% identical answers;
     - every competence ratio within **0.01** (not loosened for bf16 merge rounding).
   - **Per arm and checkpoint:** every summary metric's **pooled** (mean over seeds) in-process vs fresh difference is
     ≤ **0.02**, with no 1/n relaxation. Run 1's code checked each seed instead of pooling.
   - Per-seed differences are reported descriptively. A missing check fails.
   - **Pair rule.** A pair (I-V/S-V or I-QA/S-QA) is **evaluable** iff all 16 per-seed checks and all 4 pooled
     checks of both arms pass.
   - **Statuses:**
     - `CONS0A_INVALID`: no pair is evaluable, or a shared integrity failure.
     - `CONS0A_ADVANCE_<arm>`: an evaluable pair's I arm meets A1–A5 (I-V preferred).
     - `CONS0A_NO_ADVANCE`: otherwise.
     - `partial_validity` flags exactly one evaluable pair.
   - An invalid family is INVALID, not a scientific FAIL. Its numbers are descriptive only.
   - Run 1's code invalidated the whole run on any failure, while the protocol text said "INVALID for that arm".
4. **Descriptive secondaries** (never in A1–A5, never in merge tolerances).
   - **Metrics** for template and paraphrase probes:
     - exact (authoritative, unchanged);
     - whole-word value-prefix (the answer's first words equal the value's words);
     - raw string-prefix (the normalised answer starts with the normalised value).
   - **Probe:** a frame-free completion of the raw stem (greedy, ≤ 8 tokens), with the same metrics.
5. **Background confound.**
   - Disclosed: data/background.txt is prose of the same genre as the competence monitors (no exact overlap). Training
     on it lowered ppl in run 1 (calibration E = 4 ratios 0.74 and 0.88).
   - A descriptive I-minus-S competence contrast is reported.
   - The valid interpretation is **interleaved historical replay vs equally budgeted background-prose rehearsal**.
   - The dev competence sets are limited monitors. CONS-0B needs an independently held-out competence set.
6. **A4 unchanged.**
   - If A1–A3 and A5 pass but A4 fails, report "acquisition and retention with unsafe abstention" and move to a
     separate abstention protocol, not to 0B.
   - No negative or UNKNOWN training is added.
7. **Prediction.**
   - The frozen prediction stands.
   - **Run-1-informed addition (labelled):** corrected I-QA shows substantially better life-1 retention than S-QA.
   - No numerical threshold is derived from run 1.
