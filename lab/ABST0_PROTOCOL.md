# ABST-0 protocol: learning a support boundary during consolidation (v3 for freeze review; 2026-11-01)

Claude (build), Sol (review). This supersedes lab/ABST0_PROPOSAL.md and incorporates Sol's required revisions of
2026-11-01: the proposal review, and the v1 review (six corrections).

**Status: training on HOLD** until Sol approves this protocol and its implementation. The code is authoritative once
frozen: scripts/abst0.py and tests/test_abst0.py. Development only; not a sealed test.

## Question and scope
- **Question.** Given an **exact, externally supplied archive of its own experiences**, can consolidation into a
  cortical learner produce an evidence-sensitive answer/abstain decision, while preserving correct recall, retention
  and the ability to learn newly encountered facts, including UNKNOWN → KNOWN?
- **Scope: cortex-only.** Replay uses each episode's verbatim stream text. The episodic store is not used.
  - CLS-0B's 208/208 exact reconstructions do not establish reconstruction fidelity on these new streams, and
    reconstruction is hardware-sensitive (lab/CONS0A_RERUN_LOG.md).
  - Success would justify an integrated store-replay experiment, but would **not** establish that integration.
- **Supervision scaffolding.** The per-entity string index and the frozen generator and verifier used to build
  supervision are external scaffolding. They are not available during direct evaluation.
- **Definition.** UNKNOWN means *not supported by the experiences acquired by this checkpoint*. It does not mean
  false, and it does not mean the world lacks the value.
- **Claim boundary.** Success means generalisation **across entities within the trained attribute domain** (A → H).
  It does not mean generalisation to new attributes.

## Streams (seeds 12–15 development, seed 16 calibration; never used before)
- **Generator:** `abst0.make_stream(seed)`, deterministic.
  - Vocabulary, phonotactics and value suffixes come from lifegen.
  - The 9 non-relational attributes are used; "sister" is excluded.
  - Each fact is one episode of its 4 TRAIN phrasings, as in CLS-0B.
- **Name pools:** three pairwise-disjoint pools of entity names.
  - **S** (stream): every entity that appears in the stream.
  - **T** (training-only fresh names): used only in training negatives.
  - **V** (evaluation-only fresh names): 20 per seed, used only in evaluation.
  - No name word is shared across pools. Value pools are disjoint from all names.
- **Lives (5):**

  | Life | Facts |
  |---|---|
  | 1 | 8 core entities introduced (one fact each), plus 2 transient entities (one fact each) |
  | 2 | 8 more core entities introduced (one fact each), plus 2 *update-initial* facts on life-1 core entities |
  | 3 | 3 transient entities (one fact each), plus 7 second facts on core entities (never a planned cell) |
  | 4 | 8 *transition* facts, plus 2 *update-changed* facts, plus 3 transient entities |
  | 5 | 3 transient entities, plus 7 further facts on core entities (never a planned cell) |

  - **FIRST** (first-appearance) cohorts: life 1 has 8 core entities plus 2 transients; life 2 has 8 core entities and
    no transients; lives 3–5 have 3 transients each.
  - Life 4 has 13 facts. Budgets are matched across arms within each sleep.
  - `check_stream` freezes uniqueness. Each transition cell has exactly one fact, at life 4. Each update cell has
    exactly two, at life 2 and life 4. Planned cells never collide with ordinary facts.

- **Core entities (16):** split, seeded, into **A** (8; may appear in negative training) and **H** (8; never appear in
  any training negative). Each half has 4 life-1 and 4 life-2 entities.
- **Transition cells (8):** core entity × an attribute first taught in life 4. There are 4 A-cells and 4 H-cells, and
  2 of each on life-1 entities and 2 on life-2 entities.
- **Update cells (2):** a life-1 core entity's attribute taught in life 2, with its value changed in life 4. One
  update cell is on an A-entity and one on an H-entity.

## Training (per seed; arms share the cortex initialisation, QA material, schedule seed and positive-slot order)
- **Cortex, optimiser and QA:** identical to CONS-0A with Amendment 2.
  - Cortex: rank 16, α 32, on q/k/v/o/gate/up/down, fp32. Optimiser: AdamW lr 2e-4, batch 8, state persists.
  - QA distillation: the frozen QA_GEN prompt, filters, entailment verifier and cap of 3 per episode. The source text
    is the stream text. Targets are prompt ids + " answer" + "\n", with loss on the answer and the newline.
- **Each sleep L, three slots:**
  1. **New slot:** E passes over life-L QA items.
  2. **Replay slot:** the same item count, cycling earlier QA items in seeded order (I-QA). At sleep 1 there are no
     earlier items, so the slot is filled with background, identical across arms.
  3. **Third slot:** ⌈0.5 × E × |new items|⌉ items, frozen.

| Arm | Third slot |
|---|---|
| **P** | Background chunks token-length- and loss-tail-matched to U's third-slot items (an active control: equal-budget background rehearsal) |
| **U** | 50% A-entity negatives and 50% T-name negatives (deterministic rounding: A gets ⌈n/2⌉) |
| **G** | The same item positions as U. U's T-name half is unchanged. Each A-entity negative is **name-swapped**: the A name is replaced by a T name of identical token length, so the question construction and topic are controlled. |
| **N** | Base model, no training |

- **Budget.** The (items, input tokens, loss tokens) of U, G and P are asserted equal at every sleep, fail-closed.
  Positive exposures are identical across arms by construction.

### Negative corpora (stage `prep`; built once per seed BEFORE any training; arm-independent)
- **Inputs.** For sleep L, only the episodes of lives ≤ L. Arm outputs are never used.
- **String index.**
  - Index key: each A-entity's exact full name.
  - Memory: the concatenation of all acquired episode texts containing that exact name.
  - The index is checked for completeness (every acquired episode mentioning the name is included) and for no
    truncation (memory < 1,024 tokens, else STOP).
- **Generation.** The frozen base model, greedy, with the frozen prompt NEG_GEN: "Memory: {memory}\n\nWrite three
  different questions about {e} that this memory does NOT answer. …". It yields up to 3 candidates.
- **T-name candidates.** The frozen prompt NEG_GEN_FRESH ("Write three different questions about a person named
  {t}.") on deterministic T names.
- **Acceptance**, all required:
  1. the question is a single well-formed line ending in "?";
  2. it contains the target name exactly;
  3. it contains **no S-pool name word of any H-entity**, nor of any other S entity except the target;
  4. **three-way frozen verifier** (VERIFY3; a separate call with its own few-shot prompt): labels
     SUPPORTED / UNSUPPORTED / AMBIGUOUS. Accept only if UNSUPPORTED's log-prob exceeds both others. A-entity
     questions are judged against their memory; T-name questions against an empty memory;
  5. **leak filter:** token Jaccard < 0.8 against the fixed, entity-independent evaluation wording inventory (every
     TRAIN template and PARAPHRASE stem, with {e} removed), shared across seeds and sleeps;
  6. at most 3 per entity per sleep.
- **Selection.** Accepted candidates are taken in (entity, generation) order. If there are fewer than needed, the
  nonempty accepted set is cycled in seeded order. Repetitions and per-item exposure are logged.
- **Zero eligible A- or T-negatives at any sleep: corpus construction STOPS** (Sol). There is no substitution. A
  generator revision requires an amendment before training.
- **Construction holdout.**
  - Training target queries use the Question construction; scored evaluation queries use the Statement
    construction. The fixed FRAME demonstrations are shared by both.
  - Background prose and the source episodes are not target queries.
  - This is a single construction holdout (Question → Statement). It does not show generalisation across multiple
    unseen negative construction families (Sol).
- **Logged and hashed:** candidates, rejections with reasons, accepted sets, selections and exposures.
- **Label-quality audit (predeclared, blind).**
  - **Sample:** min(20, number of distinct accepted A-negatives) per seed per sleep, seeded, **without
    replacement**. At sleep 1 at most 12 candidates exist. The actual denominators are recorded.
  - **Procedure:** written to `audit_sample_seed{S}.json`. Claude audits it **before any training**, seeing only the
    memory text and the question. The auditor is blind to transition membership, future episodes and outcomes, and
    records supported / unsupported / ambiguous.
  - **Reported:** the candidate-level error rate, and the delivered exposure mass of audited supported or ambiguous
    items (from the training schedule).
  - Agreement between Claude and the verifier is not proof of perfect labels. Findings cannot trigger selective
    repairs.
- **Verifier-inferred transition exposure** (reporting only; Sol).
  - **When:** computed after the deterministic schedule is built and before training, from UNKNOWN items **actually
    delivered** to U at sleeps 1–3, meaning items with nonzero training exposure. Mere acceptance does not count.
  - **Test:** a delivered item about the entity counts as exposure if VERIFY3, given only the cell's life-4 episode
    text, labels it SUPPORTED. This is a proxy: it can recognise broad questions that episode answers without the
    question being equivalent to the transition probe.
  - **Recorded:** the matched questions, the episode and the exposure counts by sleep, in a **separate** file
    (`exposure_seed{S}.json`) that the selector never reads.
  - **D3 is reported for:** exposed A-cells, unexposed A-cells and H-cells. If exposed A-cells are absent or sparse
    (< 2 per seed on average), the report says "reversal of explicitly trained UNKNOWN was not adequately tested".

## Evaluation (primary representation per lab/MERGE_ENG_REPORT.md; fail-closed)
- **Representation.** Fresh process. The cortex is merged into an **fp32** copy of the base, and the adapter
  computation is removed. Autocast is off; TF32 is off (`torch.backends.cuda.matmul.allow_tf32 = False`,
  `cudnn.allow_tf32 = False`). Eager attention.
- **Frozen decoding.** CONS-0A's FRAME, greedy, ≤ 8 new tokens, batch 16, left padding, first line.
- **Integrity.** For each arm × sleep, a second fresh evaluation runs fp32 **unmerged**. It requires:
  - identical probe identities and counts;
  - **zero** answer mismatches;
  - **zero** outcome-class mismatches;
  - |Δ ppl ratio| ≤ 1e-4 on the same fp32 base denominator.
  Any failure, a missing file or a nonfinite value invalidates that **arm** (and every comparison using it).
- **Base model N.** Evaluated in the same fp32 representation at every sleep.
- **Outcome class per probe:**
  - **correct:** normalised answer == current value;
  - **unknown:** normalised answer == "unknown";
  - **empty:** no answer;
  - **substantive incorrect:** anything else.
  On negatives, "substantive" (any non-UNKNOWN, non-empty answer) is the **confabulation** class.
- **Probes at checkpoint L.** Each probe is the Statement construction of a template stem or an AUDIT-entailed
  paraphrase stem.

  | Category | Contents |
  |---|---|
  | **POS** | Every acquired non-update fact (template + entailed paraphrases) |
  | **FIRST** | Facts of entities first appearing in life L (sleep L only) |
  | **RET** | Life-1 facts that are never superseded, at every sleep |
  | **HNM** (primary negative) | H-entity × attribute not observed by L, excluding planned transition and update cells; up to 40 per checkpoint, seeded |
  | **ANM** (descriptive) | The same construction for A-entities |
  | **FRESH** (primary negative) | 20 V-pool names × one attribute each, seeded |
  | **TPRE** | Transition cells at sleeps 1–3 (should be UNKNOWN) |
  | **TPOST** | Transition cells at sleeps 4–5 (recall of the life-4 value) |
  | **UPD** (descriptive) | Update cells: which value (old, new, other, UNKNOWN) |

  - **Life-2 entities at sleep 1** are unfamiliar names. Their HNM/ANM probes are excluded at sleep 1; their TPRE
    probes at sleep 1 are reported separately as unfamiliar.
- **Competence.** ppl ratios on competence_eval and competence_modern vs the fp32 base (Gate v2: mean ≤ 1.02,
  per-seed max ≤ 1.05).

## Aggregation (frozen)
- **Per seed:** a measure = class count ÷ probe count of its category. Positive measures use paraphrase probes only;
  template probes are acquisition-only and reported.
- **Pooled:** an **equal-weight mean over the 4 seeds.**
- **Zero denominator** (a category empty for a seed): the measure is missing, and a criterion that needs it **fails**.

## Criteria (development; arm U; every threshold frozen)
**Functional success = D1 ∧ D2 ∧ D3 ∧ D4, all at every sleep unless stated.**

| # | Dimension | Requirement |
|---|---|---|
| D1 | Known answered | POS correct ≥ 0.50 and ≥ P's POS − 0.10. FIRST correct ≥ 0.50 at sleeps 2–5. |
| D2 | Unsupported rejected | For **each** of HNM and FRESH: confabulation ≤ 0.10 **and** exact-UNKNOWN rate ≥ 0.90. Empty answers are reported, never counted as rejection. |
| D3 | UNKNOWN → KNOWN | TPRE UNKNOWN rate ≥ 0.80 at sleeps 2–3 (familiar entities). TPOST correct ≥ 0.50 and ≥ P's TPOST − 0.15 at sleeps 4 and 5. Reported for exposed A-cells, unexposed A-cells and H-cells. Both A-cells (all) and H-cells must pass. |
| D4 | Persistence | RET correct at sleep 5 ≥ P's RET − 0.10. Competence in band at every sleep. |

- **Mechanism-specific evidence (separate claim).** At sleep 5, U's HNM confabulation ≤ G's − 0.30, pooled, **and**
  lower than G's in ≥ 3/4 seeds. Every earlier sleep's contrast is reported.
- **Verdict labels:**
  - `ABST0_FUNCTIONAL_AND_GROUNDED`: functional success, plus the mechanism gate.
  - `ABST0_FUNCTIONAL`: "a functional boundary emerged; an additional benefit from experience-grounded negative
    supervision was not demonstrated."
  - `ABST0_NO_ADVANCE`: functional success not met. Partial patterns are reported, e.g. "abstention at the cost of
    acquisition".
  - `ABST0_INVALID`: U's or P's integrity failed. Scientific outcomes are then descriptive only.
  - **U and P valid, G invalid:** the functional verdict remains assessable. The mechanism comparison is **NOT
    ASSESSABLE** (`ABST0_FUNCTIONAL_MECHANISM_NOT_ASSESSABLE` or `ABST0_NO_ADVANCE_MECHANISM_NOT_ASSESSABLE`). It is
    never reported as "benefit not demonstrated", and FUNCTIONAL_AND_GROUNDED is never awarded with an invalid G.
- **G and P are always reported in full:** POS, FIRST, RET, TPRE, TPOST and the negative classes, so that blanket
  abstention is visible.
- **Every table reports jointly:** answer rate, selective accuracy (correct ÷ substantive answers) and the full outcome
  classes.

## Calibration (seed 16; before any development seed)
- Arm P, sleep 1, fp32-primary evaluation.
- E is the smallest of {4, 8, 16} with template acquisition ≥ 0.80 and competence in band. **STOP if none passes.**
- Also reported: N's HNM/FRESH UNKNOWN rates. **If N's exact-UNKNOWN rate on FRESH is < 0.90, STOP** and revise the
  contract once, logged.

## Frozen specifications (code: scripts/abst0.py; the protocol and code must agree, and any discrepancy is corrected before execution)
- **Prompts.** NEG_GEN, NEG_GEN_FRESH and VERIFY3, verbatim:
```
NEG_GEN = 'Memory: {memory}\n\nWrite three different questions about {e} that this memory does NOT answer. Each question must ask for one specific fact about {e} that the memory does not mention.\nWrite each on its own line as: Q: <question>\n'
NEG_GEN_FRESH = 'Write three different questions about a person named {t}. Each question must ask for one specific fact about {t}.\nWrite each on its own line as: Q: <question>\n'
VERIFY3 = 'Memory: Mara Lind plays the cello every evening.\nQuestion: What instrument does Mara Lind play?\nDoes the memory answer the question? Answer SUPPORTED, UNSUPPORTED or AMBIGUOUS.\nAnswer: SUPPORTED\n\nMemory: Mara Lind plays the cello every evening.\nQuestion: Which city does Mara Lind live in?\nDoes the memory answer the question? Answer SUPPORTED, UNSUPPORTED or AMBIGUOUS.\nAnswer: UNSUPPORTED\n\nMemory: Mara Lind plays the cello every evening.\nQuestion: What about Mara Lind?\nDoes the memory answer the question? Answer SUPPORTED, UNSUPPORTED or AMBIGUOUS.\nAnswer: AMBIGUOUS\n\nMemory: {memory}\nQuestion: {q}\nDoes the memory answer the question? Answer SUPPORTED, UNSUPPORTED or AMBIGUOUS.\nAnswer:'
```
  QA_GEN, ENTAIL, FRAME and the QA filters are CONS-0A's (scripts/cons0a.py at the pinned commit).
- **Generation (corpus).**
  - The frozen bf16 base, greedy (`do_sample=False`), ≤ 120 new tokens, batch 1, left padding, pad = eos.
  - Candidate parsing (`parse_q`): lines matching `Q: <question>` that end in "?", with 3–30 words and no "|".
- **Verifier scoring.**
  - Each label's score is the summed token log-prob of the continuation " SUPPORTED", " UNSUPPORTED" or
    " AMBIGUOUS" after the prompt. There is no terminator token.
  - A label wins only if every score is finite and it is **strictly** greater than both others (`decide3`).
    Anything else is TIE.
  - TIE is never accepted as UNSUPPORTED, and never counts as SUPPORTED exposure.
  - T-name questions are judged against the empty memory "(nothing)".
- **Name rules.**
  - The target name must match exactly as a whole-word string.
  - "Other name": any word of any S-pool name other than the target's own words, case-insensitive. This covers every
    H-entity, the other A-entities and the transients.
  - The leak filter replaces the target name (A or T) by "E" before computing token Jaccard against the fixed
    inventory.
- **G token matching.** A swap is accepted only if the **complete framed training sequence** (qa_ids of the swapped
  question with target "UNKNOWN") has the same length as U's item, and VERIFY3(empty memory) = UNSUPPORTED. T names
  are tried in a deterministic cursor order. If none matches, STOP. Budget equality of U, G and P is asserted at every
  sleep.
- **fp32 evaluation.**
  - Weights: HF `from_pretrained(dtype=float32)` of the manifest bf16 safetensors, an exact upcast.
  - Merge: W += (B @ A) · (α / r) in fp32, in place, in a fresh process, with no adapter modules installed.
  - Numerics: `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`, no
    autocast, eager attention.
  - Generation: `model.generate(do_sample=False, max_new_tokens=8)`, default KV cache, batch 16, left padding,
    pad = eos.
  - The merged model is evaluated in memory. A persisted merged artifact is **not** claimed.

## Enforcement (v3; Sol's freeze review of 2620228; every item has a regression test)
- **Two-phase operation.**
  - **Phase 1** (`scripts/cb_abst0_p1.sh`): boot, then calibration, then prep for seeds 12–15. It records
    `PHASE1.sha256` (calibration result, preps and audit samples, plus PIN and model-manifest hash) and STOPS.
  - Claude then completes the blind audit (`audit_labels_seed{S}.json`, bound to the prep sha256 and to exactly the
    sampled items in order), and the artifacts are committed.
  - **Phase 2** (`scripts/cb_abst0_p2.sh`) verifies `PHASE1.sha256` and the recorded model before anything trains.
- **Training refuses to run without:**
  - a production manifest and a non-test prep built with the same model;
  - a development seed;
  - a **successful** calibration result whose chosen E equals `--passes`;
  - complete, bound audit labels (`validate_audit`).
  - **Exception:** calibration training (`--calibration`) is allowed on seed 16 only.
  - **Audit labels never enter selection.** `auditstats` reports the error rate and the delivered exposure mass of
    audited bad items, from the actual schedule.
- **Calibration** (`calchoose`):
  - every candidate E is validated: seed 16, the train log's E and prep, the checkpoint hash in both evaluations, the
    probe inventory, both monitors finite, and a fresh **unmerged twin** with zero mismatches;
  - N is validated (base, identity, completeness);
  - decision measures are **recomputed from the validated rows**;
  - the result is written, then the command **exits 4 on STOP or on any error**.
- **Analysis.**
  - `validate_eval` checks every evaluation against its *expected* identity: code, model manifest, seed, prep hash,
    sleep, mode, the checkpoint hash recorded for that arm and sleep, and production mode. It also requires exactly
    {eval, modern} finite, the probe inventory equal to `probes(stream, L)`, and outcome classes consistent with the
    answers.
  - Production analysis needs exactly seeds 12–15 and rejects test-mode artifacts.
  - A missing or unrecorded exposure report is an error.
  - Fewer than 2 exposed A-cells in a seed triggers the "not adequately tested" note.
  - Full outcome tables are produced (correct / substantive / unknown / empty per arm × sleep × category × probe
    kind).
  - The nested verdict always equals the top-level verdict.
  - `criteria` requires exactly both competence monitors, finite, at every sleep.
- **Resume.**
  - The code hash covers abst0.py, cons0a.py, cls0c_b.py, lifegen.py, competence.py and the background and competence
    data.
  - Stage provenance includes the model manifest and test mode.
  - Training is skipped only if every checkpoint's sha256 **and** the exposure report's sha256 equal the recorded
    values. Otherwise it retrains from scratch.
- **Corpus.**
  - Duplicates (frozen identity: entity + normalised question; T: normalised question) are rejected before
    verification.
  - The cap of 3 applies to both A and T.
  - Delivered exposure is recorded **per selected item** and summed, never overwritten.
- **Sequence length.** Any scheduled training sequence over 256 tokens (`build_inputs` truncates there) STOPS
  training. The tiny-model plumbing tests are the only exception: their 300-token BPE inflates lengths. Real QA items
  are about 80–100 tokens.

- **v3.1 (Sol's review of a0d970d).**
  - **Gold values.** `validate_eval` compares every evaluator-owned field, **including gold values** (value, old,
    new), with freshly generated probes. All measures are computed by `regrade`, against the fresh gold, never from
    a file. Two twins carrying the same forged gold fail.
  - **Upstream provenance.**
    - Prep and train provenance must equal the frozen code hash and model, checked by train, calchoose and analyze.
    - Phase 1 records `CODE_HASH` (`abst0.py codehash`). Phase 2 runs at a later commit (it adds the audited
      artifacts) and must reproduce the same executable/data hash. The relationship between the two commits is
      written to `PINS.json`.
  - **Evidence preservation.**
    - Every launcher archives and hashes all checkpoints and results on the pod, and verifies the off-pod copy
      before removing it. This includes phase 1's calibration checkpoints.
    - On any failure with evidence on the pod, if preservation fails, the pod is **stopped and kept**, never removed.
  - **Launch guards.** Launchers refuse an unresolved PIN. The boot tests are `-m "not slow"`.

## Integrity and operations
- **Provenance.** Pinned commit; model and tokenizer equal the CLS-0B manifest; run 2's exact pip freeze.
- **Resume key.** Every stage output records sha256 of the code, model manifest, stream, QA, negative corpus, its
  inputs and checkpoint. Resume only when **all** match; otherwise the seed restarts from scratch.
- **GPU (Amendment 1; 2026-11-01; before any ABST-0 data).**
  - The first available card in the fixed order A40, RTX A6000, L40S, RTX 6000 Ada (all 48 GB), on CUDA-13 hosts
    (required by the pinned torch build).
  - Phase 1 records the card type in `PHASE1.sha256`. Phase 2 requests that same card type and fails closed if the pod
    reports a different one, so calibration and every development seed run on one card type.
  - The original "A40 only" rule was copied from MERGE-ENG, which needed run 2's GPU to reproduce its answers. ABST-0
    is a new experiment with its own calibration and has no such need. A40 was unavailable for over 20 minutes on
    phase 1's first launch.
  - The platform is never changed in response to results.
  - **Sol, 2026-11-01:** GO for Amendment 1 and phase 1 at fc4c9a9, with code pinned to 34e1f9d.
    - A different card does not justify relaxing any frozen check.
    - Once calibration begins, the selected card type stays fixed, even if calibration fails.
    - Phase 2 compares GPU names only when both are nonempty: two failed lookups never count as a match.
  - **Launcher incident (2026-11-01).**
    - The first L40S pod booted (BOOT OK), but the launcher never proceeded and nothing ran for about 1 h. The
      suspected cause, not established, is a silent ssh drop with no keep-alive. The pod was removed, with no
      evidence lost.
    - The fix: detached boot polled with a deadline on every iteration, verified upload and start, and ssh keep-alive
      (27240da). There is a regression test (tests/test_abst0_boot.sh).
    - **Sol: GO for phase 1 at 27240da** (code 34e1f9d), after independently verifying both regressions and the
      launcher sha256.
- **Checkpoints.** All checkpoints are kept, with sha256, and downloaded (verified) before the pod is removed.
- **No outcome-driven edits.** Any change is a logged amendment reviewed by Sol.

## Prediction (Claude, before data)
- U meets D2 on HNM and FRESH. P fails D2 completely.
- G reduces FRESH confabulation but much less HNM confabulation, so the mechanism gate passes.
- The main risk is D3 on exposed A-cells.

## Cost (recalculated)
- **Per seed:** 3 arms × 5 sleeps × (fp32 merged + fp32 unmerged), plus N × 5 = 35 fp32 evaluations at about
  3–5 min each, about 2–3 h. Plus training (about 15 min), QA and negative generation (about 15 min).
- **Total:** 4 seeds plus calibration, about 10–13 h on an A40, roughly $6–8. Two pods in parallel would halve the
  wall time.
