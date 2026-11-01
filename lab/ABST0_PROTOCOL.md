# ABST-0 protocol: learning a support boundary during consolidation (v1 for review; 2026-11-01)

Claude (build), Sol (review). This supersedes lab/ABST0_PROPOSAL.md and incorporates Sol's required revisions of
2026-11-01.

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

### Negative corpora (stage `neg`; built once per seed BEFORE any training; arm-independent)
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
  accepted set is cycled in seeded order. Repetitions and per-item exposure are logged. If an A half has zero
  accepted items, it is filled with T-name negatives, and this is logged and reported.
- **Construction-family separation.** All training items use the Question construction ("Question: …"). All
  evaluation probes use the Statement construction ("Statement: <stem> ..."). No training item uses the Statement
  construction.
- **Logged and hashed:** candidates, rejections with reasons, accepted sets, selections and exposures.
- **Label-quality audit (predeclared, blind).**
  - A seeded sample of 20 accepted A-entity negatives per seed per sleep is written to `audit_sample.json`.
  - Claude audits it **before any training**, seeing only the memory text and the question, and records
    supported / unsupported / ambiguous.
  - The error rate is reported. Findings cannot trigger selective repairs.
- **Transition-exposure categorisation** (frozen at `neg`, before outcomes; reporting only).
  - An A-transition cell counts as *exposed* if any accepted negative about that entity at sleeps 1–3 is judged
    SUPPORTED by VERIFY3 against the cell's life-4 episode text alone.
  - This uses the future episode **only** to categorise. It never selects or forces negatives.
  - D3 is reported for: A-cells with prior UNKNOWN supervision, A-cells without it, and H-cells. If exposed A-cells
    are absent or sparse (< 2 per seed on average), the report says "reversal of explicitly trained UNKNOWN was not
    adequately tested".

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
  - `ABST0_INVALID`: U's or P's integrity failed.
- **G and P are always reported in full:** POS, FIRST, RET, TPRE, TPOST and the negative classes, so that blanket
  abstention is visible.
- **Every table reports jointly:** answer rate, selective accuracy (correct ÷ substantive answers) and the full outcome
  classes.

## Calibration (seed 16; before any development seed)
- Arm P, sleep 1, fp32-primary evaluation.
- E is the smallest of {4, 8, 16} with template acquisition ≥ 0.80 and competence in band. **STOP if none passes.**
- Also reported: N's HNM/FRESH UNKNOWN rates. **If N's exact-UNKNOWN rate on FRESH is < 0.90, STOP** and revise the
  contract once, logged.

## Integrity and operations
- **Provenance.** Pinned commit; model and tokenizer equal the CLS-0B manifest; run 2's exact pip freeze.
- **Resume key.** Every stage output records sha256 of the code, model manifest, stream, QA, negative corpus, its
  inputs and checkpoint. Resume only when **all** match; otherwise the seed restarts from scratch.
- **GPU.** A40 (run 2's GPU). The platform is recorded before calibration and never changed in response to results.
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
