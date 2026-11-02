# SUPPORT-1 protocol: can consolidation learn a support rule from exact supervision? (draft v1 for Sol's review; 2026-11-02)

Basis: lab/NEXT_SOL_DRAFT.md (the base design), lab/NEXT_CLAUDE_DRAFT.md (within-entity holdout) and the decision
recorded in lab/ABST0_PROTOCOL.md ("Exchange outcome", 20af43a). Anything not changed here is as in ABST-0 v3,
cited by section. Development experiment. Nothing here authorises spending; David decides and launches.

## 1. Question, scope and claim boundary
- **Question.** Given exact, oracle-defined supervision over its own acquired (entity, attribute) cells, does the
  consolidated cortex learn
  - a support rule, i.e. "no acquired fact for this cell → UNKNOWN", that transfers to cells never trained as
    negatives; or
  - only the particular trained negatives; or
  - nothing usable?
- **Within this run**, the causal contrasts are **grounding** (U vs G at matched dose) and **relative dose** (×1 vs ×4
  at a matched total budget). Comparison with ABST-0 is descriptive only, because the positive pipeline changes too
  (oracle templates replace distilled QA).
- **Claim boundary.**
  - Success shows learnability **under external oracle supervision**; it is not autonomous discovery of an epistemic
    rule.
  - Scope is cortex-only and the episodic store is unused, as in ABST-0 §Question and scope.
  - UNKNOWN means *not supported by the experiences acquired by this checkpoint*, not "false in the world".

## 2. Streams (seeds 20–23 development, seed 24 calibration; never used before in this line)
- **Generator.** `support1.make_stream(seed)`: ABST-0's `make_stream` construction with the RNG key
  `"SUPPORT1-stream-{seed}"`, plus one addition: 4 **fork distractor** facts on 4 reserved transient entities (§7).
  Everything else is identical to ABST-0 §Streams:
  - 9 non-relational attributes;
  - S/T/V name pools, pairwise disjoint by name word;
  - 5 lives;
  - 16 core entities, split A 8 / H 8;
  - 8 transition cells (4 A + 4 H, taught at life 4);
  - 2 update cells;
  - `check_stream` as frozen.
- **Cell strata, frozen at `prep`** from the full 5-life stream; no arm output is used.
  - **Permanently missing:** core (entity, attribute) cells with no fact in any life that are not planned cells.
  - **A, permanently missing:** split once per entity into **SEL** (U-selected) and **HOW** (held-out-within).
    - The split uses sampler key `"SUPPORT1-split-{seed}"`, with a deterministic attribute-balancing rule: for each
      attribute, across A entities, |SEL count − HOW count| ≤ 1.
    - Each A entity has at least 1 SEL and 1 HOW cell; prep STOPs if this is impossible.
    - The split is never redrawn.
  - **REV:** the 4 A transition cells. They receive UNKNOWN supervision until their life-4 fact arrives (§4).
  - **HNM:** H entities' permanently missing cells. They never receive an UNKNOWN target, in any arm (held-out
    across entities).
  - **Excluded from all negatives:** cells acquired at lives 3 or 5 ("second" and "further" facts), H transition
    cells and update cells. This prevents unplanned reversals.
- **FRESH:** 20 V-pool names × one attribute each, seeded, as in ABST-0.

## 3. Oracle teacher and templates (frozen; reviewed by Sol before implementation)
- **Labels** at sleep L, for cell (e, a):
  - the current value, if a fact for (e, a) has been acquired in lives ≤ L (an update takes the latest acquired
    value);
  - otherwise UNKNOWN.
  - Future episodes never enter labels. An independent check recomputes every label from the stream, with the
    temporal mask (§9).
- **Question templates.** Two training wordings per attribute (`QT[a][0]`, `QT[a][1]`), shared by positives and
  negatives:

  | Attribute | QT0 | QT1 |
  |---|---|---|
  | city | Which city does {e} live in? | What is the name of the city {e} calls home? |
  | occupation | What does {e} do for a living? | What is {e}'s line of work? |
  | favorite object | What object does {e} like best? | Which possession does {e} value above all others? |
  | pet species | What kind of animal does {e} keep as a pet? | What species is the pet that {e} looks after? |
  | childhood fear | What frightened {e} as a child? | What was {e} scared of when young? |
  | workplace | Where does {e} work? | What is the name of the place where {e} is employed? |
  | hobby | What does {e} do for fun? | Which pastime does {e} enjoy? |
  | birth month | In which month does {e} have a birthday? | What month was {e} born in? |
  | allergy | What is {e} allergic to? | What substance causes {e} an allergic reaction? |

  - **Leak rule** (enforced and tested). Every QT wording, with {e} removed, has token Jaccard < 0.8 against every
    Statement probe stem (lifegen TRAIN[a][0] and PARAPHRASE[a]) with {e} removed. Any wording that fails is
    replaced, and the replacement is logged before freeze.
- **Training sequence.** As in CONS-0A and ABST-0: `qa_ids(tok, QT[a][k].format(e=e), label)`, i.e. FRAME ∘
  "Question: …", then " " + label + "\n", with loss on the label and the terminator. Sequences over 256 tokens STOP
  (ABST-0 §Enforcement).
- **Positive items for life L:** each life-L fact × {QT0, QT1}, so n_L = 2 × |life-L facts|. Positive items are
  identical across arms.

## 4. Training (per seed; arms share the cortex initialisation, positives, schedule key and slot positions)
- **Cortex and optimiser:** as ABST-0 (CONS-0A Amendment 2): rank 16, α 32, q/k/v/o/gate/up/down, fp32, AdamW
  lr 2e-4, batch 8. Optimiser state persists across sleeps.
- **E = 4, fixed.** Calibration (§8) verifies it; it does not choose E.
- **Sleep L, with n = n_L.** Three slots, 16n items in total, in every arm:
  1. **New:** E·n = 4n presentations of the life-L positive items.
  2. **Replay:** 4n, cycling earlier positive items in seeded order. At sleep 1 this slot is background, identical
     across arms.
  3. **Third:** 8n positions, filled as follows.

  | Arm | Third slot (8n positions) |
  |---|---|
  | **P** | 8n background |
  | **U1** | 2n UNKNOWN items (n A + n T), then 6n background |
  | **G1** | U1's positions. Each A item is name-swapped to a T name of identical framed token length (ABST-0 G rule); T items are unchanged. |
  | **U4** | 8n UNKNOWN items (4n A + 4n T) |
  | **G4** | U4's positions, swapped as in G1 |
  | **N** | Base model, no training |

- **A-item allocation within the A positions**, with sampler key `"SUPPORT1-neg-{seed}-{L}-{dose}"`:
  1. **REV quota first.** For each REV cell whose entity is familiar (introduced ≤ L) and whose fact has not
     arrived (L ≤ 3):
     - q = 2 presentations in U1 and 8 in U4;
     - wordings alternate QT0 / QT1;
     - quotas therefore keep the 1:4 dose ratio.
  2. **The remaining A positions** cycle over SEL cells of familiar A entities, in seeded order, alternating wordings.
  3. HOW, HNM, excluded cells and unfamiliar entities never appear.
  - Prep STOPs if the REV quota exceeds the A positions at any sleep.
  - Delivered per-cell, per-wording counts are logged.
- **T items:** T-pool names × the 9 attributes × {QT0, QT1}, label UNKNOWN, cycled in seeded order. They are disjoint
  from the T names used for G swaps at the same sleep (stated in the schema).
- **Background:** CONS-0A background chunks, matched in token length and loss tail to the item they replace, as in
  ABST-0. They are identical across arms at identical positions.
- **Position order:** one seeded interleaving per (seed, sleep), shared by all arms. An arm differs only in what
  occupies third-slot positions.
- **Budget (fail-closed):** items, input tokens and loss tokens are asserted equal across P/U1/G1/U4/G4 at every
  sleep. Positive positions and contents are asserted identical.

## 5. Evaluation (as ABST-0 §Evaluation unless stated)
- **Representation and integrity:** unchanged.
  - fp32 merged primary, in a fresh process, with TF32 and autocast off.
  - An fp32 unmerged twin for every arm × sleep.
  - Integrity: identical probes, 0 answer and 0 outcome mismatches, |Δppl| ≤ 1e-4.
  - Frozen FRAME decoding: greedy, ≤ 8 new tokens, first line.
- **Normalisation (explicit; the ABST-0 clarification):**
  - lowercase;
  - characters outside [\w\s'-] become spaces;
  - whitespace collapsed and trimmed;
  - one leading "a", "an" or "the" removed.

  This is cls0c_b.norm. Outcome classes:
  - **correct:** norm(answer) == norm(current value);
  - **unknown:** norm(answer) == "unknown";
  - **empty:** norm(answer) == "";
  - **substantive:** anything else (on negatives, this is confabulation).
- **Statement probes at sleep L** (template and entailed-paraphrase stems, as ABST-0):

  | Category | Contents |
  |---|---|
  | POS, FIRST, RET, TPOST, UPD | As ABST-0 |
  | TPRE-A / TPRE-H | REV cells / H transition cells at sleeps 1–3 (unfamiliar entities reported separately, as ABST-0) |
  | **SEL** | Every SEL cell of familiar A entities (template stem) |
  | **HOW** | Every HOW cell of familiar A entities (template stem) |
  | **HNM** (primary negative) | H permanently-missing cells of familiar entities; up to 40, sampler key `"SUPPORT1-hnm-{seed}-{L}"` |
  | **FRESH** (primary negative) | As ABST-0 |

- **Question diagnostics (QDIAG; 64 rows per evaluation):** the training construction, FRAME ∘ "Question: QT[a][k]".
  - 12 POS, 12 SEL, 12 HOW and 12 HNM rows: seeded samples, with wordings alternating.
  - **16 GTQ** rows: the exact name-swapped T-name questions that G4 trained at this sleep, or the latest sleep
    before it. They are asked in every arm, which gives G's own acquisition. In U arms the same rows are
    fresh-name questions.
  - Rows are fixed per (seed, sleep) and identical across arms.
- **Competence:** unchanged (Gate v2, both monitors).

## 6. Criteria (development; every threshold frozen before data)
- **Aggregation:** as ABST-0 §Aggregation.
  - Per-seed class count ÷ category count.
  - An equal-weight mean over 4 seeds.
  - A zero denominator gives a missing measure, and a criterion that needs it fails.
  - Positive measures use paraphrase probes; template probes are acquisition-only.
- **Primary functional claim, for arm U4 against P:** D1–D4 exactly as ABST-0 §Criteria (thresholds unchanged).
  - D2 uses HNM and FRESH.
  - D3's A cohort is REV, with the A and H cohorts required separately.
- **Primary mechanism claim:** at sleep 5, U4's HNM confabulation ≤ G4's − 0.30, pooled, **and** lower in ≥ 3/4
  seeds. Every sleep's contrast is reported.
- **Verdicts:**
  - `SUPPORT1_FUNCTIONAL_AND_GROUNDED`;
  - `SUPPORT1_FUNCTIONAL`;
  - `SUPPORT1_NO_ADVANCE`;
  - `SUPPORT1_INVALID` (U4 or P integrity failed);
  - with G4 invalid, `…_MECHANISM_NOT_ASSESSABLE`.
  - These follow ABST-0's rules, with U → U4 and G → G4.
- **Diagnostic readings.** Frozen, descriptive and not verdicts. They are computed at sleep 5 for U4 (and for U1),
  pooled, with per-seed values:
  - **ACQ (trained negatives acquired):** QDIAG SEL UNKNOWN ≥ 0.80.
  - **R-UNLEARNED:** not ACQ. Optimisation, masking or capacity is then investigated; no held-out conclusion is drawn.
  - **R-CONSTRUCTION:** ACQ, but Statement SEL UNKNOWN < 0.50.
  - **Cohort-matched generalisation ratios** (Sol's definition):
    GR_within = (UNK_U HOW − UNK_P HOW) ÷ (UNK_U SEL − UNK_P SEL), on Statement probes. GR_across is the same with HNM
    in place of HOW. Both gains and per-seed values are reported.
  - **GR small-gain stability rule:** GR is **undefined** when the pooled denominator gain is < 0.10, or when fewer
    than 3 seeds have a per-seed denominator gain ≥ 0.05.
  - **R-GENERALIZES-WITHIN:** Statement HOW UNKNOWN ≥ 0.70 and HOW confabulation ≤ 0.30.
  - **R-GENERALIZES-ACROSS:** Statement HNM UNKNOWN ≥ 0.50.
  - **R-MEMORIZES:** ACQ, Statement SEL UNKNOWN ≥ 0.80 and GR_within ≤ 0.25 (or HOW UNKNOWN ≤ P's HOW + 0.20 when GR
    is undefined).
  - **R-GENERIC:** U4 HOW UNKNOWN ≥ 0.50 and |U4 − G4| on HOW < 0.20, i.e. a rule not tied to the experienced entity.
  - **Dose:** U4 − U1 on QDIAG SEL, Statement SEL, HOW and HNM. "Dose-responsive" means U4 − U1 ≥ 0.20 on QDIAG SEL.
  - **Reversal:**
    - *Pre-transition acquisition:* REV Statement UNKNOWN ≥ 0.80 at sleep 3 in U4. If this fails, the reversal is
      "not established", and that is still reported.
    - *Reversal success:* acquisition, plus TPOST-A correct ≥ 0.50 and ≥ P's TPOST-A − 0.15 at sleeps 4 and 5.
  - **G's own acquisition:** GTQ UNKNOWN in G4, reported next to U4's QDIAG SEL.
- **Interpretation** (frozen; Sol's table, with these labels). This is what each outcome earns next:
  - **R-UNLEARNED in both U1 and U4:** fix optimisation/masking.
  - **R-CONSTRUCTION:** construction transfer is the bottleneck; next experiment trains both constructions with
    held-out aliases.
  - **R-GENERIC with functional progress:** generic suppression suffices; grounding is not shown.
  - **U4 > U1 and U4 > G4, with positives intact:** an experience-sensitive benefit; a replication/transfer study
    is earned.
  - **R-MEMORIZES (ACQ, with H and fork failing):** stop cortex-only abstention tuning; next is an explicit support
    state or provenance representation.
  - **High dose breaks D1/D4:** a tradeoff failure; increased UNKNOWN alone is never reported as progress.

## 7. Paired-history fork (secondary; included only if the §8 preflight passes; decided before any scientific score)
- **When.** After sleep 3, for U4 and G4 in each seed, the full state is saved: weights, optimiser and all RNG
  states.
- **Observed branch** (the main run's sleep 4) teaches the 4 H transition facts.
- **Withheld branch** replaces them with the 4 reserved fork-distractor facts, as follows:
  - same attributes, on reserved transient entities, with values from the pools;
  - token- and loss-matched per item;
  - identical positions;
  - every other input identical, including A negatives.
- **Measurement.** The withheld branch is evaluated at sleep 4 only, merged plus unmerged: 2 arms × 4 seeds × 2 = 16
  evaluations. Measured on the 4 H target cells, Statement and QDIAG-Question:
  - "history-sensitive": observed-branch correct ≥ 0.50 and withheld-branch UNKNOWN ≥ 0.80;
  - reported for U4 and G4 side by side.
  - The unit of replication is the seed's paired branch.
- **The H targets never receive UNKNOWN targets in either branch** (mechanically checked).

## 8. Phases and preflight
- **Phase 1** (one pod; STOP for review at the end):
  1. **Prep** for seeds 20–24 (CPU-deterministic): streams, strata, split, every schedule and every probe set.
     Independent validation by Sol's validator before any GPU spend (§9).
  2. **Calibration** (seed 24, arm P, sleep 1, E = 4): template acquisition ≥ 0.80 and competence in band;
     N FRESH exact-UNKNOWN ≥ 0.90. Failure is a **STOP** plus an amendment; E is never tuned silently.
  3. **Fork restoration test** (seed 24, U4): run sleeps 1–3, save state, then run sleep 4 twice (uninterrupted, and
     resumed from the saved state). The two sleep-4 checkpoint sha256 must be identical; otherwise the fork is
     recorded as **unrun**.
  4. **Throughput:** the timing of steps 2–3 gives the projected full-run hours. If the projection exceeds the
     ceiling (§10), the fork is deferred first, then QDIAG is reduced to 32 rows. Arms and seeds are never dropped.
     The decision is logged before phase 2.
- **Phase 2:** verify the phase-1 manifest, CODE_HASH and GPU (Amendment 1 rules); seeds 20–23, all arms, fork if
  admitted; analyze with both analyzers. Preserve, verify, download, then remove the pod (ABST-0 launcher contract,
  including exactly-once start).

## 9. Artifact schema (frozen before the coding split)
- **`prep_seed{S}.json`** contains:
  - `stream` and its sha256;
  - `strata`: SEL, HOW, REV, HNM-eligible and excluded cells, each as [entity, attribute];
  - `split_key`;
  - `positives` per sleep, each as {cell, wording k, label, text};
  - `schedule[arm][sleep]`, with per-position entries {slot, kind ∈ {pos, replay, bg, negA, negT}, cell or name,
    wording, label, tokens, loss_tokens};
  - `sampler_keys`;
  - `probes[sleep]`: Statement rows and QDIAG rows, each with cat, key, construction ∈ {statement, question},
    kind, item text and gold;
  - `fork` (distractor facts, withheld schedule);
  - `normalisation` (§5 rule text and code reference).
- **`train_seed{S}.json`:** per arm/sleep checkpoint sha256; delivered per-cell, per-wording exposure counts;
  budget triple; provenance (code hash, prep sha256, model manifest).
- **`eval_{arm}_s{L}_{merged|unmerged}.json` and `eval_N_s{L}.json`:** rows {cat, key, construction, kind, item,
  value, answer, outcome}; ppl ratios; provenance.
- **Independent checks (Sol's validator)** recompute from the stream alone:
  - every gold label, with the temporal mask;
  - strata membership and the split balance rule;
  - the leak rule;
  - name-pool disjointness;
  - the REV quota and "no UNKNOWN on H / HOW / excluded / unfamiliar";
  - G token matching;
  - budget equality.
  Any failure is a STOP.

## 10. Cost and ceiling
- **Evaluations:** 5 trained arms × 5 sleeps × 2 (merged, unmerged) + N × 5 = 55 per seed, so 220. With the fork,
  236.
- **Estimate.** ABST-0 phase 2 measured about 2.7 min per evaluation including training on an L40S (140 evaluations,
  about 6.3 h). Training here is about 2.7× ABST-0's per seed.
  - Projection: about 11–13 h for phase 2, plus about 1.5 h for phase 1.
  - About $12–14 at observed L40S rates.
- **Ceiling: $30 total** (Sol's ceiling). The phase-1 throughput measurement replaces this projection.

## 11. Prediction (Claude, before data)
- U4 acquires its trained negatives (ACQ) and transfers them to Statement SEL.
- HOW rises above P but stays below 0.70, so a partial rule. HNM stays near ABST-0 levels.
- U4 − G4 on HOW is at least 0.10, but the mechanism gate (on HNM) fails.
- Most likely verdict: SUPPORT1_NO_ADVANCE, with a GR_within of about 0.3–0.5. This is my prediction, not a
  criterion.
