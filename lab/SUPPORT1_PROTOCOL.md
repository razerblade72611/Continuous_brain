# SUPPORT-1 protocol: can consolidation learn a support rule from exact supervision? (v2.1 for Sol's review; 2026-11-02)

- **Basis:** lab/NEXT_SOL_DRAFT.md (the base design), lab/NEXT_CLAUDE_DRAFT.md (within-entity holdout), and the
  decision in lab/ABST0_PROTOCOL.md "Exchange outcome" (20af43a).
- **v2** answers Sol's HOLD on v1 (1caeeec), items S1–S8. Each change below is tagged with its item.
- **v2.1** answers Sol's errata HOLD on v2 (8322c55), items E1–E4.
- Anything not changed here is as in ABST-0 v3, cited by section.
- Development experiment. Nothing here authorises spending; David decides and launches.

## 1. Question, scope and claim boundary
- **Question.** Given exact, oracle-defined supervision over its own acquired (entity, attribute) cells, does the
  consolidated cortex learn
  - a support rule ("no acquired fact for this cell → UNKNOWN") that transfers to cells never trained as negatives;
    or
  - only the particular trained negatives; or
  - nothing usable?
- **Causal contrasts within this run:**
  - **grounding:** U vs G at matched dose;
  - **relative dose:** the same negative examples presented ×1 vs ×4 within a matched total budget.

  Comparison with ABST-0 is descriptive only, because the positive pipeline changes too.
- **Claim boundary.**
  - Success shows learnability **under external oracle supervision**.
  - Scope is cortex-only; the episodic store is unused.
  - UNKNOWN means *not supported by the experiences acquired by this checkpoint*.

## 2. Streams and strata (seeds 20–23 development, 24 calibration)
- **Main stream.** `support1.make_stream(seed)` is ABST-0's `make_stream` construction byte for byte, with RNG key
  `"SUPPORT1-stream-{seed}"`. No facts are added to it.
  - 9 non-relational attributes; S/T/V pools disjoint by name word; 5 lives.
  - 16 core entities, split A 8 / H 8.
  - 8 transition cells (4 A + 4 H, at life 4); 2 update cells.
  - `check_stream` as frozen.
- **Fork record (S6).** Generated separately into `prep.fork`, never into `stream`:
  - 4 reserved transient entities from S, drawn after the stream is generated;
  - 4 distractor facts, one per H transition attribute, with values from the attribute pools excluding every stream
    value.
- **T pool partition (S1), global across all sleeps.** The T pool (400 names) is split once with key
  `"SUPPORT1-tpart-{seed}"` into:
  - **T_neg** (200): names for T-name negatives;
  - **T_swap** (200): names for G swaps.

  A T_swap name never appears in any U arm at any sleep, so every GTQ row is a fresh name for U arms.
- **§2.1 RNG and ordering contract (E3).**
  - Every random operation uses `random.Random(key)` (Python 3.12 string seeding), with the key string given at its
    use site.
  - **Canonical population orders, used before any shuffle or sample:**
    - names: `sorted(str)`;
    - attributes: the ATTRS9 order;
    - cells: `sorted` by (entity, ATTRS9 index);
    - (cell, wording) pairs: by (cell order, wording);
    - T entries: (name, ATTRS9 index, wording).
  - **Operations:**
    - shuffle = `rng.shuffle(list)`;
    - sample k without replacement = `rng.sample(list, k)`;
    - coin = `rng.random() < 0.5`.
  - **Cyclers** (SEL cycler, T cycler, replay cycler). Each is built once per seed from its canonical population,
    shuffled once with its key (`"SUPPORT1-selcyc-{seed}"`, `"SUPPORT1-tcyc-{seed}"`, and
    `"SUPPORT1-replay-{seed}-{L}"` for replay). They are consumed in order, wrapping around.
    - The SEL and T pointers persist across sleeps. An entry not eligible at L (e.g. an unfamiliar entity) is skipped:
      the traversal cursor **advances**, but no presentation quota is consumed. If a full traversal finds no eligible
      entry, STOP (Sol's clarification at GO).
    - The replay cycler is rebuilt per sleep.
  - **T partition:** `t = sorted(T)`; `random.Random("SUPPORT1-tpart-{seed}").shuffle(t)`; T_neg = t[:200] and
    T_swap = t[200:].
  - **Swap choice:** for each A entry in B_L order, the first unused name in `sorted(T_swap)` shuffled with
    `"SUPPORT1-swap-{seed}"` whose framed sequence has equal token length. STOP if none.
  - **HNM, FRESH and QDIAG samples:** `rng.sample` on the canonical population, with the keys given in §5.
- **Strata (frozen at prep, from the full 5-life stream):**
  - **PM** (permanently missing): core (e, a) with no fact in any life that is not a planned cell.
  - **SEL / HOW (S4):** each A entity's PM cells are split.
    - **Algorithm** (exact RNG operations in §2.1). For r = 0, 1, …, 99,999, with `rng = random.Random("SUPPORT1-split-{seed}-{r}")`,
      take each A entity in sorted(A) order:
      - `atts` = the PM attributes of e **in ATTRS9 order** (the §2.1 canonical attribute order); `rng.shuffle(atts)`;
      - `heads = rng.random() < 0.5`;
      - k = len(atts) and h = ⌈k/2⌉. If heads, SEL = atts[:h] and HOW = atts[h:]; otherwise HOW = atts[:h] and
        SEL = atts[h:].
      - Accept the first r for which every attribute has |#SEL − #HOW| ≤ 1 across A entities.
    - Per-entity |SEL − HOW| ≤ 1 holds by construction. Every A entity has k ≥ 2 (prep STOPs otherwise), so each has
      ≥ 1 SEL and ≥ 1 HOW.
    - STOP if no r ≤ 99,999 passes. The accepted r is recorded.
  - **REV:** the 4 A transition cells.
  - **HNM:** H entities' PM cells.
  - **Excluded from all negatives:** cells acquired at lives 3 or 5, H transition cells, update cells.
- **SEL-delivered vs SEL-pending (S4).** At sleep L, a SEL cell is *delivered* once any wording of it has been
  scheduled in **U1** at a sleep ≤ L. By nesting (§4) U4 then holds the same items. Otherwise it is *pending*. Only
  delivered cells form the trained cohort; pending cells are reported separately. REV is never part of SEL, HOW or
  GR.
- **FRESH:** 20 V names × one attribute each, as in ABST-0.

## 3. Oracle teacher, templates, normalisation
- **Oracle label** at checkpoint L for cell (e, a):
  - the value of the latest fact for (e, a) with life ≤ L, if any;
  - otherwise "UNKNOWN".

  **History** is the main stream, or for the fork's withheld branch, the main stream with the 4 H transition facts
  removed and the 4 distractor facts added (§7).
- **Question templates** QT[a][k], k ∈ {0, 1}, shared by positives and negatives. Unchanged from v1:

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

  - **Leak rule:** token Jaccard < 0.8 against every Statement stem, with {e} removed. The maximum is 0.75, verified
    independently by Claude and Sol. The threshold is kept.
- **Training sequence.** `qa_ids(tok, QT[a][k].format(e=e), label)`, i.e. FRAME ∘ "Question: …", then " " + label +
  "\n", with loss on the label tokens plus the terminator. STOP if the sequence is over 256 tokens.
- **Normalisation (S7), literal.** Python 3 `re`, Unicode-aware \w:
  ```
  def norm(s):                       # s: str or None
      s = re.sub(r"[^\w\s'-]", " ", (s or "").lower())
      s = re.sub(r"\s+", " ", s).strip()
      return re.sub(r"^(a|an|the)\s+", "", s)
  ```
  - **Outcome classes:**
    - **correct:** norm(ans) == norm(gold), gold ≠ "UNKNOWN";
    - **unknown:** norm(ans) == "unknown";
    - **empty:** norm(ans) == "";
    - **substantive:** otherwise.
  - "ans" is the first line of the greedy decode.
  - Both implementations must pass these test vectors (input → norm):

    | Input | norm |
    |---|---|
    | "a draembrernhound" | "draembrernhound" |
    | "A Draembrernhound" | "draembrernhound" |
    | "UNKNOWN" | "unknown" |
    | "Unknown." | "unknown" |
    | "unknown: Leston" | "unknown leston" (substantive) |
    | "the  Fo Works!" | "fo works" |
    | "an apple-pie" | "apple-pie" |
    | "O'Brien's pet" | "o'brien's pet" |
    | "The" | "the" |
    | "  " | "" |
    | "" | "" |
    | "the unknown" | "unknown" |
    | "Answer: kalgrirnwick" | "answer kalgrirnwick" |
    | "a  b" | "b" |
    | "théo" | "théo" |

## 4. Training (per seed; arms share initialisation, positives, schedule key and slot positions)
- **Cortex and optimiser:** as in ABST-0. **E = 4, fixed;** calibration verifies it.
- **Positives at sleep L.** Each life-L fact × {QT0, QT1}, labelled with its value: n_L = 2 × |life-L facts| items.
- **Replay (S2).** The replay slot cycles all positive items of lives < L, in seeded order (key
  `"SUPPORT1-replay-{seed}-{L}"`), **relabelled with the oracle at checkpoint L**.
  - The superseded life-2 update item is therefore replayed with the life-4 value from sleep 4 on.
  - Each replay entry records `source_item_id` and the relabelled target.
  - The validator checks every scheduled positive (new and replay) against the oracle at L.
- **Sleep L, with n = n_L.** 16n items: new 4n; replay 4n (background at sleep 1); third slot 8n.
- **Nested negative base block (S1),** with key `"SUPPORT1-neg-{seed}-{L}"`. Each sleep builds one base block B_L of
  2n entries:
  - **A part (n entries):**
    - first the REV quota: 2 entries per familiar, not-yet-acquired REV cell (L ≤ 3), as QT0 then QT1;
    - then SEL cells of familiar A entities, cycled in seeded order with a pointer that persists across sleeps,
      wordings alternating per cell.
    - STOP if the REV quota is greater than n.
  - **T part (n entries):** T_neg names × the 9 attributes × {QT0, QT1}, cycled in seeded order with a persistent
    pointer. Label UNKNOWN.
  - **Third slot by arm:**

    | Arm | Third slot (8n positions) |
    |---|---|
    | P | 8n background |
    | U1 | B_L once (2n), then 6n background |
    | U4 | B_L four times (8n) |
    | G1 | U1 with each A entry swapped |
    | G4 | U4 with each A entry swapped, using G1's mapping |
    | N | Base model, no training |

  - So **U4 presents exactly the U1 items, each 4×**. Dose changes frequency, not selection.
  - **Swap mapping** (shared by G1 and G4): each A entry (e, a, k) maps to a T_swap name t whose framed training
    sequence has token length equal to the original. t is chosen deterministically, by key
    `"SUPPORT1-swap-{seed}-{L}"`, from unused T_swap names.
  - The four repetitions within U4/G4 occupy positions in the shared seeded interleave. Their order is a seeded
    permutation of the 4 copies, identical in U4 and G4.
- **Background:** CONS-0A background chunks, token- and loss-tail-matched to the entry they stand in for, at
  identical positions across arms.
- **Budget (fail-closed):** at every sleep, the (items, input tokens, loss tokens) of all five trained arms are
  equal, and positive positions and contents are identical.
- **Delivered exposure:** per arm, sleep, cell (or name) and wording, logged in `train_seed{S}.json`.

## 5. Evaluation (as ABST-0 §Evaluation unless stated)
- **Unchanged from ABST-0:**
  - fp32 merged primary, in a fresh process, with TF32 and autocast off;
  - an fp32 unmerged twin;
  - integrity: identical probe IDs, 0 answer and 0 outcome mismatches, |Δppl| ≤ 1e-4 on **both** competence
    monitors against the same fp32 base denominator;
  - FRAME decoding: greedy, ≤ 8 new tokens, first line.
- **Statement probes at sleep L:**
  - **As ABST-0:** POS, FIRST, RET, TPOST, UPD.
    - UPD rows carry `gold_old`, `gold_new` and `gold_current`. Before sleep 4, `gold_new` is null.
  - **TPRE-A (REV) and TPRE-H:** sleeps 1–3, with unfamiliar entities reported separately.
  - **SEL-D, SEL-P and HOW:** every delivered SEL, pending SEL and HOW cell of familiar A entities (template stem).
  - **HNM:** ≤ 40, key `"SUPPORT1-hnm-{seed}-{L}"`.
  - **FRESH:** as ABST-0.
- **QDIAG (S3).** Question construction, FRAME ∘ "Question: QT[a][k]". The rows for each (seed, sleep) are fixed
  and identical in every arm.

  | Group | Full (64) | Fallback (32) | Sampling |
  |---|---|---|---|
  | POS-Q | 10 | 4 | acquired non-update facts, wording alternating |
  | SEL-Q | 12 | 8 | **delivered** (cell, wording) pairs from U1's schedule at sleeps ≤ L; if fewer are available, all are used, with no substitution, and the row count is logged |
  | HOW-Q | 12 | 6 | HOW cells of familiar entities, wording alternating |
  | HNM-Q | 10 | 4 | from the HNM Statement set |
  | REV-Q | 4 | 4 | each REV cell in its fixed wording QT0, every sleep (before and after the transition) |
  | GTQ | 16 | 6 | the exact swapped (t, a, k) entries delivered in G1 (= G4) at sleeps ≤ L |

  - Sampling uses key `"SUPPORT1-qdiag-{seed}-{L}"`, without replacement and with deterministic order.
  - **Matched construction pairs:** each SEL-Q and HOW-Q row stores `pair_probe_id` pointing to the Statement probe
    for the same cell. The construction contrast (Question − Statement) uses only these paired cells, with equal
    weight per cell.
- **Competence:** unchanged (Gate v2, both monitors).

## 6. Criteria and readings (development; frozen before data)
- **Aggregation:** as ABST-0, with equal-weight means over 4 seeds. A zero denominator gives a missing measure, and
  a criterion that needs it fails.
- **Primary functional claim (U4 vs P):** ABST-0 D1–D4 with unchanged thresholds.
  - D2 uses HNM and FRESH.
  - D3 uses TPRE-A/TPOST-A (REV) and TPRE-H/TPOST-H, each required, on Statement probes.
- **Primary mechanism claim:** at sleep 5, U4's HNM confabulation ≤ G4's − 0.30, pooled, **and** lower in ≥ 3/4
  seeds.
- **Verdicts:**
  - `SUPPORT1_FUNCTIONAL_AND_GROUNDED`, `SUPPORT1_FUNCTIONAL`, `SUPPORT1_NO_ADVANCE`;
  - `SUPPORT1_INVALID` (U4 or P invalid);
  - `…_MECHANISM_NOT_ASSESSABLE` (G4 invalid).
  - **An invalid U1 or G1 (S5)** makes every dose comparison involving it NOT ASSESSABLE. The U4 primary verdict
    remains assessable.
- **Readings (S5).** Descriptive, at sleep 5 unless stated, for U4 (and U1). Each is reported as pooled, per-seed,
  and with numerator/denominator, **always next to P-relative gains and that arm's D1/D4 guard status**. A reading
  with a failed guard is marked "guard failed" and earns no learning interpretation.
  - **Gains:** Δ_X = UNK_arm(X) − UNK_P(X) for X ∈ {SEL-D, HOW, HNM}.
  - **Pooled GR_within** = mean_s Δ_HOW ÷ mean_s Δ_SEL-D. **GR_across** is the same with HNM.
    - The pooled value is undefined if mean_s Δ_SEL-D < 0.10, or if fewer than 3 seeds have Δ_SEL-D ≥ 0.05.
    - The per-seed GR is undefined when that seed's Δ_SEL-D < 0.05.
  - **ACQ:** SEL-Q UNKNOWN ≥ 0.80.
  - **NOT-ACQ:** SEL-Q < 0.80. No held-out conclusion is drawn.
  - **CONSTRUCTION GAP:** ACQ, and on paired cells Question − Statement ≥ 0.30.
  - **WITHIN-TRANSFER:** HOW UNKNOWN ≥ 0.70, HOW confabulation ≤ 0.30 and Δ_HOW ≥ 0.40.
  - **ACROSS-TRANSFER:** HNM UNKNOWN ≥ 0.50 and Δ_HNM ≥ 0.30.
  - **MEMORIZATION PATTERN:** ACQ, SEL-D Statement UNKNOWN ≥ 0.80, GR_within defined and ≤ 0.25. If GR is undefined,
    the interpretation is "unresolved", with no fallback.
  - **HOW GROUNDING CONTRAST:** U4 − G4 on HOW UNKNOWN. Below 0.20 means "a grounding contrast on HOW was not
    demonstrated". Generic suppression is then *one compatible explanation*, not a finding.
  - **DOSE:** U4 − U1 on SEL-Q, SEL-D, HOW and HNM. "Dose-responsive" means U4 − U1 ≥ 0.20 on SEL-Q.
  - **REVERSAL (E1).** Uses the exact QT0 question per REV cell.
    - **Pre-transition acquisition:** at sleep 3, REV-Q UNKNOWN on ≥ 3 of the 4 REV cells **in at least 3 of 4
      seeds**. The pooled rate is reported too, but does not substitute.
    - Without acquisition, the reversal is "not established".
    - **With acquisition, reversal success** = REV-Q correct (the life-4 value) ≥ 0.50 pooled **and** ≥ P's REV-Q
      correct − 0.15, at **both** sleeps 4 and 5.
    - Statement TPOST-A is reported alongside, and the primary Statement D3 is unchanged.
  - **G's own acquisition:** GTQ UNKNOWN in G4, next to U4's SEL-Q.
- **Interpretation** (frozen; Sol's table):
  - NOT-ACQ in both U1 and U4 → fix optimisation or masking.
  - CONSTRUCTION GAP → next, train both constructions with held-out aliases.
  - Grounding not demonstrated, with functional progress → grounding not shown.
  - U4 > U1 and U4 > G4, with positives intact → a replication/transfer study is earned.
  - MEMORIZATION PATTERN, with ACROSS-TRANSFER and the fork failing → stop cortex-only abstention tuning; next is an
    explicit support or provenance representation.
  - Guards failing at high dose → a tradeoff failure.

## 7. Paired-history fork (secondary; admitted only by the §8 preflight)
- **Histories (S6):**
  - `observed`: the main stream;
  - `withheld`: the main stream, minus the 4 H transition facts, plus the 4 distractor facts from `prep.fork`.

  The oracle labels of each are computed from its own history. In `withheld`, the H target cells have gold
  "UNKNOWN".
- **When.** After sleep 3, for U4 and G4. The parent state ID is the canonical state hash of the sleep-3 state.
- **The withheld branch's sleep-4 schedule (E2).**
  - The 4 H transition facts give 8 base positive items (2 wordings each). At E = 4 these occupy **32 scheduled
    presentations** in the sleep-4 new slot.
  - **All 32** are replaced, position for position, by the 8 distractor items × 4, token- and loss-matched.
  - The validator asserts that the withheld schedule contains **zero** positive supervision for any H target cell.
  - These are life-4 items, so sleep-4 replay is unaffected. All other entries, including A negatives, are identical.
- **Distractor facts** have stable IDs `"F{seed}L4D{i}"` (i = 0–3), life 4 and role "fork_distractor".
  - Their 4 reserved entities are transient entities of the withheld history only, so they enter that branch's
    FIRST and POS inventories at sleep 4.
  - Branch-specific probe inventories are generated from each history (§9 `probes.withheld`).
- **Evaluation.** Sleep 4, merged plus unmerged: `eval_{arm}_s4_withheld_{mode}.json`, 16 evaluations. The probes
  are the 4 H targets (Statement template and QT0 Question), plus the full standard set for completeness.
- **Reading.** "History-sensitive" means observed-branch correct ≥ 0.50 and withheld-branch UNKNOWN ≥ 0.80 on the H
  targets, per seed. U4 and G4 are reported side by side. The unit is the seed's branch pair.
- **Restoration admission (S6).** On seed 24, U4: sleep 4 run uninterrupted vs resumed from the saved sleep-3
  state. Admission requires **canonical state equality**:
  - every LoRA tensor and every optimiser state tensor equal under `torch.equal`, after loading both onto the CPU;
  - the step counters equal;
  - the Python, NumPy, torch-CPU and torch-CUDA RNG states byte-equal at the end of sleep 4.

  Container checksums are not used. If equality fails, the fork is recorded as **unrun**.
- **Full-state identity (E2).**
  - `tensor_hash` = sha256 over, for each tensor in sorted name order (LoRA parameters, then optimiser state as
    `"opt/{param_name}/{state_key}"`), the concatenation of:
    - the name as UTF-8;
    - b"\0";
    - the dtype string (e.g. "torch.float32");
    - b"\0";
    - the shape as a JSON list;
    - b"\0";
    - the tensor's contiguous little-endian CPU bytes.
  - `rng_hash` = the canonical-JSON sha256 of:
    - `{"python": repr(random.getstate())`;
    - `"numpy"`: a list of [name, base64 of `state[1].tobytes()`, state[2], state[3], state[4]];
    - `"torch_cpu"`: base64 of `torch.get_rng_state().numpy().tobytes()`;
    - `"cuda"`: a list of the base64 per-device states`}`.
  - `full_state_hash` = canonical-JSON sha256 of `{"tensors": tensor_hash, "rng": rng_hash, "step": int}`.
  - The parent-state identity and admission both use `full_state_hash`.

## 8. Phases and preflight (S8)
- **Phase 0 (CPU, no pod):**
  - `prep` for seeds 20–24 writes every stream, stratum, schedule, swap mapping, probe set and the fork record;
  - Sol's independent validator (§9) passes on all of them;
  - the projected cost, from actual item and probe counts, is computed.

  Any failure is a STOP.
- **Phase 1 (pod):**
  - calibration (seed 24, P, sleep 1, E = 4): template acquisition ≥ 0.80 and competence in band, with N FRESH
    exact-UNKNOWN ≥ 0.90; failure is a STOP plus an amendment;
  - the fork restoration test;
  - the throughput measurement.

  Projected total cost: if it exceeds $30 even without the fork and with 32-row QDIAG, **STOP before phase 2**.
  Otherwise the fork and QDIAG decisions are logged before phase 2.
- **Phase 2:** verify the phase-0/1 manifest, CODE_HASH and GPU; seeds 20–23, all arms, and the fork if admitted;
  both analyzers. The ABST-0 launcher contract applies, including exactly-once start and preserve-before-remove.

## 9. Artifact schema v1 (frozen before the coding split; `schema: "SUPPORT1_SCHEMA_V1"`)
- **Conventions.**
  - JSON is UTF-8.
  - Hashes are the sha256 hex of canonical JSON: `json.dumps(obj, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False).encode()`. File hashes are the sha256 of the file bytes.
  - Absent values are `null`, never omitted.
  - A cell is `[entity: str, attribute: str]`.
  - Enums are lowercase strings.
- **`prep_seed{S}.json`:**
  ```
  {schema, experiment:"SUPPORT1", seed:int, test_mode:bool, code_hash:str, model_manifest:str,
   stream:{…ABST-0 stream fields…}, stream_sha256:str,
   tpart:{t_neg:[str], t_swap:[str]}, split:{r:int, sel:[cell], how:[cell]},
   strata:{pm:[cell], rev:[cell], hnm:[cell], excluded:[cell]},
   fork:{entities:[str], facts:[{entity, attr, value}]},
   positives:{L:[{item_id, cell, wording:0|1, label, life}]},
   schedule:{arm:{L:[{pos:int, slot:"new"|"replay"|"third", kind:"pos"|"replay"|"bg"|"neg_a"|"neg_t"|"neg_swap",
                     item_id:str|null, source_item_id:str|null, cell:cell|null, src_cell:cell|null,
                     wording:0|1|null, label:str|null, input_ids:[int], loss_start:int, n_loss:int}]}},
   fork_schedule:{L:4, arm:{…same entry form…}},
   swap_map:{L:[{src_cell, wording, t_name}]},
   probes:{history:{L:[{probe_id, cat, construction:"statement"|"question", kind:"template"|"para0"|"para1"|"qt0"|"qt1",
                          cell:cell, item:str, gold_current:str, gold_old:str|null, gold_new:str|null,
                          delivered_eligible:bool|null, pair_probe_id:str|null}]}},
   sampler_keys:{name:str}, normalisation:{code:"cls0c_b.norm", vectors_sha256:str},
   budget:{arm:{L:[items, input_tokens, loss_tokens]}}}
  ```
  - `history` ∈ {"observed", "withheld"}. Probe IDs are stable across arms: `"{seed}-{history}-{L}-{cat}-{n:04d}"`.
  - **Cell rule (E3):** every semantic training entry and every probe carries the **rendered** `cell = [name,
    attribute]`. This covers positives, replay, A negatives, T negatives (`[t_name, attr]`), G swaps (`[t_name,
    attr]`, plus `src_cell`), GTQ and FRESH. Only background has `cell: null`.
- **`train_seed{S}.json`:**
  ```
  {schema, experiment, seed, provenance:{code_hash, prep_sha256, model_manifest, test_mode, gpu},
   arms:{arm:{L:{ckpt_sha256, full_state_hash, delivered:[{kind, cell, src_cell|null, wording, label, count}], budget:[…]}}},
   fork:{admitted:bool, parent_full_state_hash:{arm:str}, withheld:{arm:{ckpt_sha256, full_state_hash, delivered:[…]}}}}
  ```
  - `delivered` lists explicit records, **per sleep, not cumulative**. `count` is the number of scheduled presentations
    of that exact (kind, cell, wording) at that sleep.
  - `full_state_hash` is as in §7.
- **`eval_{arm}_s{L}_{merged|unmerged}.json`, `eval_N_s{L}.json`, `eval_{arm}_s4_withheld_{mode}.json`:**
  ```
  {schema, experiment, seed, arm, sleep, mode:"merged"|"unmerged"|"base", history:"observed"|"withheld",
   provenance:{code_hash, prep_sha256, ckpt_sha256|null, model_manifest, test_mode},
   ppl_ratio:{eval:float, modern:float}, base_denominator_sha256:str,
   rows:[{probe_id, answer:str, outcome:"correct"|"unknown"|"empty"|"substantive"}]}
  ```
  Gold and probe fields are **not** copied into evaluation rows; analyzers join on `probe_id` against prep.
- **CODE_HASH (E4)** follows ABST-0's rule: `sha256(b"".join(sha256_hex(file_bytes).encode() for file in CODE_FILES))`,
  with CODE_FILES literally, in this order:
  1. `scripts/support1.py`
  2. `scripts/abst0.py` (make_stream and _names source)
  3. `scripts/cons0a.py`
  4. `scripts/cls0c_b.py`
  5. `being/v3/lifegen.py`
  6. `being/competence.py`
  7. `data/background.txt`
  8. `data/competence_eval.txt`
  9. `data/competence_modern.txt`
  10. `lab/support1_norm_vectors.json` (the §3 test vectors as `[[input, expected], …]`)

  Both implementations compute it identically.
- **Validator (Sol; independent).** It checks, from the stream alone:
  - every gold value, with the temporal mask and the per-history oracle;
  - the strata and the split rule (re-run and match r);
  - the T partition and its global disjointness;
  - the leak rule;
  - the REV quota;
  - no UNKNOWN target on any S/core cell that is H, HOW, excluded, or of an entity not yet introduced at L. T
    negatives and swaps concern unexperienced names by design and are exempt;
  - U4 = 4 × U1 per entry;
  - the G token-length match and a shared mapping;
  - every positive, including replay, against the oracle at L;
  - budget equality;
  - QDIAG delivery eligibility and pairing;
  - the normaliser test vectors.

## 10. Cost and ceiling
- **Evaluations:** 220 main plus 16 fork, plus phase 1 (calibration of about 3, and the restoration test).
- **Projection** (S8, superseding v1's), from ABST-0's measured per-seed times on an L40S (training about 12 min;
  evaluation about 2.4 min each at about 160 rows):
  - SUPPORT-1 training per seed is about 2.4× ABST-0's, so about 29 min;
  - evaluations carry about 1.4× the rows: 55 × 2.4 × 1.4 ≈ 185 min;
  - so about 3.6 h per seed, about 14.5 h for 4 seeds, plus about 0.7 h for the fork and about 1.5 h for phase 1;
  - about 17 h, roughly $14–15.

  Phase 0 recomputes this from actual counts, and phase 1 replaces it with measured throughput.
- **Ceiling: $30 total**, enforced as in §8.

## 11. Prediction (Claude, before data)
- U4 reaches ACQ and transfers it to Statement SEL-D.
- HOW rises above P but stays below 0.70, so partial within-transfer. HNM stays near ABST-0 levels.
- The U4 − G4 contrast on HOW is ≥ 0.10, but the HNM mechanism gate fails.
- Most likely verdict: SUPPORT1_NO_ADVANCE, with GR_within of about 0.3–0.5. This is a prediction, not a criterion.

## 12. Review record
- **v1 (1caeeec):** Sol HOLD, items S1–S8, all accepted. **v2 (8322c55):** Sol HOLD for errata E1–E4, all accepted.
- **v2.1 (20d3c3a): Sol GO** for the design, schema v1 and the implementation split, with the cycler clarification
  recorded in §2.1.
  - Sol verified all 15 normalisation vectors and that every existing CODE_FILES path resolves. Inline gold is not
    needed.
  - **Responsibilities:**
    - Claude: generator, schedules, training/evaluation and launcher (scripts/support1.py).
    - Sol: independent analyzer, CPU validator and adversarial tests (sol/* branches).
  - Paid execution stays behind the phase-0 validation gate, phase-1 preflight and David's launch decision.
- **Implementation clarifications (Claude, at 850a927 / 7b64078; each is visible in Sol's validator report):**
  - **Replay pool.** The written text says replay cycles "all positive items of lives < L". The implementation cycles
    the **unique** acquired (cell, wording) pairs of lives < L, each labelled by the oracle at L, with
    `source_item_id` = the latest item for that pair. Effect: an update cell is not double-weighted by its superseded
    and current items; every acquired cell has equal replay weight. Logged as a clarification (Sol, 2026-11-02).
  - **Fork record:**
    - built with the tokenizer, with one distractor per H target in canonical order;
    - rejection-sampled until both wordings' (framed length, n_loss) equal the target's;
    - `fork.facts[i].target` = [e, a];
    - fork names disjoint from all S/T/V words.
  - **Extra prep fields:**
    - `base_blocks {L: {A, T}}`;
    - `qdiag_fallback`: the 32-row set is the first k rows of each QDIAG group, in probe_id order.
  - **FORK-H / FORK-H-Q** (Statement template / QT0) rows are added at sleep 4 in both histories. Observed gold is the
    value; withheld gold is UNKNOWN.
  - **TPRE categories:** TPRE-A / TPRE-H, plus -UNFAM for entities not yet introduced. REV-Q asks all 4 REV cells at
    every sleep.
  - **Background:** the CONS-0A chunker with key `"CONS0A-bg-SUPPORT1-{seed}-{i}"`, where i = 1,000,000 + j (sleep-1
    replay) or 2,000,000 + L·10,000 + j (third slot). Each background entry matches the U4 entry at the same content
    index.
  - **Interleave:** content = new + replay + third, with one perm per (seed, L) shared by all arms. U4 = B_L
    concatenated ×4; a permutation of identical copies would be a no-op.
  - **Test-mode relaxations** (tiny BPE only; production keeps every STOP): G swap and fork-distractor token
    matching, the budget assertion over G arms, and over-length sequences.
  - **Training determinism:** `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True,
    warn_only=True)`, TF32 off. The restoration test decides admission empirically.
  - **Production training guards** for development seeds require:
    - Sol's validator report as JSON `{"verdict": "PASS", "prep_sha256": [...]}` covering the prep's sha256;
    - a passing `calcheck` result for seed 24.
- **Sol HOLD at fba75d5** (sol/support1-analysis: independent validator, analyzer, adversarial tests). Three findings,
  all accepted and fixed before any production prep exists:
  1. **Split order.** The §2 text said `sorted(...)` (alphabetical), but §2.1 and the code use ATTRS9.
     - Resolved to **ATTRS9 order**, since §2.1 is the canonical contract. The alphabetical variant exhausted 1,000
       retries for seed 20.
     - Under ATTRS9 the accepted r is 212 / 195 / 402 / 939 / 65 for seeds 20–24. Seed 23 was close to the old cap,
       so the cap is raised to **100,000** (same algorithm; this only removes a fragile STOP).
  2. **Swap order.** The code shuffled T_swap from its partition order. It now uses `sorted(T_swap)` and then the
     shuffle, as §2.1 requires.
  3. **Calibration gate.** Sol made `calcheck` pass on 1 POS and 1 FRESH row with no provenance. The gate now
     requires:
     - the complete sleep-1 inventory, in prep order, for merged, unmerged and base;
     - provenance bound to the prep, code, model, test mode and qdiag, with the checkpoint from the P sleep-1 train
       log (which itself must be this prep/code/model, arm P, sleep 1);
     - identity fields (schema, experiment, seed, sleep, mode, history, arm);
     - outcomes **recomputed** from answers and prep gold;
     - both monitors present and finite, with twins showing 0 mismatches, |Δppl| ≤ 1e-4 and one shared base
       denominator;
     - acquisition over **all** POS template probes, and N FRESH UNKNOWN over **all** 20.
     - Training on development seeds additionally requires the calcheck result to bind to the current code hash, the
       model manifest and the sha256 of the supplied seed-24 prep (`--calibration-prep`).
     - Regression tests: truncated rows, a forged outcome, a wrong checkpoint, a missing monitor and a foreign train
       log each fail the gate.
- **Amendment A1 (proposed by Claude, 2026-11-02; pending Sol's review): one fixed swap name per A entity.**
  - **Trigger.** The first production prep with the real Qwen3-8B tokenizer (files verified against the recorded
    manifest) STOPPED. The v2.1 rule ("first unused T_swap name per A entry") needs about 106 distinct names in
    total, and rare token lengths run out: 4-token names make up only 8–14 of the 200 in T_swap. Nothing was trained.
  - **Change.** Each A entity e gets ONE T_swap name: the first in `sorted(T_swap)` shuffled with
    `"SUPPORT1-swap-{seed}"` whose framed (length, n_loss) equals e's on **all 18** question forms (9 attributes × 2
    wordings). Every G entry for a cell of e uses that name, in G1 and G4 and at every sleep. STOP if none matches.
  - **Why this is also the better control.** U's negatives return repeatedly to the same 8 experienced people. Under
    A1, G's return to 8 consistent fake people with identical attributes, wordings, counts and positions, so the
    remaining difference is whether the name refers to an experienced person, **plus lexical effects of the
    particular names** (Sol's qualification). The per-entry rule instead confounded
    grounding with person-repetition structure, since every G entry was a new person.
  - **Verified on seeds 20–24:**
    - all A names match on all 18 forms, and no name's tokenisation depends on context;
    - production prep passes with every production check on (budgets equal across all 5 arms, G included; fork
      distractors token-matched; maximum sequence 80 tokens);
    - two runs produce byte-identical preps.
- **Sol: GO on A1 and PASS on all five candidate preps at 981f0e7** (sol/support1-analysis 7273e1b: the five tokenizer
  files verified independently, all preps regenerated byte-identically, 346,082 validator checks, 0 errors).
  - Sol's notes: keep sorted(A), eight distinct mapped names, and matching on all 18 forms. GTQ keeps 16 distinct
    delivered questions per sleep; the sleep-5 pool holds 58–66 pairs, clustered within 8 identities.
  - **Paid phase 1 stays on HOLD** for 5 remaining calibration-gate gaps (Sol's reproducers), all fixed. calcheck now
    also requires:
    - a valid 64-hex checkpoint sha256 in the train log, so a missing identity no longer "matches" null;
    - a 64-hex base-denominator identity in every evaluation;
    - both competence ratios finite and **strictly positive**;
    - `env.tf32 == false` in every evaluation.
  - Sol's 5 strict-xfail reproducers now XPASS (the attacks are rejected), and the same attacks are in the producer's
    slow end-to-end test.
  - Editing calcheck changes CODE_HASH, so the candidate preps are regenerated and must be revalidated before phase 0
    is frozen.
