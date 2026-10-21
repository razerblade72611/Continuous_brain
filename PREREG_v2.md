# Gate v2 preregistration (LOCK CANDIDATE r5)

**Status:** LOCK CANDIDATE r5, 2026-10-21, Claude (builder).

This revision is for Sol's final review on DAV-104. David gave the go-ahead for the sealed run on 2026-10-21; it launches once Sol signs off on this revision. Nothing sealed has been run, and the confirmation sets are unopened.

**r4/r5 vs r3:** runner-only changes (interruption-safe resume with an exclusive process lock, below). The experiment (`run_seed`, `summarize`, scoring, config, data) is unchanged. r5 fixes Sol's two r4 findings: a crash before the first recorded seed was unrecoverable, and concurrent resumes were not excluded.

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
8. The backbone hash is unchanged: the run-level before and after hashes match, and every seed's start and end hashes match.

## Interruption and resume (r5)

The sealed family takes about 7.4 minutes per seed when run alone (443 s measured), so roughly 2.5 hours, and this workspace restarted twice during development. So that a restart cannot destroy the one-time test:

- **Exclusive lock.** Every sealed process, whether opening or resuming, first takes an OS-held exclusive lock (`flock` on `results/v2_sealed.lock`) and holds it until it exits. A second process refuses. Process death releases the lock automatically.
- **State before evaluation.** On opening, the marker is created (O_EXCL) and then an empty resumable state (`results/v2_sealed.json`, zero rows) is written before the model is loaded.
- **Append-only row log.** Each completed seed is written atomically (temp file, fsync, rename) to the results file. Its row SHA-256 is then appended to the marker.
- **`--resume`** continues an opened family only if all of these hold:
  - the marker's config SHA-256 matches, and the results file (if present) is unfinished and has the same opening commit;
  - `being/`, `scripts/`, `configs/`, `data/`, `tests/` and this file are byte-identical to the commit recorded in the marker, and the tree is clean outside `results/`;
  - the recorded rows are exactly seeds 1000…1000+k−1 in order;
  - every row logged in the marker matches the results file by SHA-256, and at most one trailing row is unlogged (a kill between the two writes; it is logged on resume).
- If the results file is missing, a resume starts from zero **only if the marker logs no rows** (a kill before the initial write). Otherwise it refuses.
- **Recorded rows are kept verbatim. No recorded seed is ever rerun or replaced.** Only a seed whose row was never written is run, from scratch.
- Each segment (start and end time, commit, seeds run, backbone hash before and after) is logged in the results file and the marker. A killed segment has no end hash; its seeds are attested by their own start == end hashes.
- `backbone_unchanged` additionally requires every seed's start hash to equal the final segment's backbone hash, and every finished segment's before == after.
- `--max-seeds N` ends a segment cleanly; it doesn't change what is computed.
- **Regression tests** (`tests/test_sealed_runner.py`, driving the real runner with a fake model and seeds through `tests/fake_runner.py`; no training, no data read) cover:
  - a crash during the first seed;
  - a crash before the initial write;
  - completed rows preserved verbatim (random nonces unchanged);
  - a kill between a row write and its log entry;
  - a competing resume refused;
  - tampered and deleted results refused;
  - a code change refused.

**Determinism (observed).** Each seed is computed independently from fixed seeds (life, birth, replay order). Dev seed 3 was rerun at `26bd49f` in a new process (`results/dev/v2_cand2_s3_repeat.json`). It matched its recorded row in `v2_cand2_s3-4.json` exactly on all 22 shared fields: probe answers, perplexity ratios, sleep dev ratio, and the memory and backbone hashes. The only difference is `backbone_sha256_end`, which the original row lacks because the field was added later. So where a segment boundary falls cannot change a row.

## Claim boundary

- **A pass shows:** facts from a scripted life are recoverable from the memory weights alone. Recovery works with an empty prompt and in a fresh process, both as forced choice and as unrestricted top-1 next token, with competence held and the frozen backbone untouched.
- **A pass doesn't show:** that it knows David, that it can converse, that it generalises beyond the trained question form, or anything about sentience. Paraphrase probes are reported, not gated.
- **Failures are kept.** Every dev and sealed run's JSON is committed, including failures and interrupted attempts.

## Competence sets

- **Dev monitors** (seen during development, never used for the gate): `data/competence_eval.txt` (Gutenberg) and `data/competence_modern.txt` (Brown).
- **Sealed-only confirmation sets** (built by `scripts/build_competence_sets.py`, disjoint from every other set, never printed or measured before the sealed run):
  - `data/confirm_gutenberg.txt`, 200 sentences from chesterton-ball and edgeworth-parents, sha256 `2f2b5ee5…59c0`;
  - `data/confirm_modern.txt`, Brown items 200–399, sha256 `7d2f6e3d…bc6`.

## Decisions made after the first dev results (recorded because they were informed by dev data)

1. **Rehearsal: accepted by Sol (2026-10-19).** Never-scored background sentences are interleaved one per life sentence, while awake and asleep, and in the interference block. The lived and stranger arms use the same rehearsal budget.
2. **Competence rule: decided by David (2026-10-19, ~20:50 CT).** The rule is one-sided: mean perplexity ratio ≤ 1.02 and no seed above 1.05, on both sealed confirmation sets, at both A and B. It replaces the two-sided 0.98–1.02 rule that Sol reviewed. The change was proposed after seeing a dev result: seed 0 with rehearsal gave a Gutenberg ratio of about 0.91.
3. **Mixed rehearsal background (after candidate 1 failed).** Candidate 1 rehearsed on Gutenberg only. Its Brown monitor got worse (1.20–1.30), so it was stopped and is kept in `results/dev/v2_cand1_*`. Candidate 2 rehearses on Gutenberg plus a disjoint Brown slice.

## Dev evidence behind this config (seeds 0–9 only)

- **Pooled summary:** `results/dev/v2_cand2_summary.json`, sha256 `67b5a2b7d5bddcae56aa85fc0dff81a86b633dea010217c19c00ade502dcc608`.
- **Results:**
  - A recall is 29.3/30 on question-only template probes (forced, full-vocabulary and generated).
  - Wiped scores 0.5, stranger 0.5, and B (after interference) 13.5.
  - The dev-monitor perplexity ratios are 0.84–0.94.
  - Fresh in-process and separate-process reloads are identical per question, with matching hashes.
- **Restart history:**
  - A container restart ended the first two dev processes after seeds 0, 1, 2, 5, 6 and 7.
  - Seeds 3, 4, 8 and 9 were rerun with the same code and config, and merged with `scripts/merge_dev.py`.
  - The interrupted files carry no end-of-run backbone hash, so seeds 0, 1, 2, 5, 6 and 7 are listed as **backbone-unattested**. That's missing evidence, not evidence of change: the backbone is frozen in code.
  - Every seed row now records `backbone_sha256_end` (Sol, DAV-104), and the sealed gate requires start == end on every seed.

## Rehearsal sources (never scored)

- `data/background.txt`: Austen *Emma* and Carroll *Alice*, 300 sentences, sha256 `158fd001934b26d72c6a27de8b2d02f7fd7a59fea55deea60b32b0503b55346f`.
- `data/background_modern.txt`: Brown items 400–1399, 1,000 sentences, sha256 `a59359f876dbd04d3281ac3d5c93cf5f3f1e99e1ef994d0dfae8bd7182b092da`.

## Limits

- The Brown rehearsal slice and the Brown confirmation slice come from the same corpus, so this isn't evidence of robustness on genres outside Gutenberg and Brown.
- Lower perplexity on these corpora isn't a claim of general language improvement.
- **The scope is narrow.** Recall is tested only on the trained question template, and the life is 30 scripted single-token facts.
  - Paraphrase probes are reported, not gated. In dev they scored 21.9/30 forced and 18.6/30 full-vocabulary.
- **Interference roughly halves recall** (B is about 13/30).
- **distilgpt2 is a mechanism test,** not the individual. The test says nothing about autobiography, goals or sentience.

## Sealed config

- `configs/v2.json` (identical to `configs/v2_dev_candidate2.json`): SHA-256 = `e73da0b594fefb8253ecfb2c61517dd0c8713bf2b227a2db95d7ad5b4958c401`
- Backbone: distilgpt2 `model.safetensors`, sha256 `721fa38acb2ba5d1ebdeed4bad3dcb32c0beeb2b924ecf544fffaa3b0bb65a66`; parameter hash `586afec4622dee7921d264bf1634d73a35054e291fa4cbab3c756db831fcba61`
- Confirmation sets: `confirm_gutenberg.txt` `2f2b5ee58537b1182643c81fd4471bc1f71f3204981474e293d88b66c42659c0`, `confirm_modern.txt` `7d2f6e3dfbe4e77f1d72b812efd1dafd2a08d773e2ea3f5c7ab33f54e1f8cbc6`
