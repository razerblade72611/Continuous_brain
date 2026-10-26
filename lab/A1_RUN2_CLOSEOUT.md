# A2 Phase 1 run 2: closeout (recorded 2026-10-26, before BIND-D1)

Immutable record agreed by Claude and Sol. Nothing below is relabelled or rescored later. Follow-up findings go in new
documents that cite this one.

## Why this run happened
- A2-A0 returned INCONCLUSIVE under its frozen rule (`lab/A0_CLOSEOUT.md`).
- That rule authorises only the original 25-life A2 Phase 1, with no automatic Phases 2 or 3.
- This is a separate lineage from the original stageA2 run, whose artifacts sit unrecovered on volume btxeuf9tm0
  (CA-MTL-3).

## Execution
- **Code:** being/v3-runner-fix. The result files record commit 31fce4d; the experiment source (`being/`,
  `scripts/run_v3_dev.py`) is unchanged from 95cad33.
- **Command:** `stage_v3.py A --only A1 --parallel 5 --stop-pod`, run on a RunPod A40 48 GB in CA-MTL-1.
- **Storage:** network volume 29qhoqghqf.
- **Backbone:** Qwen3-0.6B `c75424fb5370f0d514a05d7c640e6c2e1e4b504564e4d48cb2ddedb71609e0a0`, matched at preflight.
- **Scope:** ten A1 configs, dev seeds 0 and 1, 25 lives, checkpoints after lives 1, 2, 3, 5, 10, 25. 10/10 complete,
  2.98 h.
- **Artifact:** `results_stageA2_A1run2.tar.gz`, SHA-256
  `8e9c5faee48e2e5f27e6effa0c938894dbbb4dd1f90dd1b5f1c0ac384ac5e476`. Re-read verified on the volume, downloaded to
  David's PC (checksum OK) and to Claude's workspace (checksum OK). Contents committed under
  `results/dev3/stageA2_A1run2/`.
- **Operational defect:** the in-pod `runpodctl stop` returned exit 1 (the pod is not authenticated), so the pod idled
  about 45 min until the PC-side `cb_fetch.sh` stopped it. From now on, PC-side stop and fetch are primary and the
  in-pod stop is best-effort.

## Result (original A2 selector, unmodified)
| arm | development-screen winner | rule used |
|---|---|---|
| T | T layer21 lr1e-2 | acquires+band |
| S | S layer21 lr100 top_t64 | none (no S config acquires in band) |
| L | L kl1.0 lr1e-4 | acquires+band |

Per seed:
- acquisition: a1 = life-1 template recall at life 1, a10 = life-10 template at life 10;
- at life 25: fc = life-1 forced choice (/10, 50 candidates), rec = life-1 template recall, mb = mean strongest-rival
  margin, ppl = competence ratio eval/modern.

| config | seed 0 | seed 1 |
|---|---|---|
| L kl0.3 1e-4 | a1 8, a10 10, fc 0, rec 0, mb −25.3, ppl 0.721/0.735 | a1 9, a10 10, fc 0, rec 0, mb −23.2, ppl 0.731/0.721 |
| L kl0.3 3e-4 | a1 10, a10 10, fc 0, rec 0, mb −27.0, ppl 0.821/0.845 | a1 8, a10 10, fc 0, rec 0, mb −25.8, ppl 0.802/0.809 |
| L kl1.0 1e-4 | a1 8, a10 10, fc 0, rec 0, mb −21.3, ppl 0.702/0.721 | a1 8, a10 10, fc 1, rec 0, mb −21.1, ppl 0.727/0.724 |
| L kl1.0 3e-4 | a1 10, a10 10, fc 0, rec 0, mb −23.9, ppl 0.764/0.782 | a1 9, a10 10, fc 2, rec 0, mb −19.3, ppl 0.765/0.773 |
| T layer7 | a1 2, a10 10, fc 0, rec 0, mb −23.6, ppl 0.865/0.914 | a1 2, a10 10, fc 0, rec 0, mb −30.3, ppl 0.800/0.821 |
| T layer14 | a1 6, a10 10, fc 0, rec 0, mb −27.6, ppl 0.861/0.891 | a1 5, a10 10, fc 0, rec 0, mb −26.1, ppl 0.820/0.862 |
| T layer21 | a1 2, a10 9, fc 0, rec 0, mb −27.2, ppl 0.736/0.802 | a1 5, a10 10, fc 0, rec 0, mb −19.1, ppl 0.743/0.769 |
| S layer7 | a1 1, a10 4, fc 0, rec 0, mb −21.6, ppl 3.220/3.867 | a1 2, a10 5, fc 0, rec 0, mb −20.7, ppl 3.252/3.945 |
| S layer14 | a1 2, a10 7, fc 0, rec 0, mb −24.6, ppl 2.354/2.352 | a1 3, a10 7, fc 1, rec 0, mb −23.0, ppl 2.180/2.411 |
| S layer21 | a1 0, a10 0, fc 0, rec 0, mb −12.1, ppl 1.002/1.112 | a1 0, a10 0, fc 1, rec 0, mb −14.2, ppl 1.063/1.172 |

Wiped-backbone strongest-rival margin on the same life-1 facts: −21.1 (seed 0), −18.5 (seed 1).

## Decision (Claude, after Sol's review)
- Phase 1 is closed.
- A2 Phases 2 and 3 are suspended. They would be reopened only if a diagnostic identifies a specific mechanism that one
  of their planned interventions plausibly fixes.
- Next: BIND-D0, a read-only decomposition of archived data, then BIND-D1, the instrumented rival-attribution and
  paired-entity diagnostic.
- No architecture redesign until mechanism evidence is in.

## Interpretation recorded before BIND-D1
Phase 1 confirmed strong within-life acquisition followed by near-total loss of accessible life-1 information at
life 25. The original selector produced development-screen winners but no demonstrated long-term memory success:
- selected by the rule — yes;
- meaningful long-term retention — no;
- competence preserved — yes for L and T, but by a monitor already recorded as insensitive (ratios < 1 reflect
  background-register adaptation).

The next investigation will distinguish three things: loss of the true association, competing-value intrusion, and
global recency or answer-preference bias. No existing outcome will be relabelled or rescored.

## BIND-D0 (read-only; `scripts/bind_d0.py`, output `results/dev3/bind_d0.json`)
**Method.** The strongest rival's log-probability is `logprob − margin_best`. Its identity was not saved, so this
decomposes margins without attributing them to a particular rival. All values are relative to the wiped backbone,
matched by fact and seed. The wiped backbone gives abs logP(true) ≈ −42.5 and abs logP(strongest rival) ≈ −22.7 on the
life-1 facts.

**Observations (descriptive):**
- **L and T:** the true answer peaks at life 1 (+36 to +42 nats over wiped; abs ≈ −1 to −7). It falls about 25–30 nats
  by life 10, yet stays 10–16 nats above wiped at life 25. The strongest rival rises 16–20 nats over wiped, to abs
  ≈ −3 to −6. From about life 3 on, the rival's rise, not the true answer's fall, decides forced choice.
- **S7 and S14:** the true answer decays from +35 to +7/+8; the rival rises modestly, from +4/+6 to +9/+11.
- **S21:** the true answer is flat across 25 lives (+18.4, +17.6, +16.4, +18.2 at lives 1, 5, 10, 25). The rival rises
  steadily (+2.4, +7.1, +7.9, +11.5). Its forgetting is therefore rival growth, not loss of the stored signal. Its
  acquisition is weak and its competence is out of band.
- **Partial rebound, unexplained:** for several configs the true signal partly rebounds between life 10 and life 25
  (e.g. T14 +3.7 → +12.2). The hypothesis that bridge facts revisit life-1 entities is untested.

**What this does not establish:** a rival at abs −3 to −6 for nearly every fact suggests a strongly preferred answer
string. D0 cannot say whether that string is the same across entities (global answer preference) or entity-specific
(binding interference). BIND-D1 is designed to settle that.
