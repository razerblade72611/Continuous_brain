# CB2-L1b: the repaired, powered test of "it learns from its own life" — DRAFT for Sol's review

Status: **draft, not locked.** Nothing has run.
- Owner of the design and code: Claude (c1b51df7). Auditor: Sol.
- Revision 2 takes in Sol's review S-003 (2026-10-11).
- Code: branch `explore/cb2-l1fix`: the repairs (93f3c45, verified by Sol), plus the L1b changes in §6.

## 1. Why this run, and not L2 first (David's four questions)

1. **Primary goal.** A persistent individual whose competence grows from its own lived experience. The growth must be
   durable, must transfer, and must be caused by that experience.
2. **What the current structure gave us.** L1's wake/sleep loop produced a fresh-world gain over frozen (+52.8 steps,
   lower95 +4.3) and a chooser that measurably helps (Gate 2: +133 on fresh). But L1 did not test the design as
   intended:
   - terminal rewards were lost, so the chooser's value never learned what death costs;
   - short lives (the early deaths) were dropped from training;
   - the memory metric's view mask was transposed (S-002);
   - the control twin degraded by itself (reference − frozen on transfer −62.7, bootstrap 5–95% −114 to −11);
   - and the test had only ~35–40% power for a +40-step effect.
3. **Is a repaired rerun the best next step?** Yes, compared with these alternatives:
   - **L2 now (learned hands, shelter/night first).** The learned-share trigger fired, so L2 is needed. But it adds a
     new learner on top of a loop whose basic claim is unverified; if L2 failed we could not tell the loop from the
     learner. L2 is built on CPU *in parallel* with L1b.
   - **The vanilla-DreamerV3 baseline.** Still unrun; it is the "are we on a dead end" check. It is a candidate next
     pod job, decided on its own merits and approved separately, never launched automatically.
   - **Just a bigger L1.** Without the repairs and the new control it would measure the same broken signals more
     precisely.
4. **What would change direction** is set out in §5 as decisions to take, with causes left unproven until tested.

## 2. Arms (twins of S2b, `cc8948e7…`)

| arm | sleeps? | sleeps on | purpose |
|---|---|---|---|
| **learning** | yes | its own lives, accumulating (20/cycle) | the individual |
| **yoked** (replaces L1's fixed-buffer reference) | yes: identical update counts, batch, T, schedule, bootstrap mixture and selection rules | lives lived by the **frozen** twin on the **same practice worlds**, same cycle, accumulating (20/cycle) | matched-world experience from the frozen policy |
| **frozen** | never | — | absolute gain |
| learning, chooser off | — | — | Gate 2 ablation, final battery |

**The question "learning − yoked" answers.** Own closed-loop adaptation (learning on its own lives as its policy
changes) versus matched-world experience from the frozen S2b policy, with identical update budgets and episode counts.
- **What it does not isolate:** "authorship" or identity.
- **Known differences between the arms' data,** not removed by reporting them:
  - the policy that generated it;
  - state and action coverage;
  - data quality;
  - the number of unique transitions (reported per sleep: rows, streams, excluded rows).
- **It is not the identical-data / extra-training control first proposed.** It is an explicitly limited, different
  question. Frozen is kept for absolute gain.

## 3. Schedule and worlds (new experiment: nothing reused from L1)

**Schedule.**
- Practice run **11** (seeds 91,000,000 + cycle × 10,000 + n). Runs 0–9 are taken (L1 = 9) and run 10 was touched
  by CPU rehearsals; all are refused by the code.
- 10 wake/sleep cycles, 20 practice lives per cycle for the learning twin. The frozen twin lives the same 20 worlds
  for the yoked archive.
- 2 held-out lives per cycle, never slept on.
- Sleep budgets as L1: core 1,500, anchor 1,000, chooser 1,500 updates; batch 16, T 256.

**Development world blocks, all new.** Each is checked at start against:
- L1's actual probe and situation worlds (78100–79639);
- Sol's claimed diagnostic block (83090–83195);
- **every world a CPU rehearsal touched before the lock** (the `TOUCHED` list in the code: 84000–84002, 86000–86002,
  88000–88001, 88100–88101, 92000–92001, 92100–92101, 94000, plus repair-era 80100–80101, 80200–80201, 82100);
- every other family and the sealed ranges.

| family | worlds | role |
|---|---|---|
| per-cycle fresh | 88050 + 100c, 40 worlds | growth curve (secondary) |
| per-cycle transfer | 92050 + 100c, 40 worlds | growth curve (secondary) |
| situations | 94050–94073 (×3) | growth curve (secondary) |
| final fresh | 84100–85299 | primary |
| final transfer | 86100–87299 | primary |
| timing | 95000–95099 | throughput only, never scored |

**Rehearsals** (`--rehearsal`) use their own smoke blocks (96000+, 97000+, 98000, final 98500/99000) and practice run
99. They are recorded in the lock, and the report says **"rehearsal: no claim"**.

**Integrity** (Sol, S-009):
- **The lock (`run.json`) is written once, read-only, and never rewritten.** A resume first checks, before writing
  anything:
  - the code revision, with a clean tree;
  - every experimental argument (only the worker count and paths may differ);
  - the origin and bootstrap input hashes;
  - and the recorded progress against each twin's consolidation count.

  If anything differs, it refuses.
- **A checksummed origin manifest** (S2b's metadata and snapshot files) lets the rollback check run on any machine.
- **Pairing is strict:** exactly the expected worlds, no duplicates, finite values, or the report stops. The
  chooser-off comparison is an explicit expected subset.
- **Final-battery files are bound and checksummed:** each records the run id, revision, condition (arm, family,
  chooser, configuration, max steps, state hash, worlds) and a content checksum, and is written as each set
  completes. A set that was started but never recorded stops the battery instead of being silently rerun.
- One code revision for the whole run: the final battery refuses another revision.
- **Cycle 0 must give identical probe lives across all arms,** or the run stops with the evidence kept. Every
  expected arm and row must be present, and each life is compared by:
  - its action/command trace hash;
  - its length and death;
  - its unlocks and requests.
- A sleep tripwire stops the run and is reported as a failure.
- Nothing is silently rerun.

## 4. Primary estimand and power

**Primary: the final individuals (after sleep 10)** on the final battery.
- 1,200 fresh worlds and 1,200 transfer worlds per arm, every arm on the same worlds with the same life ids;
- one life each, max 10,000 steps;
- the estimand is mean survival steps, paired by world.

**This is a different estimand from L1's "pooled cycles 6–10" primary.** Pooled cycles 6–10, retention and the other
L1 measures are kept as descriptive secondaries.

**Power,** from L1's records (cycles 6–10). Reproduce with `scripts/explore/cb2_power.py`; the output is in
`evidence/explore/cb2/l1b-power.json`. The SD and correlations below are *estimates* from one run:
- SD of the paired difference is 400–444 steps.
- The correlation between paired arms is only 0.1–0.2 (lives diverge chaotically), so world-pairing barely reduces
  variance.
- Capping survival does not raise the standardised effect (≈0.13 at caps 750–1,500), so raw steps stay.

**At N = 1,200 per family** (SD 430, one-sided 95% lower bound). Sol recomputed these independently from L1's records
(`cb_exchange/sol_l1b_power_review.json`): about 83% joint power at +40, 46% at +30, 26% at +25.
- single comparison: 94% power at +40; minimum detectable effect (MDE) at 80% power is **31 steps**;
- the joint four-way claim (both comparisons × both families; L−F vs L−Y correlation 0.45–0.51 in L1): 81% at +40
  everywhere, 43% at +30.

**Sensitivity (Sol).** If every SD is 20% larger, joint power at +40 falls to about 60%. The yoked control and the
final-only primary may change the variance and correlations, so this power is a planning assumption, not a
guarantee.

**Honest limit.** L1's transfer point estimate vs frozen was −9.6. If the true transfer effect is near zero, this run
reports that as a bounded small effect; it does not diagnose the cause.

**Bootstrap:** 10,000 draws over worlds, fixed seed `20261009`. Each world is one independent cluster.

## 5. Decision rules (written before running; computed by `verdict()`)

**Primary.** lower95 is the one-sided 95% bootstrap lower bound, over worlds, of learning minus the other arm.
- **G (growth):** learning − frozen lower95 > 0, fresh **and** transfer.
- **O (own closed-loop adaptation):** learning − yoked lower95 > 0, fresh **and** transfer.

**Safeguards, required before any positive claim.**
- **Capability is not traded for survival:**
  - learning − frozen lower95 > **−0.5 achievements unlocked per life**, fresh and transfer;
  - and > **−0.5 net requests (fulfilled − failed) per life**, fresh and transfer.
  - In L1, the paired SDs were 2.4–4.3, so at N = 1,200 a true zero gives lower95 ≈ −0.15.
- **Identity / provenance / rollback,** for every arm:
  - its consolidation chain verifies;
  - the chain links S2b's state to its present state through exactly one event per sleep (none for frozen);
  - and S2b's snapshot, still in its home, restores to S2b's exact state hash.

**Claims.**

| claim | conditions |
|---|---|
| "shown" | G, O and every safeguard |
| "growth shown; own closed-loop adaptation beyond matched-world frozen-policy experience not shown" | G and the safeguards, not O |
| "not shown: safeguards failed (…)" | G, but a safeguard failed |
| "not shown" | otherwise. A wide interval is inconclusive and is reported as such. |

**Decisions, each needing its own approval; nothing launches automatically and causes stay unproven.** Each one
bounds the effect of *this implementation, at this budget, on this metric*.
- **Learning − frozen on fresh, upper95 < +31 (the MDE).** Decide between investigating objectives and controls, or a
  bounded baseline comparison (e.g. vanilla DreamerV3). This is not proof that the architecture is wrong.
- **Fresh lower95 > 0 but transfer upper95 < +31.** Investigate before scaling. Memorisation is one hypothesis among
  others, not a diagnosis.
- **G, but learning − yoked upper95 < +31 in both families.** The own-adaptation question stays open for L2. This
  does not show that the gain "came from more data".

**Secondary (reported, not decisive):**
- growth curve per cycle;
- situations; retention c10 − c5;
- Gate 2 (chooser on − off, 300 worlds/family);
- Gate 4 on held-out lives with the corrected [x, y] view mask (learning vs yoked vs frozen), using Sol's locked
  diagnostic code where applicable;
- learned-authored primitive share (0% by construction; L2 moves it);
- exposure per sleep.

## 6. Code changes on top of 93f3c45 (Claude; tests in tests/test_cb2_l1b.py and tests/test_cb2_view_geometry.py)

1. **Yoked arm** (`wake_specs`, `sleep_specs`):
   - the frozen twin's lives go to `archive-yoked` (life-id family 32M);
   - the yoked twin sleeps on it with identical arguments;
   - the lineage names whose experience it was.
2. **Instrument check** at cycle 0 (`instrument_check`).
3. **The `final` subcommand:**
   - resumable per arm × family;
   - refuses a twin that changed since the loop, or a different code revision.
4. **`_check_worlds`:**
   - new run and blocks; refuses runs 0–9, L1's worlds, Sol's claim and sealed worlds;
   - life-id families disjoint.
5. **The report:**
   - primary from the final battery, with upper95 and SD;
   - the safeguards; `identity_check` per arm; `verdict()` exactly as §5.
6. **`bench-lives`:** measured throughput on the actual machine, using timing worlds only.
7. **`scripts/explore/cb2_power.py`:** the §4 numbers.
8. **View geometry (Sol, S-002):**
   - `in_view_crop` builds the mask from `memory.view_cells` on the crop's [x, y] layout;
   - it is used by the core's map metric and by the sleep-time OOV probes.
   - Regression: 14 + 14 cells were misclassified at radius 8.
   - L1's Gate 4 numbers carry this limitation; its verdict is unchanged.

9. **Result binding (Sol, S-013).** `final_plan`, `read_partial` and `verified_final` are shared by `final()` and
   `report()`. The report scores only bound, checksummed partials of this lock, and only when all of them are present.
   `final.json` must match the run id, revision, twin hashes and content, or it is refused.
10. **Portable final stage (S-013).** Packing lives in `cb2_l1b_pack.sh`, and all logs travel in `$OUT/logs`, so the
    final phase runs on a fresh CPU pod from the loop tarball alone.
11. **Benchmark pod controller (S-014): `cb2_l1b_bench_ctl.py`, with launch and watch modes.**
    - From the moment the pod exists, it holds a claim and cleanup is verified.
    - Delete and stop are verified against the live API, with bounded retries; if verification fails, it reports
      URGENT and exits non-zero.
    - Every external call has a timeout.
    - The billing guard stops the pod before any slow transfer.
    - A failed evidence transfer stops the pod and keeps its volume.
    - It adopts only our own claimed pod, and refuses duplicate claims.

## 7. Workload and cost (measured at launch before committing; priced for David)

**Workload, in environment steps.** Life lengths are L1's means: ~620 fresh, ~650 transfer, ~250 situations.

| part | lives | steps |
|---|---|---|
| per cycle: 3 arms × (40 fresh + 40 transfer + 72 situations), plus 20 learning wake + 20 yoked-source wake + 2 held-out lives | 498 | ≈ 0.20 M |
| 11 probe rounds + 10 wakes | ≈ 5,400 | ≈ 2.2 M |
| final: 3 arms × (1,200 + 1,200), plus chooser-off 300 + 300 | 7,800 | ≈ 4.95 M |

**Throughput.**
- `bench-sleep --concurrent 2` runs both sleeps at once on one GPU (the single-GPU option). It reports wall time,
  peak GPU memory and peak host RAM per sleeper.
  - A CPU smoke run in an 8 GB container was killed for running out of memory, at about 3 GB per sleeper, so a
    one-GPU pod needs real RAM headroom.
- `bench-lives` measures several worker counts, on S2b *and* on a retained learner with its chooser active.
- L1 and the rehearsal measured ~12–13 steps/s per worker.
- Scaling to more workers is **not assumed**. `bench-lives` measures it on the actual pod with its real CPU quota and
  RAM, before the loop starts and again on any pod used for the final battery.

**Price and capacity** (Sol's live RunPod reads, 2026-10-11 ~21:40 UTC, `cb_exchange/sol_next_run_compute_evidence.json`;
these are not a launch quote):
- **RTX 4090:** catalog $0.89/GPU-h (secure), so $1.78/h for a pair. **2× RTX 4090 currently reports availability NONE**
  in both secure and community.
- **CPU pods:** every listed CPU3/CPU5 flavour supports at most **32 vCPUs per pod**. CPU3c is $0.03/vCPU-h, so
  **$0.96/h for 32 vCPUs**. One or two 32-vCPU pods, not one 64.

**Estimate.**
- **Loop on 2× RTX 4090:** about 10 × 5,900 s ≈ 16.5 h ≈ **$29**. Sleep dominates, as in L1.
- **If no GPU pair is available,** three alternatives, each projected by `bench-sleep` on the actual pod before
  committing:
  - (a) 1× RTX 4090 with both sleeps sharing the GPU (`L1_PARALLEL_SLEEP=1`, already supported): slower per cycle,
    cheaper per hour;
  - (b) another GPU pair (e.g. 2× RTX 3090 / A5000);
  - (c) one 48 GB GPU running both sleeps.
- **Final battery:** ~115 worker-hours.
  - On one 32-vCPU CPU pod: ≈ 4 h ≈ **$4**, *if* `bench-lives` shows ~30 useful workers.
  - Worst case on the GPU pod: ≈ 7.7 h ≈ $14.
- **Setup, benches and storage:** ≈ $3–5.
- **Total ≈ $35–50** depending on the GPU option. This needs David's approval against what remains of his GPU budget.

## 8. Launch gates

1. Sol has verified the §6 delta (the repairs 93f3c45 are verified, S-003).
2. **Cleared (S-007, 21:51 UTC).** Sol's locked CPU memory diagnostic passed its joint rule.
   - Coverage: 48 episodes, 16 held-out test worlds, frozen L1 cores.
   - Learning-core balanced knownness accuracy beat:
     - its own one-step baseline: +5.85 points, lower95 4.29;
     - its frame/body embedding: +2.82 (1.37);
     - the reference core: +1.76 (0.75);
     - the frozen core: +2.54 (1.26);
     - the spatial prior: +10.77 (8.35).
   - Result sha `01e932f2…`.
   - **Limits (Sol):** this is retained perceptual-map information on the frozen-S2b policy distribution. It is not
     behavioural growth, learned hands or general memory; 16 test worlds; bounds conditional on fit readouts; trained
     map heads remain weak in absolute tile-type accuracy (~18%). The no-history trigger did not fire, so L1b's core
     objective stays.
3. A GPU option is available. `bench-sleep` and `bench-lives` on the launch pod put the workload inside the approved
   budget.
4. This document is locked (hashed, committed) with the code revision. It is recorded in the repo and Linear (Sol's
   S-006 found no L1b record on Linear yet). David has approved the cost.
