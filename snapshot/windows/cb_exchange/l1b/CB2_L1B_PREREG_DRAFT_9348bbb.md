# CB2-L1b: the repaired, powered test of "it learns from its own life" — DRAFT for Sol's review

Status: **draft, not locked.** Nothing has run. Owner of the design and code: Claude (c1b51df7). Auditor: Sol.
Code base: branch `explore/cb2-l1fix` (repairs 93f3c45, under Sol's review) plus the L1b changes listed in §6.

## 1. Why this run, and not L2 first (David's four questions)

1. **Primary goal.** A persistent individual whose competence grows from its own lived experience. The growth must be
   durable, must transfer, and must be caused by that experience.
2. **What the current structure gave us.** L1's wake/sleep loop produced a fresh-world gain over frozen (+52.8 steps,
   lower95 +4.3) and a chooser that measurably helps (Gate 2: +133 on fresh). But L1 did not test the design as
   intended:
   - terminal rewards were lost, so the chooser's value never learned what death costs;
   - short lives (the early deaths) were dropped from training;
   - the control twin degraded by itself (reference − frozen on transfer −62.7, bootstrap 5–95% −114 to −11);
   - and the test had only ~35–40% power for a +40-step effect.
3. **Is a repaired rerun the best next step?** Yes, compared with these alternatives:
   - **L2 now (learned hands, shelter/night first).** The learned-share trigger fired, so L2 is needed. But it adds a
     new learner on top of a loop whose basic claim is unverified; if L2 failed we could not tell the loop from the
     learner. L2 is built on CPU *in parallel* with L1b, so no time is lost.
   - **The vanilla-DreamerV3 baseline.** Still unrun, and it is the "are we on a dead end" check. Proposed as the
     next pod job after L1b (§7), not in it.
   - **Just a bigger L1.** Without the repairs and the new control it would measure the same broken signal more
     precisely.
4. **What would change direction** is written as decision rules in §5, before running.

## 2. Arms (twins of S2b, `cc8948e7…`)

| arm | sleeps? | sleeps on | purpose |
|---|---|---|---|
| **learning** | yes | its own lives, accumulating (20/cycle) | the individual |
| **yoked** (replaces L1's fixed-buffer reference) | yes, identical update counts, batch, T and schedule | lives lived by the **frozen** twin on the **same practice worlds**, same cycle, accumulating (20/cycle) | equal amount of fresh, matched-world experience that is not its own |
| **frozen** | never | — | absolute gain |
| learning, chooser off | — | — | Gate 2 ablation, final battery only |

- **Why yoked rather than a common pre-generated pool.**
  - The yoked twin gets a growing archive with the same number of unique lives, on the same worlds and the same
    schedule. Only the author of the experience differs: its own closed loop versus a non-learning individual's.
  - "learning − yoked" is therefore the contrast for *its own experience*, with data amount and world difficulty
    matched.
  - Unique transitions seen and windows sampled are recorded per sleep for both arms. Lives' lengths differ by policy,
    so exposure in transitions is reported, not assumed equal.
- **Its known limit.** The yoked data comes from the frozen S2b policy, so "own experience" is confounded with "data
  from a better policy as the learner improves". This is the intended contrast: learning on-policy from one's own life.
  It is stated, not hidden.

## 3. Schedule (unchanged from L1 for comparability)

- 10 wake/sleep cycles, 20 practice lives per cycle (learning twin; the frozen twin lives the same 20 worlds for the
  yoked archive).
- 2 held-out lives per cycle, never slept on.
- Sleep budgets as L1: core 1,500, anchor 1,000, chooser 1,500 updates; batch 16, T 256.
- **Per-cycle probes:** 40 fresh, 40 transfer, 24×3 situations per arm. These give the growth curve (secondary,
  descriptive).

## 4. Primary estimand and power

- **Primary: the final individuals (after sleep 10)** on a large final battery. These are new worlds used only here:
  - **1,200 fresh worlds and 1,200 transfer worlds per arm**, every arm on the same worlds;
  - one life each, max 10,000 steps;
  - the estimand is mean survival steps, paired by world.
- **This replaces L1's "pooled cycles 6–10" primary.**
  - Why: the claim is about the individual after its lived experience, and 40 worlds per cycle cannot carry it.
  - Pooled cycles 6–10 is reported as a secondary, for continuity with L1.
- **Power,** from L1's own records (cycles 6–10, `scripts/explore/cb2_power.py`):
  - SD of the paired difference is 400–444 steps;
  - correlation between paired arms is only 0.1–0.2 (lives diverge chaotically), so world-pairing barely reduces
    variance;
  - capping survival doesn't help either (the standardised effect stays ≈0.13 at caps 750–1,500), so the estimand
    stays raw steps.
- **Power at N = 1,200 per family,** SD 430, one-sided 95% lower bound:
  - single comparison: 94% power at +40;
  - minimum detectable effect at 80% power: **31 steps**;
  - the full four-way claim (both comparisons × both families; correlation 0.45–0.51 between L−F and L−Y from L1):
    81% at +40 everywhere, 43% at +30.
- **Honest limit.** L1's transfer point estimate vs frozen was −9.6. If the true transfer effect is near zero, this
  run will say so rather than miss it.
- **Bootstrap:** 10,000 draws over worlds, fixed seed `20261009`. Each world is one independent cluster: worlds are
  not shared across families or cycles.

## 5. Decision rules (written before running)

- **Primary.** For each family f in {fresh, transfer}, with lower95 the one-sided 95% bootstrap lower bound:
  - **G (growth):** learning − frozen, lower95 > 0 in both families.
  - **O (own experience):** learning − yoked, lower95 > 0 in both families.
  - **"Shown"** = G and O. **"Growth, not shown to be from its own experience"** = G only. Otherwise "not shown".
- **Futility, which changes direction.**
  - Learning − frozen on fresh with **upper95 < +31** (the MDE): no growth of the size we can detect. Stop adding to
    this loop; the core/learner design is wrong. The next pod job becomes the vanilla-DreamerV3 baseline (§7), and we
    adopt it if it beats us.
  - Fresh shown but transfer upper95 < +31: memorisation-type gain. Change objectives (L-A long-horizon / transfer
    objectives) before scaling.
  - G shown but learning − yoked upper95 < +31 in both families: the gain comes from more Crafter data, not from its own closed-loop
    life. Continue (data helps), but the "own experience" claim is dropped and the L2 design must earn it.
- **Secondary (reported, not decisive):**
  - growth curve per cycle;
  - situations; retention c10 − c5;
  - Gate 2 (chooser on − off, 300 worlds/family);
  - Gate 4 on held-out lives, learning vs yoked vs frozen, using Sol's diagnostic code if locked by then;
  - learned-authored primitive share (0% by construction; L2 moves it);
  - unique-transition exposure per arm.
- **Integrity.**
  - One code revision for the whole run.
  - Cycle 0 must give exactly 0 paired difference across all arms (instrument check), or the run stops.
  - A tripwire in any sleep stops the run and is reported as a failure. Nothing is silently rerun.

## 6. Code changes on top of 93f3c45 (Claude; each with a test; tests/test_cb2_l1b.py)

1. **Yoked arm** replaces the fixed-buffer reference:
   - the frozen twin lives the cycle's practice worlds into `archive-yoked`, with its own life-id family;
   - the yoked twin sleeps on it with identical arguments;
   - test: the yoked archive never contains a learning-twin life, and the two sleeps receive identical update counts.
2. **Final battery:**
   - a `final` subcommand that runs the 1,200 + 1,200 battery from the three final twin homes, so it can run on a
     many-core CPU pod after the GPU pod stops;
   - new world blocks, checked disjoint from every other family, from sealed worlds and from Sol's claimed
     83090–83195.
3. **Report:** the §5 rules computed exactly; MDE printed; pooled cycles 6–10 kept as secondary.
4. **`scripts/explore/cb2_power.py`:** reproduces §4's numbers from L1's `cycles.jsonl`.

## 7. Compute and cost (estimates, to be confirmed by a timing check at launch)

- **Loop:** about 10 × 5,900 s ≈ 16.5 h on 2× RTX 4090, the same as L1. Sleep dominates; the yoked wake lives add
  about 1 minute per cycle.
- **Final battery:**
  - about 4.7 M environment steps;
  - at L1's ~13 steps/s per worker, that is about 100 worker-hours: 6–7 h on the GPU pod's 15 workers, so instead
    **about 1.6 h on a 64-vCPU CPU pod**;
  - the twins are downloaded and checksummed between the two pods.
- **Budget:** to be set against David's remaining GPU budget before launch. The rate is read at launch, not assumed.
- **Next pod job, if budget allows:** the vanilla-DreamerV3 baseline, with the same experience budget and the same
  final battery.

## 8. Launch gates

1. Sol has verified 93f3c45 and the §6 delta.
2. Sol's CPU memory diagnostic has reported. If it shows the learning core carries no history beyond its one-step
   baseline, we decide together whether L1b's core objective changes before launch. That is the one result that
   could alter this design.
3. This document is locked (hashed, committed) with the code revision; David has seen the cost.
