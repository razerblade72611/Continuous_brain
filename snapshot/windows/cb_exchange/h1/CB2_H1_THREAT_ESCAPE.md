# CB2-H1: learned threat-escape hands — DRAFT protocol (not locked, nothing run)

Status: **DRAFT for review (Sol, David)**. No code is run and no seeds are opened until this is approved and a locked
preregistration (`evidence/explore/cb2/h1-prereg.json`) is committed. Exploration lane, ENGINEERING_ONLY.

## 1. Question and scope

With the decision-maker (the canonical S2b selector) and every other executor routine held fixed, can a learned
threat-escape module make the individual live longer, without costing score or commitments, while it stays the same
individual and can be rolled back exactly?

David's qualifications, kept in view throughout:

* Stopping the S2 line says this PPO setup on the selector plateaued; it does not show outcome-based learning is
  exhausted.
* Damage causes ~43% of S2b's training-time deaths, but some of those deaths start with earlier decisions (being
  outside at night, entering a fight already weak). Phase 0 measures how much of the damage is attributable to the
  hands at all before anything is trained.

**What changes:** only what the body does during an eligible threat encounter (section 2). **What does not change:**
the S2b selector (weights 8cffde79…), the scripted executor's routines (drink, eat, rest/shelter, explore, requests,
build), the r2 core, vision v1, the language stand-in, and the executor's lava safety check, which also applies to
every escape action.

## 2. Definitions (computed from the individual's own perception, so the organ can gate itself at run time)

**Threat:** a zombie or skeleton among the objects the individual perceives at that step.

**Eligible encounter onset** at step *t*, when all of these hold:
- it is awake and not inside its sealed shelter;
- and one of the following:
  - a zombie at Manhattan distance ≤ 3;
  - a skeleton at Manhattan distance ≤ 4;
  - it was hit (zombie or arrow) on step *t−1*. Hit events come from the evaluator's event log and are used only
    for this eligibility rule. The organ itself sees only its health drop.

**Encounter end:**
- *cleared*: the first step after which no threat is inside those radii for 5 consecutive steps;
- *died*: the individual dies during the encounter;
- *timeout*: 40 steps pass.

Onsets within 5 steps of an encounter's end merge into that encounter.

**Per-encounter metrics.** These come from privileged evaluator data (events and true health) and never reach the
agent:
- **Escape success:** the encounter is cleared, alive, within 40 steps.
- **Combat damage:** health lost on steps with a `zombie_hits` or `arrow_hits` event.
- **Subsequent survival:** steps alive after the encounter ends, censored at 300 steps or at the life's end.
- **Encounter death:** death during the encounter or within 50 steps after it.

**Onset diagnostics.** These address David's second qualification. At each onset we record:
- health, food, drink and energy;
- phase of day;
- inside the shelter or not, and distance to the shelter;
- the selector's current command.

Encounters are then classed as:
- **hands-attributable:** health ≥ 6, all needs > 2, and not outside in the dark;
- **upstream-attributable:** the encounter began weak or exposed.

Every damage death is traced to its last encounter and given that encounter's class.

## 3. Phase 0 — census and headroom (CPU only, no learning, no sealed seeds)

**Worlds:** S2b lives on fresh development worlds **77001–77120**. These have never been used; development worlds
are for measurement and selection only.

1. **Census:**
   - encounters per life;
   - escape success, combat damage and encounter death rates;
   - the share of all deaths preceded by an encounter;
   - the hands-attributable vs upstream split.
2. **Branch fidelity check.** The environment snapshot/restore is bit-exact (`W1Env.snapshot/restore`). The
   individual is restored from its in-process state. For every sampled onset, a *control branch* (S2b's own hands,
   restored at the onset) must reproduce the original continuation action for action. If any branch differs,
   Phase 0 stops and the problem is reported.
3. **Headroom (a privileged look-ahead teacher, for measurement and as a training target only, never at run time).**
   At each hands-attributable onset, branch the state and try a fixed, preregistered family of escape scripts:
   - flee away from threats for k ∈ {4, 8, 16} steps;
   - flee toward the known shelter;
   - flee toward a water wall;
   - fight when adjacent;
   - hold still;
   - S2b's own hands (control).

   Report the best script's reduction in combat damage and encounter death, *paired against the control from the
   same state*.

**Go/no-go to Phase 1, preregistered.** Proceed only if both hold:
- (a) encounters precede ≥ 30% of S2b's deaths;
- (b) on hands-attributable encounters, the best-of-family teacher cuts encounter death by ≥ 25% (relative) with a
  paired one-sided 95% lower bound > 0.

Otherwise H1 stops. The report then points to upstream decisions (night exposure, entering fights weak) as the
lever.

## 4. Phase 1 — the escape organ (pod; only after Phase 0 passes)

**Interface:**
- At each executor step inside an eligible encounter, the organ outputs one of
  `{defer, noop, move_left, move_right, move_up, move_down, do}`.
- `defer` returns the executor's own action unchanged. Outside encounters the organ is never consulted.
- Every organ action passes through the executor's lava check.
- The organ cannot sleep, place, craft or eat.

**Inputs:**
- the frozen r2-core feature and body sense (the same ones the selector sees);
- the local belief-map crop;
- the 4 nearest threats (kind and relative position);
- facing and daylight;
- whether it is in the shelter, and distance to the shelter;
- steps into the encounter.

**Initialisation ≡ S2b.** The `defer` logit is biased so that iteration 0 reproduces S2b exactly. This is verified by
replaying development worlds action for action.

**Learning, two stages, both on practice worlds (run 8: 88,000,000 + …, never used):**
1. **Distillation.** Learn the Phase 0 teacher's best script at hands-attributable onsets. This is supervised and
   low-variance.
2. **Encounter-level PPO.**
   - Episodes are encounters branched from snapshots.
   - Reward: −(combat damage) − 10·(encounter death); nothing else. This is local credit, unlike S2's life-level
     returns.
   - KL is anchored to the distilled policy.

**Selection guard (the iteration-80 lesson).** There is no checkpoint sweep and no peak-picking:
- The single candidate is the **exponential moving average of the weights over the last 25% of training**, fixed
  in advance.
- Development data is used only for go/no-go on that one candidate, never to choose among candidates.

**Development A: 77121–77240 (fresh), go/no-go**
- Comparisons are paired against S2b on the same worlds.
- (i) Branch metrics at the same onset states:
  - combat-damage reduction, with lower95 > 0;
  - encounter-death reduction, with mean > 0.
- (ii) Life-level paired survival, with mean > 0.
- If this fails, the result is reported, and nothing goes further.

**Development B: 77241–77360 (fresh holdout), confirmation**
- The same single candidate:
  - life-level paired survival, with mean > 0;
  - score and requests non-inferior, with the same rules as the sealed test.
- Only if Development B passes is a sealed range declared in `sealed-ledger.json`.

## 5. Sealed test (declared only after Development B passes)

**Worlds:** 200 worlds from block 2. The proposed range is **91081–91280**; it is written into the ledger before it
is opened.

**Arms:** the H1 candidate, frozen S2b and the routine, all on the same worlds. Each sealed seed is used once.

Bootstrap: paired per-world differences; one-sided 95% lower bound = 5th percentile of 10,000 bootstrap means, with
numpy `default_rng(20261009)`.

**PASS requires all of these:**
1. **Survival:** lower95 of paired survival (H1 − S2b) > 0.
2. **Score, non-inferior:**
   - aggregate Crafter score ≥ 0.97× S2b's;
   - lower95 of paired per-world unlocks (H1 − S2b) > −0.5.
3. **Commitments, non-inferior:**
   - on-time request rate ≥ 0.95× S2b's;
   - lower95 of paired per-world (fulfilled − failed) (H1 − S2b) > −0.5.
4. **Mechanism:** lower95 of the paired per-world reduction in combat damage per encounter > 0. This ensures that
   any survival gain comes through escape.
5. **Identity, provenance and rollback:**
   - integrity of every input;
   - save/restore;
   - the organ install is a recorded, hash-chained event with the state hash and identity digest unchanged;
   - exact replay of world 91081;
   - **100% action provenance** with three tags: selector command, `scripted-executor-organ`,
     `learned-escape-organ`;
   - **exact rollback:** swapping the organ back to `defer` reproduces S2b's actions bit for bit on a replay world.
6. No protocol deviation.

The routine comparison is reported but does not gate the verdict; the target is improvement over the canonical S2b.

**Power.** Paired survival differences between adjacent controllers have SD ≈ 390 steps (S2c vs S2b). With 200
worlds the lower bound sits about 45 steps below the mean, so a true gain of about 45+ steps is needed to pass.

**If PASS:** David decides whether to install it into cb2-ind-0001 as a recorded organ event, with S2b kept as the
rollback state. Gate 4 (prediction / persistent memory) stays OPEN.

## 6. Identity and rollback engineering (built and tested before Phase 1 runs)

- The escape organ gets its own binding slot (`hands.escape`). Install, swap and rollback reuse the Gate 7 organ-swap
  machinery (`organ_swap.swap_organ`), generalised beyond the perception slot.
- Tests:
  - every non-hands channel, the tick and the life are unchanged by install and by rollback;
  - with `hands.escape = defer`, behaviour is identical to S2b;
  - a snapshot refuses to restore under a different escape organ.
- The organ and its weights card are stored by hash in `<home>/organs/`.

## 7. Integrity

- Any crash, missing world, corrupted artifact or protocol deviation is reported, never silently rerun.
- Development worlds are used for measurement, go/no-go and confirmation only.
- Sealed seeds are used once.
- Nothing is tuned after sealed outcomes are visible.

## 8. Cost and order

1. Phase 0: CPU only. About 120 lives plus branches, a few hours on a CPU pod or overnight in the container. No GPU
   needed.
2. Engineering and tests for section 6.
3. Phase 1 on one 4090 pod, about 4–6 h.
4. Development A, then Development B, then the sealed test, each on the same pod session or a follow-up.

## Open questions for review

1. Radii (zombie ≤ 3, skeleton ≤ 4) and the 40-step timeout: are these the right definition of an encounter?
2. Go/no-go thresholds in Phase 0 (30% of deaths; 25% teacher reduction): too lax or too strict?
3. Should the mechanism criterion (5.4) gate PASS, or only be reported?
4. Is distillation from a privileged look-ahead teacher acceptable as a training target? It is never used at run
   time.
