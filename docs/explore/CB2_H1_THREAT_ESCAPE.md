# CB2-H1: learned threat-escape hands — protocol (revision 2)

## Status

- **Phase 0 (measurement only, local CPU) is AUTHORISED** by David, 2026-10-10, with the revisions below.
- **Phase 1 training, both development gates, any paid pod and any sealed test remain PENDING.** Each needs its own
  approval.
- Exploration lane, ENGINEERING_ONLY.

Revision 2 records David's answers on DAV-97 (2026-10-10 00:04 CT). The changes from revision 1 are marked **[r2]**.

## 1. Question and scope

With the decision-maker (the canonical S2b selector) and every other executor routine held fixed, can a learned
threat-escape module make the individual live longer, without costing score or commitments, while it stays the same
individual and can be rolled back exactly?

Qualifications kept throughout (David):

- Stopping the S2 line says this PPO setup on the selector plateaued; it does not show that outcome-based learning
  is exhausted.
- Damage causes ~43% of S2b's training-time deaths, but some of those deaths begin with earlier decisions. Phase 0
  measures how much is attributable to the hands before anything is trained.

**What changes:** only the body's actions during an eligible threat encounter.

**What does not change:**
- the S2b selector (weights 8cffde79…);
- the scripted executor's routines;
- the r2 core, vision v1 and the language stand-in;
- the executor's lava check, which applies to every escape action.

## 2. Definitions

**Threat:** a zombie or skeleton in the objects the individual itself perceives at that step (its episodic memory's
`objects`).

**Eligible encounter onset** at step *t*, from the individual's own perception and body sense, when all of these
hold:
- it is awake;
- it is *not enclosed*: on its believed map, at most 2 walkable tiles (grass, sand, path) are reachable from its
  tile within 3 steps — its sealed shelter pocket, or a pocket equally closed;
- and one of:
  - a zombie at Manhattan distance ≤ 3;
  - a skeleton at Manhattan distance ≤ 4;
  - its health dropped on step *t−1* while food, drink and energy were all above 0 (felt pain, not a need).

Distances and the timeout are **provisional [r2]**; Phase 0 reports how well they fit (below).

**Encounter end:**
- **cleared:** 5 consecutive steps with no threat inside the radii;
- **died;**
- **timeout:** 40 steps. **[r2]** A timeout is recorded separately from cleared and never counts as an escape.

Onsets within 5 steps of an end merge into that encounter.

**Per-encounter metrics.** These come from privileged evaluator data (`events.zombie_hits`, `events.arrow_hits`,
true health and objects) and never reach the agent:
- **escaped:** cleared, alive.
- **combat damage:** health lost on steps with a zombie or arrow hit.
- **encounter-associated death [r2]:** death during the encounter or within 50 steps after its end, *with the
  cause "damage"* (health 0 while food, drink and energy are all above 0). All-cause deaths in the same window are
  reported too.
- **subsequent survival:** steps alive after the end, censored at the life's end.

**Fit of the definition [r2, reported in Phase 0]:**
- **missed threats:** steps where the true state has a zombie within distance ≤ 3 (or a skeleton within ≤ 4) that
  the individual did not perceive, and hits that land outside any eligible encounter;
- **timeouts:** encounters that hit the 40-step limit.

**Onset diagnostics.** At each onset: health, food, drink, energy, phase of day, enclosed or not, the selector's
command. Encounters are classed as:
- **hands-attributable:** health ≥ 6, needs > 2, and not outside in the dark;
- **upstream-attributable:** otherwise.

Every damage death is traced to its last encounter.

## 3. Phase 0 — census and headroom (AUTHORISED; local CPU; measurement only)

**Worlds:** S2b, greedy, unchanged evaluation stack, on fresh development worlds **77001–77120**. They have never
been used; development worlds are for measurement and selection only. **No sealed world (90001–91400) is used for
anything in H1 except a declared sealed test.**

### 3a. Census, one full life per world

Reports, each with a bootstrap 95% interval:
- encounters per life;
- escape, timeout and death rates;
- combat damage per encounter and **total combat damage per life [r2]**;
- the share of all deaths that are encounter-associated;
- the hands-attributable vs upstream split;
- the missed-threat counts.

### 3b. Headroom by branching (the "best-escape tester")

**Branching.** For each world, the first hands-attributable onset (fixed in advance) is replayed deterministically:
- `W1Env.snapshot/restore` is bit-exact;
- the individual's state is copied with `Individual.save/restore`.

From that state, each of the following runs **to the end of the life** (death, or 1,000 steps after the onset,
censored):
- **control:** S2b's own hands.
- **flee-k, k ∈ {4, 8, 16}:** for up to k steps of the encounter, step to the walkable, unoccupied neighbour that
  maximises the minimum distance to perceived threats. If no neighbour improves on the current tile, defer to S2b.
- **water:** for up to 16 steps, walk toward the nearest believed water tile. Zombies cannot cross water; arrows can.
- **fight:** attack an adjacent zombie; otherwise approach the nearest one, up to 16 steps.
- **hold:** noop for up to 8 steps.

Within the encounter the script replaces only the executor's action (through a tester hook in the skill host, whose
provenance tag is `phase0-tester-override`). Outside it, and after the script's step limit, S2b acts unchanged.
Every override passes the lava check.

**What the tester sees [r2, documented].**
- Each script's *actions* use only what S2b has: its own perceived objects, believed map, body sense and position.
- The tester's *choice* of the best script per onset is made **in hindsight from privileged outcomes**: true deaths,
  hits and health in the branched futures.
- That choice is the optimistic part. It is a *ceiling estimate, not an attainable improvement*.
- Each fixed script's own average effect is reported separately; it is a non-hindsight estimate.

**Fidelity check.** The pass-2 replay up to the onset, and the control branch from the restored state, must
reproduce the original life action for action. Any mismatch stops Phase 0 and is reported.

**Known approximation.** During an override the individual still perceives every step. Its record of its last
action is the executed one, because the hook sits inside the host.

### 3c. Screening thresholds [r2: screening, not proof]

| | Numerator | Denominator | Screening threshold |
|---|---|---|---|
| (a) Opportunity | encounter-associated (damage-cause) deaths | all deaths in the census | ≥ 30% |
| (b) Headroom | relative reduction in encounter-associated deaths, best-of-family vs control | sampled hands-attributable onsets | ≥ 25% |

Both are reported with bootstrap intervals (numpy `default_rng(20261009)`, 10,000 resamples). (b) is also reported
with the paired change in **overall survival** (end of life, censored at 1,000 steps), because postponing death past
the encounter window is not enough.

**Outcome:**
- If (a) and (b) both clear: Phase 0 reports **"opportunity screened in"**. That is a reason to *propose* Phase 1,
  not a result.
- If not: H1 stops, and the report points at upstream decisions.

## 4. Phase 1 — the escape organ (PENDING approval; outline only)

**Interface:**
- Inside an eligible encounter only, the organ outputs one of
  `{defer, noop, move_left, move_right, move_up, move_down, do}`.
- `defer` is the executor's own action.
- Every action passes the lava check.
- The organ cannot sleep, place, craft or eat.

**Runtime inputs [r2: no privileged input at run time, enforced by a test].** Only what S2b receives:
- the frozen r2-core feature;
- body sense;
- the believed-map crop;
- the perceived threats (4 nearest, kind and relative position);
- facing and daylight;
- enclosure;
- steps into the encounter.

**Initialisation ≡ S2b.** A `defer` bias; verified by exact replay.

**Training (practice run 8, 88,000,000+, never used):**
1. Distil the tester's best-script targets.
2. Encounter-level PPO with reward −(combat damage) − 10·(encounter-associated death), KL to the distilled policy.

**[r2] No sealed world is ever used in training or target generation.**

**Candidate.** The exponential moving average of the weights over the last 25% of training, fixed in advance. No
checkpoint selection.

**Development A: 77121–77240 (fresh). Locked go/no-go rule [r2].** All comparisons are paired against S2b on the same
worlds, with the bootstrap as above. GO only if all hold:
- (i) matched-state branches (same onsets, S2b vs candidate): lower95 of the combat-damage reduction > 0, and mean
  encounter-associated death reduction > 0;
- (ii) full lives: mean paired survival > 0;
- (iii) score and commitments not worse by more than the sealed margins (section 5).

**Development B: 77241–77360 (fresh holdout). Locked confirmation rule [r2].**
- The same single candidate.
- Full lives: mean paired survival > 0, and lower95 > −25 steps.
- Score and commitments within the margins.

**[r2] A failure at A or B ends that candidate.** Any revision must be preregistered anew and evaluated on **new**
development worlds. A and B are never reused for it.

## 5. Sealed test (PENDING; allocation deferred [r2])

**Allocation.** No sealed range is reserved for H1 now. Block 2 has 320 untouched worlds (91081–91400). After Phase 0
(and Development B), the size is fixed by a **power calculation on paired-difference variance** (below), then
written into the ledger before opening.

**Power [r2].** The paired per-world survival difference (H1 − S2b) has SD σ_d, estimated from the Phase 0 branches
and from the Development A/B paired lives — not from S2c.

- With n worlds, the one-sided 95% bootstrap lower bound sits ≈ 1.645·σ_d/√n below the mean.
- To pass with power 1−β on a true gain Δ: **n ≈ ((1.645 + z_{1−β}) · σ_d / Δ)²**.
- Example only: σ_d = 390 (S2c vs S2b) and Δ = 60 steps give n ≈ 115 at 50% power, and n ≈ 262 at 80% power.
  So 200 worlds would *not* reliably detect a 45-step gain.
- n is chosen from the measured σ_d and the Phase 0/Development B effect size. If the required n exceeds what the
  ledger can spare, the test is not run as designed and the design is revised.

**PASS requires all of these.** Every comparison is paired against S2b on the same worlds, with a one-sided 95%
bootstrap:
1. **Survival:** lower95 of (H1 − S2b) per-world survival > 0.
2. **Score:** aggregate Crafter score ≥ 0.97× S2b's, and lower95 of paired per-world unlocks > −0.5.
3. **Commitments:** on-time rate ≥ 0.95× S2b's, and lower95 of paired per-world (fulfilled − failed) > −0.5.
4. **Escape mechanism [r2: required for the threat-escape claim].**
   - Reduced encounter damage on **matched encounters**: same onset states, branched, S2b's hands vs H1's. Lower95 of
     the combat-damage reduction > 0.
   - Full lives report total combat damage per life and encounter frequency, because damage per encounter alone can
     improve when a policy changes which encounters occur.
5. **Identity, provenance and rollback:**
   - integrity of every input;
   - save/restore;
   - the organ install is a hash-chained event with the state hash and identity digest unchanged;
   - exact replay of the first sealed world;
   - 100% action provenance with three tags: selector command, `scripted-executor-organ`, `learned-escape-organ`;
   - exact rollback: swapping the organ back to `defer` reproduces S2b bit for bit.
6. No protocol deviation.

The routine comparison is reported only.

**If PASS:** David decides on installation, recorded as an organ event with S2b as the rollback state. Gate 4 stays
OPEN.

## 6. Identity and rollback engineering (before Phase 1)

- A `hands.escape` organ binding slot using the Gate 7 swap machinery.
- Tests:
  - non-hands channels, the tick and the life are unchanged by install and rollback;
  - `defer` ≡ S2b;
  - a snapshot refuses to restore under a different escape organ;
  - **[r2]** the organ's input builder reads no privileged field.

## 7. Integrity

- Crashes, missing worlds, corrupted artifacts and deviations are reported, never silently rerun.
- Development worlds are for measurement, go/no-go and confirmation.
- Sealed seeds are used once, only by a declared test, never for training or target generation.
- Nothing is tuned after sealed outcomes are visible.
