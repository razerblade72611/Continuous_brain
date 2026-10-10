# CB2-L: learning from lived experience — DRAFT for review

**Status: draft.** Nothing has run; no training, no pod, no seeds opened. H1 (threat escape, DAV-97) is **paused**
after Phase 0. Exploration lane.

## 0. The principle this track exists to serve

> **Competence must increasingly come from learning, not from rules we write for the individual.**

We still design the machinery: learning mechanisms, interfaces, storage, tests. What we stop doing is supplying the
answers about *how to behave*. Every component is classified one of two ways:
- **machinery:** a learner, a memory store, an objective about its own body and life, an interface, a test;
- **answer:** a hand-written routine, a rule about when to do what, a teacher label, a shaped reward that encodes
  our strategy.

Answers are allowed only as **scaffolding**. Each one is listed, its effect measured, and given a retirement gate
(section 5).

**What this track does not claim.** Identity hashes, the consolidation chain and exact rollback establish
*technical continuity*: the same stored individual, changed only by recorded events. They do not establish
subjective selfhood, friendship or consciousness. We pursue a persistent individual without treating those
questions as settled.

### Decision checklist (David's four questions, asked of every step in this track)

1. What is our primary goal?
2. How has the current structure helped us achieve it?
3. Is this truly the best choice for success?
4. What alternatives are we overlooking, and what evidence would make us change direction?

## 1. Where we are, honestly

**Learned today, but fixed after offline training:**
- The predictive core: a DreamerV3 RSSM, trained in r2 and frozen since.
- The vision organ: supervised on privileged labels.
- The command selector: imitation, then PPO (S2b).

**Answers today, written by us:**
- **The executor organ.** All "hands": drink, eat, shelter, explore, requests, build, retreat, reflexes.
- **The 8-command vocabulary,** and the option-feasibility rules ("no wandering off in the dark", …).
- **The episodic memory map.** Its writing rules, confidence threshold, bump memory and object memory.
- **The commitments parser, self-model and speech** (templates).
- **The routine teacher** whose labels started everything.
- **S2b's death penalty,** our reward choice.

**Gate 4 failed in r2** (DAV-88): the core learned *prediction, not persistent world state*.
- It never beat a one-frame snapshot on feature probes (0.278 vs 0.298).
- Nothing recovered out-of-view memory (R² < 0).

**The individual does not learn from its own life at all.** Every improvement so far was made by us, offline, and
installed.

**Headline metric for this track — learned-authored share of actions.** Provenance tags already record who authored
every action:
- `scripted-executor-organ`, the selector's command, and future learned tags;
- **today the learned share of the primitive actions is ~0%:** the selector picks a command, and a script picks every
  action.

This track must move that number, and say why it moved.

## 2. What learns (in order of leverage)

**L-A: Persistent world state in the core** (repairs Gate 4)
- **Objectives that require memory,** not only next-frame prediction:
  - predict what it will see when it returns to a place it saw earlier;
  - reconstruct earlier observations from the present state;
  - predict the timing of night and needs over long horizons.
- **Long sequences in training,** so the state must carry information.
- **Success:** the Gate 4 probes (out-of-view recovery R² > 0, and beating the snapshot baseline) on held-out lived
  lives.

**L-B: A learned episodic memory**
- A store of the individual's own past moments, keyed by the core's state.
- **Usefulness is learned:** what to keep and what to retrieve is trained by whether retrieval improves its
  predictions.
- **Successor to the engineered map** under the existing capability-ledger gate: it must recover remembered
  out-of-view places, and come within 2 points of agreement.

**L-C: Learned value and choice over imagined futures** (Gate 2, "choice")
- A value function trained on its own lived returns *and* on imagined continuations (dreams, labelled DREAMED,
  never autobiography).
- **First use:** choosing among the existing commands by imagined consequence. This is the first decision the
  individual makes from its own model rather than our routine's habits.
- **Success:** removing the value (frozen or ablated) changes its choices and outcomes on fresh worlds — Gate 2 as
  written.

**L-D: Learned hands, one routine at a time**
- Low-level policies trained in imagination and on lived experience (a Dreamer-style actor-critic on the learned
  core), each replacing one executor routine.
- **The command vocabulary itself is scaffolding.** The end state is a policy over primitive actions, with options
  that are learned, not named by us.
- **Order by measured need:** about 74% of damage deaths begin upstream (outside at night, entering threats weak).
  So **shelter and night behaviour** comes before escape.

## 3. How experience changes it: the wake/sleep loop

1. **Wake.** It lives N lives on fresh practice worlds. Every moment goes into the hash-chained archive as LIVED ×
   SELF (this exists and is verified).
2. **Sleep.** It learns from its own archive:
   - the core (L-A), with replay across its *whole lifetime*, not just recent lives, to protect old knowledge;
   - its memory store (L-B);
   - its value and actors in imagination (L-C, L-D).
3. **Consolidate.** One recorded, hash-chained event with lineage: what it learned from (archive range), what
   changed (weights by hash), the objectives and their losses. Exact rollback is kept.
4. **Next wake.**

**Explicit rules:**
- **No answers inside the loop.** No routine-teacher labels after bootstrap, and no shaped rewards that encode our
  strategy.
- **Drives only.** Objectives come from its body and life: needs, damage, death, completing the caretaker's requests,
  and prediction error.
- **Where we write the objective, we say so.** Death and damage penalties are drive *signals*, not instructions on how
  to avoid them. Each one is listed in the scaffolding ledger with its rationale.
- **Measurable effect per cycle.** Each sleep reports what changed in later behaviour on probe worlds (section 4).
  A cycle with no measurable effect is recorded as such.

## 4. How growth is demonstrated

A rising score alone could reflect easier experiences or memorisation. The claim is: **experience caused durable,
transferable improvement.**

**Twins: the causal contrast.** The canonical individual is cloned at cycle 0 into:
- the **learning** twin: sleep enabled;
- the **frozen** twin: lives the same wake schedule on the same world seeds, sleep disabled.
- An optional third twin, **amnesic**: sleeps, but its archive is withheld beyond the current cycle. This separates
  "learning from its accumulated past" from "training compute".

The difference is attributed to learning only through the *probe battery*, never through training lives, because the
twins' training lives diverge.

**Probe battery** (fresh each checkpoint; never archived for learning; reused by both twins with identical seeds):
1. **Fresh worlds:** full lives; survival, score, requests; paired by world.
2. **Situation probes:** branched from identical saved states (the Phase 0 machinery). Examples:
   - outside at dusk without shelter;
   - low food far from cows;
   - a zombie at distance 3 at health 5;
   - a request it has failed before.

   Matched starting states remove "easier experiences".
3. **Transfer probes:** situations and worlds *unlike its training distribution*:
   - held-out request families;
   - altered arrival rates;
   - start-at-night;
   - scarce water.

   Improvement there cannot be memorisation of training worlds.
4. **Retention probes:** situation types from earlier cycles, re-tested K cycles later after training lives that
   contained none of them.

**Analysis:**
- The primary claim is a paired difference (learning − frozen) on each probe family, with one-sided 95% bootstrap
  bounds.
- Growth curves are reported over cycles.
- The power calculation uses paired-difference variance (DAV-97 rev 2).
- Development probes are used during the track; a **sealed** probe battery is used once, for the final claim, from
  block 2 (91081–91400) or a new reserved block.

**What would count as growth:**
- the learning twin beats the frozen twin on fresh *and* transfer probes, with lower95 > 0;
- the advantage persists on retention probes;
- the learned-authored action share rises without losing those gains.

## 5. Retiring scaffolding

The capability ledger (`continuous_brain_cb2/capabilities.json`) already requires every capability to be kept, have
a measured successor, or be removed on purpose. This track extends it to **every scripted answer**, with:
- what it supplies;
- the learned successor;
- the evidence required to retire it.

**The evidence rule is the same for every item:**
- The successor matches or beats the scaffold on paired situation and fresh-world probes.
- The scaffold is then *removed from the acting path*, kept only as a rollback.
- The learned-authored share is reported before and after.

| Scaffold (answer) | Successor (machinery) | Retirement evidence |
|---|---|---|
| Option choice by imitation habits | L-C value over imagined futures | Gate 2: ablating the value changes behaviour; lower95 > 0 vs habit-only on probes |
| Engineered memory map | L-A core state + L-B learned memory | Gate 4 probes; ledger memory gate |
| Shelter/night routine | L-D learned shelter policy | situation probes (dusk, night) non-inferior, then superior |
| Drink/eat routines | L-D | situation probes on need states |
| Explore/frontier routine | L-D, with curiosity from prediction error | coverage and later-use probes |
| Requests/build routines | L-D | request completion on held-out families |
| Reflexes (fight adjacent, lava check) | L-D, with the lava check kept as a safety floor | probes; the safety floor is retired last, if ever |
| 8-command vocabulary | learned options or a primitive policy | equal or better probes with the vocabulary removed |
| Option-feasibility rules | learned | probes |
| Commitments template parser | language organ (Qwen) | the existing ledger gate |
| Routine teacher labels | none (bootstrap only) | already absent from the loop |

## 6. Alternatives we are overlooking, and evidence that would change direction (question 4)

**Baselines, not just alternatives.** Run a **vanilla DreamerV3 agent with no scaffolding** on the same world and
the same amount of experience.
- If our individual cannot beat it on probes, our extra structure is not helping competence.
- We then adopt the vanilla learner as the core and keep only identity and memory infrastructure.

**A different world.** Crafter may cap what can be shown.
- **Pre-declared signal:** the learning twin saturates (two consecutive cycles with no probe gain) while transfer
  probes still show deficits we cannot express in Crafter.
- Then we plan a richer world.

**Pretrained models as organs** (language, vision-language). This is the path you raised for open models trained
together.
- **Signal to pursue it:** language-dependent probes (requests, reports) stall under the template organ while other
  probes improve.

**Change-of-direction triggers, written before running:**
- No learning-vs-frozen advantage on fresh probes after the planned cycles (lower95 ≤ 0) → the learner design is
  wrong; stop and redesign.
- An advantage on fresh but not on transfer probes → memorisation; change objectives before scaling.
- The learned-authored share does not rise across cycles → scaffolding is still doing the work; stop adding learners
  on top of scripts.

## 7. Continuity and testing infrastructure (kept)

- Hash-chained archive with LIVED/DREAMED separation.
- Consolidation with lineage and exact rollback.
- Organ swaps.
- Provenance on every action.
- Preregistration, fresh development worlds, sealed worlds used once, and no peak-picking.
- **New:** the twin protocol and the probe battery become standard instruments.

## 8. First milestone proposal (for review; nothing runs before approval)

**L1 — "it learns from its own life, and we can tell":**
1. **L-A:** add memory-requiring objectives to the core, and continue training it from the archive in sleep. Gate 4
   probes are the check.
2. **L-C:** learned value over imagined command outcomes, acting in the selector. Gate 2 is the check.
3. **Twins:** learning vs frozen, over 10 wake/sleep cycles of about 20 lives each, with development probe batteries
   at every cycle.
4. **Reported:** the learned-authored share, and the vanilla-DreamerV3 baseline on the same experience budget.

**Compute:** wake lives run on David's WSL (fast: 120 lives plus branches in about 30 min). Sleep training (the core)
runs on a pod, about one 4090-hour per cycle, so roughly $10 for 10 cycles — **pending approval**.

## Open questions for review

1. Is L-C (value over the existing commands) an acceptable *first* use of learned choice, given that it keeps the
   command bottleneck for now? The alternative is starting L-D at primitive level immediately.
2. Should the amnesic twin be in L1, or deferred?
3. Should the vanilla-DreamerV3 baseline gate the track (if it wins, we adopt it), or be reported only?
4. Death and damage penalties as drive signals: acceptable, or should drives come only from the body's own variables
   (health, food, drink, energy)?
