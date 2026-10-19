# I1 — one persistent individual (v0.38.1.dev7, ENGINEERING_ONLY)

**Status.** Built, reviewed twice by independent agents, reviewed by ChatGPT, amended, and dev-verified on cloud
CPU. Preregistration `I1-INTEGRATED-INDIVIDUAL-v1` is **PROPOSED**. No scientific family has been registered or
opened. Every number below is engineering data that the author has seen; none of it is a scientific result.
Approval is David's alone (§7).

**Scope.** I1 is a narrower **five-mechanism self-lineage milestone**. It does not close, replace or satisfy
the six-mechanism composition-retention objective of G1. v0.25.2 consequence replanning is excluded (it has no
shared interface with the self lineage), v0.23 remains blocked, and G1-v2 stays PROPOSED on its own terms.

## 1. What I1 is

Until dev6, each of the five self-lineage mechanisms had been re-qualified in isolation (G0), and dev6
showed that each one runs unchanged when its calls go through the runtime's canonical store (E1). I1 is
the first time they act together on **one individual**.

One individual is one `UnifiedRuntime`. Its persistent state lives in the store, in blob channels that
each host reads at the start of every tick:

| slot | host | frozen code it runs | state channel |
|---|---|---|---|
| self_state | dev6 `MechanismHost` (stateless) | v0.31 recurrent self-trajectory model | — |
| autobiography | `AutobiographyHost` | v0.32.2 trajectory memory, neutral autobiographical model | `autobiography.state` |
| skills | `DevelopmentHost` | v0.34.1 history-relative developmental controller | `skills.state` |
| goal_genesis | `GoalHost` | v0.35.3 endogenous goal generator and goal lifecycle | `goal_genesis.state` |
| curriculum | `CurriculumHost` | v0.36.1 `goal_is_advanced`, `same_goal_semantics` | `curriculum.state` |

**Store-authority invariant** (preregistered, `design.store_authority_invariant`). Anything that can change
the individual's future behaviour is in one of three places:

1. the store;
2. the driver's environment state, which is the world and is saved with the snapshot for persistence;
3. objects that are immutable for the whole process.

It is checked four ways:

| what could carry hidden state | check |
|---|---|
| host Python fields | hosts keep no mechanism state between ticks, and projections are recomputed from the stored bytes; the AUTHORITY arms show the store *is* the state |
| frozen models and fixed-parameter objects (shared trajectory model, neutral model, controller, generator) | SHA-256 digests before and after every life of every arm, which must form a single set (validity `no_hidden_mutable_state`) |
| global randomness (torch, numpy, random) | the **RNG_PERTURBED** arm re-seeds all three with unrelated values before birth and after every tick; every life must equal its NORMAL twin bit-exactly (validity `no_hidden_mutable_state`). The environment's own randomness is per-episode `random.Random(seed)` from (individual, cycle), part of the frozen v0.33 world |
| anything else held by the process | PERSIST_RESUME restores in a **new process** from the snapshot plus the environment state and must be bit-exact (C10) |

**The life** (`integration/life.py`). The driver plays the environment, using the frozen v0.33 world. The steps:

1. **Birth:** the bootstrap history is imprinted into the autobiography.
2. **Setup:** the controller initializes its phases from the autobiography's reliability, the individual authors its first goal, and the curriculum observes it.
3. **Interruption:** one admitted episode.
4. **First goal:** up to 16 episodes, with the strategy chosen by the controller.
5. **Second-goal decision:** gated on the goal lifecycle held in the store.
6. **Second goal:** authored, observed by the curriculum, and pursued.

At every goal check the driver also records, without any store call or RNG use, the goal commitment's
self-model reliability, its true capability, and each episode's successful and failed work steps. These
records are used for S4.

**Redirects.** All five specs are installed, and only v0.31 is bound as a dev6 host. A call to one of the
other four mechanisms' APIs from inside its own host runs the frozen code, and is counted. The same call
from the driver reaches a host that has no such API, and is blocked.

## 2. Arms

| arm | what changes | what it shows |
|---|---|---|
| NORMAL | nothing | the integrated individual |
| FIXED_SCAFFOLD | pursuit strategy fixed to SCAFFOLD (the frozen v0.35/v0.36 rule) | fidelity reference; comparison arm for S3 and S4 |
| LEGACY_REF | no runtime: the frozen v0.36 `complete_first_goal` | fidelity reference |
| ERASED | ERASE `autobiography.state` right after birth | goal genesis depends on the store-held history (C6) |
| DONOR | REPLACE it with the birth autobiography of individual **sid XOR 1** | the history determines which goal is authored (C7) |
| ROLLBACK | REPLACE it with the individual's own birth autobiography before the second-goal decision | the second goal depends on the updated history (C8, C9) |
| PERSIST_SAVE / PERSIST_RESUME | save the runtime before the second-goal decision; a new process restores it | the individual persists bit-exactly (C10) |
| RNG_PERTURBED | global random generators re-seeded around every tick | validity control: no hidden dependence on global randomness |
| AUTHORITY_SKILLS / _GOAL / _CURRICULUM | ERASE that host's state before the second-goal decision (4 individuals each) | validity control: the store, not a private copy, is the state |

**Donor rule** (preregistered, `design.donor_rule`). Each individual's donor is the other member of its
even/odd pair, `sid XOR 1`. The rule uses only the ids, so it is fixed before any arm runs and cannot depend
on any outcome. Every plan entry records `donor_sid`. Validity `donor_pairing_fixed_by_ids` requires that it
equals `sid XOR 1` and that the payload is byte-identical to that partner's NORMAL birth autobiography.

Intervention ticks come from the NORMAL life of the same individual. Each intervened arm must reproduce
NORMAL's store digest and tick exactly up to the tick of the intervention.

## 3. Validity checks (blocking) and their negative controls

**Fourteen** validity checks are evaluated and reported for every run. A science run also has three refusals
that act *before* the family is spent, so nothing is opened if any of them fails:

- the manifest must equal the preflight's;
- the approval must name this design;
- the central registry entry must be APPROVED.

`scripts/dev/i1_tamper_check.py RUN_DIR` copies a finished run and injects one fault per copy. It confirms
that the check meant to catch each fault turns false, and that the sealed (preflight) view contains no
outcome-bearing field. On `i1_dev7_64e` all **21 of 21** are detected, and the untampered copy is fully
valid (`evidence/i1/dev7/i1_dev7_64e/tamper_check.json`).

| check | guards against | injected fault that it catches |
|---|---|---|
| fidelity_to_frozen_v036_driver | the integration changing the frozen behaviour | an altered store-held goal field; altered capabilities |
| replay_bit_exact_before_every_intervention | an intervened arm diverging before its intervention | a wrong mark tick; a wrong pre-intervention digest |
| interventions_applied_exactly_as_planned | interventions at the wrong tick or target, or in the wrong arm | a wrong tick; an intervention in NORMAL |
| manifest_consistent_across_arms | arms of one run produced by different code | one arm's manifest differs (this also fired for real; see §5) |
| store_authority_controls_change_the_individual | a host secretly keeping its own state; a crash passing as authority | each authority erase made to have no effect; an abort by an unrelated exception |
| host_state_read_from_store | hosts not reading their state | zero state reads for one host |
| driver_record_matches_store_lifecycle | the driver and the store disagreeing about completion | a flipped store completion flag |
| no_driver_side_model_forwards | model computation done outside a host | a driver-side forward of the neutral model |
| no_driver_side_mechanism_calls | private mechanism copies; cross-host shortcuts | a blocked call; a cross-slot nested call |
| **no_hidden_mutable_state** | state outside the store (global RNG, mutated models or objects) | an RNG twin that differs; a frozen model whose digest changes during a life |
| **donor_pairing_fixed_by_ids** | outcome-dependent donor selection | a donor that is not `sid XOR 1` |
| no_errors | silent failures | an injected arm error |
| legacy_tree_unchanged, pins_verified | modification of, or drift in, the frozen legacy tree | — (checked by metadata hash and 28 pins) |
| (sealed view) | the preflight revealing outcomes | counts that would reveal C2 appearing in `i1_validity.json` |

## 4. Dev result: `i1_dev7_64e` (906232–906295, cloud CPU, ENGINEERING_ONLY, the code as delivered)

All 14 validity checks hold. The runner reports **PASS** as the verdict this run *would* receive on a
scientific family. Report SHA-256: `c827c209…0642`. Manifest: `875ccd54…b465` (84 code files, 28 legacy pins).

| check | value | n | frozen threshold |
|---|---|---|---|
| C1 first goal endogenous (targets the fragile commitment) | 1.000 | 64 | 0.95 |
| C2 first goal completes (store lifecycle) | 0.938 | 64 | 0.80 |
| C3 history guides the initial strategy | 1.000 | 64 | 0.95 |
| C4 second goal authored | 1.000 | 60 | 0.95 |
| C5 curriculum advances | 1.000 | 60 | 0.95 |
| C6 erased autobiography → no goal | 1.000 | 64 | 1.0 |
| C7 donor autobiography flips the goal | 1.000 | 64 | 0.95 |
| C8 rolled-back autobiography repeats the goal | 1.000 | 60 | 0.95 |
| C9 updated vs rolled-back second goals differ | 1.000 | 60 | 0.95 |
| C10 persistence bit-exact (digest, tick, life trace) | 1.000 | 64 | 1.0 |

Other measurements from the same run:

- **Accounting:** 802,462 model forwards, all host-side, none driver-side. Zero blocked, unbound or
  cross-slot calls. 200 replay checks.
- **Hidden state:** all 64 RNG-perturbed twins match NORMAL in final store digest, final tick and life
  trace. The frozen-object digests form one set across every life of every runtime arm. All 64 donor
  entries follow `sid XOR 1`, and each DONOR store held exactly the partner's autobiography after the
  intervention.
- **Authority controls:** in every one of the 12 lives, the second half changed or aborted.
  - Erasing the skills state makes the controller raise "developmental phases are not initialized", the
    only abort accepted.
  - Erasing the goal lifecycle means no second goal is requested.
  - Erasing the curriculum means the curriculum has no previous goal to compare against.

**Secondary (reported, never gating):**

- **S1, first half identical under the controller and under fixed SCAFFOLD: 1.0** (n = 64), as predicted.
- **S2, goal-2 episodes pursued DIRECT: 1.0** (818 episodes), as predicted.
- **S3, goal-2 completion: controller 0.350 vs fixed SCAFFOLD 0.517** (n = 60). Paired difference −0.167,
  95% CI [−0.283, −0.033]. Registered as estimation-only.
- **S4, the mechanism at matched goal-2 checks** (preregistered after ChatGPT's review, estimation-only).
  Population: first-goal completers of both arms. At check k, only individuals still pursuing in **both** arms
  are counted. Deltas are measured from check 0; the CI is a paired bootstrap.

  | k | n at risk | reliability gain: controller / fixed | paired difference [95% CI] | capability gain: controller / fixed | paired difference [95% CI] |
  |---|---|---|---|---|---|
  | 1 | 60 | −0.056 / −0.021 | −0.035 [−0.050, −0.020] | 0.037 / 0.045 | −0.009 [−0.014, −0.004] |
  | 2 | 60 | −0.066 / −0.013 | −0.053 [−0.072, −0.035] | 0.069 / 0.082 | −0.013 [−0.020, −0.007] |
  | 4 | 55 | −0.079 / −0.006 | −0.073 [−0.098, −0.049] | 0.123 / 0.138 | −0.014 [−0.022, −0.008] |

  Descriptively, goal-2 episodes under the controller average 2.84 successful and 0.16 failed work steps. Under
  fixed SCAFFOLD they average 1.98 successful and 0.018 failed.

**What S3 and S4 mean.** The developmental controller and the goal system value different things. The
mechanism is in the frozen v0.33 world (`self_development_loop_v1.py`, `evaluate_self_development_loop_v1.py`):

| per episode at capability c | SCAFFOLD | DIRECT |
|---|---|---|
| work steps | 2 (of 4) | 3 (of 4) |
| surprise probability per work step | 0.15 (1−c) | 0.85 (1−c) |
| capability gain per successful work step | 0.13 (1−c) | 0.085 (1−c) |

DIRECT buys **throughput**: about 43% more successful work per episode in this run. It pays in learning, with
smaller capability gains, and in surprises, about 9 times as many failed steps. The second goal is scored on
the self-model's reliability, which surprises lower, and no I1 outcome measures throughput. The controller
graduates to DIRECT on reliability alone, and the goal then asks for reliability that DIRECT practice erodes.
So the frozen mechanisms are coherent and causally coupled, but they disagree about what progress is.

An earlier reading in the dev7 discussion was wrong: it said the controller arm ended with *higher*
capability. That was an episode-count confound, since unfinished goals run all 16 episodes.

The first engineering diagnostic `i1_s3_trace60` (906940–906999, `scripts/dev/i1_s3_trace.py`,
`evidence/i1/dev7/s3_trace/`) found the same pattern with a read-only tracer. Each traced life equalled its
untraced twin. Because that diagnostic was seen before S4 was written down, S4 is labelled
engineering-motivated.

**S3 on other disjoint cloud slices** (engineering; the author saw every one):

| run | individuals | code | difference | 95% CI | evidence |
|---|---|---|---|---|---|
| i1_dev64 | 906600–906663 | pre-review | −0.138 | [−0.276, −0.017] | `earlier/` |
| i1_dev7_64 (BLOCKED) | 906680–906743 | 9f0949d, mid-run edit | −0.224 | [−0.362, −0.086] | `i1_dev7_64_blocked/` |
| i1_dev7_64b | 906744–906807 | 9f0949d | −0.150 | [−0.283, −0.017] | `i1_dev7_64b/` |
| i1_dev7_64c | 906808–906871 | before ChatGPT amendments | −0.310 | [−0.431, −0.190] | `i1_dev7_64c/` |
| i1_dev7_64d | 906032–906095 | amended, before the third verification | −0.203 | [−0.339, −0.068] | `i1_dev7_64d/` |
| i1_dev7_64e | 906232–906295 | as delivered | −0.167 | [−0.283, −0.033] | `i1_dev7_64e/` |

All six point the same way, including the five runs whose validity checks all held, but the size varies with the
individuals. The scientific family will give the
estimate that counts.

**After I1** (agreed with ChatGPT). The next mechanism is a **goal-aware developmental controller**. While an
open goal targets a commitment and still needs reliability, it keeps SCAFFOLD; otherwise it keeps the frozen
graduation rule. It will be compared against the frozen controller and against fixed SCAFFOLD on fresh
development families:

- **Primary question:** whether it recovers goal-2 completion without harming first-goal completion or
  curriculum advance.
- **Secondary:** the reliability and capability trajectories, throughput, surprise burden and time to
  completion.

It will get its own preregistration. I1 is not modified: it closes first, as the frozen integration test.

## 5. Disclosures

- **`i1_dev7_64` (906680–906743) is BLOCKED by its own manifest check.** I bumped the version strings in the
  middle of the run. The check worked as designed. The run is kept in `evidence/i1/dev7/i1_dev7_64_blocked/`
  and was repeated with the tree unchanged.
- **Overlap:** `i1_dev7_64b` overlaps the smoke slice `i1_dev7_smoke8` (906760–906767). Both are engineering runs.
- **Independent verification of the first dev7 package** (a fresh agent). It confirmed the hashes, a clean run
  from the zip, a byte-identical re-evaluation and every number in the docs. It also found four problems:
  - an incomplete approval procedure;
  - a C2 leak in the preflight's visible output;
  - authority aborts that accepted any exception;
  - a conservative derived-id coincidence for the scientific family.

  All four are fixed, and `i1-v1-dev` moved to 328000–328127 before anything was registered.
- **ChatGPT's second-author review** (Linear contract, 2026-10-02) required:
  - a lifecycle-proof test suite;
  - an explicit scope;
  - a mechanically checked store-authority invariant;
  - a frozen donor rule;
  - an explicit science freeze;
  - S4 preregistered as estimation-only;
  - approval by David alone.

  All of these are in this version (§6).
- **A third independent verification** (a fresh agent) of the amended package reproduced every hash,
  count and digest. Its positive control, which injected global-RNG use inside a host, turned
  `no_hidden_mutable_state` false, so the check is not vacuous. It found four things, all now fixed:
  - the tamper tool crashed on runs whose first individual completes no goal;
  - the donor check read the plan only, and now also verifies the applied payload;
  - RNG_PERTURBED was missing from the no-intervention list;
  - the science run did not re-check the registry at the end, and now blocks if it changed.

  `i1_dev7_64e` is the run of the delivered code.
- **Cloud preflight incident.** `i1-v1-preflight-eng` (907000–907063) was consumed by mistake in the cloud on
  2026-10-02. It is recorded as spent and replaced by `i1-v1-preflight-eng-b` (907100–907163). The sealed
  report (`8c061c1e…`) has not been opened.
- **"Sealed"** means withheld from display and committed by SHA-256. The file is not encrypted.

## 6. Reviews and what changed

| source | finding | fix |
|---|---|---|
| reviewer agent | the manifest covered too little; `route.py` names collided | all package `.py` files plus the I1 scripts and legacy JSONs, keyed by repo path (84 files); every arm computes it itself |
| reviewer agent | dev seeds were not confined to the named family | `_dev_sids` confines the slice to one family; the slice and its derived ids collide with nothing at any offset pair |
| reviewer agent | "sealed" overstated secrecy; `no_unbound_calls` was vacuous; host authority was weakly shown | wording fixed; all five specs redirected; AUTHORITY arms |
| reviewer agent | gating, projections, core path, replay/plan checks, BLOCKED report, design-bound approval, cycle labels, verification order | all fixed (store-gated second goal, recomputed projections, repo-anchored path, tick + plan comparison, BLOCKED report, `approved_design_sha256`, 100+k labels, verify before registering) |
| verifier agent | approval procedure incomplete; preflight view revealed C2; any crash passed as an abort; derived-id coincidence | `i1_approve.py`; value-free sealed view; exact abort rule; family moved to 328000–328127 |
| ChatGPT | tests failed after approval by design | no test asserts a lifecycle state; invariants are checked in every legal state; `scripts/dev/i1_lifecycle_suite.py` runs the whole suite in PROPOSED, PREFLIGHT_SPENT, APPROVED and SCIENCE_SPENT copies (`evidence/i1/dev7/lifecycle_suite.json`) |
| ChatGPT | scope | stated as a five-mechanism milestone that does not close G1 (`scope.milestone`, `relation_to_G1`) |
| ChatGPT | store authority must be explicit and mechanically checked | `design.store_authority_invariant`; RNG_PERTURBED arm; frozen-object digests; validity `no_hidden_mutable_state` |
| ChatGPT | donor pairing must be frozen and outcome-independent | `design.donor_rule`; `donor_sid` in every plan entry; validity `donor_pairing_fixed_by_ids` |
| ChatGPT | science freeze | `decision_rule.science_freeze` lists everything frozen at approval, the manifest rules, and the binding of the mutable registry records (`registry_binding` in `seed_opened.json` and the science report) |
| ChatGPT | register the S3 mechanism | S4: k ∈ {1, 2, 4}, the at-risk rule, paired bootstrap; estimation-only and labelled engineering-motivated |
| ChatGPT | approval | David's alone; Claude, the author, and ChatGPT, the verifier, do not approve |

Tests: 225 pass in every legal lifecycle state.

## 7. Path to the scientific I1

1. **ChatGPT verifies the binaries** (the source zip and the evidence bundles) against their SHA-256s.
2. **WSL Stage A:** 8 engineering individuals on David's machine; spends nothing single-use
   (`evidence/i1/dev7/WSL_COMMANDS.md`).
3. **WSL preflight** on `i1-v1-preflight-eng-b` (907100–907163). Only validity is displayed. It writes the
   manifest for the machine that will run the science.
4. **Approval is David's decision, and only his.** He runs `scripts/i1_approve.py --preflight-dir
   runs/i1/wsl_i1_preflight --approver David --i-approve I1-INTEGRATED-INDIVIDUAL-v1` on the preflight machine.
5. **Science run** on `i1-v1-dev` (328000–328127). The family is spent before launch, and the verdict is PASS,
   FAIL or BLOCKED.

What a PASS would establish: the five frozen mechanisms, sharing one store-held state, keep authoring,
pursuing and completing their own goals and advancing their own curriculum; their causal dependence on the
autobiography survives integration; and the individual persists bit-exactly across processes, with no hidden
state outside the store. It would not establish consciousness, sentience or subjective experience, or anything
outside the frozen two-commitment developmental world.
