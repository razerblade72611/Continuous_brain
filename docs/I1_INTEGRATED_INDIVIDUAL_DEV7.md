# I1 — one persistent individual (v0.38.1.dev7, ENGINEERING_ONLY)

**Status.** Built, independently reviewed, fixed, independently verified and dev-verified on cloud CPU. Preregistration
`I1-INTEGRATED-INDIVIDUAL-v1` is **PROPOSED**. No scientific family has been registered or opened. Every
number below is engineering data that the author has seen; none of it is a scientific result.

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

No host keeps mechanism state in Python between ticks. Whatever the store holds is the individual, so a
store intervention (erase, replace, freeze, donor) changes the individual itself, and a runtime snapshot
saves and restores it. The canonical projection channels are recomputed from the stored bytes whenever
those bytes change, so they cannot go stale after an intervention.

**The life** (`integration/life.py`). The driver plays the environment, using the frozen v0.33 world
(`build_bootstrap_memory`, `initial_capabilities`, `execute_development_episode`). The steps:

1. **Birth:** the bootstrap history is imprinted into the autobiography.
2. **Setup:** the developmental controller initializes its phases from the autobiography's reliability, the individual authors its first goal, and the curriculum observes it.
3. **Interruption:** one admitted episode.
4. **First goal:** up to 16 episodes, with the strategy chosen by the developmental controller from current reliability.
5. **Second-goal decision:** gated on the goal lifecycle held in the store.
6. **Second goal:** authored from the updated autobiography, observed by the curriculum, and pursued.

**Redirects.** All five specs are installed, and only v0.31 is bound as a dev6 host. A call to one of the
other four mechanisms' APIs from inside its own host runs the frozen code, and is counted. The same call
from the driver reaches a host that has no such API, and is blocked. The driver therefore cannot run a
private copy of a mechanism without the run recording it.

## 2. Arms

| arm | what changes | what it shows |
|---|---|---|
| NORMAL | nothing | the integrated individual |
| FIXED_SCAFFOLD | pursuit strategy fixed to SCAFFOLD (the frozen v0.35/v0.36 rule) | fidelity reference; comparison arm for S3 |
| LEGACY_REF | no runtime: the frozen v0.36 `complete_first_goal` | fidelity reference |
| ERASED | ERASE `autobiography.state` right after birth | goal genesis depends on the store-held history (C6) |
| DONOR | REPLACE it with the paired individual's birth autobiography | the history determines which goal is authored (C7) |
| ROLLBACK | REPLACE it with the individual's own birth autobiography before the second-goal decision | the second goal depends on the updated history (C8, C9) |
| PERSIST_SAVE / PERSIST_RESUME | save the runtime before the second-goal decision; a new process restores it | the individual persists bit-exactly (C10) |
| AUTHORITY_SKILLS / _GOAL / _CURRICULUM | ERASE that host's state before the second-goal decision (4 individuals each) | validity control: the store, not a private copy, is the state |

Intervention ticks come from the NORMAL life of the same individual. Each intervened arm must reproduce
NORMAL's store digest and tick exactly up to the tick of the intervention.

## 3. Validity checks (blocking) and their negative controls

Twelve validity checks are evaluated and reported for every run. A science run also has three refusals
that act *before* the family is spent:

- the manifest must equal the preflight's;
- the approval must name this design;
- the central registry entry must be APPROVED.

If any of these fails, nothing is opened.

`scripts/dev/i1_tamper_check.py RUN_DIR` copies a finished run and injects one fault per copy. It confirms
that the check meant to catch each fault turns false. It also confirms that the sealed (preflight) view
contains no outcome-bearing field. On `i1_dev7_64c` all **18 of 18** are detected, and the untampered copy
is fully valid (`evidence/i1/dev7/i1_dev7_64c/tamper_check.json`).

| check | guards against | injected fault that it catches |
|---|---|---|
| fidelity_to_frozen_v036_driver | the integration changing the frozen behaviour | an altered store-held goal field; altered capabilities |
| replay_bit_exact_before_every_intervention | an intervened arm diverging before its intervention | a wrong mark tick; a wrong pre-intervention digest |
| interventions_applied_exactly_as_planned | interventions at the wrong tick or target, or in the wrong arm | a wrong tick; an intervention in NORMAL |
| manifest_consistent_across_arms | arms of one run produced by different code | one arm's manifest differs (this also fired for real; see §5) |
| store_authority_controls_change_the_individual | a host secretly keeping its own state; a crash passing as authority | each authority erase made to have no effect; an abort by an unrelated exception |
| driver_record_matches_store_lifecycle | the driver and the store disagreeing about completion | a flipped store completion flag |
| no_driver_side_model_forwards | model computation done outside a host | a driver-side forward of the neutral model |
| no_driver_side_mechanism_calls | private mechanism copies; cross-host shortcuts | a blocked call; a cross-slot nested call |
| host_state_read_from_store | hosts not reading their state | zero state reads for one host |
| no_errors | silent failures | an injected arm error |
| legacy_tree_unchanged, pins_verified | modification of, or drift in, the frozen legacy tree | — (checked by metadata hash and 28 pins) |
| (sealed view) | the preflight revealing outcomes | counts of planned individuals or replay checks (which reveal C2) appearing in `i1_validity.json` |

## 4. Dev result: `i1_dev7_64c` (906808–906871, cloud CPU, ENGINEERING_ONLY, the code as delivered)

All 12 validity checks hold. The runner reports **PASS** as the verdict this run *would* receive on a
scientific family. Report SHA-256: `da09cecb…12a6dc`. Manifest: `9357369b…a74b8` (84 code files, 28
legacy pins).

| check | value | n | frozen threshold |
|---|---|---|---|
| C1 first goal endogenous (targets the fragile commitment) | 1.000 | 64 | 0.95 |
| C2 first goal completes (store lifecycle) | 0.906 | 64 | 0.80 |
| C3 history guides the initial strategy | 1.000 | 64 | 0.95 |
| C4 second goal authored | 1.000 | 58 | 0.95 |
| C5 curriculum advances | 1.000 | 58 | 0.95 |
| C6 erased autobiography → no goal | 1.000 | 64 | 1.0 |
| C7 donor autobiography flips the goal | 1.000 | 64 | 0.95 |
| C8 rolled-back autobiography repeats the goal | 1.000 | 58 | 0.95 |
| C9 updated vs rolled-back second goals differ | 1.000 | 58 | 0.95 |
| C10 persistence bit-exact (digest, tick, life trace) | 1.000 | 64 | 1.0 |

Other measurements from the same run:

- **Accounting:** 679,491 model forwards, all host-side, none driver-side. Zero blocked, unbound or
  cross-slot calls. 198 replay checks.
- **Authority controls:** in every one of the 12 lives, the second half changed or aborted.
  - Erasing the skills state makes the controller raise "developmental phases are not initialized", the
    only abort accepted.
  - Erasing the goal lifecycle means no second goal is requested.
  - Erasing the curriculum means the curriculum has no previous goal to compare against.

**Secondary (reported, not gating):**

- **S1, first half identical under the controller and under fixed SCAFFOLD: 1.0** (n = 64), as predicted.
  The controller graduates a commitment at baseline + 0.12, and the goal completes at baseline + 0.12, so
  goal-1 pursuit is SCAFFOLD under both.
- **S2, goal-2 episodes pursued DIRECT: 1.0** (874 episodes), as predicted.
- **S3, goal-2 completion: controller 0.224 vs fixed SCAFFOLD 0.534** (n = 58). Paired difference −0.310,
  bootstrap 95% CI [−0.431, −0.190]. Registered as estimation-only.

  This is the new thing that integration reveals: the developmental controller and the goal system
  value different things. The mechanism is in the frozen v0.33 world (`self_development_loop_v1.py` and
  `evaluate_self_development_loop_v1.py`):

  | per episode at capability c | SCAFFOLD | DIRECT |
  |---|---|---|
  | work steps | 2 (of 4) | 3 (of 4) |
  | surprise probability per work step | 0.15 (1−c) | 0.85 (1−c) |
  | capability gain per successful work step | 0.13 (1−c) | 0.085 (1−c) |
  | expected successful work steps, c = 0.8 | 1.94 | 2.49 |

  DIRECT buys **throughput**, more successful work per episode, and pays in learning and surprises.
  The second goal is scored on the self-model's reliability, which surprises lower, and no I1 outcome
  measures throughput. The controller graduates to DIRECT on reliability alone, and the goal then asks
  for reliability that DIRECT practice erodes.

  **Engineering diagnostic** `i1_s3_trace60` (906940–906999; `scripts/dev/i1_s3_trace.py`;
  `evidence/i1/dev7/s3_trace/`). The arms run unchanged under a read-only tracer. Each traced life
  reproduces its untraced twin's final store digest, tick and life trace. For the 54 lives that complete
  the first goal in both arms, at matched goal-2 episode counts:

  | after k goal-2 episodes | k = 1 | k = 2 | k = 4 |
  |---|---|---|---|
  | reliability gain, controller (DIRECT) | −0.021 | −0.037 | −0.042 |
  | reliability gain, fixed SCAFFOLD | +0.002 | +0.005 | +0.009 |
  | capability gain, controller | 0.040 | 0.070 | 0.120 |
  | capability gain, fixed SCAFFOLD | 0.046 | 0.080 | 0.134 |
  | lives still pursuing (controller / fixed) | 54 / 54 | 54 / 54 | 53 / 51 |

  Comparisons at larger k suffer from survivorship, because the fixed arm's remaining lives are its hard
  cases, so they are not used.

  An earlier reading in the dev7 discussion was wrong: it said the controller arm ended with *higher*
  capability (0.993 vs 0.976). That gap is an episode-count confound. Unfinished goals run all 16
  episodes, and the controller arm practised 15.1 episodes on average against 11.6.

  So the frozen mechanisms are coherent and causally coupled, and they disagree about what progress is.
  That is the problem for after I1: development and goals need a shared account of what the individual
  is trying to become (for example, a controller that keeps scaffolding while a goal on that commitment
  is open, or goals that value competence and throughput, not only self-predictability). Either would
  be a new mechanism, with its own preregistration.

**The same estimate on other disjoint cloud slices** (engineering; the author saw every one):

| run | individuals | code | difference | 95% CI | evidence |
|---|---|---|---|---|---|
| i1_dev64 | 906600–906663 | pre-review | −0.138 | [−0.276, −0.017] | `earlier/` |
| i1_dev7_64 (BLOCKED) | 906680–906743 | 9f0949d, mid-run edit | −0.224 | [−0.362, −0.086] | `i1_dev7_64_blocked/` |
| i1_dev7_64b | 906744–906807 | 9f0949d | −0.150 | [−0.283, −0.017] | `i1_dev7_64b/` |
| i1_dev7_64c | 906808–906871 | as delivered | −0.310 | [−0.431, −0.190] | `i1_dev7_64c/` |

All four point the same way, but the size varies with the individuals. The scientific family will give
the estimate that counts. `i1_dev7_64b` also passed all 12 validity checks under the slightly earlier code
(report `7599af8b…`): C2 0.938, n = 60 completers.

## 5. Disclosures

- **`i1_dev7_64` (906680–906743) is BLOCKED by its own manifest check.** I bumped the version strings in
  the middle of the run, so arms after NORMAL ran a different code state. The check worked as designed. The
  run is kept in `evidence/i1/dev7/i1_dev7_64_blocked/` and was repeated with the tree unchanged.
- `i1_dev7_64b` overlaps the smoke slice `i1_dev7_smoke8` (906760–906767). Both are engineering runs.
- **The independent verification of the first dev7 package** (a fresh agent) confirmed hashes, a clean
  run from the zip, a byte-identical re-evaluation and every number in the docs. It also found:
  - an incomplete approval procedure;
  - a C2 leak in the preflight's visible output;
  - authority aborts that accepted any exception;
  - a conservative derived-id coincidence for the scientific family (320000–320127 +70000 = g0-v023-dev
    +90000).

  All four are fixed, and `i1-v1-dev` moved to 328000–328127 before anything was registered.
  A re-verification of the final package found no blocker. Its remaining notes:
  - approval checks that the validity block is all true, but not its exact keys; the manifest match makes
    this hard to abuse;
  - per-arm run times weakly track C2, so the WSL tar sends no stdout and normalises file times;
  - after approval, 4 tests failed by design because they asserted the PROPOSED state. ChatGPT's review
    asked for this to be fixed, and it is (§6, last row).
  `i1_dev7_64c` is the run of the fixed code.
- **Cloud preflight incident.** `i1-v1-preflight-eng` (907000–907063) was consumed by mistake in the cloud
  on 2026-10-02. It is recorded as spent and has been replaced by `i1-v1-preflight-eng-b` (907100–907163).
  The sealed report (`8c061c1e…`) has not been opened.
- **"Sealed" means withheld from display and committed by SHA-256.** The sealed file is not encrypted. The
  commitment makes later alteration detectable; it does not prevent anyone with the run directory from
  reading it.

## 6. The independent review (second) and what changed

A fresh reviewer agent, with no access to the authoring context, audited the I1 code. Every finding is
fixed in dev7:

| finding | severity | fix |
|---|---|---|
| The manifest covered too little; `route.py` names collided | HIGH | all package `.py` files plus the I1 scripts and legacy JSONs, keyed by repo path (84 files); every arm process computes the manifest itself |
| Dev seeds were not confined to the named family | HIGH | `_dev_sids`: the slice must lie inside the family; it and its derived ids (+70000, +90000) may touch no other registered family except a containing engineering reserve, and none of the I1 families |
| "Sealed" overstated secrecy | MEDIUM | the wording is fixed everywhere (`SEAL_NOTE`) |
| `no_unbound_calls` was vacuous | MEDIUM | all five specs are redirected; driver-side calls block; new checks on driver-side calls and forwards |
| Host-state authority was weakly evidenced | MEDIUM | AUTHORITY_SKILLS / _GOAL / _CURRICULUM controls |
| The driver gated on its own copy of completion | LOW–MEDIUM | goal host `status` API; the second goal is gated on the store; agreement is checked |
| Deviation-log wording | LOW | corrected |
| Stale canonical projections | LOW | recomputed from the stored bytes, inside the host |
| Core extraction depended on the working directory | LOW | repo-anchored path |
| Replay and plan checks were weak | LOW | the mark tick is also compared; interventions are compared to the plan (tick, target, kind) |
| No report when an arm failed after the spend | LOW | a BLOCKED `i1_report.json` is written |
| Approval was not bound to the design | LOW | science requires `approved_design_sha256 == design_digest(prereg)` |
| Goal-2 completion cycle labels | LOW | 100+k, matching the episode indices |
| Verification happened after registration | LOW | pins and the historical core are verified before any family is registered or spent |
| *(verification)* Approval procedure incomplete | SHOULD-FIX | `scripts/i1_approve.py`: one checked act that sets both registry files |
| *(verification)* Preflight output revealed C2 | SHOULD-FIX | the sealed view carries only value-free diagnostics; tamper-checked |
| *(verification)* Any crash passed as an authority abort | SHOULD-FIX | only the erased mechanism's own missing-state error counts |
| *(verification)* Derived-id coincidence; one-sided dev check | NOTE | `i1-v1-dev` moved to 328000–328127; all-offset-pair checks in the runner and the tests |
| *(ChatGPT review)* Tests failed after approval by design | SHOULD-FIX | No test asserts one lifecycle state any more. `i1_lifecycle_problems` checks the repository against invariants that hold in every legal state, and legal and illegal states are simulated on copies. `scripts/dev/i1_lifecycle_suite.py` runs the whole suite on copies put into PROPOSED, PREFLIGHT_SPENT, APPROVED (through the real `i1_approve.py`) and SCIENCE_SPENT: 223 pass in each (`evidence/i1/dev7/lifecycle_suite.json`). The run found a fifth state-dependent test, the family-collision test, which is also fixed |

Tests: 223 pass, in every legal lifecycle state (see the last row of the table above). They now cover dev confinement (including derived-id collisions), design-digest binding,
manifest coverage, authority plans, projection freshness after an erase, and blocking of driver-side calls.

## 7. Path to the scientific I1

1. **ChatGPT review** of `registry/i1_integration_v1.json` and this document. Any amendments become v2.
2. **WSL engineering check** on a cloud-dev slice. Confirm all validity checks hold on David's machine
   before the single-use family is spent (`evidence/i1/dev7/WSL_COMMANDS.md`).
3. **WSL preflight** on `i1-v1-preflight-eng-b` (907100–907163). Only validity is displayed. It writes the
   manifest for the machine that will run the science.
4. **Approval.** David runs `scripts/i1_approve.py --preflight-dir runs/i1/wsl_i1_preflight --approver David
   --i-approve I1-INTEGRATED-INDIVIDUAL-v1` on the preflight machine. It refuses unless the preflight is a
   valid, spent preflight of the preflight family and the machine reproduces its manifest. It then writes
   both registry files: the status, the approver, the time, the preflight manifest and the design digest.
5. **Science run** on `i1-v1-dev` (328000–328127): `scripts/i1_run.py --confirm-open
   I1-INTEGRATED-INDIVIDUAL-v1`. The family is spent before launch, and the verdict is PASS, FAIL or
   BLOCKED.

What a PASS would establish: the five frozen mechanisms, sharing one store-held state, keep authoring,
pursuing and completing their own goals and advancing their own curriculum; their causal dependence on the
autobiography survives integration; and the individual persists bit-exactly across processes. It would
not establish consciousness, sentience or subjective experience, and it would not establish anything
outside the frozen two-commitment developmental world.
