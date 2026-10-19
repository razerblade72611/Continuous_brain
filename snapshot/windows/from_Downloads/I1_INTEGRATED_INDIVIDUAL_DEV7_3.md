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

**Fourteen** validity checks are evaluated and reported for every run. A science run also has refusals
that act *before* the family is spent, so nothing is opened if any of them fails:

- the manifest must equal the preflight's;
- the approval must name this design;
- the central registry entry must be APPROVED;
- the whole preflight chain of custody must verify again (below), including the approval basis recorded at
  approval;
- `CONTINUOUS_BRAIN_REGISTRY` must not point anywhere but this checkout's `registry/`;
- the approval must have been given by the frozen approval authority, David;
- immediately before the spend, the family's base and derived ids (sid, sid+70000, sid+90000) must collide with
  no registered family and no I1 preregistered family at any offset pair (`runner.offset_collisions`, the same
  check that confines dev slices). The preflight runs the same check before its own spend.

**Opening a family is one atomic transition.** `runner.open_family()` holds an exclusive cross-process lock
(`runner.protocol_lock`, an `flock` on `registry/i1_protocol.lock`) for the whole transition, and the steps
run in this order:

1. reload the canonical ledger inside the lock;
2. run the derived-id collision and usability checks;
3. register the family if it is new, and spend it;
4. verify the raw ledger lineage: at most one registration, exactly one spend, made by this run, and
   nothing after it;
5. write and fsync `seed_opened.json`;
6. release the lock.

A second process trying to open the same family waits for the lock, then finds the family spent and is
refused. Any failure after the spend and before the opening record is durable is `I1SpentButBlocked`: a
lineage refusal, an error building the binding or the record, a failed write or fsync. `runner.open_for_run`,
which the runs use, writes a BLOCKED `i1_report.json` whenever the raw ledger shows a spend by this run and the
opening failed in any way, interrupts included. So a spent family always leaves a record of why it was lost;
if even that report cannot be written, it says so on stderr. Approval and
approval recovery hold the same lock. The science run checks the lineage of its own spend again before any
verdict stands. A two-process test requires exactly one of two concurrent openers to succeed. A control
with the lock disabled reproduces ChatGPT's double spend, so the test exercises the real race window.

`scripts/dev/i1_tamper_check.py RUN_DIR` copies a finished run and injects one fault per copy. It confirms
that the check meant to catch each fault turns false, and that the sealed (preflight) view contains no
outcome-bearing field. On `i1_dev7_64k` all **21 of 21** are detected, and the untampered copy is fully
valid (`evidence/i1/dev7/i1_dev7_64k/tamper_check.json`).

The 14 checks are one schema, `evaluate.VALIDITY_KEYS`. Every report asserts it. The lifecycle suite and the
tests import the same tuple, so none of them can drift from it.

**The chain of custody behind the approval** (`runner.verify_preflight_chain`, used by `approve()` and again
by the science run before it opens `i1-v1-dev`). Each link is checked in this order, and approval happens
only if all of them hold:

1. **A real spend.** This checkout's ledger records exactly one spend of `i1-v1-preflight-eng-b`, made by
   this run (source `I1-INTEGRATED-INDIVIDUAL-v1:preflight:<label>:opened`), with no later row for that family.
   A `"spent": true` flag written into a directory counts for nothing.
2. **That spend belongs to this run.** The directory is `runs/i1/<label>`, and its `seed_opened.json`, written
   by `runner.open_family()` at the moment of the spend, names the run, the spend source, the manifest, and the
   registry state at opening: design digest, both registry files, and the seed ledger.
3. **Opened under the design being approved.** The preflight was opened under the PROPOSED preregistration
   with no approval fields. Its design digest equals the current one. Both registry files and the seed ledger
   are byte-identical to their state at opening, so not even an unrelated ledger row may be added between
   the preflight and the approval.
4. **The same machine and code.** The manifest is self-consistent, it is the one the run was opened under,
   and this machine reproduces it exactly.
5. **Validity.** Exactly the 14 checks, all `true`.
6. **Arms.** Exactly the 12 arms of `evaluate.ARMS`, each bound to that manifest.
7. **The sealed report.** `sealed/i1_report.json` exists, its SHA-256 equals the committed one, and its
   validity block and value-free diagnostics agree with the visible view. Its values are never shown.
8. **The registries agree.** There is a single PROPOSED I1 entry, and `runner.lifecycle_problems` holds before
   and after the write.
9. **The approver is the approval authority.** The preregistration freezes `approval_authority: "David"`, and
   that field is part of the design digest. `approve()` refuses any `--approver` that is not exactly `David`.
   The science run and `lifecycle_problems` refuse an approval recorded under any other name. This makes
   "David alone approves" executable, but it is not authentication: what makes it David's act is that he runs
   it on his own machine.

The approval is one journaled act over the two registry files. If it is interrupted, the journal stays, and
everything refuses until `scripts/i1_approve.py --recover` completes it. The approval records an
`approval_basis`: the hashes of the opening record, the validity view, the manifest and the sealed report, and
of the registry state at opening. The science run must reproduce that basis exactly from the preflight
directory. It also binds the registry state, now including the seed ledger, when it opens, and BLOCKS if
anything changed by the end.

What this does and does not protect against: it catches mistakes, stale or mismatched files, and a fabricated
or copied preflight directory. It cannot stop someone with write access to the checkout from forging a
consistent ledger and files, since that person could edit the code too. Against that, the protection is
review: the ledger, the evidence hashes and the code are all inspected by someone other than the author.

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

## 4. Dev result: `i1_dev7_64k` (908064–908127, cloud CPU, ENGINEERING_ONLY, the code as delivered)

The run uses the second cloud engineering family `cloud-dev-claude-eng-2` (908000–908999). All 14 validity
checks hold. The runner reports **PASS** as the verdict this run *would* receive on a scientific family. Report
SHA-256: `88b21f8a…bedb`. Manifest: `96301e1c…c73f` (85 code files, 28 legacy pins). The approval gate, run on this
run's genuine sealed view through the full chain of custody, behaves as specified in 59 of 59 cases
(`scripts/dev/i1_approval_gate_check.py`; §3, §6).

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

- **Accounting:** 817,626 model forwards, all host-side, none driver-side. Zero blocked, unbound or
  cross-slot calls. 200 replay checks.
- **Hidden state:** all 64 RNG-perturbed twins match NORMAL in final store digest, final tick and life
  trace. The frozen-object digests form one set across every life of every runtime arm. All 64 donor
  entries follow `sid XOR 1`, and each DONOR store held exactly the partner's autobiography after the
  intervention.
- **Authority controls:** in every one of the 12 lives, the second half changed, or aborted with the
  accepted error.

**Secondary (reported, never gating):**

- **S1, first half identical under the controller and under fixed SCAFFOLD: 1.0** (n = 64), as predicted.
- **S2, goal-2 episodes pursued DIRECT: 1.0** (862 episodes), as predicted.
- **S3, goal-2 completion: controller 0.367 vs fixed SCAFFOLD 0.500** (n = 60). Paired difference −0.133,
  95% CI [−0.267, −0.017]. Registered as estimation-only.
- **S4, the mechanism at matched goal-2 checks** (preregistered after ChatGPT's review, estimation-only).
  Population: first-goal completers of both arms. At check k, only individuals still pursuing in **both** arms
  are counted. Deltas are measured from check 0; the CI is a paired bootstrap.

  | k | n at risk | reliability gain: controller / fixed | paired difference [95% CI] | capability gain: controller / fixed | paired difference [95% CI] |
  |---|---|---|---|---|---|
  | 1 | 60 | −0.042 / −0.006 | −0.036 [−0.052, −0.022] | 0.038 / 0.047 | −0.009 [−0.015, −0.005] |
  | 2 | 60 | −0.059 / −0.004 | −0.055 [−0.076, −0.036] | 0.072 / 0.086 | −0.014 [−0.021, −0.008] |
  | 4 | 57 | −0.062 / +0.013 | −0.075 [−0.099, −0.052] | 0.121 / 0.139 | −0.018 [−0.028, −0.009] |

  Descriptively, goal-2 episodes under the controller average 2.84 successful and 0.16 failed work steps. Under
  fixed SCAFFOLD they average 1.99 successful and 0.015 failed.

**What S3 and S4 mean.** The developmental controller and the goal system value different things. The
mechanism is in the frozen v0.33 world (`self_development_loop_v1.py`, `evaluate_self_development_loop_v1.py`):

| per episode at capability c | SCAFFOLD | DIRECT |
|---|---|---|
| work steps | 2 (of 4) | 3 (of 4) |
| surprise probability per work step | 0.15 (1−c) | 0.85 (1−c) |
| capability gain per successful work step | 0.13 (1−c) | 0.085 (1−c) |

DIRECT buys **throughput**: about 43% more successful work per episode in this run. It pays in learning, with
smaller capability gains, and in surprises: about 11 times as many failed steps here (14 in 64j, 11 in 56i,
7 in 64h, 10 in 64g). The second goal is scored on
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
| i1_dev7_64e | 906232–906295 | before ChatGPT's binary verification | −0.167 | [−0.283, −0.033] | `i1_dev7_64e/` |
| i1_dev7_64f | 906332–906395 | before the arm-coverage fix | −0.186 | [−0.288, −0.102] | `i1_dev7_64f/` |
| i1_dev7_64g | 906432–906495 | before the chain-of-custody fix | −0.267 | [−0.383, −0.167] | `i1_dev7_64g/` |
| i1_dev7_64h | 906532–906595 | before the fourth gate audit | −0.169 | [−0.288, −0.068] | `i1_dev7_64h/` |
| i1_dev7_56i | 906144–906199 | before the fifth audit (56 individuals) | −0.224 | [−0.347, −0.122] | `i1_dev7_56i/` |
| i1_dev7_64j | 908000–908063 | before the sixth audit (`cloud-dev-claude-eng-2`) | −0.197 | [−0.295, −0.098] | `i1_dev7_64j/` |
| i1_dev7_64k | 908064–908127 | as delivered | −0.133 | [−0.267, −0.017] | `i1_dev7_64k/` |

All twelve point the same way, including the eleven runs whose validity checks all held, but the size varies with the
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

- **ChatGPT's independent binary verification** of `95da5d5` reproduced every hash, the 530-file evidence
  manifest, 14/14 validity, C1–C10, 21/21 tamper detection and all 225 tests. Re-evaluated in a different
  environment, it differed only in 13 last-bit floating-point values of S4. It also found one real pre-approval
  blocker: `approve()` accepted any all-true validity block, and the lifecycle suite and a test helper still
  used the old 12-check list. ChatGPT demonstrated an approval with the two dev7 checks omitted. The previous
  verifier had noted the key-set weakness, and I left it as a note instead of fixing it; that was my error.
  It is fixed (§6). ChatGPT's check of that fix confirmed it and found one narrow remaining gap: approval
  checked each arm binding that was present, but did not require all 12 arms to be present. That is fixed too.
- **ChatGPT's attack on the approval chain** (`4606396`, run 64g) reproduced two more blockers:
  - a preflight opened under design `13fb2f…` was approved after a frozen threshold had been changed (the
    opening binding was recorded but not checked);
  - a fabricated directory was approved. It had a `"spent": true` flag, the current manifest, 14 true checks
    and 12 bound arms, but no ledger spend and no sealed report.

  Both are fixed by the chain of custody in §3. ChatGPT also asked for the canonical registry, the seed ledger
  in the run binding, and science re-verifying the whole chain; all three are in. The procedure text changed,
  so the design digest became `bcf896bc…`.
- **ChatGPT's fourth audit** (`0a0ae40`, run 64h) independently reproduced the package: 283/283 tests,
  531/531 evidence sums, 21/21 tamper detections and 56/56 gate cases. It found three more blockers before
  the preflight:
  - approval by anyone: `approver="Claude"` was accepted;
  - the seed ledger was recorded at opening but not compared at approval, so an unrelated row appended after
    the preflight went unnoticed;
  - the derived-id check confined dev slices only, so a family registered at 398000–398127 (`i1-v1-dev` +
    70000) did not stop the scientific family from opening.

  All three are fixed (§3). The zip no longer carries a dangling `origin/HEAD`. The design digest is now
  `4655f339…`. The run of that code, `i1_dev7_56i`, has 56 individuals, because 906144–906199 was the last
  slice of `cloud-dev-claude-eng` that no earlier I1 run used.
- **ChatGPT's fifth audit** (`c627c93`, run 56i) reproduced the package again: 305/305 tests, 475/475
  evidence sums and 21/21 tamper detections. It found one more blocker: the single-use spend was not atomic
  across processes. Two concurrent science openings both succeeded, and the ledger held two registrations and
  two spends of `i1-v1-dev`. Fixed by the locked opening transition (§3). The design digest is now
  `c7c238ea…`. A second cloud engineering family, `cloud-dev-claude-eng-2` (908000–908999), was registered in the
  cloud ledger for dev runs. Its first run, `i1_dev7_64j` (908000–908063), was the first with primary values
  below 1.0. **C1, C3 and C7 (0.984, 0.984 and 0.969)** all trace back to one individual, 908055:
  - **C1 and C3:** its first goal targets commitment 1, not the evaluator's fragile commitment 2, and its initial
    strategies are reversed. The frozen v0.36 driver (LEGACY_REF) makes the same choice on this seed, and the
    fidelity check holds, so this is behaviour of the frozen mechanisms, not of the integration.
  - **C7:** because 908055 and its donor partner 908054 both target commitment 1, swapping their autobiographies
    cannot flip either goal. That gives C7 its two misses.

  All three stay above their frozen thresholds. The scientific family may contain such individuals too; the
  thresholds were set with that room.
- **ChatGPT's sixth audit** (`db4fa48`, run 64j) reproduced the package (311/311 tests, 531/531 evidence sums,
  fsck clean, 21/21 tamper, and both the locked race and the lock-disabled control). It found one more
  blocker. A simulated disk failure while writing `seed_opened.json`, after the spend, left the family spent
  with no opening record and no report. A PASS could not have come out of that, but a single-use family could
  have been lost without a trace. Fixed (§3). The design digest is now `f3a68bc4…`. `i1_dev7_64k`
  (908064–908127) is the run of the final code.
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
| ChatGPT (binary verification) | `approve()` accepted any all-true validity block; the suite and tests used the old 12-check list | `evaluate.VALIDITY_KEYS` is the single schema; approval requires exactly those 14 keys, all true, and every arm bound to `manifest.json`; the suite and tests import it; parametrized negative tests for each omission, the old schema, unknown keys and binding faults; `scripts/dev/i1_approval_gate_check.py` checks the gate on a real run. The S3 archive was repackaged with a root-relative `SHA256SUMS` |
| ChatGPT (verification of the fix) | approval did not require all I1 arms in `arm_manifests` (a NORMAL-only binding passed) | approval requires `set(arm_manifests) == set(evaluate.ARMS)`, each bound to `manifest.json`; all fixtures use `evaluate.ARMS`; negative tests per unbound arm, NORMAL-only and unknown arm |
| ChatGPT (attack on the approval chain) | a design changed after the preflight was opened was approved; a fabricated preflight with no ledger spend and no sealed report was approved; `CONTINUOUS_BRAIN_REGISTRY` was honoured; the seed ledger was outside the run binding | `runner.verify_preflight_chain` (§3) in `approve()` and again in science; `registry_root` refuses a foreign registry; `seed_ledger_sha256` in the binding, checked at science end; journaled approval with `--recover`; `approval_basis`; `runner.lifecycle_problems` in production, run before and after the write; preflight prints no per-arm timings or error counts; `evaluate.seal_report` shared by production and fixtures, with byte-identical output; the gate check and the lifecycle suite build the chain through `runner.open_family()`; ChatGPT's two reproductions are negative tests, with 20 further breaks at approval and 7 after approval, and the gate check on 64h covered 56 cases |

| ChatGPT (fourth audit) | `approve()` accepted any approver; the seed ledger was not compared at approval; derived ids were checked for dev slices only | `approval_authority: "David"` frozen and digested, with an exact match in `approve()`, science and `lifecycle_problems`; the opening seed-ledger digest must equal the current one at approval; `runner.offset_collisions` is one function, used for dev slices and in `open_family()` before every preflight and science spend; negative tests cover every non-David approver (including empty and whitespace) and a collision at each of the six offset pairs, in both directions, for both the preflight and the science spend; the gate check adds the approver-is-Claude, wrong-case and unrelated-ledger-row cases (59 cases) |

| ChatGPT (fifth audit) | two concurrent openings of one single-use family both succeeded | `runner.protocol_lock` (cross-process `flock`) around `open_family()`, `approve()` and `recover_approval()`; the ledger is reloaded inside the lock; the raw lineage is verified before `seed_opened.json` is written and fsynced; `I1SpentButBlocked` writes a BLOCKED report; science re-checks its own spend lineage before a verdict; two-process tests for science and preflight (exactly one opener succeeds), a lock-disabled control that reproduces the double spend, and lock-timeout tests |

| ChatGPT (sixth audit) | a failure after the spend other than a lineage refusal left the family spent with no opening record and no report | every exception after the spend becomes `I1SpentButBlocked`; `runner.open_for_run` writes a BLOCKED report whenever the raw ledger shows this run's spend and opening failed in any way (interrupts too), and falls back to stderr; fault-injection tests at the lineage check, the registry binding, building the record, the file write, the file fsync, the directory fsync and an interrupt, for preflight and science |

Tests: 330 pass in every legal lifecycle state.

## 7. Path to the scientific I1

1. **ChatGPT verifies the binaries** (the source zip and the evidence bundles) against their SHA-256s.
2. **WSL Stage A:** 8 engineering individuals on David's machine; spends nothing single-use
   (`evidence/i1/dev7/WSL_COMMANDS.md`).
3. **WSL preflight** on `i1-v1-preflight-eng-b` (907100–907163). Only validity is displayed. It writes the
   manifest for the machine that will run the science.
4. **Approval is David's decision, and only his.** He runs `scripts/i1_approve.py --preflight-dir
   runs/i1/wsl_i1_preflight --approver David --i-approve I1-INTEGRATED-INDIVIDUAL-v1` on the preflight machine,
   in the same checkout. It refuses unless every link of the chain in §3 holds, and unless the approver is
   exactly `David`.
5. **Science run** on `i1-v1-dev` (328000–328127). It verifies the whole chain again, then spends the family
   before launch. The verdict is PASS, FAIL or BLOCKED.

What a PASS would establish: the five frozen mechanisms, sharing one store-held state, keep authoring,
pursuing and completing their own goals and advancing their own curriculum; their causal dependence on the
autobiography survives integration; and the individual persists bit-exactly across processes, with no hidden
state outside the store. It would not establish consciousness, sentience or subjective experience, or anything
outside the frozen two-commitment developmental world.
