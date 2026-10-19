# Continuous Brain — Claude Collaboration Dossier
**Current state, evidence chain, and review target for G1 composition retention**
**Date:** 30 September 2026

## Purpose
This dossier is a handoff and review packet for Claude as co-author of the Continuous Brain project. It describes the verified project state, the evidence chain that led to v0.38.1.dev5, the scientific constraints that must not be weakened, and the exact G1 preregistration now awaiting second-author critique. It is not a request for agreement; it is a request for adversarial review before David is asked to approve G1.

## Executive status
- **Canonical engineering baseline:** v0.38.1.dev4 at commit `a71d673a...`, 186/186 tests passed on David's WSL CPU environment.
- **Current review artifact:** v0.38.1.dev5, commit `3e8459a...`, G1 v2 is PROPOSED only; no G1 family is registered or spent.
- **G0 scientific result:** six historical mechanisms independently requalified on fresh preregistered DEV families, all PASS, provenance intact.
- **Blocked item:** v0.23 episodic two-item memory remains outside G1 because its TEST_IDS are task-condition IDs rather than generic RNG seeds; its first high-number G0 family was an infrastructure-only spend and is never reused.
- **Scientific boundary:** G0 demonstrates isolated reproduction, not composition, integration, consciousness, sentience, or superiority of one substrate.

## What Claude is asked to do now
1. Review `G1-COMPOSITION-RETENTION-v2` as a skeptical second author.
2. Look for a way that six isolated legacy stacks could still satisfy the shared-state/scheduler attestation without being one genuinely composed individual.
3. Decide whether engineering preflight needs stronger evidence of non-focal mechanism activity or shared-state use before science can open.
4. Check whether the frozen evaluator vectors are sufficient retention tests under composition or could miss destructive interference outside each focal evaluator.
5. Return feedback as **code defect**, **spec amendment**, or **preregistered experiment**. Agreement is not evidence.
6. Do not approve or run G1 on David's behalf; David remains final approver.

## Collaboration rules
- Proposal → critique → prediction → preregistration → David approval → experiment → artifact → admission/rejection.
- Failures remain failures. Missing evidence is never upgraded by narrative.
- Every run/pass/fail claim must cite a primary artifact or remain UNVERIFIED.
- Historical sealed TEST ranges are never reopened.
- Fresh scientific families are single-use and are marked spent before launch.
- The project remains simulation-only during development.

## Scientific program
The main line is function-first integration: one persistent continuously running individual with state, memory, self/world models, endogenous goals, skills/planning and causal evaluability. In parallel, H1 retains the SNN-first developmental substrate and H2 tests whether the same target properties can be integrated in a conventional recurrent substrate. The transformer/LLM, when added, is a removable semantic/perceptual coprocessor beside the brain; it is not the executive, clock, autobiography authority, persistent goal generator or hidden memory owner.

Consciousness-related work uses functional/computational indicators as evidence targets, not as proof of subjective experience. G1 is specifically a retention-under-composition gate and should not be described as a consciousness gate.

## Current architecture
- `continuous_brain_snn/`: E0 SNN substrate, unchanged from commit `882e584`.
- `continuous_brain_protocols/`: track-neutral contracts and semantic firewall.
- `continuous_brain_substrates/`: shared encoder, SNN adapter, dense-GRU comparator.
- `continuous_brain_unified/`: runtime shell, 15 slots, single-writer channels, DAG scheduler, interventions, checkpointing, null coprocessor, legacy adapters.
- `continuous_brain_eval/`: G1–G5 harness, battery, statistics, seed registry, content-addressed runs.
- E1/E2 are not silently folded into this runtime. Workspace/GWT and attention-schema mechanisms remain interfaces/registration points rather than placeholder implementations.

## Verified G0 evidence
Evidence bundle SHA-256: `a191bf3a06160a867c177f6377470879fe6b233a089f768b093250c7aaa277a5`
Historical-core archive SHA-256: `a73545ba5729214101e17f96578cdbeb3e9f27e16773c61b7a5764edad6aa554`
Protected checkpoint SHA-256: `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`

| Mechanism | Fresh DEV family | Frozen checks | G0 | Raw result SHA-256 |
|---|---:|---:|---|---|
| autobiographical_discrepancy_v0322 | 304000–304255 | 14 | PASS | `37f4489c2b1525b3…` |
| consequence_replanning_v0252 | 301000–301255 | 17 | PASS | `269ab9b10573b458…` |
| endogenous_goal_genesis_v0353 | 306000–306127 | 10 | PASS | `3a2a499fd8cd3922…` |
| event_consistent_self_development_v0341 | 305000–305127 | 16 | PASS | `866c10d49ca08f97…` |
| recurrent_self_trajectory_v031 | 302000–303023 | 17 | PASS | `7bba68ae76bb6e12…` |
| recursive_autobiographical_curriculum_v0361 | 307000–307127 | 11 | PASS | `2a825494e6cee4c4…` |

All six completed with return code 0, frozen top-level pass field true, complete frozen check vector, no threshold/intervention override, no reopened historical TEST, unchanged legacy-tree metadata, pinned evidence and dependency closure.

### Why a historical-core overlay exists
The first real v0.23 G0 attempt exposed an infrastructure incompatibility: the accumulated legacy tree had later been changed from a 5-block nucleus growth stack to 20 blocks, while the protected checkpoint and historical frozen evaluators came from the earlier 5-block core. The first v0.23 family was therefore consumed before a result existed and is permanently classified as infrastructure-only, not scientific FAIL. dev3 introduced a hash-pinned historical-core overlay from the original v0.12.2 archive and a full-runtime engineering preflight before scientific opening.

## Current review target — G1-COMPOSITION-RETENTION-v2
Status: **PROPOSED**

When the six G0-requalified historical mechanisms are simultaneously hosted in one Unified Individual runtime with one canonical shared state store and an active common scheduler, does each retain the complete causal/admission behavior of its own frozen evaluator?

### Core preconditions
- All six G0 evidence rows in registry/g0_requalifications.json must verify.
- All six mechanisms must be bound through hash-pinned audited adapters/state codecs into one continuously running UnifiedRuntime; no stand-ins.
- One canonical runtime-owned state store must be authoritative for shared self_state, autobiography, world_model, goal and planning channels. Compatibility adapters may translate views but may not hold independent authoritative shadow copies of those shared domains.
- Every focal run must use one runtime instance containing all six adapters in the same scheduler graph. No non-focal mechanism may be administratively disabled, bypassed, detached, replaced by a mock, or run in a private duplicate runtime.
- The runtime must emit a composition attestation containing runtime_instance_id, scheduler_graph_hash, binding/adaptor hashes, state-codec hashes, canonical state-store identity, base-checkpoint identity, and per-mechanism scheduler eligibility/invocation counts.
- Each compatibility/evaluator adapter must be hash-pinned and must not modify historical evaluator thresholds, interventions, required check semantics, or pass-field semantics.
- A full-composition ENGINEERING_ONLY preflight must complete for each of the six focal evaluator routes under the exact adapter/binding/state-store/scheduler identities intended for science. Preflight may verify structure, execution, provenance, complete check-key presence and scheduler participation but must not be used to tune against scientific pass/fail values.
- After engineering preflight, the exact composition manifest must be frozen. Any adapter, binding, state codec, scheduler graph, shared-state mapping or historical-core change invalidates the preflight and requires a new preregistration version before science.
- No parameter training, threshold tuning, task-specific routing repair, or post-G0 mechanism repair may occur against proposed G1 DEV families before they open.

### Proposed G1 families
| Mechanism | Frozen checks | Proposed scientific DEV | Proposed engineering preflight |
|---|---:|---:|---:|
| autobiographical_discrepancy_v0322 | 14 | 313000–313255 | 904400–904655 |
| consequence_replanning_v0252 | 17 | 310000–310255 | 903000–903255 |
| endogenous_goal_genesis_v0353 | 10 | 315000–315127 | 904900–905027 |
| event_consistent_self_development_v0341 | 16 | 314000–314127 | 904700–904827 |
| recurrent_self_trajectory_v031 | 17 | 311000–312023 | 903300–904323 |
| recursive_autobiographical_curriculum_v0361 | 11 | 316000–316127 | 905100–905227 |

No range in this table is registered or spent in dev5.

### Decision rule
- **Gate PASS:** PASS only if every one of the six mechanism-level G1 results is PASS.
- **Mechanism PASS:** The frozen evaluator/compatibility contract completes with return code 0; top-level admission/pass field is true; every required_frozen_check is present and true; all six mechanisms were instantiated in one runtime and remained scheduler-eligible; canonical shared-state identity and composition-manifest hash match the frozen preflight; no threshold/intervention override occurred; and all provenance/integrity checks pass.
- **Mechanism FAIL:** A completed scientifically valid composed-runtime evaluator contract returns a false admission/pass field or any required frozen check false while composition/provenance attestations remain valid.
- **Blocked/infrastructure:** Missing check, missing/broken binding, private/shadow authoritative state, disabled/detached non-focal mechanism, composition-manifest mismatch, scheduler-attestation failure, adapter mismatch, result missing, provenance change, dependency mismatch, or process/setup failure. Such outcomes are not scientific FAIL and stop before another family opens.

## Main open risks / review questions for Claude
- Can adapter-local caches or codec behavior still create de facto shadow authoritative state while technically satisfying the one-store rule?
- Is scheduler eligibility/invocation count enough, or should the attestation prove actual shared-state reads/writes for each non-focal mechanism where applicable?
- Could a focal evaluator accidentally serialize the other five mechanisms into harmless inactivity while still showing nonzero scheduler counts?
- Should G1 add a pre-registered 'disable one non-focal mechanism' engineering contrast purely to prove real interaction, or should that wait for G2 to avoid contaminating G1's retention construct?
- Do the six frozen evaluator vectors sufficiently cover interference, or are there cross-mechanism state conflicts that require a new G2 matrix rather than expanding G1?
- Does the composition manifest freeze enough routing detail (channel ownership, state schema, trigger graph, state-codec versions) to make preflight and science genuinely identical?
- v0.23 needs a separate scientifically faithful amendment because condition IDs are not interchangeable with arbitrary seed IDs.

## Known historical failures that must remain visible
- v0.21–v0.22 option-support components include formal failures in their own records; later v0.24 integration success does not erase those failures.
- v0.29 self-broadcast line is a formal failure/control, not an admitted capability.
- Earlier confidence-head metacognition was weak/negative and was not promoted as a capability.
- Risk-gating evidence became less convincing after calibration and remains experimental.
- A MemorySalienceHead proposal was withdrawn after discovering existing memory gating.
- G0 v0.23 high-number family was infrastructure-only and cannot be reused or reinterpreted as a scientific FAIL.

## Version/evidence lineage
- Claude v0.38.0.dev0: `c011d40` — stronger base chosen for unified runtime; David's WSL later passed 166 tests.
- v0.38.1.dev0: `337f2a2d` — D1–D3 fixes, G0 evidence foundation; 174/174 WSL tests and E0 infrastructure PASS.
- dev1: `2a9a6d9a` — fail-closed G0 runner; 178/178 WSL tests.
- dev2: `cf4f978a` — David-approved G0 runner; first v0.23 execution exposed infrastructure flaw.
- dev3: `91bbc854` — historical-core overlay + engineering runtime preflight; six scientific G0 mechanisms later PASS.
- dev4: `a71d673a` — freezes six G0 PASS artifacts; 186/186 WSL tests; current engineering-pass baseline.
- dev5: `3e8459a9` — G1 v2 prereg proposal only; 188 tests collect, targeted clean-clone prereg/eval tests pass; full WSL validation still pending at dossier creation time.

## Linear/project status
- DAV-5 E0 Genesis: In Progress.
- DAV-74 legacy audit: In Progress.
- DAV-76 versioning/experiment registry: Todo, although much of its evidence discipline is already implemented in code.
- DAV-77 v0.34.1 sealed TEST closure: In Review; primary result was recovered and shows the historical sealed test already passed, so it must not be rerun.
- Shared contract document: `Continuous Brain — Joint Implementation Contract for v0.38.0 (DRAFT)`; current implementation has evolved through dev5 under that collaboration protocol.

## Artifact inventory for Claude
- dev5 review artifact: `Continuous_brain_v0.38.1.dev5_g1_prereg_v2.zip`
- dev5 ZIP SHA-256: `2e38d5815307cb3303f6b3abdf577ea0f3091291e76563df692df710ebc6cbea`
- dev5 commit: `3e8459a96b4dac6ad6910a5e8080de8b4a9035e2`
- dev5 branch: `v0.38.1-dev5-g1-prereg-v2`
- G0 evidence bundle SHA-256: `a191bf3a06160a867c177f6377470879fe6b233a089f768b093250c7aaa277a5`
- G0 manifest SHA-256: `aa853ebccce57a477d61cb62d1a6916baf9daf44bde9ed407481060c419e2c3b`
- historical-core SHA-256: `a73545ba5729214101e17f96578cdbeb3e9f27e16773c61b7a5764edad6aa554`
- protected checkpoint SHA-256: `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`

## Requested response format from Claude
Please return only evidence-grounded findings, grouped as:
1. **Code defects** — concrete implementation bugs or loopholes.
2. **Spec amendments** — changes to G1 v2 needed before approval.
3. **Preregistered experiments** — additional tests that answer a specific uncertainty without teaching to the result.
4. **No-objection items** — clauses you believe are sufficiently specified, with a short reason.

For any proposed change, state what observation would make you reverse it. Do not convert the six G0 PASSes into a broader integration claim, do not reopen spent TEST/DEV families, and do not approve G1 on David's behalf.

## Bottom line
The project is past the stage of asking whether six historical mechanisms can still work individually: six have now been requalified on fresh DEV families. The immediate question is harder and more valuable: **can all six remain causally intact while living inside one continuous shared-state individual, without hiding six isolated stacks behind a common shell?** G1 v2 is the proposed test. Claude's role is to attack that design before David decides whether to approve it.