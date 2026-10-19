# v0.38.1.dev6 — store-mediated composition route (ENGINEERING_ONLY)

Author: Claude (AI agent). Status: engineering infrastructure and engineering evidence only. No G1 family was registered or opened, and no scientific claim is made. Every run used the registered engineering family `cloud-dev-claude-eng` (906000–906999); each use is logged in `evidence/dev6/seed_usage.jsonl`.

## 1. What was built

`continuous_brain_unified/composition/` runs a frozen legacy evaluator **unmodified** as the runtime's task driver. Each top-level call it makes to a mechanism API becomes one runtime tick, in three steps:

1. **The call goes into the store.** The arguments are encoded by an exact, fail-closed codec (`codec.py`). The driver writes them to the blob channel `driver.<slot>.request`.
2. **A host serves it.** The `MechanismHost` bound in that slot reads the request from the store and decodes fresh objects, so nothing passes by Python reference. It executes the original legacy callable, then writes the encoded result, any in-place argument mutation, and its slot's canonical channels.
3. **The result comes back from the store.** The driver reads `<slot>.response` and applies the argument post-state in place, keeping the caller's object identities. It then returns the decoded result.

Other pieces:

- **`spec.py`** is the boundary table. It lists the six mechanisms, their hosting slot, their API callables, shared model loaders, stateless constructors and canonical-channel projections.
- **Runtime changes.**
  - Blob channels (`shape=None`).
  - A per-reader read log that counts reads of written versus default values and keeps a hash chain.
  - Declared driver channels (`RuntimeConfig.driver`), which leave the config hash unchanged when unused.
  - A `DRIVER_OWNER` constant.
- **`route.py`** writes a composition attestation for every run. It contains:
  - runtime instance id, config hash and scheduler graph hash;
  - source hashes of the composition package, the codec version and the spec digest;
  - environment fingerprint;
  - per-host statistics, per-API tick and response traces, and content flows;
  - the read log, store digest and model-forward accounting;
  - the census: binding loads, verified shared acquisitions, private instances, constructed instances and `torch.load` paths.
- **Dev scripts:** `scripts/dev/g0_route_dev.py`, `composition_route_dev.py`, `e1_null_composition.py` and `e1b_store_authority.py`.

## 2. What crosses the store and what does not

Every number below is measured in the attestations of the final E1 run (32 seeds per mechanism).

| Focal route | Hosts used | Store-mediated calls | Model forwards in hosts | Model forwards directly in the driver | Mediation |
|---|---|---:|---:|---:|---|
| recurrent_self_trajectory_v031 | v031 | 6,882 | 131,360 | 101 | PARTLY_DRIVER_DIRECT |
| autobiographical_discrepancy_v0322 | v031 | 1,920 | 20,352 | 1,408 | PARTLY_DRIVER_DIRECT |
| event_consistent_self_development_v0341 | v031, v0322, v0341 | 22,474 | 188,766 | 0 | STORE_MEDIATED |
| endogenous_goal_genesis_v0353 | v031, v0322, v0353 | 7,683 | 79,052 | 0 | STORE_MEDIATED |
| recursive_autobiographical_curriculum_v0361 | v031, v0322, v0353, v0361 | 6,568 | 66,333 | 0 | STORE_MEDIATED |
| consequence_replanning_v0252 | v0252 | 336 | 53,680 | 11 | PARTLY_DRIVER_DIRECT |

"Model forwards" counts the forward calls of every submodule of a runtime-owned model instance. Driver-direct forwards come from three places:

- v0.31's teacher-forced metrics;
- v0.32.2's own evaluator, which calls its neutral model directly; its host is never called, so its self-tests are not store-mediated;
- one v0.25.2 model-metrics call.

These are mechanisms testing their own models. They are measured and reported here, not hidden.

Calls not mediated: a hosted API that calls another spec'd API internally runs that call directly inside the host's tick, because ticks cannot nest. These are counted (for example, `active_goal_from_intention` inside `update_explicit_intention`), and they are refused when the callee's slot is unbound.

## 3. Engineering results

All runs were on cloud CPU, torch 2.14.0, with 2 threads. Each claim cites a run report under `runs/dev/` and is summarized in `evidence/dev6/`.

- **G0-route self-reproducibility.** For every mechanism, two runs of the unmodified G0 route on the same seeds give byte-identical `result.json`.
- **E1, null-composition equivalence: PASS for all six.** The G0 route, full composition (all six hosts bound) and used-hosts-only composition all give byte-identical `result.json`, with equal return codes. Neither composed run was blocked or failed.
  - Faithfulness checks also pass. Every shared-model acquisition was bit-identical to a fresh load, every controller or generator instance has the canonical default configuration, and no call reached an unbound mechanism.
  - Report: `evidence/dev6/e1_report.json` (run tag e1v3; the legacy tree was unchanged in every run).
- **Focal-only composition.** The four nested-lineage routes (v0.32.2, v0.34.1, v0.35.3, v0.36.1) are BLOCKED when only their own host is bound, because they call v0.31's `rollout`. v0.31 and v0.25.2 reproduce on their own host alone.
- **E1b, store authority: PASS.** For all 13 (route, host) pairs with traffic, a store-level REPLACE of the host's response changed the frozen evaluator's outcome (18 CHANGED and 2 ABORTED_AFTER_REPLACEMENT rows). Four per-API rows were UNCHANGED, which is robustness, and one was not testable. The weakest case is v0.31's host in the v0.36.1 route, shown only by whole-stream replacement (tier 2). Details: `evidence/dev6/E1B_RESULT.md`.
- **Unit tests:** 205 pass, including 17 composition tests covering:
  - codec exactness, aliasing and refusals;
  - blob channels and the read log, and mandatory driver channels;
  - store override reaching the caller;
  - post-state identity, and views between arguments being refused;
  - nested calls into unbound mechanisms being blocked;
  - verified shared acquisition, reload probes, forward accounting, and reversible install.

## 4. What this shows and what it does not

- **It shows that the six frozen mechanisms can be hosted in one UnifiedRuntime** with their API traffic going through the canonical store (more than 45,000 store-mediated calls across the six routes). One shared instance per frozen model was verified bit-identical to what each evaluator would have loaded. The frozen evaluators reproduce G0 byte-for-byte under that wiring.
- **It does not show integration.** In every route, the persistent individual state is built by the frozen evaluator itself, freshly for each subject and each experimental arm. That state is the autobiographical memory, goal lifecycle, developmental phases, capability and curriculum. The hosted mechanisms are pure functions of that state: the frozen models are fixed, and the controller and generator hold configuration only.

## 5. Consequence for G1 (decision for David and ChatGPT)

**A retention-style G1 is predetermined by E1.** The mechanisms carry no state of their own, so with faithful wiring the composed result equals the G0 result for the same seeds. E1 shows exactly that, byte for byte. Running G1 v2 as written on six fresh DEV families would spend those families to re-observe the E1 identity. My recommendation:

1. **Treat E1 as G1's engineering gate.** Re-run it on David's WSL with the exact manifest on the proposed preflight families (after the S8 fix for v0.25.2), and freeze the result. Do not spend scientific DEV families on retention.
2. **The real composition question needs a runtime-owned individual.** One autobiography, one goal lifecycle and one set of developmental phases persist in the runtime and are shared by all mechanisms over time. That is a new integration experiment and needs its own preregistered predictions. The concrete conflicts it must settle:
   - v0.35.3 and v0.36.1 pursue goals with a hard-coded SCAFFOLD strategy (`run_subject_horizon16`, `complete_first_goal`). In one individual, v0.34.1's controller would choose the strategy.
   - The frozen checks define ground truth from each evaluator's own constructed histories, for example fragile commitment by subject parity. On a shared, accumulated history, those checks must be re-derived and preregistered, not reused as-is.
   - v0.25.2 (route world, episodic recall) shares no interface with the self lineage. Measured: zero cross-lineage calls in all six routes. Any coupling would be a new mechanism.
3. **Slot decisions still to be ratified:**
   - v0.34.1 is hosted in `skills` (strategy selector), because `curriculum` is taken by v0.36.1;
   - v0.25.2's consequence model lives inside the `planner` host, so `world_model` is unbound;
   - `reliability_from_memory` is hosted by the autobiography host.

## 6. Independent review and fixes

A separate reviewer agent audited the composition layer before hand-off. It found nine confirmed issues and several minor ones. All are fixed, with regression tests where they apply:

1. Driver-direct model computation was invisible. Fixed with forward accounting and a mediation class in every attestation.
2. The E1 and E1b verdicts were too weak:
   - E1 now requires equal return codes, clean runs and the faithfulness checks.
   - E1b is now per API. It separates INFRA from genuine aborts and validates that each control was applied.
3. Post-state application lost nested identities and kwarg results. It now merges recursively, keeps identities, and refuses views between arguments.
4. The codec was not fully fail-closed. It now refuses or wraps:
   - structured dtypes, ndarray subclasses, and legacy classes that subclass builtins;
   - non-leaf tensors, tensors carrying `.grad`, and conj/neg bits;
   - storage-sharing views and unresolvable classes.

   Every encode or decode failure is now a `CodecError`, so it is reported as BLOCKED, never as a legacy exception.
5. Overriding the constructor's `__new__` also captured `copy` and pickle reconstruction. The override was removed. Stateless instances are now registered per construction and checked against the default configuration.
6. Nested calls skipped the unbound-host check. Fixed.
7. Flows were content-equality based and recorded from the host's own encoding. They are now recorded from the bytes the driver actually read (after any store intervention), restricted to informative content, and renamed `content_flows`, with the meaning stated.
8. Install-time model loads moved the global RNG. RNG state is now saved and restored around binding loads, and the evaluator's own first load still runs, so side effects match G0.
9. Host canonical state was not checkpointed. `state_dict` and `load_state_dict` were added.

## 7. Known limits

- **Binding policy.** Hosts bind under the `TEST_FIXTURES` policy (`scientific=False`). Binding through the verified-manifest path (`LEGACY_VERIFIED`) is required before any scientific use.
- **Tensor layout.** Tensors are decoded contiguous. Storage-sharing views between arguments are refused, not reproduced.
- **Content flows** show equal informative content moving from one host's response into another host's request through the evaluator's objects. They do not show causation.
- **Environment.** All results so far come from the cloud container. Cross-environment identity with David's WSL is the next check (§9).

## 8. E1b result

See `evidence/dev6/E1B_RESULT.md`.

## 9. Reproducing on WSL

See `evidence/dev6/WSL_COMMANDS.md`.
