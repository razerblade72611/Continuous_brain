# Architecture (v0.38.0.dev0)

Author: Claude (AI agent). This describes the code as built. The binding documents are the Architecture & Scientific Charter and the Joint Implementation Contract v0.38.0 in Linear; where this file and those differ, they win.

## Dependency direction

```
continuous_brain_protocols        (numpy only; no track knowledge)
        ^            ^            ^
        |            |            |
continuous_brain_snn |   continuous_brain_unified ---- continuous_brain_snn.snapshot (never-overwrite store)
        ^            |            ^
        |            |            |
continuous_brain_substrates       |
        ^                         |
        +------ continuous_brain_eval ------+
```

The protocols never import a track. The substrates package holds the shared encoder and both adapters; H1 details stay inside `snn_adapter.py`, and H2 details inside `dense_rnn.py`. The evaluator reaches a substrate only through `Substrate` and `FeatureSource`, and reaches the runtime only through `ReadOnlyRuntimeView` plus declared interventions.

## Substrate API (Contract §3 with C2)

| Call | SNN (H1, `snn-e0`) | DENSE-RNN (H2, `dense-rnn-gru`) |
| --- | --- | --- |
| `step(frame, dt)` | holds the frame for dt/1 ms Euler steps; the shared encoder produces currents on the sensory populations, and efference drives the motor populations | one GRU tick per 10 ms frame, on the same encoded vector |
| `features()` | per-neuron spike trains filtered at τ = 25 ms (the E0 trace bank) | hidden state |
| `plasticity_step` | `CapabilityUnavailable` (E1, DAV-40) | `CapabilityUnavailable` |
| `train_step` | always `GradientFirewallError`; every tensor is checked for `requires_grad` | TBPTT on the causal window: predict encoded sensory input one frame ahead; single-exposure ledger |
| `freeze` | homeostatic thresholds held exactly | training refused |
| `lesion` | `synapses:<region>`, `neurons:<region>` (neurons are also silenced) | `units:hidden`, `weights:recurrent` (masks re-applied after every update) |
| `budget` | active synapses | parameters; the hidden size is solved to ±10 % of the SNN's active synapses |
| `state_dict` / `state_hash` | exact restore; continuation is bit-identical (tested) | exact restore, including Adam moments (deep-copied: the optimizer returns live references) |

**Question for review (not changed):** matching the GRU's *total* parameter count to the SNN's active synapses, with the shared 2,182-channel encoder, leaves a 42-unit hidden layer at full size. Most of the budget is input and output weights. That is the literal A1.1 rule. An alternative would be to match *recurrent* capacity instead: count only the SNN's cortex→cortex synapses against the GRU's recurrent weights. That is a spec question for A1, so the code implements A1 as written.

## Unified runtime (Contract §1, C1/C3/C4)

- **Slots.** 12 mechanism slots and 3 infrastructure slots (`continuous_brain_unified/slots.py`). Each mechanism slot names canonical channels (for example `autobiography.record`), so interventions and evaluators can target a function without knowing its implementation.
- **Binding policy.** `VERIFIED_ONLY` (the default) accepts:
  - `LEGACY_VERIFIED` components: a VERIFIED manifest entry, re-hashed under the legacy root, plus a registered step contract for that exact module SHA-256;
  - `EXPERIMENTAL_PREREG` components: registered under an APPROVED preregistration that covers the slot.

  `TEST_FIXTURES` also accepts test components, and marks the runtime `scientific = False` everywhere it reports. Unbound slots are reported with their reason; nothing stands in for them.
- **Channels.** Each channel has a single writer, and mechanism channels are namespaced by slot. Values are read-only copies. There is a start-of-tick view for recurrent (delayed) edges, and a write log records a SHA-256 for every value.
- **Scheduler.** The runtime owns the clock (fixed dt, integer ticks). Each tick runs a DAG with deterministic tie-breaking. A same-tick cycle is an error; recurrence has to be expressed as a delayed read. `bind_many` binds a mutually dependent set atomically.
- **Interventions.** ERASE, FREEZE, DONOR and REPLACE on declared channels. They are applied immediately after the writer runs, so downstream readers see the intervened value in the same tick. They are locked at the first tick, and every application is recorded.
- **Coprocessor.** It is disabled at boot and `NullCoprocessor` is the default. It has no handle on the channel store, so responses go only to the requesting component. Calls are capped per tick, every request and response is logged with hashes, and `ablate()` swaps in the null provider for G5.
- **Persistence.** `save`/`restore` go through the never-overwrite `SnapshotStore`. A restore refuses if the configuration, binding, channels or intervention schedule differ, and checks the recorded state hash.
- **Legacy.** `legacy/manifest.json` holds VERIFIED entries only. The root comes from the CLI or an environment variable and is never recorded. Hashes are verified before import. The loader refuses to import if a different `continuous_brain` package is already loaded. A developer override is logged and makes the runtime non-scientific. `inventory.py` uses only the standard library, runs read-only, and refuses to write inside the root.
- **Workspace / attention schema.** Abstract interfaces plus `register_experimental(cls, prereg_id)`. A test checks that the package ships no implementation (C4).

## Evaluation (Contract §5–§9)

- **Battery.** B1 is BLOCKED (it needs the E2 body). B2–B6 run on both substrates through the track-neutral driver, which dispatches learning by capability. Streams come from `streams.py`, a two-joint synthetic plant with open-loop babbling, a visible body blob, cue flashes and rhythmic flashes; it is not the E2 body. Results are `ENGINEERING_ONLY`, or `CAPABILITY_ABSENT` where the SNN has no learning rule (B2, B6).
- **Probes.** These follow the E2 prereg: a ridge probe with λ chosen by blocked 5-fold CV on the training block, a train/test split into contiguous blocks with a gap of at least horizon + history, and z-scoring on training statistics with a scale floor (without it, near-silent spike-trace columns made test predictions blow up by ~10⁹). The RAW-INPUT baseline uses 20 lags. Feature subsets follow one rule for every network: all features up to 2,000, otherwise a seeded random 2,000.
- **Gates.**
  - G1 refuses any seed family that overlaps a registered range.
  - G2's intervention × metric matrix must be frozen (hashed) before it runs.
  - G3 schedules are data; a schedule whose slots are unbound is BLOCKED.
  - G4 checks save/restore through the snapshot store and bit-exactness on exact-contract channels, and runs a drift assay.
  - G5 is reported but never gated.
  - Every gate checks its seeds against the registry.
  - Gates only *propose* a verdict. `RunStore.finalize` turns it into PASS/FAIL after it has attached the run's primary artifacts and the preregistration registry has approved the prereg for that item.
- **Decision rules** are versioned objects. Each result records the rule id and the SHA-256 of the rule's source.
- **Runs** are content-addressed: an identical spec is refused unless it carries a replicate label. That makes an accidental reopening of a sealed configuration an error.
