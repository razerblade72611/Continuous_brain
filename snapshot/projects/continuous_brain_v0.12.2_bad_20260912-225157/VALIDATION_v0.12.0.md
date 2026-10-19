# Release validation — v0.12.0

Build: `v120-local-development-r1`, 10 September 2026.

This release supplies a runnable developmental experiment and the retained
v0.11.4 source and diagnostics. The included 667,130-parameter starter is a new
model. The original trained 53.68M child was not available for this build, and
no result here establishes recovery of its skills or scores.

## Automated checks

**56 targeted tests passed** under Python 3.12.14, PyTorch 2.4.1+cpu, and
NumPy 1.26.4. The run completed with exit code 0. Fourteen warnings came from
the legacy Transformer nested-tensor configuration. No test failures remained.
The compact machine-readable receipt is `validation/test_results.json`.

The selected suites cover the local runtime, original growth integration,
checkpoint integrity, world retention, and the v0.11.1–v0.11.3 device fixes:

```text
tests/test_v120_local.py
tests/test_v110.py
tests/test_v111.py
tests/test_v112.py
tests/test_v113.py
tests/test_v114.py
tests/test_v114_integrity.py
tests/test_v104.py
```

Material checks include:

- A candidate learns while the reference model's tensors remain exact.
- A 38-to-24 success regression is rejected by the relative retention rule.
- A 35-step uninterrupted run matches a 19-step run resumed for 16 steps,
  including model tensors, forecaster state, recurrent context, and lived
  events across two candidate learning blocks.
- Database transactions reject an invalid summary without committing a
  partial state. A backup remains consistent and a second writer is rejected.
- Save acknowledgement requires reopening and verifying the completed
  database checkpoint; a stale 224-step commit cannot acknowledge a 256-step
  session. The writer lock remains held throughout this check.
- World state and random state resume correctly; copied observations cannot
  directly change the simulated body's true resource state.
- Legacy gate crossover copies change exactly the two designated gate
  buffers and preserve every other tensor and the source checkpoint.
- A bounded paired legacy trace starts both controllers from the same state;
  the reserved blind seeds are rejected by this diagnostic interface.
- Read-only dashboard HTTP responses and rejection of write requests work.

The CPU device preflight also passed a matrix operation, model
forward/backward/Adam update, and both demonstration and lived-replay training
routes. Its report is `validation/doctor_cpu.json`. Requesting DirectML in
this CPU-only environment stopped with an error, as intended; it did not
silently fall back to CPU.

## Included starter

Tensor fingerprint:

```text
34d1b8baede51987f8e9204bd05039b065f075ad1ca5783b136473a925ed95a5
```

The starter received 2,800 CPU training updates: 1,200 initial demonstration
updates, then two rounds of 800 updates using additional teacher-labelled
training trajectories. The authored teacher is used for training only. It
does not choose actions in the default live session or in evaluation.
Training used PyTorch 2.4.1+cpu and NumPy 2.3.5; compatibility and release tests
subsequently passed with the fresh-install NumPy 1.26.4 pin.

A subsequent frozen-model evaluation used 100 new seeds in each family. No
gradients or model selection used these 200 episodes. Success means collecting
at least two resources without depletion during a 64-step episode.

| Resource-world family | Seeds | Successes | Mean collisions |
|---|---|---:|---:|
| Open | 800001–800100 | 67/100 | 0.17 |
| Detour | 810001–810100 | 47/100 | 2.49 |

Complete per-episode evidence is in `assets/starter_validation.json`. These
are new layouts from the same authored resource-world generator. They do not
establish transfer to another environment, general intelligence, or performance
on the original P4b task. They are not directly comparable to the old child's
38/100 or 61/100 results.

## Complete learning session

One CPU session with seed 2026 completed 1,024 lived steps and four candidate
learning blocks of 16 updates each. It collected 21 food and 27 water resources.
Three candidates were rejected and one was admitted. The admitted policy
completed 5/6 open and 6/6 detour episodes on the small, reused development
suite. Those development scores are selection evidence, not a new blind test.

The development forecaster had four examples. Its mean absolute error was
0.228, compared with 0.059 for predicting no change. It therefore has **not**
demonstrated useful forecasting. It remains advisory and does not authorize
promotion. The compact session evidence is `validation/full_session_summary.json`.

The 95% retention setting is a finite development-suite admission rule, not a
95% statistical confidence guarantee. Repeated reuse of its small suite can
overfit it. Broader held-out evaluation is needed before treating repeated
development gains as reliable generalization.

## Extracted launcher and interruption check

The archive was extracted into a folder whose path contains spaces. The actual
shell launcher reused the test environment, verified the manifest, loaded the
packaged starter, and passed its CPU training preflight. Initial 16-step and
resumed 16-step sessions preserved the same identity.

An earlier candidate build produced one inconsistent interruption result: its
report said step 256 was saved, but the next process loaded step 224. That
result did not reproduce consistently. The release now requires a completed
WAL checkpoint and a closed-database readback before acknowledging a save;
automated tests also reject a stale committed state explicitly.

The final extracted build was interrupted by sending SIGINT to both the shell
launcher's Python process and its runtime during a normal 16-update learning
block. The block finished, the process exited successfully, and both the
session report and reopened database recorded step 256. A separate launcher
invocation resumed to step 264, retaining the identity and learning block.
Its 264 lived-event records contained exactly 264 distinct step numbers.
The receipt is `validation/extracted_launcher.json`.

## Platform and interface limits

- Actual DirectML or RX 6600 hardware was not available here. Published
  DirectML wheel dependencies were checked, and the package uses the legacy
  DirectML-compatible recurrent implementation. The startup preflight on your
  machine remains the required hardware check.
- Native Windows batch launchers and driver installation were not executed
  in this Linux environment. The batch files use Windows line endings and
  quoted paths. The WSL/Linux shell launcher passed syntax checking.
- Dashboard HTTP behavior and JavaScript rendering logic were checked using
  real session data. Browser layout was not visually verified because a
  browser executable could not be downloaded in this environment.
- Positive and negative signals are simulated learning feedback. Persistent
  identity and history-dependent behavior do not establish felt joy, pain,
  subjective experience, or consciousness.

For startup instructions, see `START_HERE.md`. For using your original frozen
checkpoints, see `LEGACY_CHILD.md`.
