# Continuous Brain v0.6.1 — Validation

Date: 2026-09-04. Source-only release. Experimental checkpoints are not bundled.

## Executed checks

- Full pytest suite: **118 passed in 60.66 seconds**. Includes the 94 prior tests
  and 24 additional/parameterized v0.6.1 cases.
- Python compileall for package and scripts: pass.
- Bash syntax for new and retained v0.6 launchers: pass.
- Upgrade and individual command-line help: pass.
- Reported v0.4.3 format-6 metadata and exact 64/128/48 motor tensor dimensions
  accepted using synthetic weights. Exact imported weights remain frozen after
  execution; recorded mastery [0, 1] and prior 23,941 steps/1,803 goals preserved.
- Wrong/missing legacy schema, incorrect version/format, different vision,
  nonfinite weights and shape mismatch rejected before creating upgrade destination.
- Copy-on-upgrade tested with and without attachment. Tested both tiny and full
  standard profiles. Executive/target tensors, optimizer, replay, workspace,
  intentions, environment, RNG, original experience rows and identity preserved.
  Original individual database and legacy checkpoint remain byte-identical.
- After upgrade, stale-control replay remains present but inactive. Eight new
  decisions resume training; retained optimizer state is not reset.
- Running source, existing destination, nested paths, existing motor replacement,
  incomplete destination and injected archive I/O failure handled without
  overwriting either original. Interrupted destination cannot run.
- Search moves after a full unsuccessful scan; rejects translation when given
  unknown depth, blocked clearance or failure acknowledgement. Reacquires an
  occluded target in a targeted development-room regression case.
- Search-state restart and split/uninterrupted movement/training produce identical
  final tensors, replay, workspace, physical state and reward on CPU.
- Old protected replay cannot block collection of new-controller formative samples.
  Replay sampling does not bridge controller revisions or sensor-fault gaps.
- Per-experience authored search provenance is not mislabeled as neural motor output.
- Existing read-only evaluation, process locking, SIGTERM save, fault quarantine,
  memory retrieval, dashboard HTTP and mock THOR integration tests still pass.

## Short source-execution experiments

| Run | Result |
|---|---|
| Standard CPU smoke, reference motor | 32,114,036 parameters; 8 decisions; 30 ticks; 1 optimizer update; 8 memories; 0 faults |
| Tiny CPU, seed 7, reference motor, assisted training | 300 decisions; 1,226 ticks; 74 updates; 300 memories; 40/44 tasks successful; 0 faults |
| Evaluation from that tiny checkpoint | 200 decisions; 21/24 evaluation tasks successful; executive teacher/exploration off; 74 updates and 300 stored memories unchanged |

Training teacher probability at decision 300 was 0.946. Evaluation retained the
analytical reference motor and authored skill controllers. These single-seed,
small-sample results are not a general-intelligence benchmark, a certification of
90–95% reliability, or evidence about the user's actual legacy motor. Environment
trajectories/task samples differ from v0.6.0 after search changes, so this is not a
controlled performance comparison with that release.

## Environment and unverified areas

Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.5.2 and pytest 9.1.1 in isolated test
dependencies. The user's installed PyTorch/ROCm environment was not changed.

Not tested: the user's actual trained motor tensors; native custom Unity executable;
RX 6600/ROCm execution; physical robots; multi-day operation; generalization across
seeds/environments; lifetime retention; autonomous skill discovery or consciousness.

No dependency installation was attempted for this patch. The earlier optional
package-install verification was blocked by an approval cancellation and was not
retried. Bundled-source module and script commands were executed successfully.

## Packaging

The release builder generates one fresh SHA-256 manifest, checks archive CRC and
verifies every manifested file digest. State, trained weights, caches, bytecode,
virtual environments and build output are excluded. Historical v0.6.0 documents
remain available; RUN_v0.6.1.md contains the current instructions.
