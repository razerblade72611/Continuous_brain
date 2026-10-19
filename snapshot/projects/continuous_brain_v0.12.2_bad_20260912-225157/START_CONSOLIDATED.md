# Start the consolidated project

Build `v120-consultation-consolidated-r1`.

This is the complete supplied v0.12.0 DirectML release, which already includes
the v0.11.4 implementation, plus the useful consultation evaluation work. The
original model, training, simulator, device, checkpoint, and storage code is
preserved. There is no automatic merge of differently shaped neural models.

## Launch the included local learner

Extract the ZIP into a new folder. Use 64-bit Python 3.10, 3.11, or 3.12.
From the extracted `continuous_brain_consolidated` directory:

| Platform | Start |
|---|---|
| Native Windows / DirectML | Double-click `START_DML.cmd` |
| WSL with DirectML | `bash run_local.sh` |
| CPU diagnostic session | `python setup_local.py --cpu run --steps 256 --no-dashboard` |

The existing launcher checks source hashes and dependencies, creates a private
environment when needed, performs a real model/optimizer preflight, then runs
the learner. Initial dependency installation needs internet; subsequent local
simulator sessions use the installed environment. No remote teacher is needed
for this default runtime. Installation failures are reported; CPU is selected
explicitly, not substituted silently for your GPU.

The default session is bounded to 1,024 lived steps. A local dashboard reports
activity, learning, and checkpoint status. Press Ctrl+C once to request a clean
stop and wait for the save acknowledgement. The runtime handles the request at
its existing step/learning-block boundary. No fixed millisecond response bound
is claimed. See `START_HERE.md` for dashboard controls and recovery details.

Relaunch to continue the saved individual. Default state is outside this source
folder: `~/ContinuousBrain/individual_v120` on WSL/Linux or
`%USERPROFILE%\ContinuousBrain\individual_v120` on native Windows. Supply
`--state-dir PATH` for a separate individual. Existing state formats and identity
checks are unchanged. Use a new state path for experiments you do not want to
add to your current individual's experience.

The included trained starter has 667,130 parameters. It starts a **new v0.12
individual**, or the runtime resumes an existing v0.12 state. It is not your
older 53.68M child. That child's source code is retained, but its trained
checkpoint and nursery data were not in the available source archive. Use your
existing old checkpoint for the legacy evaluation below; do not pass it to the
small learner's state loader. See `LEGACY_CHILD.md`.

## What the consultation contributed

The useful progress is an auditable evaluation procedure and a reproducible
mock-grid baseline. The main project already had learned options and imagined
planning. The consultation has not demonstrated a new intelligence gain in
that project, or supplied a tested recipe for consciousness.

This bundle integrates its generalization evaluator against the real project
APIs, with additional corrections:

- An exact-checkpoint fixed-layout replication is required before randomized
  evaluation. Protocol, thresholds, and a real Boolean pass flag are checked.
- Reactive, direct learned options, and hierarchical planning run from the
  same environment seeds. The direct-options arm separates option behavior
  from the extra contribution of imagined planning.
- Paired bootstrap intervals, benchmark thresholds, primary completion
  evidence, and secondary reward/collision outcomes remain separate fields.
- Reports record source and checkpoint hashes, runtime, settings, every
  episode, layout, observations, actions, and planner decisions. Diagnosis
  identifies failures but does not automatically prescribe more training.
- Original held-out seeds 51001–51100 remain protected. Diagnostic runs default
  to 41001–41100. Confirmation ranges are reserved separately below.

The confidence/risk experiments are not installed into the agent's decision
path. The corrected handwritten foraging experiment is runnable separately in
`consultation/reference_lab/`. Original consultation deliveries are retained in
`consultation/provenance/` for provenance, not as current installation guides.
Negative results and mock-grid effects are not claims about the real child.

## Run the verification and evaluation

Run these commands using the same compatible Python environment as the learner.
If setup created a private environment, use `.venv_dml/bin/python` in WSL,
`.venv_dml\Scripts\python.exe` on Windows, or the corresponding `.venv_cpu` path.

First run a real PyTorch integration smoke. It needs no trained legacy child:

```bash
python scripts/run_consultation.py smoke --device cpu --output results/smoke.json
```

Use `--device dml` to test DirectML on your machine. The smoke constructs an
untrained proof-profile model, runs the real simulators and all three controller
paths, invokes the old evaluator, and checks that its unqualified result is
rejected by the new API and CLI. Passing means the code executes, not that the
model has learned planning. No fake PyTorch or no-op optimizer is used.

For the actual older child, use its existing checkpoint:

```bash
python scripts/run_consultation.py evaluate --checkpoint /path/to/infant_training.pt --device cpu --output-dir results/diagnostic
```

Use a quoted Windows path if needed. The driver first runs the existing fixed
P4b protocol: 100 paired episodes, 120 steps, spent seeds 12001–12100. It stops
if that exact checkpoint does not pass. If it passes, both randomized-detour
and two-barrier comparisons run: 100 paired episodes, 160 steps, diagnostic
seeds 41001–41100. The inherited benchmark is completion at least 20%, advantage
over reactive greater than 5 percentage points, collision regression no more
than 2 per episode. A threshold pass is distinct from a positive paired
completion interval. The direct-options comparison is an additional diagnostic.

Inspect the reports before changing training. When the implementation,
checkpoint, and diagnostic choices are frozen, confirmation is an explicit run:

```bash
python scripts/run_consultation.py evaluate --checkpoint /path/to/frozen_checkpoint.pt --device cpu --role confirmation --output-dir results/confirmation
```

This uses 61001–61100 for randomized detour and 71001–71100 for two barriers.
Confirmation enforces fixed counts and planner settings. Do not tune on those
outcomes and call a rerun fresh evidence. Code can protect reserved ranges; it
cannot verify that a person has never inspected an outcome elsewhere.
Each environment's completion interval is reported separately, not as an
omnibus claim across environments or secondary metrics.

The integrated low-level command is also available after installation:

```bash
python -m continuous_brain.parented_cli evaluate-generalization-gate --checkpoint /path/to/infant_training.pt --replication-report results/diagnostic/replication.json --kind randomized_detour --output results/custom_diagnostic.json
```

The report check establishes protocol and file consistency, not authenticity
of arbitrary editable JSON. Prefer the driver, which executes the replication
with the actual project evaluator. The evaluator is read-only with respect to
the source checkpoint and performs no training. Ctrl+C interrupts evaluation;
completed matching reports may be reused, but an unfinished evaluation restarts
from its first episode. This is different from the local learner's persistent
session restoration; no mid-episode evaluator resumption is promised.

## Reference reproduction and compute

```bash
python scripts/run_consultation.py reference
```

This needs only NumPy and runs the corrected collection/respawn test plus the
two previously reported 200-episode seed sets. These are reproductions, not
newly reserved confirmation data. Controllers and simulator are handwritten.

The real proof-model smoke takes seconds on the validation CPU. A complete
legacy sequence can execute up to 132,000 environment steps (three controller
arms in the fixed gate and both randomized tasks), plus imagined rollouts.
Actual time depends on checkpoint size, termination frequency, and backend;
timing a 10.46M random proof model cannot predict your trained 53.68M child's
speed reliably. Evaluation uses no training optimizer state. The 667k local
starter is much smaller and runs on CPU; DirectML/RX 6600 execution remains
to be verified on your hardware. Exact measurements are in the validation report.

For developer regression checks, install `pytest==8.3.5` in the same environment:

```bash
python -m pytest -q tests/test_consultation_integration.py tests/test_v103.py tests/test_v114_integrity.py tests/test_v120_local.py
```

To rebuild after deliberate source edits, use a new output filename:

```bash
python scripts/build_consultation_release.py --output ../my_consolidated_build.zip
```

This regenerates the source inventory and verifies the resulting ZIP. Do not
rebuild merely to bypass an unexpected integrity failure; restore the intended
source first. The existing README, architecture, installation, and lineage
documents remain available unchanged apart from the new README introduction.
