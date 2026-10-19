# Continuous Brain v0.37.0

This is an additive DEV experiment for your existing continuous_brain_v0.12.2
project. It follows the v0.36.1 recursive curriculum TEST PASS. It requires your
existing trained v0.31/v0.32 models and earlier project files; it does not bundle
or recreate those checkpoints.

## Install and run in WSL

```bash
cd "$HOME/projects/continuous_brain_v0.12.2"
source "$HOME/.venvs/continuous_brain_v110_dml/bin/activate"

unzip -o \
  /mnt/c/Users/davei/Downloads/continuous_brain_v0.37.0_compositional_autobiographical_goals_v1.zip \
  -d .

chmod +x run_compositional_autobiographical_goals_v1.sh
./run_compositional_autobiographical_goals_v1.sh
```

The wrapper compiles the files, runs the software fixture tests, verifies the
source files and prior result, and executes the CPU DEV experiment. It does not
run v0.36 TEST again. An optional preflight without any DEV execution is:

```bash
python evaluate_compositional_autobiographical_goals_v1.py --preflight-only
```

Each complete run prints an absolute path for result.json and result_bundle.zip
under state/compositional_autobiographical_goals_v1/runs/. Return that bundle or
paste the complete terminal output. Both scientific PASS and FAIL produce a
bundle. If preflight stops, paste that error; no subjects have run.

A source hash mismatch means the installed legacy file differs from the
recovered package. Preserve the file and report the message. Do not edit hashes,
remove the check, replace checkpoints, or rerun any spent TEST to bypass it.

## What the experiment adds

The goal becomes two ordered clauses, each specifying a context, a known A/B
primitive and a target autobiographical reliability. The author selects the
clauses from three context histories. Context transplant, erased history,
zero discrepancy, missing-clause and wrong-context controls test which
information causes the result. Checkpoint probes test continuation, not just
serialization of a goal object.

The contexts reuse the existing A/B simulator. The rule that composes the goal
is explicit Python logic. Neural models remain frozen and capability gains
remain simulated environment changes. See the protocol for exact thresholds
and the limits of the claim.

## Validation included with this release

The software fixture suite runs without PyTorch and uses no scientific subjects.
Compilation, shell syntax and archive integrity are checked during packaging.
Your trained checkpoints were unavailable in the build environment. Neither
the neural DEV experiment nor TEST has been executed there, and no v0.37
scientific PASS is claimed by this ZIP.
