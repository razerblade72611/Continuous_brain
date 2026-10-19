# Continuous Brain v0.37.1 diagnostic

This additive package investigates the formal v0.37.0 DEV FAIL you reported.
It requires the installed v0.37.0 package and the original run directory.
It preserves all earlier code, checkpoints and results. It performs no training,
no new admission test, and no TEST evaluation.

## Run in your existing WSL environment

```bash
cd "$HOME/projects/continuous_brain_v0.12.2"
source "$HOME/.venvs/continuous_brain_v110_dml/bin/activate"

unzip -o \
  /mnt/c/Users/davei/Downloads/continuous_brain_v0.37.1_compositional_goal_completion_diagnostic_v1.zip \
  -d .

chmod +x run_compositional_goal_completion_diagnostic_v1.sh

./run_compositional_goal_completion_diagnostic_v1.sh \
  --source-run state/compositional_autobiographical_goals_v1/runs/20260920T055442Z_8b224fc43c
```

The wrapper runs software fixtures, audits the saved data, and replays only the
original bootstrap/interruption boundary for the spent DEV subjects. It then
prints the path to a new result_bundle.zip. Return that bundle or paste the
complete terminal output.

The source run above is also the default when no --source-run is supplied. The
program never selects a newer run automatically. If that directory is missing,
restore it or report the error; do not rerun the old experiment just to create it.

For a records-only audit without model inference, add --records-only. This mode
cannot confirm the interruption hypothesis and labels it a source-path
inference. The full replay command above is the intended next step.

All previous files are checked against their original hashes. If preflight
reports a mismatch, preserve the file and paste the error. Do not edit hashes
or bypass the checks.

## What the diagnostic is testing

The first evaluator set targets before an unrelated interruption. That event
updated the joint self-estimate and displaced an older record from its 12-record
window. The new package checks whether the supposedly missing target already
looked complete immediately afterward, and records what happens when the oldest
record is removed separately from adding the interruption.

v0.37.0 remains DEV FAIL regardless of the diagnostic findings. No repaired
mechanism is included. The next repair should follow the measured result.

## Validation limits

Local software tests use deterministic fixtures, including an apparent target
completion caused solely by an unrelated-history change. Python compilation,
shell syntax and ZIP integrity are checked. The build environment has neither
your trained checkpoints nor your scientific run bundle, so it has not executed
the actual neural replay or determined the subject-specific diagnosis.
