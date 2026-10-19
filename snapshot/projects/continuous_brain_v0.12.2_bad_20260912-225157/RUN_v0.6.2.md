# Run Continuous Brain v0.6.2

The new motor is trained separately. Your saved 32.1M-parameter executive at
decision 3,800 is not used or modified during motor acquisition. Your successful
v0.6.1 backup is the migration source below. Keep all earlier processes stopped.

## 1. Extract side by side; keep the working virtual environment

Download continuous_brain_v0.6.2.zip into Windows Downloads. With the same working
`.venv` active in WSL:

```bash
mkdir -p ~/projects/continuous_brain_v0.6.2
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.6.2.zip \
  -d ~/projects/continuous_brain_v0.6.2
cd ~/projects/continuous_brain_v0.6.2/continuous_brain_project
python -m pytest -q
```

Do not replace PyTorch, move the venv, rebuild Unity or install new dependencies.
These commands use the bundled source; changing an editable package installation
is unnecessary. Tests include full-size preservation checks and motor training,
so allow a few minutes and several GB of temporary disk/RAM. Keep state on WSL's
Linux filesystem, not `/mnt/c` or a network filesystem. Stop if a command fails.

## 2. Train the new motor

```bash
python -m continuous_brain.motor_cli train \
  --state-dir state/motor_dev_v062 --updates 2000
```

This initializes a separate 5,060-parameter motor and collects teacher-labelled
observations from 256 development-room curriculum cases (seed 7). Class-balanced
supervised learning fits the local steering policy. Teacher data and learning are
explicit, not counted as autonomous success. The motor uses six local sensor values
and an orient/navigate mode; it receives no world coordinates or rendered pixels.
The executive still uses its normal visual features, unchanged.

The trainer runs on CPU by default and needs no GPU setup. `--updates` is additional,
not a lifetime stopping count. Re-running preserves its weights, optimizer, dataset
and sampling RNG and performs more updates. Ctrl+C/SIGTERM saves; periodic snapshots
are written every 250 updates. If stopped before initial data collection completes,
there are no learned updates to lose; the seeded data can be regenerated.

The lab files are `motor_training.pt`, its previous checkpoint, `motor.lock`, and
later an evaluation report/qualified snapshot. Keep this directory separate from
individual and legacy state. Only load your own trusted checkpoints (PyTorch uses
pickle here). Do not run two commands against the same motor lab concurrently.

## 3. Test without teacher actions

```bash
python -m continuous_brain.motor_cli evaluate \
  --state-dir state/motor_dev_v062
```

This runs the learned motor for 100 orientation and 100 short-range navigation
episodes under a distinct, fixed evaluation seed. It uses no reference motor or
teacher corrections. It does not train or change the motor training checkpoint;
it writes `motor_evaluation.json` and, on success, `qualified_motor.pt`.

Continue only when the report says `"passed": true`. The attachment gate requires
at least 95% success for each skill, at least 100 episodes per skill, no infrastructure
faults and actual forward translations. Qualification is tied to the exact learned
weights. More training invalidates the old qualification for attachment until you
evaluate again. A failed evaluation exits nonzero; do not bypass or edit the report.

The test is intentionally narrow: visible goals and clear direct corridors, not
arbitrary obstacle avoidance, exploration, a new simulator or physical robotics.
Its fixed held-out seed is a regression/qualification set, not an untouched
scientific benchmark after repeated human-guided tuning. Report failures instead
of repeatedly training to the test without examining what failed.

## 4. Attach it while preserving the current executive

Use the backup you just created. The destination must be new:

```bash
python scripts/upgrade_individual_v062.py \
  "$HOME/projects/continuous_brain_v0.6.1/continuous_brain_project/state/individual_v061_before_motor_training_01" \
  state/individual_v062 \
  --motor-state state/motor_dev_v062
```

Expected report: same UUID `1f58521f-b334-4fde-898f-7ada266b378c`, decisions 3,800,
memories 3,800 and updates 948, assuming that backup is the one shown in your log.
Expect `motor_source: learned_development_motor`.

This preserves the executive/target weights, optimizer, recurrent workspace,
memory rows, replay, intentions, counters, physical state and RNG. The previous
motor's exact tensors/hidden state are archived under `motor_history/`; the full
legacy history stays in `legacy_v053/`. Both source individuals/backups remain
unchanged. No personal history is deleted or relabelled as new experience.

The copy is the same individual continued, not a new independent identity. Do not
run source and destination concurrently. Use the untouched source for rollback;
older executables cannot read the new v0.6.2 checkpoint. If a partial destination
has `UPGRADE_INCOMPLETE`, do not remove that marker to bypass the check. Preserve
the error and use the untouched source. Do not overwrite a previous destination.

## 5. Check the new motor beneath a fixed planner

```bash
python -m continuous_brain.individual_cli --profile standard \
  --state-dir state/individual_v062 --decisions 200 \
  --evaluate --planner-diagnostic --fast
```

This replaces the long inline diagnostic from v0.6.1. It is deliberately
planner-guided and changes no saved state. The output includes `run_window` and
`primitive_actions_by_skill`. Look for real `navigate/advance` actions, completed
tasks and no faults. Search's forward moves must not be mistaken for motor moves.

This broader room probe is distinct from the clear-corridor motor qualification.
Some room tasks still fail because route selection/obstacle handling remain limited.
If it still produces no navigation advances or no successes, stop and share the
report instead of launching long executive training.

## 6. Continue the executive with the learned motor

After the diagnostic shows useful movement:

```bash
PROFILE=standard STATE_DIR=state/individual_v062 \
  bash scripts/run_individual_v062.sh --decisions 200
```

Expect the same UUID and 32,114,036 trainable executive parameters, plus a separate
5,060-parameter learned motor. The motor snapshot is held fixed during this stage
so the executive's world model sees a stable controller; it is trainable through
the lab, not updated automatically in this run. There is no hidden reference-motor
fallback. Orient/navigate use neural action selection; search, observe, rest and
interaction remain explicitly authored. The executive planner teacher still follows
its existing decaying schedule. These are not claims of autonomous general intelligence.

`active_replay` starts at zero after attachment. Historical replay/memory/optimizer
state remain intact; new-controller examples populate the active training set.
The old finite replay capacity still applies. `controller_updates` counts new
executive updates since attachment; learned-world action reranking waits for 200
of these instead of trusting the old motor's 948-update count. Old model weights
are preserved, not reset. This warmup concerns learned consequence reranking only.

`run_window`/`run_tasks` report results since this process started; lifetime totals
still include earlier controllers. Keep these separate when judging improvement.

## 7. Evaluate the executive and continue

After training stops, remove the planner diagnostic flag to evaluate learned
executive selection without planner forcing, exploration, learning or saved changes:

```bash
python -m continuous_brain.individual_cli --profile standard \
  --state-dir state/individual_v062 --decisions 200 --evaluate --fast
```

This still uses the authored skill definitions/search/controller supervisor. It
is a learned-executive evaluation, not an all-components-learned benchmark.

For more training, repeat the bounded run or omit `--decisions` to run until Ctrl+C.
Prefer reviewing the first bounded run before committing substantial compute.

## Later motor training

More motor updates use the same train command and preserve its optimizer/data.
They do not alter an attached snapshot. Evaluate again, then use the upgrade tool
with your LATEST stopped individual as source and a new destination. Previous motor
snapshots remain archived. This is explicit staged training, not joint online motor/
executive adaptation. The initial demonstration set is fixed, not lifelong replay
collected from the individual's subsequent movements.

## What to send back

The motor evaluation report, upgrade report, planner diagnostic `run_window` and
primitive counts, then the bounded executive training summary. No large database
upload is needed. Native Unity/ROCm/physical robot behavior was not tested here.
