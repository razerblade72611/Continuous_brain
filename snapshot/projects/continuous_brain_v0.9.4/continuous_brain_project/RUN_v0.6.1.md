# Run Continuous Brain v0.6.1

This update does not require starting your executive over. The older motor is
optional: preserve it as an asset and measure whether it transfers successfully.
The reference motor remains available for development without that dependency.

## 1. Stop and install side by side

Press Ctrl+C once in any running training process and wait for its final report
and shell prompt. Stop both v0.6 and legacy THOR processes before copying state.
Download continuous_brain_v0.6.1.zip to Windows Downloads. Keep the working `.venv`
you are already using active; do not copy it or replace its PyTorch installation.

```bash
mkdir -p ~/projects/continuous_brain_v0.6.1
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.6.1.zip \
  -d ~/projects/continuous_brain_v0.6.1
cd ~/projects/continuous_brain_v0.6.1/continuous_brain_project
python -m pytest -q
```

The provided scripts run the bundled source from this directory, so reinstalling
dependencies or changing the editable package installation is unnecessary. If you
want the global `continuous-individual` entry point to point here as well, an
optional `python -m pip install -e . --no-deps` does that. That packaging route was
not re-tested in this release; the source/module commands below were tested.

## 2. Continue your existing 32.1M executive and attach the motor

Run from the new project directory, with the old processes stopped:

```bash
python scripts/upgrade_individual_v061.py \
  "$HOME/projects/continuous_brain_v0.6.0/continuous_brain_project/state/individual_inherited_v060" \
  state/individual_v061 \
  --legacy-motor \
  "$HOME/projects/continuous_brain_v0.4.3_preserved/state/embodied_brain_v043_clean"
```

This creates a new destination and leaves both source directories untouched. It
preserves the current UUID, executive weights, recurrent workspace, optimizer,
target network, replay, memories, intentions, counters, environment and RNG. It
adds the exact frozen motor weights, a fresh motor hidden state and explicit
lineage. It does not inject legacy latent memories into the new executive.

The reported legacy v0.4.3 checkpoint is format 6, 23,941 steps, 1,803 goals,
recorded mastery [0, 1], with a 64-embedding/128-GRU motor and pixel-stats 48 input.
Those recorded achievements are provenance, not proof of transfer in the new room.
Only use trusted checkpoints you created: PyTorch checkpoint loading uses pickle.

The destination must not already exist. If an upgrade fails, preserve any directory
marked `UPGRADE_INCOMPLETE`, keep using the untouched original, and share the error.
Do not remove the marker to bypass the check. Use a fresh destination when retrying.
Keep several GB of disk free for standard checkpoints, SQLite journals and copies.

An upgrade copy is a continuation, not a new individual. Do not run it and its source
concurrently. The old source remains an exact pre-upgrade fallback. v0.6.0 cannot
read a checkpoint saved as v0.6.1; use that untouched source for rollback.

### Without importing the old motor

To preserve the current executive but continue with the reference controller,
omit the `--legacy-motor` argument. Choose this variant instead of the preceding
command; do not run both into the same destination:

```bash
python scripts/upgrade_individual_v061.py \
  "$HOME/projects/continuous_brain_v0.6.0/continuous_brain_project/state/individual_inherited_v060" \
  state/individual_v061
```

No old data needs to be deleted. A genuinely new individual is also possible by
running the next command with a new, empty state path, but is not necessary here.

## 3. Run a bounded transfer check

```bash
PROFILE=standard STATE_DIR=state/individual_v061 \
  bash scripts/run_individual_v061.sh --decisions 200
```

Expect version 0.6.1, 32,114,036 executive parameters, the SAME current individual
UUID (1f58521f-b334-4fde-898f-7ada266b378c in your supplied log), and decisions
continuing from wherever you stopped. With attachment, expect `motor=imported_learned`.
Without it, expect `motor=development_reference_teacher`. Do not change the profile
of an existing state directory. The imported motor's frozen parameters are separate
from the reported trainable executive count.

Search now scans then relocates using depth clearance and action feedback. Its
progress survives the four-tick option limit and restart. It still has no omniscient
target coordinates, and it is an authored controller, not a newly learned skill.
Its heuristics are not a certified safety controller for physical hardware.

`active_replay` initially starts at zero after a control revision. Old episodic
memory, replay and optimizer state are preserved; obsolete-control replay is not
sampled for current dynamics/value training. At least six new contiguous healthy
decisions are needed before a new training batch. This is not a learning reset.
Old replay entries still age out of the existing bounded replay buffer normally;
their lifetime experience rows remain in SQLite and the source backup remains intact.

Legacy motor transfer is an experiment: the development room's images and dynamics
differ from THOR. Search/observe/rest/interact remain authored even when the orient/
navigate motor is neural. Per-experience provenance now records that distinction.
Cumulative skill statistics include pre-upgrade history; they are not a fresh score
for the new controller.

## 4. Evaluate and then continue

After the bounded run exits, evaluate without executive teacher forcing, exploration,
learning or saved changes:

```bash
python -m continuous_brain.individual_cli --profile standard \
  --state-dir state/individual_v061 --decisions 200 --evaluate --fast
```

Read `evaluation_window` for new evaluation tasks/successes, not lifetime totals.
This removes the executive teacher, not authored skills or a reference motor. It
is not a benchmark of general intelligence. A low score at this stage does not
by itself distinguish executive training needs from unsuccessful motor transfer.

Resume with another bounded run, or run until Ctrl+C:

```bash
PROFILE=standard STATE_DIR=state/individual_v061 \
  bash scripts/run_individual_v061.sh
```

No `--import-v053-state` flag is needed after attachment. That older runtime flag
remains available only for creating NEW individuals and now recognizes the exact
reported v0.4.3 schema. It must not be used to replace an existing motor.

Please share the upgrade report, startup lines, final training summary and
`evaluation_window`. Do not upload the large database unless specifically needed.

## Scope

No Unity rebuild or GPU change is required. Existing THOR scripts and v0.5.3 sensor
recovery remain present. Native custom Unity and RX 6600/ROCm were not tested here;
mock THOR integration and CPU source execution were tested. This patch adds no
semantic VLM/LLM executive, autonomous skill discovery or evidence of consciousness.
