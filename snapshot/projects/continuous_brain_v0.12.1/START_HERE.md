# Start Continuous Brain v0.12.1

This release contains both the v0.12 persistent lifelong runtime and the frozen 14,000-update InfantBrain used by the competitive-workspace experiment.

## Fresh WSL installation using your existing DirectML environment

Assuming the ZIP is in Windows Downloads:

```bash
cd "$HOME/projects"
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.12.1_unified_workspace.zip
cd "$HOME/projects/continuous_brain_v0.12.1"

source "$HOME/.venvs/continuous_brain_v110_dml/bin/activate"
python -m pip install -e .

python -c 'import continuous_brain; print(continuous_brain.__version__)'
```

Expected version: `0.12.1`.

## 1. Qualify DirectML before competitive training

```bash
cd "$HOME/projects/continuous_brain_v0.12.1"
source "$HOME/.venvs/continuous_brain_v110_dml/bin/activate"

DEVICE=directml \
CHECKPOINT=state/child_v113_dml_smoke/infant_training.pt \
./scripts/qualify_directml_v121.sh
```

The final line must be:

```text
DIRECTML_V121_QUALIFIED
```

This runs recurrent backward at 1, 2, 5, 10 and 20 steps in fresh processes, then the real child core, full competitive one-step backward, and a complete delayed-cue episode update. If any gate fails, stop there and keep the traceback. Do not alter Windows TDR, detach hidden state, or fall back to CPU just to make the gate green.

## 2. Start or continue the v0.12 lifelong individual

The lifelong runtime remains compatible with the existing v0.12 state directory:

```bash
bash run_local.sh --existing doctor
bash run_local.sh --existing run --steps 1024
```

Default WSL state remains `~/ContinuousBrain/individual_v120`.

## 3. Competitive-broadcast smoke

Only after DirectML qualification passes:

```bash
SMOKE_DIR="state/competitive_broadcast_v3_dml_smoke"
rm -rf "$SMOKE_DIR"

DEVICE=directml \
BASE_CHECKPOINT=state/child_v113_dml_smoke/infant_training.pt \
UPDATES=25 \
EPISODES_PER_LENGTH=10 \
STATE_DIR="$SMOKE_DIR" \
./scripts/run_competitive_broadcast_experiment.sh \
  2>&1 | tee "$SMOKE_DIR.run.log"
```

The run trains matched shared, independent and uniform conditions and evaluates frozen baseline, shared, independent, uniform, hidden-reset and disabled conditions. `baseline_disabled_exact_parity` must be true.

## 4. Full competitive experiment

Only after the 25-update smoke completes cleanly:

```bash
FULL_DIR="state/competitive_broadcast_v3_dml_full"
mkdir -p "$FULL_DIR"

DEVICE=directml \
BASE_CHECKPOINT=state/child_v113_dml_smoke/infant_training.pt \
UPDATES=4000 \
EPISODES_PER_LENGTH=100 \
STATE_DIR="$FULL_DIR" \
./scripts/run_competitive_broadcast_experiment.sh \
  2>&1 | tee "$FULL_DIR/run.log"
```

## Preservation rules

Keep the bundled 14k checkpoint immutable. Keep v2 diagnostics separate from v3. Do not use `--reset-memory` on legacy learned state. Do not interpret a successful competitive experiment as proof of consciousness; use it to decide whether shared broadcast deserves integration into the main developmental architecture.
