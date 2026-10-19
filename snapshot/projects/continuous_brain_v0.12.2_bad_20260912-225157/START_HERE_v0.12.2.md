# Start Continuous Brain v0.12.2

This release contains the v0.12 persistent lifelong runtime and the frozen 14,000-update InfantBrain used by the competitive-workspace experiment. v0.12.2 specifically hardens the DirectML training graph after v0.12.1 passed the GRU/core gates but failed the integrated competitive one-step backward on the target RX 6600.

## Fresh WSL installation

Keep v0.12.1 and v0.11.3 as backups. Assuming the ZIP is in Windows Downloads:

```bash
cd "$HOME/projects"
unzip /mnt/c/Users/davei/Downloads/continuous_brain_v0.12.2_unified_workspace.zip
cd "$HOME/projects/continuous_brain_v0.12.2"

source "$HOME/.venvs/continuous_brain_v110_dml/bin/activate"
python -m pip install -e .
python -c 'import continuous_brain; print(continuous_brain.__version__)'
```

Expected version: `0.12.2`.

## 1. Qualify DirectML

```bash
cd "$HOME/projects/continuous_brain_v0.12.2"
source "$HOME/.venvs/continuous_brain_v110_dml/bin/activate"

DEVICE=directml \
CHECKPOINT=state/child_v113_dml_smoke/infant_training.pt \
./scripts/qualify_directml_v122.sh
```

The final line must be:

```text
DIRECTML_V122_QUALIFIED
```

The script reports Python / PyTorch / torch-directml versions, then runs basic tensor/backward checks, GRU chains at 1/2/5/10/20 steps, the real 14k recurrent core, the hardened competitive one-step backward and a complete 0.3-corridor optimizer update. Every gate starts a fresh Python process.

Do not change Windows TDR, detach hidden state, or launch the 4,000-update experiment if qualification fails. Save the first failing traceback.

## 2. Continue the v0.12 lifelong individual

Existing state remains compatible:

```bash
bash run_local.sh --existing doctor
bash run_local.sh --existing run --steps 1024
```

Default state remains `~/ContinuousBrain/individual_v120`.

## 3. Competitive-broadcast smoke

Only after `DIRECTML_V122_QUALIFIED`:

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

`baseline_disabled_exact_parity` must be `true`.

## 4. Full experiment

Only after the smoke completes cleanly:

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

Keep the bundled 14k checkpoint immutable. Do not use `--reset-memory` on learned state. DirectML materialization in v0.12.2 uses differentiable `clone()`, never `detach()`, so temporal credit assignment remains intact. A competitive-broadcast win would justify integration as a protected developmental mechanism; it would not establish consciousness.
