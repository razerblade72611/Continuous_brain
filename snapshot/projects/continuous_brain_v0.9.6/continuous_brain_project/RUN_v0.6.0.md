# Run Continuous Brain v0.6.0 on your WSL computer

## 1. Install side-by-side; preserve v0.5.x and its state

Stop any old training run with Ctrl+C before importing or backing up its state.
Download `continuous_brain_v0.6.0.zip` into your Windows Downloads folder. In WSL:

```bash
mkdir -p ~/projects/continuous_brain_v0.6.0
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.6.0.zip \
  -d ~/projects/continuous_brain_v0.6.0

source ~/projects/continuous_brain_project/.venv/bin/activate
cd ~/projects/continuous_brain_v0.6.0/continuous_brain_project
python -m pip install -e . --no-deps
python -m pytest -q
```

Activate the existing working PyTorch environment if yours lives elsewhere. Do
not move/copy a virtual environment: its entry-point paths may be absolute. Do not
replace the working ROCm PyTorch install with a generic pip torch wheel. If pytest
is missing, `python -m pip install pytest` adds the test runner separately.

The editable package entry points now use v0.6 in this environment. Old source and
state directories are untouched. To restore the old editable install later,
activate the same environment and run `python -m pip install -e . --no-deps` from
the old project directory. Both releases retain the old `continuous-brain` command;
the new architecture is invoked with `continuous-individual` or the scripts below.

Check your compute device:

```bash
python -c "import torch; print('torch:', torch.__version__, 'HIP:', torch.version.hip, 'GPU available:', torch.cuda.is_available())"
```

ROCm uses the PyTorch `cuda` device name. If unavailable, the runtime selects CPU.

## 2. Run a short, separate smoke individual

```bash
PROFILE=tiny STATE_DIR=state/smoke_v060 \
  bash scripts/run_individual_v060.sh --decisions 30 --fast
```

The browser dashboard opens at `http://localhost:8765`. If auto-opening fails,
paste the URL into your browser. This is an egocentric 2-D development room, not
Unity. Expect the same individual ID when resuming this same profile/directory,
increasing memory counts, nonzero workspace activity and learning updates after
the first few decisions. These are functional checks, not consciousness tests.

The tiny smoke individual is a test branch. Do not switch its directory to a
different profile. Preserve it if you want, but use the separate main directory below.

## 3. Start the main 32.1M-parameter individual

```bash
PROFILE=standard STATE_DIR=state/individual_v060 \
  bash scripts/run_individual_v060.sh --decisions 1000
```

This initial development run uses analytical motor primitives and an explicit,
decaying executive planner teacher. The new executive, memory integration and
world predictor train; the analytical motor controller is not a neural achievement.
Teacher probability starts at 1 and decays to 0.1 over 5,000 executive decisions.

To continue the same individual without a fixed stopping count:

```bash
PROFILE=standard STATE_DIR=state/individual_v060 \
  bash scripts/run_individual_v060.sh
```

Ctrl+C saves and stops. The next identical command restores workspace, memories,
intentions, self-state, optimizer, replay, world state and RNG. Do not delete state
between sessions. Use `--fast` to remove the real-time pacing, or `--device cpu`
for an explicit CPU run. CPU throughput/GPU memory are hardware dependent.
Allow several GB of free disk for standard checkpoints, journal growth and backups.

## 4. Optionally inherit the previously learned motor skills

Import is only allowed when creating a NEW individual directory. The source must
be stopped and contain an embodied `brain.pt`; a foraging checkpoint with the
same tensor dimensions is rejected. This is a new executive inheriting the older
motor history, not a claim that the older system already had this executive.

For your protected mastered L0/L1 state, for example:

```bash
PROFILE=standard STATE_DIR=state/individual_inherited_v060 \
  bash scripts/run_individual_v060.sh --decisions 1000 \
  --import-v053-state "$HOME/projects/continuous_brain_v0.4.3_preserved/state/embodied_brain_v043_clean"
```

The option name is `--import-v053-state`, but it also accepts compatible embodied
v0.4.x/0.5.x checkpoints. A healthy v0.5.3 state can be used instead. Do not choose
the recovery-storm diagnostic state as canonical. Resume without the import flag.

The import copies the entire old state to `legacy_v053/`, loads its exact motor
weights, freezes them, and records their provenance and SHA-256. Original files
are never modified. Its old latent memories/replay remain preserved archives;
they are not fabricated into new semantic memories. Different visual dimensions
are rejected. If the old motor used DINO, select its original vision backbone.
The development room differs from THOR, so motor transfer performance is not guaranteed.

## 5. Optional THOR run with inherited motor state

The development-room run is sufficient to begin v0.6 work; a perfect L2 run is
not required. To try the existing THOR environment, use another new state directory:

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"
PROFILE=standard STATE_DIR=state/individual_thor_v060 \
  bash scripts/run_individual_thor_v060.sh --decisions 200 \
  --import-v053-state "$HOME/projects/continuous_brain_project/state/embodied_brain_v053_spatial"
```

Replace the import path with your actual stopped, healthy embodied checkpoint if
needed. On later runs omit `--import-v053-state`. The THOR adapter retains v0.5.3
sensor quarantine/recovery and internal-map L2. The executive uses the learned
motor unchanged. The actual custom Unity executable could not be tested here;
mock-controller integration is tested. This is not a physical robot release.

An existing development individual cannot simply be switched to THOR by changing
the flag. Explicit cross-environment cognitive transfer remains later work. THOR
restarts re-ground the physical scene; they do not claim exact Unity state restoration.

## 6. Evaluate without contaminating the training state

Stop the training process first. This evaluates its existing checkpoint in memory,
without changing learned weights, stored memories, intentions, state or RNG on disk:

```bash
python -m continuous_brain.individual_cli \
  --environment development --profile standard \
  --state-dir state/individual_v060 --decisions 200 --evaluate --fast
```

Executive teacher forcing and exploration are disabled. Motor provenance remains
unchanged: a reference-controller run is still reference-motor-assisted. Read task
counts in the `evaluation_window` section, not as cumulative evaluation scores.
Short or zero-completed-task samples are inconclusive. Repeat evaluations across
separate seeded training branches before interpreting success rates as competence.

## 7. Consolidation and backups

Replay-only learning (no additional environment actions):

```bash
python -m continuous_brain.individual_cli --profile standard \
  --state-dir state/individual_v060 --consolidate-updates 20
```

Use the same environment/profile/vision flags as the original individual. This
is ordinary rehearsal, not generative dreaming or autonomous introspection.

Stopped-state backup, with a new destination each time:

```bash
python scripts/snapshot_individual.py state/individual_v060 state/individual_v060_backup_01
```

An explicitly labelled experimental fork (new identity ID, inherited weights/history):

```bash
python scripts/snapshot_individual.py state/individual_v060 state/individual_v060_branch_01 --fork
```

Both refuse existing destinations or a running source. Restore by running the
backup directory with its original flags, not by overwriting the live directory.
Do not run a backup and its source concurrently under the same identity.

## What to share after the first main run

- The startup line with parameter count, device and individual ID.
- Several periodic log lines and any exception traceback.
- `state/individual_v060/individual_status.json` (small, readable summary).
- A separate unassisted evaluation summary when enough training has occurred.

Do not upload the large database unless diagnosis requires it. A low early success
rate is expected for a newly initialized executive. We need measured learning,
retention and transfer evidence before calling the architecture broadly capable.
