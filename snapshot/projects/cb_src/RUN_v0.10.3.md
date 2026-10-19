# Run Continuous Brain v0.10.3

v0.10.3 requires **no retraining**. Continue from the exact trained v0.10.2 option checkpoint.

## 1. Install

```bash
cd ~/projects
mkdir -p ~/projects/continuous_brain_v0.10.3
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.10.3.zip \
  -d ~/projects/continuous_brain_v0.10.3
cd ~/projects/continuous_brain_v0.10.3/continuous_brain_project
source ~/projects/continuous_brain_project/.venv/bin/activate
python -m pip install -e . --no-deps
python - <<'PY'
import continuous_brain
print(continuous_brain.__version__)
PY
```

Expected version: `0.10.3`.

## 2. Branch the exact v0.10.2 child

```bash
export SOURCE_V102="$HOME/projects/continuous_brain_v0.10.2/continuous_brain_project/state/child_v102_options"
export V103_BRANCH="$HOME/projects/continuous_brain_v0.10.3/continuous_brain_project/state/child_v103_hierarchical"
mkdir -p "$V103_BRANCH"
cp "$SOURCE_V102/infant_training.pt" "$V103_BRANCH/infant_training.pt"
cp "$SOURCE_V102/parents.json" "$V103_BRANCH/parents.json"
cp "$SOURCE_V102/identity.json" "$V103_BRANCH/identity.json"
sha256sum "$SOURCE_V102/infant_training.pt" "$V103_BRANCH/infant_training.pt"
```

The two hashes must match exactly. The known trained v0.10.2 checkpoint from the development run was:

`ba1792d340b761ae436585f77a6a80dd7188b4daf108e4ab4c0c7bf8091fae07`

If your source file has changed since that run, stop and inspect it before proceeding.

## 3. Run the predeclared development test

Use only already-spent development seeds 9001-9010:

```bash
mkdir -p results
python -m continuous_brain.parented_cli evaluate-hierarchical-options \
  --state-dir "$V103_BRANCH" \
  --episodes 10 \
  --steps 120 \
  --seed-start 9001 \
  --option-depth 2 \
  --selector-topk 2 \
  --beam-width 8 \
  --max-option-steps 12 \
  --discount 0.95 \
  --min-commit 3 \
  --terminate-threshold 0.50 \
  --device cpu \
  --output results/P4B_v103_hierarchical_smoke_9001_9010.json
```

Do not run seeds 12001-12100 yet.

## 4. Interpretation

The result includes:

- `reactive`;
- `option_direct_diagnostic`;
- `hierarchical_option_planning`;
- `advantages_vs_reactive`;
- `p4b_passed`.

The formal gate is unchanged:

- completion >= 20%;
- completion advantage > 5 percentage points;
- collision regression <= 2 per episode.

A development `p4b_passed: true` is not yet blind proof. It earns the final blind examination after the configuration is frozen.

## 5. If the development test passes

Freeze both code/configuration and checkpoint. Then the untouched 100-episode final blind run is:

```bash
python -m continuous_brain.parented_cli evaluate-hierarchical-options \
  --state-dir "$V103_BRANCH" \
  --episodes 100 \
  --steps 120 \
  --seed-start 12001 \
  --option-depth 2 \
  --selector-topk 2 \
  --beam-width 8 \
  --max-option-steps 12 \
  --discount 0.95 \
  --min-commit 3 \
  --terminate-threshold 0.50 \
  --device cpu \
  --output results/P4B_v103_FINAL_12001_12100.json
```

Only run this after the development result justifies spending the blind seeds.

## 6. After a blind P4b pass

Re-run the sealed P3 world-model and bootstrap-retention evaluations, then use `scripts/check_v103_colab_readiness.py` before major whole-brain Colab training.
