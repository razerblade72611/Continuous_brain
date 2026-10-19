# Run Continuous Brain v0.10.4

v0.10.4 continues from the exact v0.10.3 checkpoint that passed blind P4b. Do not train on `nursery_v095_WORLD_BLIND_FINAL`; that dataset has already informed the diagnosis.

## 1. Install

```bash
cd ~/projects
mkdir -p ~/projects/continuous_brain_v0.10.4
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.10.4.zip \
  -d ~/projects/continuous_brain_v0.10.4
cd ~/projects/continuous_brain_v0.10.4/continuous_brain_project
source ~/projects/continuous_brain_project/.venv/bin/activate
python -m pip install -e . --no-deps
python - <<'PY'
import continuous_brain
print(continuous_brain.__version__)
PY
```

Expected: `0.10.4`.

## 2. Identify source and training nurseries

```bash
export SOURCE_V103="$HOME/projects/continuous_brain_v0.10.3/continuous_brain_project/state/child_v103_hierarchical"
export V104_BRANCH="$HOME/projects/continuous_brain_v0.10.4/continuous_brain_project/state/child_v104_consolidated"

export OLD_WORLD="$HOME/projects/continuous_brain_v0.9.5/continuous_brain_project/state/nursery_v095_mixed_003"
export DETOUR_TRAIN="$HOME/projects/continuous_brain_v0.9.6/continuous_brain_project/state/nursery_v096_detour_train"

export FRESH_FORAGING_BLIND="$HOME/projects/continuous_brain_v0.10.4/continuous_brain_project/state/nursery_v104_FORAGING_RETENTION_BLIND"
export FRESH_DETOUR_BLIND="$HOME/projects/continuous_brain_v0.10.4/continuous_brain_project/state/nursery_v104_DETOUR_RETENTION_BLIND"

mkdir -p results
```

The expected v0.10.3 source checkpoint from the blind P4b run is:

`ba1792d340b761ae436585f77a6a80dd7188b4daf108e4ab4c0c7bf8091fae07`

Verify it before proceeding:

```bash
sha256sum "$SOURCE_V103/infant_training.pt"
```

## 3. Create the NEW blind retention holdouts BEFORE training

These commands use no DINO/LLM parent and no child model.

```bash
python -m continuous_brain.parented_cli collect-world-holdout \
  --dataset "$FRESH_FORAGING_BLIND" \
  --domain foraging \
  --samples 1200 \
  --seed 24017 \
  --exploration 0.45 \
  | tee results/V104_FORAGING_BLIND_COLLECTION.json

python -m continuous_brain.parented_cli collect-world-holdout \
  --dataset "$FRESH_DETOUR_BLIND" \
  --domain detour \
  --samples 1200 \
  --seed 24019 \
  --exploration 0.45 \
  | tee results/V104_DETOUR_BLIND_COLLECTION.json
```

If either destination already contains records, stop rather than overwriting it.

## 4. Seal holdouts and allowed training sources

```bash
python -m continuous_brain.parented_cli seal-world-holdouts \
  --foraging-dataset "$FRESH_FORAGING_BLIND" \
  --detour-dataset "$FRESH_DETOUR_BLIND" \
  --training-datasets "$OLD_WORLD" "$DETOUR_TRAIN" \
  --output results/V104_WORLD_HOLDOUT_SEAL.json \
  | tee results/V104_WORLD_HOLDOUT_SEAL.stdout.json

sha256sum results/V104_WORLD_HOLDOUT_SEAL.json \
  | tee results/V104_WORLD_HOLDOUT_SEAL.json.sha256
```

Do not evaluate the fresh holdouts yet.

## 5. Branch the same persistent child

```bash
mkdir -p "$V104_BRANCH"
cp "$SOURCE_V103/infant_training.pt" "$V104_BRANCH/infant_training.pt"
cp "$SOURCE_V103/parents.json" "$V104_BRANCH/parents.json"
cp "$SOURCE_V103/identity.json" "$V104_BRANCH/identity.json"

sha256sum "$SOURCE_V103/infant_training.pt" "$V104_BRANCH/infant_training.pt"
```

The hashes must be identical before consolidation.

Preserve another immutable source copy if desired:

```bash
cp "$V104_BRANCH/infant_training.pt" "$V104_BRANCH/infant_training.before_v104_consolidation.pt"
chmod a-w "$V104_BRANCH/infant_training.before_v104_consolidation.pt"
```

## 6. Run the fixed consolidation training

```bash
python -m continuous_brain.parented_cli train-world-consolidation \
  --datasets "$OLD_WORLD" "$DETOUR_TRAIN" \
  --state-dir "$V104_BRANCH" \
  --holdout-seal results/V104_WORLD_HOLDOUT_SEAL.json \
  --updates 3000 \
  --batch-per-domain 16 \
  --learning-rate 0.0001 \
  --seed 10410 \
  --device cpu \
  | tee results/V104_WORLD_CONSOLIDATION_TRAINING.json
```

Required integrity fields:

- `world_only: true`
- `changed_non_world_tensors: 0`
- `balanced_domain_weighting: true`
- `action_balanced_within_domain: true`

Do not rerun with altered hyperparameters based on blind results.

## 7. Freeze the trained checkpoint

```bash
cp "$V104_BRANCH/infant_training.pt" \
   "$V104_BRANCH/infant_training.at_V104_WORLD_CONSOLIDATED.pt"
sha256sum "$V104_BRANCH/infant_training.at_V104_WORLD_CONSOLIDATED.pt" \
  | tee "$V104_BRANCH/infant_training.at_V104_WORLD_CONSOLIDATED.pt.sha256"
```

## 8. Spend the fresh world-retention holdouts once

```bash
python -m continuous_brain.parented_cli evaluate-world-consolidation \
  --state-dir "$V104_BRANCH" \
  --foraging-holdout "$FRESH_FORAGING_BLIND" \
  --detour-holdout "$FRESH_DETOUR_BLIND" \
  --holdout-seal results/V104_WORLD_HOLDOUT_SEAL.json \
  --device cpu \
  --output results/V104_WORLD_RETENTION_FINAL.json \
  | tee results/V104_WORLD_RETENTION_FINAL.stdout.json
```

The required top-level field is:

`world_consolidation_passed: true`

and both `domains.foraging.world_model_passed` and `domains.detour.world_model_passed` must be true.

## 9. Bootstrap retention

```bash
export BOOTSTRAP="$HOME/projects/continuous_brain_v0.9.0/continuous_brain_project/state/nursery_v090_bootstrap"
python -m continuous_brain.parented_cli evaluate \
  --dataset "$BOOTSTRAP" \
  --state-dir "$V104_BRANCH" \
  --device cpu \
  --seed 17 \
  | tee results/V104_BOOTSTRAP_RETENTION.json
```

## 10. Colab readiness

Use the already-frozen 100-episode v0.10.3 P4b result. World consolidation cannot modify the planner if the exact tensor-isolation gate passed.

```bash
python scripts/check_v104_colab_readiness.py \
  --source-v103-checkpoint "$SOURCE_V103/infant_training.pt" \
  --current-checkpoint "$V104_BRANCH/infant_training.pt" \
  --p4b-result "$HOME/projects/continuous_brain_v0.10.3/continuous_brain_project/results/P4B_v103_FINAL_BLIND_12001_12100.json" \
  --world-retention-result results/V104_WORLD_RETENTION_FINAL.json \
  --bootstrap-retention-result results/V104_BOOTSTRAP_RETENTION.json \
  | tee results/V104_COLAB_READINESS.json
```

Only `colab_ready: true` authorizes the planned major whole-brain Colab phase.
