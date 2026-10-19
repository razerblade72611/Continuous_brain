# Run Continuous Brain v0.10.1

This procedure continues the **same persistent child** from the already-trained v0.10.0 planning checkpoint. It does not initialize a new lineage.

## 1. Install

```bash
cd ~/projects
mkdir -p ~/projects/continuous_brain_v0.10.1
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.10.1.zip \
  -d ~/projects/continuous_brain_v0.10.1
cd ~/projects/continuous_brain_v0.10.1/continuous_brain_project
source ~/projects/continuous_brain_project/.venv/bin/activate
python -m pip install -e . --no-deps
```

Verify:

```bash
python - <<'PY'
import continuous_brain
print(continuous_brain.__version__)
PY
```

Expected: `0.10.1`.

## 2. Run the parented-development regression line

```bash
pytest -q \
  tests/test_v090.py tests/test_v092.py tests/test_v093.py tests/test_v094.py \
  tests/test_v095.py tests/test_v096.py tests/test_v100.py tests/test_v101.py
```

Expected for this release build: `27 passed`.

## 3. Branch from the trained v0.10.0 planner

```bash
export SOURCE_V100="$HOME/projects/continuous_brain_v0.10.0/continuous_brain_project/state/child_v100_planning"
export V101_BRANCH="$HOME/projects/continuous_brain_v0.10.1/continuous_brain_project/state/child_v101_bellman"
export DETOUR_DATA="$HOME/projects/continuous_brain_v0.9.6/continuous_brain_project/state/nursery_v096_detour_train"

mkdir -p "$V101_BRANCH"
cp "$SOURCE_V100/infant_training.pt" "$V101_BRANCH/infant_training.pt"
cp "$SOURCE_V100/parents.json" "$V101_BRANCH/parents.json"
cp "$SOURCE_V100/identity.json" "$V101_BRANCH/identity.json"

sha256sum "$SOURCE_V100/infant_training.pt" "$V101_BRANCH/infant_training.pt"
```

The two hashes must be identical before training.

## 4. Train only conservative Bellman Q

The release defaults are precommitted development settings. Do not tune them using the reserved blind seeds.

```bash
mkdir -p results

python -m continuous_brain.parented_cli train-planning-q \
  --dataset "$DETOUR_DATA" \
  --state-dir "$V101_BRANCH" \
  --updates 3000 \
  --batch 256 \
  --gamma 0.95 \
  --learning-rate 0.0003 \
  --cql-alpha 0.10 \
  --target-tau 0.01 \
  --seed 10110 \
  --device cpu \
  | tee results/V101_bellman_q_training.json
```

Required isolation result:

```text
q_only: true
changed_non_q_tensors: 0
```

All five `train_action_counts` must be nonzero.

## 5. Freeze the trained valuation checkpoint

```bash
cp "$V101_BRANCH/infant_training.pt" \
   "$V101_BRANCH/infant_training.at_V101_BELLMAN_Q.pt"
sha256sum "$V101_BRANCH/infant_training.at_V101_BELLMAN_Q.pt" \
  | tee "$V101_BRANCH/infant_training.at_V101_BELLMAN_Q.pt.sha256"
```

## 6. Development-only action-value probe

Use only the already-spent seeds:

```bash
python scripts/p4b_bellman_q_probe_v101.py \
  --checkpoint "$V101_BRANCH/infant_training.pt" \
  --seed-start 9001 \
  --seeds 10 \
  --discount 0.95 \
  --device cpu \
  --output results/P4B_v101_bellman_q_probe_9001_9010.json
```

The key question is whether UP now becomes competitive in `initial_q`, `prewall_q`, and especially the model-based Bellman ranking. This is diagnostic only, not blind evidence.

## 7. Development-only P4b behavior

```bash
python -m continuous_brain.parented_cli evaluate-planning \
  --state-dir "$V101_BRANCH" \
  --episodes 10 \
  --steps 120 \
  --seed-start 9001 \
  --horizons 1 3 6 12 \
  --beam-width 32 \
  --discount 0.95 \
  --device cpu \
  --output results/P4B_v101_smoke_9001_9010.json
```

Do not use seeds `12001–12100` unless at least one development horizon produces real detour completion and the search configuration is then frozen.

## 8. Final P4b gate (later, only after development success)

Reserved blind range: `12001–12100`.

The gate remains:

```text
absolute completion >= 20%
AND completion advantage over reactive > 5 percentage points
AND collision regression <= 2 collisions/episode
```

A P4b pass plus P3/retention/lineage checks is the trigger for the intended major whole-brain Colab phase on the current persistent architecture.
