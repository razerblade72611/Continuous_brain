# Run Continuous Brain v0.10.2

This procedure continues the **same persistent child** from the trained v0.10.1 Bellman-Q checkpoint.

## 1. Install

```bash
cd ~/projects
mkdir -p ~/projects/continuous_brain_v0.10.2
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.10.2.zip \
  -d ~/projects/continuous_brain_v0.10.2
cd ~/projects/continuous_brain_v0.10.2/continuous_brain_project
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

Expected: `0.10.2`.

## 2. Regression line

```bash
pytest -q \
  tests/test_v090.py tests/test_v092.py tests/test_v093.py tests/test_v094.py \
  tests/test_v095.py tests/test_v096.py tests/test_v100.py tests/test_v101.py tests/test_v102.py
```

Release expectation: `31 passed`.

## 3. Branch from the trained v0.10.1 child

```bash
export SOURCE_V101="$HOME/projects/continuous_brain_v0.10.1/continuous_brain_project/state/child_v101_bellman"
export V102_BRANCH="$HOME/projects/continuous_brain_v0.10.2/continuous_brain_project/state/child_v102_options"
export DETOUR_DATA="$HOME/projects/continuous_brain_v0.9.6/continuous_brain_project/state/nursery_v096_detour_train"

mkdir -p "$V102_BRANCH"
cp "$SOURCE_V101/infant_training.pt" "$V102_BRANCH/infant_training.pt"
cp "$SOURCE_V101/parents.json" "$V102_BRANCH/parents.json"
cp "$SOURCE_V101/identity.json" "$V102_BRANCH/identity.json"

sha256sum "$SOURCE_V101/infant_training.pt" "$V102_BRANCH/infant_training.pt"
```

The hashes must match before option training.

## 4. Train only the temporal option cortex

```bash
mkdir -p results

python -m continuous_brain.parented_cli train-options \
  --dataset "$DETOUR_DATA" \
  --state-dir "$V102_BRANCH" \
  --updates 3000 \
  --batch 256 \
  --lookahead 6 \
  --gamma 0.95 \
  --goal-gamma 0.97 \
  --learning-rate 0.0003 \
  --q-margin 0.08 \
  --seed 10210 \
  --device cpu \
  | tee results/V102_option_training.json
```

Required isolation result:

```text
option_only: true
changed_non_option_tensors: 0
```

Training labels are anonymous temporal clusters discovered from successful lived sequences. `detour_geometry_labels_used` must be `false`.

Useful validation health metrics:

- `selector_accuracy`
- `action_accuracy`
- `termination_f1`
- `executed_action_rank_accuracy`
- `option_q_mae`

These are diagnostics, not substitutes for behavioral completion.

## 5. Freeze the trained option milestone

```bash
cp "$V102_BRANCH/infant_training.pt" \
   "$V102_BRANCH/infant_training.at_V102_OPTIONS.pt"
sha256sum "$V102_BRANCH/infant_training.at_V102_OPTIONS.pt" \
  | tee "$V102_BRANCH/infant_training.at_V102_OPTIONS.pt.sha256"
```

## 6. Development-only option/planning diagnostic

Use the already-spent seeds only:

```bash
python -m continuous_brain.parented_cli evaluate-options \
  --state-dir "$V102_BRANCH" \
  --episodes 10 \
  --steps 120 \
  --seed-start 9001 \
  --horizons 1 3 6 12 \
  --beam-width 32 \
  --discount 0.95 \
  --min-commit 3 \
  --terminate-threshold 0.50 \
  --device cpu \
  --output results/P4B_v102_options_smoke_9001_9010.json
```

Inspect both:

- `option_direct_diagnostic` — shows whether temporal abstractions themselves organize successful behavior;
- `horizons.*.option_planning` — the formal child-owned imagination path.

Do not tune using seeds `12001–12100`.

## 7. When to use the blind P4b range

Only after a fixed development configuration shows real completion, preferably >=20% on the development sample, freeze all settings and use the untouched blind range:

```text
12001–12100
```

The formal gate remains:

```text
absolute completion >= 20%
AND completion advantage over reactive > 5 percentage points
AND collision regression <= 2 collisions/episode
```

A blind P4b pass is followed by P3 retention, bootstrap retention and lineage checks before the major whole-brain Colab phase.
