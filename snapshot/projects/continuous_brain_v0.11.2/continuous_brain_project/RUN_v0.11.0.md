# Run v0.11.0

This file gives the canonical local progression from the validated v0.10.4 child to the expanded v0.11.0 child. For the cloud phase, continue with `COLAB_v0.11.0.md`.

## 1. Source checkpoint

The lineage checkpoint expected from the successful v0.10.4 run is:

```text
1936e046b21292a5d0519d03fe087f2568b3899e91e9bd6796a0b020a966b3a9
```

Verify it before expansion.

## 2. Function-preserving expansion

```bash
export SOURCE_V104="$HOME/projects/continuous_brain_v0.10.4/continuous_brain_project/state/child_v104_consolidated"
export V110_BRANCH="$HOME/projects/continuous_brain_v0.11.0/continuous_brain_project/state/child_v110_growth"
mkdir -p results

python -m continuous_brain.parented_cli expand-capacity \
  --source-state-dir "$SOURCE_V104" \
  --state-dir "$V110_BRANCH" \
  --growth-profile balanced-v110 \
  --device cpu \
  | tee results/V110_EXPANSION.json
```

Do not proceed unless the result reports:

```text
legacy_tensor_mismatches = 0
exact_function_preservation = true
max_abs_difference = 0.0
expanded_trainable_parameters >= 50000000
```

Freeze a copy of the immediate post-expansion checkpoint before training.

## 3. Collect generalized training experience

```bash
export RAND_TRAIN="$HOME/projects/continuous_brain_v0.11.0/continuous_brain_project/state/nursery_v110_randomized_detour_train"
export TWO_TRAIN="$HOME/projects/continuous_brain_v0.11.0/continuous_brain_project/state/nursery_v110_two_barrier_train"

python -m continuous_brain.parented_cli collect-generalized-curriculum \
  --dataset "$RAND_TRAIN" --kind randomized_detour \
  --samples 8000 --seed 31001 --exploration 0.25 --max-steps 160 \
  | tee results/V110_RANDOMIZED_COLLECTION.json

python -m continuous_brain.parented_cli collect-generalized-curriculum \
  --dataset "$TWO_TRAIN" --kind two_barrier \
  --samples 8000 --seed 31003 --exploration 0.20 --max-steps 180 \
  | tee results/V110_TWO_BARRIER_COLLECTION.json
```

Collection may use environment geometry to produce useful lived sequences, but `motor_valid=false`: this policy is not installed into the child.

## 4. Data used for growth-first training

```bash
export BOOTSTRAP="$HOME/projects/continuous_brain_v0.9.0/continuous_brain_project/state/nursery_v090_bootstrap"
export OLD_WORLD="$HOME/projects/continuous_brain_v0.9.5/continuous_brain_project/state/nursery_v095_mixed_003"
export DETOUR_TRAIN="$HOME/projects/continuous_brain_v0.9.6/continuous_brain_project/state/nursery_v096_detour_train"
```

The first routed phase should use `BOOTSTRAP` and `OLD_WORLD` for parented rehearsal, all four physical datasets for the world route, and the randomized-detour dataset for planning/option sequence learning.

## 5. Growth-first training

Canonical first phase:

```bash
python -m continuous_brain.parented_cli train-whole-brain \
  --rehearsal-datasets "$BOOTSTRAP" "$OLD_WORLD" \
  --world-datasets "$OLD_WORLD" "$DETOUR_TRAIN" "$RAND_TRAIN" "$TWO_TRAIN" \
  --planning-dataset "$RAND_TRAIN" \
  --state-dir "$V110_BRANCH" \
  --updates 6000 \
  --batch 8 \
  --learning-rate 0.0001 \
  --anchor-weight 0.10 \
  --checkpoint-every 500 \
  --seed 11010 \
  --device cuda \
  | tee results/V110_WHOLE_BRAIN_GROWTH_PHASE_A.json
```

A rerun against the same state directory resumes the routed step index and attempts to restore the optimizer. Do not use a different seed or schedule merely because an intermediate metric looks disappointing.

## 6. Post-training regression

Re-run bootstrap retention, v0.10.4 fresh world retention, and the already-spent P4b seed range as a **regression test**, not new blind evidence. Then evaluate new randomized layouts on development seeds beginning at 41001.

Do not use the future generalized blind range 51001–51100 while tuning v0.11.0.
