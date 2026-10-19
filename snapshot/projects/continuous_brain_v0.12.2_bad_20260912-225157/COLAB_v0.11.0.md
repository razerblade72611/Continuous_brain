# Colab workflow — v0.11.0 growth-first whole-brain phase

## Principle

Upload the **expanded** child, not merely the v0.10.4 checkpoint. Expansion is deterministic and can be performed locally first so the exact continuity audit is preserved with the state.

## Recommended preparation on the WSL machine

After expansion and generalized data collection:

```bash
python scripts/make_v110_colab_bundle.py \
  --project-root . \
  --state-dir "$V110_BRANCH" \
  --datasets "$BOOTSTRAP" "$OLD_WORLD" "$DETOUR_TRAIN" "$RAND_TRAIN" "$TWO_TRAIN" \
  --output "$HOME/continuous_brain_v110_colab_bundle.tar.gz"
```

Record the printed bundle SHA-256 before upload.

## Colab setup

After uploading and extracting the bundle:

```bash
cd /content/continuous_brain_project
python -m pip install -e . --no-deps
python - <<'PY'
import torch, continuous_brain
print('version', continuous_brain.__version__)
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
```

Use the paths under `/content/payload` or move the state/datasets to a persistent mounted Drive directory before a long run.

## Training command

```bash
export V110_BRANCH=/content/payload/state/child_v110_growth
export DATA=/content/payload/datasets

python -m continuous_brain.parented_cli train-whole-brain \
  --rehearsal-datasets "$DATA/nursery_v090_bootstrap" "$DATA/nursery_v095_mixed_003" \
  --world-datasets "$DATA/nursery_v095_mixed_003" "$DATA/nursery_v096_detour_train" \
                   "$DATA/nursery_v110_randomized_detour_train" "$DATA/nursery_v110_two_barrier_train" \
  --planning-dataset "$DATA/nursery_v110_randomized_detour_train" \
  --state-dir "$V110_BRANCH" \
  --updates 6000 --batch 8 --learning-rate 0.0001 \
  --anchor-weight 0.10 --checkpoint-every 500 --seed 11010 --device cuda \
  | tee results/V110_WHOLE_BRAIN_GROWTH_PHASE_A.json
```

The trainer creates periodic `infant_training.v110_route_*.pt` snapshots. Copy the state directory and `results/` to persistent storage at ordinary checkpoint boundaries.

## First post-training gates

Before any legacy-weight unfreezing, require:

1. `legacy_tensors_changed == 0` in the training result;
2. bootstrap vision/language retention still passes;
3. fresh v0.10.4 ForagingWorld and DetourWorld retention still pass;
4. P4b hierarchical planning still passes when rerun as a regression;
5. randomized-detour development completion is at least 20% on the predeclared development range.

Use `scripts/check_v110_growth_readiness.py` to combine those results.

A future controlled-plasticity phase is not authorized merely because training loss falls.
