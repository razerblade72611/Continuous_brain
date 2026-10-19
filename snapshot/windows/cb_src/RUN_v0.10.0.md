# Continuous Brain v0.10.0 — Run Guide

This guide continues the **same persistent child lineage**. Do not initialize a new child and do not modify the frozen v0.9.6 P4a checkpoint.

## 1. Branch from the frozen P4a checkpoint

```bash
export SOURCE_P4A="$HOME/projects/continuous_brain_v0.9.6/continuous_brain_project/state/child_v090_p4_detour/infant_training.at_11000_P4A_PASS.pt"
export SOURCE_STATE="$HOME/projects/continuous_brain_v0.9.6/continuous_brain_project/state/child_v090_p4_detour"
export V100_BRANCH="$HOME/projects/continuous_brain_v0.10.0/continuous_brain_project/state/child_v100_planning"

mkdir -p "$V100_BRANCH"
cp "$SOURCE_P4A" "$V100_BRANCH/infant_training.pt"
cp "$SOURCE_STATE/parents.json" "$V100_BRANCH/parents.json"
cp "$SOURCE_STATE/identity.json" "$V100_BRANCH/identity.json"
sha256sum "$SOURCE_P4A" "$V100_BRANCH/infant_training.pt"
```

The two hashes must match before planning training.

## 2. Train only the new planning cortex

Reuse the already collected v0.9.6 detour nursery:

```bash
export DETOUR_DATA="$HOME/projects/continuous_brain_v0.9.6/continuous_brain_project/state/nursery_v096_detour_train"

python -m continuous_brain.parented_cli train-planning \
  --dataset "$DETOUR_DATA" \
  --state-dir "$V100_BRANCH" \
  --updates 4000 \
  --batch 32 \
  --unroll 12 \
  --gamma 0.95 \
  --goal-gamma 0.97 \
  --learning-rate 0.0003 \
  --seed 10010 \
  --device cpu \
  | tee results/V100_planning_training.json
```

This command trains only `planning.*`. The original 11,000 update count is preserved as the source whole-brain/world-development count; planning updates are recorded separately in checkpoint metadata.

## 3. Verify legacy weights stayed frozen

Do not continue if any legacy tensor changed:

```bash
python - <<'PY'
import os, torch
source=torch.load(os.environ['SOURCE_P4A'],map_location='cpu',weights_only=False)
current=torch.load(os.path.join(os.environ['V100_BRANCH'],'infant_training.pt'),map_location='cpu',weights_only=False)
changed=[]
for k,v in source['model'].items():
    if k.startswith('planning.'):
        continue
    if k not in current['model'] or not torch.equal(v,current['model'][k]):
        changed.append(k)
print('changed_legacy_tensors =',len(changed))
if changed: print(*changed[:50],sep='\n')
else: print('PASS: every pre-v0.10 tensor is bit-for-bit unchanged.')
PY
```

Expected: `changed_legacy_tensors = 0`.

## 4. Development diagnostic — already used seeds

Seeds 9001+ are diagnostic only because previous P4 work has already inspected them.

```bash
mkdir -p results
python -m continuous_brain.parented_cli evaluate-planning \
  --state-dir "$V100_BRANCH" \
  --episodes 10 \
  --steps 120 \
  --seed-start 9001 \
  --horizons 3 6 12 \
  --beam-width 32 \
  --discount 0.95 \
  --device cpu \
  --output results/P4B_v100_smoke_9001_9010.json
```

If all planning horizons still have zero completion, stop. Do not inspect the final blind seeds. The architecture has not yet earned the blind exam.

## 5. Final P4b blind exam — only after diagnostic completions

The final range remains untouched until the mechanism is fixed:

```bash
python -m continuous_brain.parented_cli evaluate-planning \
  --state-dir "$V100_BRANCH" \
  --episodes 100 \
  --steps 120 \
  --seed-start 12001 \
  --horizons 3 6 12 \
  --beam-width 32 \
  --discount 0.95 \
  --device cpu \
  --output results/P4B_v100_FINAL_12001_12100.json
```

P4b passes only if at least one horizon has absolute completion >=20%, >5 percentage-point completion advantage over reactive, and <=2 collisions/episode regression.

## 6. Re-run P3 and childhood retention on the v0.10 checkpoint

Use the already sealed P3 and bootstrap datasets; do not retrain on them.

```bash
export P3_BLIND="$HOME/projects/continuous_brain_v0.9.5/continuous_brain_project/state/nursery_v095_WORLD_BLIND_FINAL"
export BOOTSTRAP="$HOME/projects/continuous_brain_v0.9.0/continuous_brain_project/state/nursery_v090_bootstrap"

python -m continuous_brain.parented_cli evaluate \
  --dataset "$P3_BLIND" --state-dir "$V100_BRANCH" --device cpu --seed 1601 \
  | tee results/V100_P3_FINAL_RETENTION.json

python -m continuous_brain.parented_cli evaluate \
  --dataset "$BOOTSTRAP" --state-dir "$V100_BRANCH" --device cpu --seed 17 \
  | tee results/V100_BOOTSTRAP_RETENTION.json
```

## 7. Mechanical Colab-readiness gate

```bash
python scripts/check_v100_colab_readiness.py \
  --source-checkpoint "$SOURCE_P4A" \
  --current-checkpoint "$V100_BRANCH/infant_training.pt" \
  --planning-result results/P4B_v100_FINAL_12001_12100.json \
  --p3-result results/V100_P3_FINAL_RETENTION.json \
  --retention-result results/V100_BOOTSTRAP_RETENTION.json \
  | tee results/V100_COLAB_READINESS.json
```

Only `"colab_ready": true` is the handoff to the first major whole-brain Colab development phase **at the current v0.10 dimensions**. Do not switch this persistent checkpoint to the `balanced` or `scale` profile in place. A later explicit capacity-expansion migration must preserve the learned lineage before larger dimensions are used.
