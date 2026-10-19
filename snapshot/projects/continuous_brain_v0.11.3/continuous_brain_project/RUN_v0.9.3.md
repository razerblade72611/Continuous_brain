# v0.9.3 continuation from the existing child

Use the same child and existing mixed nursery. Do not restart the lineage.

```bash
export CHILD="$HOME/projects/continuous_brain_v0.9.0/continuous_brain_project/state/child_v090"
export MIXED="$HOME/projects/continuous_brain_v0.9.2/continuous_brain_project/state/nursery_v092_mixed_001"
export FORAGING="$HOME/projects/continuous_brain_v0.9.2/continuous_brain_project/state/nursery_v092_foraging_001"
```

Back up the current 2,000-update checkpoint, then train 1,000 updates with 50% lived/world-valid minibatch sampling:

```bash
cp "$CHILD/infant_training.pt" "$CHILD/infant_training.pre_v093.pt"
python -m continuous_brain.parented_cli train \
  --dataset "$MIXED" \
  --state-dir "$CHILD" \
  --profile proof \
  --updates 1000 \
  --batch 8 \
  --world-fraction 0.50 \
  --device cpu
```

The first v0.9.3 resume should report `world_dynamics_migrated: true` and `optimizer_reinitialized: true` once. The child should continue from update 2000 to 3000.

Evaluate world dynamics on the lived-only nursery so the held-out split has 50 world-valid records:

```bash
python -m continuous_brain.parented_cli evaluate \
  --dataset "$FORAGING" \
  --state-dir "$CHILD" \
  --device cpu \
  --seed 41
```

Then evaluate the bootstrap nursery again for retention.
