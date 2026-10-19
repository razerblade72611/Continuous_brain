# v0.9.4 continuation from the existing 3,000-update child

Do not restart the child.

## Paths

```bash
export CHILD="$HOME/projects/continuous_brain_v0.9.0/continuous_brain_project/state/child_v090"
export BOOTSTRAP="$HOME/projects/continuous_brain_v0.9.0/continuous_brain_project/state/nursery_v090_bootstrap"
export OLD_FORAGING="$HOME/projects/continuous_brain_v0.9.2/continuous_brain_project/state/nursery_v092_foraging_001"
```

Back up the 3,000-update checkpoint:

```bash
cp "$CHILD/infant_training.pt" "$CHILD/infant_training.pre_v094.pt"
```

## Collect structured lived transitions

v0.9.4 stores exact `next_context` targets and uses more exploratory action diversity.

```bash
python -m continuous_brain.parented_cli collect-foraging \
  --dataset state/nursery_v094_foraging_001 \
  --samples 1000 \
  --seed 73 \
  --exploration 0.30 \
  --dino-model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --vision-device cpu
```

## Merge developmental replay

Reuse both the old upbringing and the earlier lived transitions:

```bash
python -m continuous_brain.parented_cli merge-nurseries \
  --sources \
    "$BOOTSTRAP" \
    "$OLD_FORAGING" \
    state/nursery_v094_foraging_001 \
  --destination state/nursery_v094_mixed_001
```

Expected total: 5,000 lessons (3,500 bootstrap + 500 old lived + 1,000 new structured lived).

## Continue the same child

```bash
python -m continuous_brain.parented_cli train \
  --dataset state/nursery_v094_mixed_001 \
  --state-dir "$CHILD" \
  --profile proof \
  --updates 1000 \
  --batch 8 \
  --world-fraction 0.65 \
  --device cpu
```

First v0.9.4 resume should report:
- `optimizer_reinitialized: true`
- `world_dynamics_migrated: true`
- `world_target_cortex_migrated: true`
- `spatial_cortex_migrated: true`
- `world_context_head_migrated: true`

The child continues from update 3000 to 4000.

## Strict world evaluation

```bash
python -m continuous_brain.parented_cli evaluate \
  --dataset state/nursery_v094_foraging_001 \
  --state-dir "$CHILD" \
  --device cpu \
  --seed 73
```

Look for:
- `evidence_eligible.world: true`
- `world_spatial_passed`
- `world_structured_state_passed`
- `world_model_passed`

A true P3 proof requires `world_model_passed: true`.

## Retention check

```bash
python -m continuous_brain.parented_cli evaluate \
  --dataset "$BOOTSTRAP" \
  --state-dir "$CHILD" \
  --device cpu \
  --seed 17
```

Do not record additional weaning yet. First prove that the new dorsal/world-model learning coexists with the earlier ventral/language upbringing.
