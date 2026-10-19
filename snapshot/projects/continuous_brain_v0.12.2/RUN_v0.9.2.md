# Run Continuous Brain v0.9.2

v0.9.2 continues the same v0.9 child. Do not reset child weights.

## 1. Install code

```bash
cd ~/projects
mkdir -p ~/projects/continuous_brain_v0.9.2
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.9.2.zip -d ~/projects/continuous_brain_v0.9.2
cd ~/projects/continuous_brain_v0.9.2/continuous_brain_project
source ~/projects/continuous_brain_project/.venv/bin/activate
python -m pip install -e . --no-deps
```

## 2. Point to the existing child and bootstrap nursery

```bash
export CHILD="$HOME/projects/continuous_brain_v0.9.0/continuous_brain_project/state/child_v090"
export BOOTSTRAP="$HOME/projects/continuous_brain_v0.9.0/continuous_brain_project/state/nursery_v090_bootstrap"
```

Correct nursery backup filename:

```bash
cp "$BOOTSTRAP/lessons.jsonl" "$BOOTSTRAP/lessons.pre_v092.jsonl"
cp "$BOOTSTRAP/manifest.json" "$BOOTSTRAP/manifest.pre_v092.json"
cp "$CHILD/infant_training.pt" "$CHILD/infant_training.pre_v092.pt"
```

## 3. Record the first partial weaning using three different held-out splits

```bash
for SEED in 17 23 31; do
  python -m continuous_brain.parented_cli evaluate \
    --dataset "$BOOTSTRAP" \
    --state-dir "$CHILD" \
    --device cpu \
    --seed "$SEED" \
    --record-weaning
done

python -m continuous_brain.parented_cli status --state-dir "$CHILD"
```

Do not keep repeatedly evaluating this same nursery to force full graduation. One 20% authority reduction is enough for this proof stage.

## 4. Collect a fresh lived transition nursery

```bash
python -m continuous_brain.parented_cli collect-foraging \
  --dataset state/nursery_v092_foraging_001 \
  --samples 500 \
  --seed 41 \
  --dino-model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --vision-device cpu
```

Each record now separates:
- parent motor target (`action`)
- actual executed action (`executed_action`)
- recurrent previous action (`previous_action`)
- next image / reward / success

## 5. Merge old lessons with new lived transitions

```bash
python -m continuous_brain.parented_cli merge-nurseries \
  --sources "$BOOTSTRAP" state/nursery_v092_foraging_001 \
  --destination state/nursery_v092_mixed_001
```

Expected records: 4000.

## 6. Continue the same child on mixed developmental replay

```bash
python -m continuous_brain.parented_cli train \
  --dataset state/nursery_v092_mixed_001 \
  --state-dir "$CHILD" \
  --profile proof \
  --updates 1000 \
  --batch 8 \
  --device cpu
```

Watch `world=` and `temporal=` become nonzero and generally decrease.

## 7. Evaluate mixed competence and held-out world prediction

```bash
python -m continuous_brain.parented_cli evaluate \
  --dataset state/nursery_v092_mixed_001 \
  --state-dir "$CHILD" \
  --device cpu \
  --seed 41
```

Important v0.9.2 metrics:
- `motor_accuracy` no longer has target leakage
- `world_next_bus_error`
- `world_no_change_baseline_error`
- `world_wrong_action_error`
- `world_relative_improvement`
- `world_action_advantage`
- `world_reward_mae`
- `world_success_accuracy`
- `temporal_next_bus_error`
- `world_model_passed`

A preliminary world-model proof requires:
1. learned next-state error < 98% of no-change baseline error,
2. correct executed action predicts better than a wrong-action control,
3. success accuracy >= 0.70.

## 8. Re-evaluate the old bootstrap nursery for catastrophic forgetting

```bash
python -m continuous_brain.parented_cli evaluate \
  --dataset "$BOOTSTRAP" \
  --state-dir "$CHILD" \
  --device cpu \
  --seed 17
```

The visual active-parent and historical-parent retention scores should remain strong after embodied learning.
