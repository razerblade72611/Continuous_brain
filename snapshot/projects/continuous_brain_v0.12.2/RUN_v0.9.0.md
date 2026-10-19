# Run Continuous Brain v0.9.0

## 1. Install beside the old experiments

```bash
cd ~/projects
mkdir -p continuous_brain_v0.9.0
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.9.0.zip -d ~/projects/continuous_brain_v0.9.0
cd ~/projects/continuous_brain_v0.9.0/continuous_brain_project
source ~/projects/continuous_brain_project/.venv/bin/activate
python -m pip install -e . --no-deps
python - <<'PY'
import continuous_brain
print(continuous_brain.__version__)
PY
```

Expected: `0.9.0`.

## 2. Create a fresh individual

```bash
python -m continuous_brain.parented_cli init \
  --state-dir state/child_v090 \
  --name "Continuous Brain"
```

This creates identity/parent state only. No old student weights are imported.

## 3A. Fastest first training: reuse old *lessons*, not old child weights

If the v0.7.1 P2 dataset exists:

```bash
python -m continuous_brain.parented_cli import-semantic \
  --source ~/projects/continuous_brain_v0.7.1/continuous_brain_project/state/semantic_dataset_v071_p2 \
  --destination state/nursery_v090_bootstrap \
  --parent-id vision.dinov3.vits_legacy
```

This imports images, DINO parent embeddings and Qwen-created lessons. It imports **zero trained child parameters**.

Train the raw 7.5M proof child:

```bash
python -m continuous_brain.parented_cli train \
  --dataset state/nursery_v090_bootstrap \
  --state-dir state/child_v090 \
  --profile proof \
  --updates 2000 \
  --batch 8 \
  --device cpu
```

Evaluate teacher-free:

```bash
python -m continuous_brain.parented_cli evaluate \
  --dataset state/nursery_v090_bootstrap \
  --state-dir state/child_v090 \
  --device cpu \
  --record-weaning
```

Do not expect one run to graduate every capability. Re-run the same train command to resume.

## 3B. Fresh DINOv3-B lived nursery

For a clean new visual/world-model dataset:

```bash
python -m continuous_brain.parented_cli collect-foraging \
  --dataset state/nursery_v090_fresh \
  --samples 200 \
  --dino-model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --vision-device cpu
```

This is intentionally small because DINO-B CPU embedding is slow. Use a GPU/cloud for thousands of transitions.

Then train/evaluate using the same commands with `state/nursery_v090_fresh`.

## 4. Watch parent authority fall

```bash
python -m continuous_brain.parented_cli status --state-dir state/child_v090
```

The `parent_authority` field changes only after repeated held-out passes.

## 5. Run the child in the toy lived environment

Parent-guided according to the current motor authority:

```bash
python -m continuous_brain.parented_cli live-foraging \
  --state-dir state/child_v090 \
  --steps 500 \
  --device cpu
```

Force a true child-only test:

```bash
python -m continuous_brain.parented_cli live-foraging \
  --state-dir state/child_v090 \
  --steps 500 \
  --device cpu \
  --child-only
```

The continuity database records lived episodes transactionally and marks whether a parent guided the action.

## 6. Child-only language

```bash
python -m continuous_brain.parented_cli chat \
  --state-dir state/child_v090 \
  --device cpu
```

No Qwen call occurs in this command.

## 7. Serious GPU profile after proof

Use `--profile balanced` from a fresh state directory. It is ~47.7M trainable child parameters. Do not jump to `scale` (~172.7M) until the proof profile shows actual weaning and world-model benefit.

## 8. Upgrade a visual parent without replacing the child

After the fast bootstrap from older DINO-S lessons, re-teach the same images with DINOv3-B:

```bash
python -m continuous_brain.parented_cli reparent-vision \
  --dataset state/nursery_v090_bootstrap \
  --model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --parent-id vision.dinov3.base \
  --vision-device cpu
```

The old target paths remain in per-record history. Resume the same `train` command; v0.9 automatically adds a new detachable parent head while retaining all child cortex/bus weights.

## 9. Add lessons from a newer language parent

Serve any multimodal OpenAI-compatible parent (for example Qwen3.5-4B/9B) and add grounded visual-language lessons without replacing the child:

```bash
python -m continuous_brain.parented_cli teach-language \
  --dataset state/nursery_v090_bootstrap \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen3.5-4B-local \
  --parent-id language.qwen3_5.4b \
  --limit 50
```

Each lesson records the parent ID. Older language lessons remain. Resume child training normally afterward.
