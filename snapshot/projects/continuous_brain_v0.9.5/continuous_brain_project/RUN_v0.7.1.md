# Run v0.7.1 from the successful local v0.7.0 first-conversation state

These commands assume the successful v0.7.0 project is at:

`~/projects/continuous_brain_v0.7.0/continuous_brain_project`

and v0.7.1 is extracted at:

`~/projects/continuous_brain_v0.7.1/continuous_brain_project`

## 1. Activate the known environment and install this source

```bash
cd ~/projects/continuous_brain_v0.7.1/continuous_brain_project
source ~/projects/continuous_brain_project/.venv/bin/activate
python -m pip install -e . --no-deps
```

Keep the proven local inference stack rather than upgrading it casually. v0.7.1 semantic extras require Transformers 4.x:

```bash
python -m pip install 'transformers>=4.56,<5' 'huggingface_hub>=0.34,<1.0' pillow
```

Keep the already-working Torch/torchvision pair from the first-conversation environment (the user's successful run reported Torch 2.4.1 + torchvision 0.19.1). Do not let a semantic dependency install silently replace Torch.

## 2. Clone the successful 3,500-example v0.7.0 dataset

```bash
python -m continuous_brain.semantic_cli clone-dataset \
  --source ~/projects/continuous_brain_v0.7.0/continuous_brain_project/state/semantic_dataset_v070 \
  --destination state/semantic_dataset_v071
```

The old dataset remains untouched.

## 3. Add temporal/state-grounded nursery experiences

DevelopmentRoom:

```bash
python -m continuous_brain.semantic_cli collect-development \
  --dataset state/semantic_dataset_v071 \
  --samples 500
```

AI2-THOR:

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"

python -m continuous_brain.semantic_cli collect-thor \
  --dataset state/semantic_dataset_v071 \
  --samples 1000 \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE"
```

These new records encode the previous skill/reward/success in the same memory-tail layout used by the persistent individual.

## 4. Start the local Qwen2-VL visual teacher

Use the same working llama.cpp server and GGUF pair from v0.7.0. Example:

```bash
MODEL="/mnt/e/DaveInc/OllamaModels/gguf_files/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ="/mnt/e/DaveInc/OllamaModels/gguf_files/mmproj-Qwen2-VL-2B-Instruct-f16.gguf"

~/projects/llama.cpp/build/bin/llama-server \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  --host 127.0.0.1 \
  --port 8000 \
  --alias Qwen2-VL-2B-local \
  --ctx-size 8192 \
  --parallel 1 \
  --n-gpu-layers 0 \
  --no-mmproj-offload
```

## 5. Label only the new unlabeled frames

In the Continuous Brain terminal:

```bash
python -m continuous_brain.semantic_cli label \
  --dataset state/semantic_dataset_v071 \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --teacher-model Qwen2-VL-2B-local \
  --timeout 600
```

Copied v0.7.0 labels are skipped automatically.

## 6. Cache DINO only for the new frames

```bash
python -m continuous_brain.semantic_cli cache-dino \
  --dataset state/semantic_dataset_v071 \
  --dino-model facebook/dinov3-vits16-pretrain-lvd1689m \
  --vision-device cpu
```

Existing copied DINO vectors are skipped.

## 7. Build the v0.7.1 dialogue curriculum

```bash
python -m continuous_brain.semantic_cli prepare-dialogue \
  --dataset state/semantic_dataset_v071 \
  --individual-name "Continuous Brain"
```

This normalizes self-state/care answers and adds identity, paraphrase, unknown and temporal Q/A examples.

## 8. Train v0.7.1 by continuing from the qualified v0.7.0 cortex

```bash
python -m continuous_brain.semantic_cli train \
  --dataset state/semantic_dataset_v071 \
  --lab state/semantic_lab_v071_local \
  --bootstrap-semantic \
    ~/projects/continuous_brain_v0.7.0/continuous_brain_project/state/semantic_lab_v070_local/qualified_semantic.pt \
  --updates 5000 \
  --batch 8 \
  --device cpu
```

Shared visual/language/care weights are retained. The new dialogue-intent head starts fresh and must learn/qualify.

## 9. Teacher-free v0.7.1 qualification

```bash
python -m continuous_brain.semantic_cli evaluate \
  --dataset state/semantic_dataset_v071 \
  --lab state/semantic_lab_v071_local
```

Qualification now includes held-out intent classification and an OOD question set that was not part of the training strings.

## 10. Attach only after qualification

Use the preserved first-conversation individual as the continuation source:

```bash
python scripts/upgrade_individual_v071.py \
  ~/projects/continuous_brain_v0.7.0/continuous_brain_project/state/individual_v070_local_first_conversation \
  state/individual_v071_grounded_dialogue \
  --semantic-lab state/semantic_lab_v071_local
```

The source is never overwritten.

## 11. Test grounded dialogue

```bash
python -m continuous_brain.semantic_cli talk \
  --state-dir state/individual_v071_grounded_dialogue \
  --question "Who are you?" \
  --vision-device cpu \
  --show-intent
```

```bash
python -m continuous_brain.semantic_cli talk \
  --state-dir state/individual_v071_grounded_dialogue \
  --question "What do you think about mathematics?" \
  --vision-device cpu \
  --show-intent
```

The second question should be rejected rather than mapped to a visual/state answer.

Temporal probes:

```bash
python -m continuous_brain.semantic_cli talk \
  --state-dir state/individual_v071_grounded_dialogue \
  --question "What did you do just before this?" \
  --vision-device cpu \
  --show-intent
```

```bash
python -m continuous_brain.semantic_cli chat \
  --state-dir state/individual_v071_grounded_dialogue \
  --vision-device cpu \
  --show-intent
```
