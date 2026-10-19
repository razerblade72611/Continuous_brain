# Run Continuous Brain v0.7.0 — First Conversation

This release does not throw away v0.6.2.  Train the semantic cortex in a separate
lab first, qualify it without Qwen, then attach it to a continuation copy of the
existing individual.

## 1. Extract and test

```bash
mkdir -p ~/projects/continuous_brain_v0.7.0
unzip -n /mnt/c/Users/davei/Downloads/continuous_brain_v0.7.0.zip \
  -d ~/projects/continuous_brain_v0.7.0
cd ~/projects/continuous_brain_v0.7.0/continuous_brain_project
source ~/projects/continuous_brain_v0.6.2/continuous_brain_project/.venv/bin/activate
python -m pip install -e . --no-deps
python -m pytest -q
```

Install the optional semantic/vision dependencies only if they are not already in
the venv:

```bash
python -m pip install 'transformers>=4.56' 'pillow>=10'
```

DINOv3 may require accepting Meta/Hugging Face access terms and authenticating.
Do not replace the working PyTorch build merely to install Transformers.

## 2. Collect a small mechanics nursery first

```bash
python -m continuous_brain.semantic_cli collect-development \
  --dataset state/semantic_dataset_v070 \
  --samples 500
```

This does not load or modify the persistent individual.

## 3. Collect the real 3D semantic nursery

Use the working Discovery build.  Start with one scene, then add several scenes as
the pipeline proves stable:

```bash
export THOR_EXECUTABLE="$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64"

python -m continuous_brain.semantic_cli collect-thor \
  --dataset state/semantic_dataset_v070 \
  --samples 3000 \
  --scene FloorPlan1Discovery_physics \
  --thor-executable "$THOR_EXECUTABLE"
```

For broader grounding, a later run can use a comma-separated set of scenes.  These
frames are training-environment observations, not automatically autobiographical
memories of the persistent individual.

## 4. Run Qwen3.8-27B as a frozen teacher

Qwen3.8-27B is far larger than an 8 GB GPU can practically serve at full precision.
For speed, run it on a cloud GPU and expose an OpenAI-compatible vLLM/SGLang endpoint
through a private tunnel.  Do **not** fine-tune Qwen.

Typical cloud command:

```bash
vllm serve Qwen/Qwen3.8-27B --host 127.0.0.1 --port 8000
```

Then label locally through the tunnel:

```bash
python -m continuous_brain.semantic_cli label \
  --dataset state/semantic_dataset_v070 \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --teacher-model Qwen/Qwen3.8-27B
```

An alternate open-weight/abliterated Qwen checkpoint can be used by changing only
`--teacher-model`, provided the serving endpoint exposes the same VLM chat API.
For the first release, the official model is the reproducibility baseline; caring
behavior is trained explicitly rather than assumed from or removed with refusal
behavior.

## 5. Cache frozen DINOv3 features

```bash
python -m continuous_brain.semantic_cli cache-dino \
  --dataset state/semantic_dataset_v070 \
  --dino-model facebook/dinov3-vits16-pretrain-lvd1689m \
  --vision-device auto
```

DINO is inference-only.  No DINO parameter receives a gradient.  Cache once so
student training does not repeatedly run the foundation model.

## 6. Train the student's semantic/communication cortex

```bash
python -m continuous_brain.semantic_cli train \
  --dataset state/semantic_dataset_v070 \
  --lab state/semantic_lab_v070 \
  --updates 10000 \
  --batch 8 \
  --device auto
```

`--updates` is additional.  Re-running resumes model weights, optimizer and update
count.  Ctrl+C/SIGTERM saves.  Increase to 25k/50k after inspecting validation.
This is the training that creates the individual's future communication cortex;
Qwen and DINO remain frozen.

## 7. Qualify without Qwen

```bash
python -m continuous_brain.semantic_cli evaluate \
  --dataset state/semantic_dataset_v070 \
  --lab state/semantic_lab_v070
```

A passing run writes:

```text
state/semantic_lab_v070/qualified_semantic.pt
```

The evaluator does not call Qwen.  It tests held-out Q/A token accuracy, concepts,
care targets, visual-code compatibility and free-running generated answers.  Do not
attach a failed model by editing the report.

## 8. Attach to the existing individual without resetting it

Keep the v0.6.2 source stopped.  Use a new destination:

```bash
python scripts/upgrade_individual_v070.py \
  state/individual_v062 \
  state/individual_v070 \
  --semantic-lab state/semantic_lab_v070
```

The report should show the same individual UUID, decision count and memory count.
The executive, motor, optimizer, target network, workspace, replay, intentions,
world state and RNG are preserved.  The source remains unchanged.

## 9. First teacher-free conversation

The current individual lives in DevelopmentRoom, whose state can be reconstructed
exactly.  Ask one question:

```bash
python -m continuous_brain.semantic_cli talk \
  --state-dir state/individual_v070 \
  --question "What do you see?"
```

Or enter the interactive shell:

```bash
python -m continuous_brain.semantic_cli chat \
  --state-dir state/individual_v070
```

At this point Qwen is not running.  DINO supplies frozen vision and the attached
student generates the answer from its learned weights plus the individual's current
permitted self/goal/memory context.

## 10. What to train next

C1 is reached when teacher-free answers are short, grounded and stable across
held-out 3D frames and paraphrased questions.  Then proceed in this order:

1. more scenes/objects/lighting/viewpoints;
2. questions about current goal/action and recent memory;
3. uncertainty calibration — teach "I don't know" rather than hallucination;
4. social/care scenarios with stop/help/consent/harm consequences;
5. feed the learned semantic code into the recurrent executive through a tested
   compatibility migration;
6. longer world-model rollouts and counterfactual training;
7. only then increase reliance on imagination.

## Do we need a 3D workspace?

For the first plumbing test, no.  For robust grounded intelligence, yes.  Use
AI2-THOR now rather than building another simulator.  It already gives camera
images, depth, objects, interaction and physics.  A custom richer world can come
later if THOR becomes the bottleneck.

## Can we train through imagination?

Yes, but imagination must be downstream of reality.  Current v0.6/v0.7 world-model
rollouts are short and should not be treated as lived memories.  Once prediction
error is calibrated, generate lower-weight counterfactual training examples from
real starting states, keep explicit provenance, and compare them against later real
outcomes.  Never let imagined episodes silently become autobiographical facts.
