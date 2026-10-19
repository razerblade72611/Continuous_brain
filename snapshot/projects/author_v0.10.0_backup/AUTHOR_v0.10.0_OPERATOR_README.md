# AUTHOR v0.10.0 Operator README

## Purpose

v0.10.0 rewrites accelerated training so A and B learn reusable body competence, not just a route table. Qwen remains the semantic decision-maker. The trainer does not decide what an agent wants during normal autonomous life.

### Memory / learning layers

1. **LLM context** — short-lived prompt/context state.
2. **Per-agent episodic memory** — `state/memory/a|b/memory.jsonl`.
3. **Shared persistent map/navigation policy** — map + `navigation_skill.json`.
4. **Per-agent embodied skill checkpoint** — `state/checkpoints/a|b/embodied_skills.json`.
5. **Embodied replay** — `state/replay/embodied_replay.jsonl`.
6. **Optional future model-training dataset** — `state/datasets/embodied_sft.jsonl`.
7. **Qwen weights** — unchanged in v0.10.0.

## Upgrade from v0.9.3 without losing memory

```bash
cd ~/projects
mv author author_v0.9.3_backup
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.0_embodied_skill_curriculum.zip -d ~/projects
cp -a ~/projects/author_v0.9.3_backup/state ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected release regression count: `132 passed`.

No Unity rebuild is required for v0.10.0 if the v0.9.3 Discovery smoke test already passes. v0.10.0 is Python-side training/runtime skill logic.

## Verify the current Discovery player once

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_discovery_world.py \
  --scene FloorPlan1Discovery_physics \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

You already want nonzero north/east extension and PASS before training.

## Recommended first curriculum

Start by teaching the exact skill currently blocking escape:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

EPISODES=200 WORKERS=1 \
./scripts/run_training_wsl.sh --curriculum open_cross
```

Then inspect A/B competence:

```bash
python scripts/inspect_skills.py
```

You want `open`, `cross_door`, `face_target`, and `navigate` to have multiple attempts with a healthy success rate.

Next teach manipulation:

```bash
EPISODES=400 WORKERS=1 \
./scripts/run_training_wsl.sh --curriculum pickup,place,toggle
```

Then run the full mixed curriculum:

```bash
EPISODES=2000 WORKERS=2 ./scripts/run_training_wsl.sh
```

If two Unity workers are unstable on the machine, use `WORKERS=1`; no learning is lost.

## Continue training vs reset

Continue from the current lineage:

```bash
EPISODES=2000 WORKERS=1 ./scripts/run_training_wsl.sh
```

Use `RESET_MEMORY=1` only when intentionally starting a completely new learning lineage:

```bash
RESET_MEMORY=1 EPISODES=1000 WORKERS=1 ./scripts/run_training_wsl.sh
```

That deletes the selected `STATE_DIR`, including episodic memory and learned skills.

## What the trainer does

### Navigate

Teacher computes a legal shortest-grid route, executes it rapidly, updates the existing navigation Q table/map, and records whether Agent A or B could reliably reach the target.

### Open + cross

Teacher selects a real `Door`, approaches it, faces it, executes `OpenObject`, then executes the v0.10 `CrossThreshold` body skill. `CrossThreshold` computes a reachable point immediately beyond **that same selected door** and walks through it. It does not choose a different door or goal.

### Pickup

Teacher chooses a real pickupable simulator object, navigates/faces/picks it, verifies inventory state, then cleans up the object so later episodes remain usable.

### Place

Teacher prefers an object with a known parent receptacle, picks it up, navigates back to that receptacle, executes `PutObject`, and verifies parent-receptacle state.

### Toggle

Teacher navigates/faces/toggles a toggleable object and restores its original state after the measured episode.

### Peer meet

One body navigates to the other body's real position and verifies physical encounter distance. No private peer memory is copied.

## Runtime effect

Normal sessions load the checkpoint. A trained task sequence can look like:

```text
LLM A sees/selects: "closed door to the left"
        ↓
trained body: NavigateTo(the already-selected Door)
        ↓
trained body: FaceTarget(the same Door)
        ↓
OpenObject
        ↓
trained body: CrossThreshold(the same open Door)
        ↓
ExitRegion verifier sees physical region change
        ↓
task complete
```

For pickup:

```text
LLM B selects: "red apple"
        ↓
trained NavigateTo / FaceTarget if needed
        ↓
PickupObject
        ↓
inventory verifier
```

Training never exposes hidden object IDs to Qwen. The body controller uses IDs internally only after semantic grounding has already associated the model's phrase with the visible target.

## Training outputs

```text
state/
├── memory/
│   ├── a/memory.jsonl
│   └── b/memory.jsonl
├── maps/
│   └── FloorPlan1Discovery_physics.json
├── checkpoints/
│   ├── navigation_skill.json
│   ├── a/embodied_skills.json
│   └── b/embodied_skills.json
├── replay/
│   ├── navigation_replay.jsonl
│   └── embodied_replay.jsonl
├── datasets/
│   └── embodied_sft.jsonl
└── metrics/
    ├── training_summary.json
    └── training_history.jsonl
```

## Later: actual Qwen weight learning

`embodied_sft.jsonl` is intentionally only a dataset export in this release. It contains successful generalized action sequences by skill/object type and strips simulator object IDs from the training example.

A later phase can fine-tune a Qwen LoRA from this material (ideally combined with selected image/action trajectories). That is separate from v0.10's fast body-skill learning. The quantized GGUF used by llama.cpp should remain the inference artifact; actual LoRA/SFT preparation should use an appropriate trainable source-model checkpoint and then return an adapter/converted model to the inference stack.

## Recommended normal session after training

Set the WSL LLM endpoint first:

```bash
WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')
export AGENT_LLM_BASE_URL="http://$WINDOWS_HOST:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"
curl -sS "$AGENT_LLM_BASE_URL/models" | python -m json.tool
```

Then:

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1Discovery_physics \
  --session \
  --agents 2 \
  --llm \
  --vision \
  --interact \
  --dashboard \
  --vision-every 1 \
  --step-delay 0 \
  --roam-decisions 500 \
  --max-steps 3000 \
  --peer-text-every 3 \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

A useful first test after `open_cross` training is `/a Escape.` or `/both Help each other escape.` and watch for `motor-handoff` plus `CrossThreshold` before the verified task-complete event.
