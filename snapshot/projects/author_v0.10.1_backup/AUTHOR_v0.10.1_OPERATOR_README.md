# AUTHOR v0.10.1 Operator Guide

## Purpose

v0.10.1 corrects the physical-learning failure exposed by v0.10.0: the agents often knew the interaction mechanic but were being trained to approach object centers or were being scored on impossible episode setup. The new release keeps Qwen responsible for semantic intent while the body privately resolves and executes simulator-valid interaction poses.

The normal-life boundary is:

```text
Qwen/user chooses semantic target
        ↓
grounded objectId
        ↓
GetReachablePositions (current component)
        ↓
GetInteractablePoses(selected target)
        ↓
connected/path-cost pose selection
        ↓
physical walk + yaw + camera horizon
        ↓
verified interaction
        ↓
Qwen receives public outcome
```

Coordinates and simulator IDs used for motor execution are never inserted into Qwen's ordinary language prompt.

## Before upgrading: preserve v0.10.0

Do **not** use `--reset-memory` during this upgrade.

In WSL:

```bash
cd ~/projects
STAMP=$(date +%Y%m%d-%H%M%S)
cp -a author "author_v0100_backup_${STAMP}"
echo "Backup: ~/projects/author_v0100_backup_${STAMP}"
```

Keep that terminal open so `$STAMP` remains available, or note the printed backup path.

Unpack the v0.10.1 ZIP somewhere temporary and replace code while preserving the live `state/` directory:

```bash
rm -rf /tmp/author_v0101_unpack
mkdir -p /tmp/author_v0101_unpack

unzip -q /path/to/author_phase9_v0.10.1_interactable_pose_curriculum.zip \
  -d /tmp/author_v0101_unpack

rsync -a --delete \
  --exclude 'state/' \
  /tmp/author_v0101_unpack/author/ \
  ~/projects/author/
```

If your current tree has no `state/` yet, that exclusion is harmless.

## Reinstall and run the offline checks

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

python -m pip install -e ".[dev]"
python -m pytest -q
```

Expected for this packaged release:

```text
141 passed
```

Then verify the trainer parses the new level/curriculum options without launching Unity:

```bash
EPISODES=10 WORKERS=1 LEVEL=0 \
./scripts/run_training_wsl.sh \
  --curriculum door \
  --dry-run
```

The plan should say:

```text
version: 0.10.1
trainer: staged_interactable_pose_curriculum
level: 0
llm_calls: 0
invalid_setup_policy: skip_not_failure
legacy_checkpoint_policy: preserve_and_migrate
```

`open_cross` is still accepted as an alias for `door` so old commands do not suddenly break.

## Migrate the old A/B skill checkpoints

After the full backup above:

```bash
python scripts/inspect_skills.py
```

Old v0.10 statistics should appear under `legacy_v010`; staged v0.10.1 buckets will initially be empty.

Persist the lossless migration:

```bash
python scripts/inspect_skills.py --migrate
```

Then inspect again:

```bash
python scripts/inspect_skills.py --object-details
```

Important: old `open`, `cross_door`, `pickup`, `place`, `toggle`, and `peer_meet` remain usable as legacy mechanical evidence where appropriate. Old `navigate` is **not** treated as v0.10.1 approach competence because that statistic was contaminated by incompatible and impossible episodes.

## Live pose preflight — do this before real training

This is the most important new preflight because custom Unity objects must actually support the pose query we are relying on.

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_interaction_poses.py \
  --scene FloorPlan1Discovery_physics \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

Do not begin a long curriculum unless the JSON ends with:

```json
"ok": true
```

At minimum it should report reachable cells and one or more `Door` interaction poses. If Door pose count is zero, fix the custom Door metadata/collider/scene issue first. More episodes cannot repair a target the simulator says is non-interactable.

## What the levels now mean

```text
LEVEL 0  direct mechanics
         trainer teleports to an exact valid interaction pose

LEVEL 1  near approach
         roughly 0.45–1.05 m from a valid pose

LEVEL 2  local-room / connected-room approach
         room doors are administratively closed

LEVEL 3  open multi-room approach
         connectors are administratively opened

LEVEL 4  door-aware/full routing
         manipulation setups are accepted only if one reachable closed Door
         genuinely blocks the already-selected target

LEVEL 5  interference
         same blocked-route requirement plus a nearby peer nuisance; the trainer
         does not deliberately place the peer directly on the required target pose
```

Setup actions are trainer/admin operations. They are not scored and are not exported as learned demonstrations.

## Door curriculum first

### Level 0 — isolate the hand/door mechanic

```bash
EPISODES=100 WORKERS=1 LEVEL=0 \
./scripts/run_training_wsl.sh --curriculum door

python scripts/inspect_skills.py
```

Look specifically under each agent's `by_level -> "0"` section.

Gate before moving on:

```text
door_open: at least ~95% success
```

If Level 0 cannot open a Door from a simulator-returned interactable pose, stop. That is a scene/action integration problem, not a navigation-learning problem.

### Level 1 — near approach

```bash
EPISODES=150 WORKERS=1 LEVEL=1 \
./scripts/run_training_wsl.sh --curriculum door

python scripts/inspect_skills.py
```

Desired Level-1 gate per agent:

```text
door_approach  >= 90%
door_open      >= 95%
door_cross     >= 90%
```

### Level 2 — local-room starts

```bash
EPISODES=200 WORKERS=1 LEVEL=2 \
./scripts/run_training_wsl.sh --curriculum door

python scripts/inspect_skills.py
```

Suggested gate:

```text
door_approach  >= 88%
door_open      >= 95%
door_cross     >= 90%
```

### Level 3 — open multi-room starts

```bash
EPISODES=250 WORKERS=1 LEVEL=3 \
./scripts/run_training_wsl.sh --curriculum door

python scripts/inspect_skills.py
```

Suggested gate:

```text
door_approach  >= 85%
door_open      >= 93%
door_cross     >= 88%
```

### Level 4 — full door approach/cross

```bash
EPISODES=300 WORKERS=1 LEVEL=4 \
./scripts/run_training_wsl.sh --curriculum door

python scripts/inspect_skills.py
```

Suggested gate before the first live Escape test:

```text
door_approach  >= 85%
door_open      >= 93%
door_cross     >= 88%
```

The thresholds are operational gates, not hardcoded claims that the model is ready. Use the per-level data plus the current run summary.

## Inspect setup skips after every run

```bash
python -m json.tool state/metrics/training_summary.json

tail -n 5 state/metrics/training_history.jsonl
```

The summary contains `skipped_setups` and `setup_skip_reasons`.

A few skips are normal. A dominant repeated reason is a diagnostic signal. **Do not compensate for a broken setup generator by raising `EPISODES`.**

If the trainer aborts with:

```text
Only X/Y valid episodes could be generated ...
```

read the skip reasons, fix the topology/object issue, or temporarily return to the previous level.

## First live `/a Escape.` test

### Windows — start Qwen-VL

Use the same llama.cpp server, with the image-token floor explicitly enabled because llama.cpp warned that Qwen-VL grounding needs it:

```powershell
cd E:\DaveInc\llama.cpp

.\llama-server.exe `
  -m "E:\DaveInc\OllamaModels\gguf_files\Qwen2-VL-2B-Instruct-Q4_K_M.gguf" `
  --mmproj "E:\DaveInc\OllamaModels\gguf_files\mmproj-Qwen2-VL-2B-Instruct-f16.gguf" `
  --alias qwen2-vl-2b `
  -ngl 999 `
  -c 16384 `
  --image-min-tokens 1024 `
  --host 0.0.0.0 `
  --port 8080
```

### WSL — start AUTHOR

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')

export AGENT_LLM_BASE_URL="http://$WINDOWS_HOST:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"

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
  --roam-decisions 200 \
  --max-steps 1500 \
  --peer-text-every 3 \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

Give only A:

```text
/a Escape.
```

Do not give B a task for this first acceptance run.

What we want to see is materially equivalent to:

```text
[A-task ...] focus="Door"
[A motor-handoff] door_approach
[A motor-handoff] OpenObject ... verified isOpen=true
[A motor-handoff] door_cross ... verified crossed=true
[A task-complete] Escape.
```

The exact logging wording may differ. The non-negotiable evidence is:

1. a real selected Door,
2. physical approach primitives,
3. a real `OpenObject` event,
4. authoritative `isOpen=true`,
5. physical movement across the doorway,
6. the task verifier accepting Escape only after the world changed.

## Then train manipulation

Do not jump straight to a giant mixed run. Establish mechanics first:

```bash
EPISODES=450 WORKERS=1 LEVEL=0 \
./scripts/run_training_wsl.sh --curriculum pickup,place,toggle

python scripts/inspect_skills.py --object-details
```

Then:

```bash
EPISODES=600 WORKERS=1 LEVEL=1 \
./scripts/run_training_wsl.sh --curriculum pickup,place,toggle

EPISODES=800 WORKERS=1 LEVEL=2 \
./scripts/run_training_wsl.sh --curriculum pickup,place,toggle

EPISODES=1000 WORKERS=1 LEVEL=3 \
./scripts/run_training_wsl.sh --curriculum pickup,place,toggle
```

Level 4 is where `NavigateTo(selected target)` is deliberately trained through a genuinely blocking closed Door:

```bash
EPISODES=1200 WORKERS=1 LEVEL=4 \
./scripts/run_training_wsl.sh --curriculum pickup,place,toggle
```

Only add Level 5 once Level 4 is healthy:

```bash
EPISODES=1200 WORKERS=1 LEVEL=5 \
./scripts/run_training_wsl.sh --curriculum pickup,place,toggle
```

## Why level-4 manipulation matters

A target may be semantically selected while it is in another currently disconnected room. v0.10.1 can use an accessible closed Door as controller-private route infrastructure:

```text
Qwen: "pick up that apple"
          ↓
Apple remains the semantic target
          ↓
no Apple interaction pose in current component
          ↓
find reachable closed Door
          ↓
reach Door pose
OpenObject
cross
re-query reachable component
          ↓
find Apple interaction pose
walk / rotate / look
PickupObject
          ↓
verify inventory contains selected Apple
```

Intermediate Door outcomes are recorded as `door_approach`, `door_open`, and `door_cross` auxiliary experience. The semantic target is never silently changed from Apple to Door.

## Verify training data quality

```bash
wc -l state/replay/embodied_replay.jsonl
wc -l state/datasets/embodied_sft.jsonl

tail -n 1 state/datasets/embodied_sft.jsonl | python -m json.tool
```

`embodied_replay.jsonl` may contain successful and failed verified attempts because failures are useful diagnostics/replay material.

`embodied_sft.jsonl` contains only successful verified trajectories. Treat it as **motor-demonstration data**, not as proof that Qwen has been fine-tuned. v0.10.1 does not change Qwen's weights.

## Peer-text verification

After a task that explicitly requires texting:

```bash
tail -n 20 state/messages/peer_texts.jsonl
```

Real deliveries now contain a unique `message_id`. `NO_MESSAGE`, `[NO_MESSAGE]`, `null`, and malformed JSON-like model output are rejected by the transport and are not persisted as peer messages.

## Online learning during real sessions

Verified live motor handoffs update the executing agent's own staged checkpoint. A and B remain separate. Intermediate Door route stages are also recorded when a target approach has to cross a closed Door.

This is still not an end-to-end neural motor policy. v0.10.1 is a verified motor/control curriculum with persistent competence statistics and a data-generation path toward later learned policies.

## Recovery / rollback

If the upgrade has a code regression, stop the session and restore the whole backup you made before installation:

```bash
cd ~/projects
mv author "author_v0101_failed_$(date +%Y%m%d-%H%M%S)"
cp -a /path/to/author_v0100_backup_TIMESTAMP author
```

Do not delete the failed v0.10.1 tree until you have copied out any useful new logs/state for comparison.

## Recommended acceptance order

```text
1. pytest
2. trainer --dry-run
3. inspect + migrate old checkpoints
4. GetInteractablePoses live preflight
5. Door L0
6. Door L1
7. Door L2
8. Door L3
9. Door L4
10. /a Escape.
11. Manipulation L0-L3
12. Manipulation L4 blocked-route curriculum
13. optional L5 interference
14. mixed curriculum only after staged metrics are healthy
```

That sequence is intentionally conservative: it makes failures diagnosable instead of burying them inside thousands of mixed episodes.
