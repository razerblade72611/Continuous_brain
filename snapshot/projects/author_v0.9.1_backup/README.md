# AI2-THOR Agent v0.9.1 — Verified Task Execution

v0.9.1 is a behavioral-control and Unity-build reliability update on top of the v0.9.0 Two-Room Discovery World.

## New in v0.9.1

- **Verified completion only:** an LLM `task_complete=true` claim no longer ends an unsupported task. Recognized manipulation, text, and Discovery-world exit tasks require real evidence.
- **Text tasks require delivery:** `Text B ...` / `message each other` is converted into a user-task mechanical contract. `say` does not count; the planner retries if it fails to emit `TextPeer`, and completion requires a real delivered message event.
- **Approach → interact motor handoff:** while an explicit user manipulation task is active, if the model has already grounded/focused the requested target and the requested action is mechanically available, preparatory `Observe`/`Approach`/`Face`/movement transitions to the requested native AI2-THOR interaction. Autonomous mode is unchanged.
- **Composite pickup/place staging:** `Pick up X and put it on/in Y` mechanically stages pickup first, then placement once the hand actually holds X.
- **Physical escape verification:** in `FloorPlan1Discovery_physics`, `Escape`/`Exit` cannot complete until the body crosses a real doorway threshold into a different connected region. If the model itself has selected a reachable door, the motor layer can finish the `Approach → OpenObject` transition without selecting a route for it.
- **Unity compiler hotfix:** `DiscoveryWorldBuilder.cs` now fully qualifies `UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene`, avoiding the AI2-THOR `SceneManager` name collision that produced CS0117.
- Task-contract payloads contain only facts derived from the active user task; no object IDs, coordinates, hidden catalog, reward, or autonomous preference is injected.

### Upgrade while preserving state

```bash
cd ~/projects
mv author author_v0.9.0_backup
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.9.1_verified_task_execution.zip -d ~/projects
cp -a ~/projects/author_v0.9.0_backup/state ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected release suite: `117 passed`.

## Discovery World inherited from v0.9.0

A Unity patch kit in `unity_world/` creates `FloorPlan1Discovery_physics` without overwriting stock FloorPlan1. The two existing kitchen doors become real openable AI2-THOR objects. One leads to an Archive/Signal room full of readable fragments and manipulable artifacts; the other leads to an Experiment/Workshop room with causal-state language, tools/devices, and manipulable objects. Both extensions are part of the baked NavMesh and are entered physically—never by teleport.

The content is deliberately interesting but behavior-neutral: there is no hidden curiosity reward, no automatic goal to enter, and no injected preference for either room.

### Apply the world patch

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python unity_world/apply_discovery_world_patch.py ~/projects/ai2thor-humanoid
python unity_world/verify_discovery_world_patch.py ~/projects/ai2thor-humanoid
```

See `unity_world/README.md` and `scripts/build_discovery_world.ps1` for the Windows Unity 2020.3.48 generation/build step.

### Normal Discovery session

```bash
cd ~/projects/author
./scripts/run_discovery_session_wsl.sh
```

### Accelerated training in the two-room world

```bash
RESET_MEMORY=1 EPISODES=1000 WORKERS=1 ./scripts/run_training_wsl.sh --headless
```

v0.8.0 builds on v0.7.0 accelerated learning and v0.6.1 humanoid rendering.

## New in v0.8.0

- **TextPeer**: Agent A and B can explicitly text each other at any distance. Text carries only the chosen message; no private camera, memory, prompt, task, focus, or hidden reasoning is copied.
- Persistent text transcript at `state/messages/peer_texts.jsonl` and sent/received text in each agent's safe episodic memory.
- Grounded manipulation expanded and verified: `PickupObject`, `PutObject`, `DropObject`, `OpenObject`, `CloseObject`, `ToggleObjectOn`, and `ToggleObjectOff`.
- The LLM never receives simulator object IDs. Semantic references are translated only when the target is currently rendered/visible (or is the model's own previously grounded focus), has the required affordance, and is within interaction range.
- Every manipulation action is checked against post-action AI2-THOR metadata before it is recorded as successful.
- `scripts/smoke_test_interactions.py` provides a live pickup/drop/open/toggle integration check against the real Unity player.
- Training coverage reporting is clamped to 100% (fixes the v0.7 reachable-cell accounting edge case).

AI2-THOR itself supports pickup/put/drop for pickupable/receptacle objects and open/close/toggle actions for objects whose metadata advertises those affordances. v0.8 uses those native actions rather than faking manipulation.

### Social action schema

```json
{"kind":"TextPeer","target":"B","params":{"text":"I opened the fridge."}}
```

Texting is optional and does not create a cooperation preference. Incoming messages appear in `social.texts_since_last_turn` on the recipient's next turn.

### Manipulation examples

```json
{"kind":"PickupObject","target":"apple","params":{}}
{"kind":"PutObject","target":"counter","params":{}}
{"kind":"DropObject","target":null,"params":{}}
{"kind":"OpenObject","target":"fridge","params":{}}
{"kind":"ToggleObjectOn","target":"lamp","params":{}}
```

For `PutObject`, the semantic target is the destination receptacle; the held object is inferred from the body's actual hand state. `DropObject` requires that the body is currently holding something.

### Live manipulation smoke test

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_interactions.py \
  --scene FloorPlan1 \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
```

## About connected rooms / a Discovery Lab

v0.8.0 deliberately does **not** fake room-to-room travel by teleporting an agent after an arbitrary cabinet/fridge interaction. Stock iTHOR scenes are individual room scenes. The agents can now genuinely open any native `openable` object they perceive, and the controller can recompute navigation after physical state changes, but a continuous second room requires an edited/custom Unity scene with a real doorway and traversable geometry. The release keeps the interaction layer ready for that custom scene rather than pretending a stock object is a door.


# v0.7.0 — Accelerated Learning + Persistent Skills

v0.7.0 adds a separate fast-training path for low-level embodied learning while preserving the neutral normal-session boundary. The training engine does **not** call the LLM. It rapidly practices navigation with randomized starts, stores actual transitions, builds an experience-derived topological map, trains a small local navigation policy, records explicit peer encounters, and persists everything for later runs.

The normal embodied session still lets the LLM choose its own behavior. Training does not inject a hidden desire to seek, greet, cooperate with, or follow the other agent. Peer-encounter practice exists only because it is an explicit training objective supplied by the human.

## What persists across runs

```text
state/
  memory/
    a/memory.jsonl
    b/memory.jsonl
    solo/memory.jsonl
  maps/FloorPlan1.json
  checkpoints/navigation_skill.json
  replay/navigation_replay.jsonl
  metrics/training_summary.json
  metrics/training_history.jsonl
```

- **episodic/social memory** reloads on the next normal session; stored camera-description prose and simulator object IDs are deliberately excluded from long-term LLM recall;
- **world map** stores only cells and transitions physically visited during training;
- **navigation checkpoint** is a compact CPU-trained local motor policy;
- **replay buffer** keeps navigation transitions for later training/analysis;
- **training summary/history** tracks coverage, route/action success, peer encounters, and throughput.

The detailed map/checkpoint remains in the motor layer. It is not serialized into the LLM prompt. When the LLM independently chooses a semantic `Approach`, the motor layer can use the learned map/skill; if learned coverage is insufficient, it falls back to AI2-THOR's deterministic reachable-position navigation.

## Clean accelerated training (recommended first v0.7 run)

The `--reset-memory` flag clears **only** the configured persistent `state/` directory immediately before training. Source code and `runs/` logs are not deleted.

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.training \
  --config config.yaml \
  --scene FloorPlan1 \
  --episodes 1000 \
  --workers 1 \
  --reset-memory \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
```

No llama.cpp server is required for this trainer. It intentionally spends compute on physical navigation experience rather than multimodal LLM inference.

Equivalent helper script:

```bash
RESET_MEMORY=1 EPISODES=1000 WORKERS=1 ./scripts/run_training_wsl.sh
```

After the first successful run, increase parallelism cautiously:

```bash
EPISODES=4000 WORKERS=2 ./scripts/run_training_wsl.sh
```

Each worker owns a separate Unity process and worker-local artifacts while training, then v0.7 merges the experience into one persistent map/checkpoint/replay state. Start with one worker to verify the custom player, then try two.

Optional off-screen rendering can be tested with `--headless`. If the custom WSL player is more reliable under WSLg/D3D12 on a particular machine, simply omit `--headless`; the trainer still uses low 160x120 render resolution, no instance segmentation, no artificial step delay, and zero LLM calls.

## Standalone memory reset

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python scripts/reset_training_memory.py --state-dir state --yes
```

The reset utility refuses obviously dangerous broad paths such as the home directory.

## Dry-run the trainer without starting Unity

```bash
python -m agent_core.training \
  --config config.yaml \
  --episodes 1000 \
  --workers 2 \
  --dry-run
```

## Normal learned-skill session afterward

Start llama.cpp as usual, then run the normal social session. `config.yaml` enables `learning.use_skill_navigation: true`, so the motor layer automatically loads `state/maps/FloorPlan1.json` and `state/checkpoints/navigation_skill.json`.

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1 \
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
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
```

To intentionally wipe learned state before a normal session too, add `--reset-memory`. Normally you should **not** do that after training because it would remove the map, checkpoint, replay buffer, and persistent episodic/social memory you just created.

## Training objective and neutrality

The fast trainer alternates randomized room-navigation episodes with a configurable fraction of peer-encounter episodes (`training.peer_encounter_rate`, default 0.25). In an encounter episode, one body is the seeker and the other is the stationary peer; the training motor layer practices reaching the peer's physical vicinity. This is an explicit training task, not a personality or autonomous preference. Normal free-roam receives no hidden seek-friend reward or controller-selected behavior.

The trainer uses AI2-THOR environment queries to construct teacher routes, but those route details remain controller-private. The large model later sees only safe self-history/training summaries plus its normal camera/body state.

---

# AI2-THOR Agent v0.6.1 — Humanoid Visual Replacement Hotfix

v0.6.1 is a visual hotfix over v0.6.0. The social/session/LLM architecture is unchanged.
It fixes the custom Unity avatar after runtime verification showed the v0.6.0 hook executed for
both agents but the stock AI2-THOR body was still the dominant visible peer body.

The two models now know a stable factual identity:

- Agent A knows it is **Agent A**.
- Agent B knows it is **Agent B**.
- Each knows the other label refers to **another independently embodied agent like itself**.
- Neither receives the other's private camera, memory, task, focus, intention, or hidden reasoning.
- Knowing the peer exists does **not** imply friendship, cooperation, trust, conversation,
  competition, avoidance, or any other social preference.

The package contains an updated AI2-THOR Unity patch that keeps AI2-THOR's native `TallVisCap`
as the visibility host, permanently suppresses its stock renderers, and inserts a clearly
humanoid head/torso/arms/hands/legs/feet visual inside that hierarchy. CharacterController,
camera, navigation, hand, interaction reach, and the action API remain unchanged.

## v0.6.1 visual hotfix

v0.6.0 correctly reached both agents at runtime, but its generated visual was attached as a
separate root and used floor-relative Y coordinates even though AI2-THOR's tall controller is
centered on the agent transform. v0.6.1 changes only the Unity renderer path:

- `TallVisCap` remains `VisibilityCapsule`.
- Existing stock `TallVisCap` renderers are set `forceRenderingOff=true`.
- The humanoid is parented inside `TallVisCap`, on the stock visible layer.
- Humanoid renderers use `ShadowCastingMode.Off` and are re-registered with first-person culling.
- Coordinates span roughly `-0.9 .. +0.85 m` around the centered controller origin.
- Primitive colliders remain disabled; physics/navigation are untouched.
- The patcher upgrades an already-patched v0.6.0 AI2-THOR checkout in place and creates v0.6.1 backups.

Runtime success markers now contain `v0.6.1 humanoid visual replacement applied`.

## What is fully validated in this ZIP

The Python/social/controller layer and the Unity patching tool are covered by the automated
test suite. The final ZIP is expected to pass the shipped tests before the custom Unity
player is built.

The actual Unity player binary is **not included** because it must be compiled with the
Unity editor against an AI2-THOR source checkout. See `unity_humanoid/README.md`.

## v0.6 architecture

```text
                ONE AI2-THOR SCENE

       Agent A                         Agent B
  humanoid robot body            humanoid robot body
        camera A                       camera B
           |                               |
           v                               v
        Qwen A                          Qwen B
           |                               |
     private context                 private context
     private memory                  private memory
     own intention                   own intention
           |                               |
           +---------- speech ------------+
                 only explicit Speak

Human Observer Dashboard sees both cameras and public telemetry.
It never feeds data back into either model.
```

## Context hygiene in v0.6

Model output is split into separate channels:

```json
{
  "observation": "I see the refrigerator and Agent B near the counter.",
  "intention": "I want to look at the sink.",
  "say": "",
  "focus": "sink",
  "pace": "normal",
  "task_complete": false,
  "actions": [
    {"kind": "RotateRight", "target": null, "params": {}}
  ]
}
```

Persistence rules:

- `observation` is the **current** camera/body report and is not replayed as narrative
  history on the next turn.
- `intention` is a compact self-generated goal/interest and can persist if the model keeps
  endorsing it.
- `say` is optional status for the human observer only.
- `Speak` is the only action that sends words to the peer.
- mechanical failures are returned as compact machine feedback rather than being blended
  into visual observations.

This addresses the v0.5.2 failure where repeated descriptions such as “I am near the
fridge” recursively contaminated later generations.

## Explicit peer identity

The social context now includes stable facts equivalent to:

```json
{
  "self_identity": {
    "label": "A",
    "kind": "independently embodied agent"
  },
  "peer_identity": [
    {
      "label": "B",
      "relation": "another independently embodied agent like you",
      "private_state_shared": false
    }
  ]
}
```

When B is in A's camera cone, A can additionally receive:

```json
{
  "peer_label": "B",
  "peer_is_another_independently_embodied_agent_like_you": true,
  "in_camera_cone": true,
  "distance_m": 1.6,
  "bearing_degrees": 8.0,
  "within_hearing_range": true
}
```

This means A no longer has to infer whether the peer body is a coffee machine, metal rack,
or ordinary prop. It knows the sensed body corresponds to B. The RGB image still provides
appearance and occlusion.

## Install the Python package

After downloading the ZIP to Windows Downloads:

```bash
cd ~/projects

mv author author_v0.5.2_backup

unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.6.0_humanoid_social_embodiment.zip \
  -d ~/projects

cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

## Run immediately with the stock AI2-THOR body

The v0.6 identity/context improvements work even before the custom Unity player is built:

```bash
WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')
export AGENT_LLM_BASE_URL="http://$WINDOWS_HOST:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1 \
  --session \
  --agents 2 \
  --llm \
  --vision \
  --interact \
  --dashboard \
  --vision-every 1 \
  --step-delay 0.08 \
  --roam-decisions 500 \
  --max-steps 3000
```

Open `http://localhost:8765` in the Windows browser.

## Use the humanoid custom player

After building the custom AI2-THOR player described in `unity_humanoid/README.md`, either
put its path in `config.yaml`:

```yaml
simulation:
  local_executable_path: /path/to/custom/thor-Linux64
```

or pass it directly:

```bash
python -m agent_core.cli \
  --scene FloorPlan1 \
  --session \
  --agents 2 \
  --llm --vision --interact --dashboard \
  --thor-executable /path/to/custom/thor-Linux64
```

## Humanoid renderer design

`unity_humanoid/HumanoidPeerAvatar.cs` constructs the visible robot from Unity primitives:

- spherical head with face plate and emissive eyes
- rectangular torso/chest panel
- two arms and hands
- two legs and feet
- AI2-THOR's distinct per-agent body colors are retained

The generated visual primitives have their colliders disabled. The actual AI2-THOR
CharacterController remains authoritative for physics and navigation.

## Observer Dashboard

The dashboard continues to show both live RGB streams and now separates:

- current observation
- self-chosen intention
- observer/status text
- semantic focus
- chosen actions
- body/progress feedback
- peer identity and camera-cone geometry
- speech/hearing events
- fallback/error state

No hidden chain-of-thought is displayed or transferred between agents.

## Task routing

```text
/a Find a cup.
/b Open the refrigerator.
/both Put the apple in the sink.
/a roam
/b roam
/both roam
/status
/quit
```

Plain unprefixed text remains an explicit shared user task.

## Unity source target

The project remains pinned to `ai2thor>=4.3.0,<5.0.0`. AI2-THOR 4.3.0 corresponds to
commit `a634e28`. The humanoid patcher refuses to guess if the expected `AgentManager.cs`
hook is absent, and creates `AgentManager.cs.v060.bak` before modifying the source tree.

---

# AI2-THOR Embodied Agent — v0.4.0

v0.4.0 separates **LLM perception/choice** from **AI2-THOR motor grounding**.

The purpose is simple: in autonomous mode, the model should not be told that an Apple, Fridge, Statue, or any other simulator object exists. It receives the current camera view plus minimal body state and decides what to do from that. AI2-THOR metadata remains available internally only to translate a semantic choice such as `"the object I see on the counter"` into the object ID Unity requires.

## The v0.4 boundary

```text
AI2-THOR viewport
       |
       v
   Qwen2-VL  <----- minimal body state
       |
       |  "I want to approach the object I see."
       v
semantic action
       |
       v
Hidden SemanticGrounder <----- AI2-THOR object metadata
       |
       |  objectId only after the model chose the target
       v
ActionExecutor / navigation / recovery
       |
       v
AI2-THOR
```

The autonomous LLM prompt does **not** contain:

- the simulator object catalog
- object IDs
- object names/types
- object visibility/distance metadata
- pickup/open/receptacle capability metadata for individual objects
- raw AI2-THOR error strings that might reveal object IDs

It does contain:

- the viewport image when vision is enabled
- position, heading, camera horizon
- whether it is holding something
- previous action name and success/failure
- its own recent semantic actions/status
- an active user task, but only when the user explicitly assigned one

## Modes

### Autonomous roam

There is no required objective. The system prompt describes capabilities but gives the model no preferred object, destination, exploration strategy, or interest.

```bash
python -m agent_core.cli --scene FloorPlan1 --roam --llm --vision --interact
```

### Persistent interactive session

This is the recommended v0.4 mode. The same Unity scene stays alive while the model alternates between autonomous behavior and user tasks.

```bash
python -m agent_core.cli --scene FloorPlan1 --session --llm --vision --interact
```

While it is running, type any normal sentence and press Enter:

```text
Pick up the apple and put it in the sink.
Open the refrigerator.
Look around and tell me what you see.
Find something you think is interesting.
```

Commands:

```text
/roam     cancel the current task and return to autonomy
/status   show current autonomous/task state
/help     show session commands
/quit     close the persistent session
```

For structured pickup/place/open/close tasks, task completion is checked using hidden AI2-THOR metadata. The metadata verifies the result but is not supplied to Qwen as perception.

For unstructured tasks such as `look around and tell me what you see`, the model may set `task_complete=true` when it considers the request finished.

## Semantic grounding

Object-directed LLM actions use ordinary language:

```json
{"kind":"Approach","target":"the cup I can see","params":{}}
{"kind":"PickupObject","target":"the apple","params":{}}
{"kind":"OpenObject","target":"the refrigerator","params":{}}
{"kind":"PutObject","target":"the sink","params":{}}
```

`SemanticGrounder` translates those phrases into AI2-THOR IDs internally.

A crucial v0.4 rule is that a newly named semantic target must correspond to a **currently visible** simulator object. Hidden metadata cannot make a hallucinated unseen object actionable. A previously grounded object can still be referenced with a pronoun such as `it`.

## Failure behavior

Older free-roam builds used controller-generated movement when the LLM failed. v0.4 does not do that. If the model request fails or remains invalid after one corrective retry, the controller sends `Pass` and idles. This keeps autonomous movement attributable to the LLM rather than to a hidden fallback explorer.

## Files added or substantially changed

```text
src/agent_core/roam.py       neutral embodied decision loop
src/agent_core/grounding.py  hidden semantic -> objectId translator
src/agent_core/session.py    persistent autonomous/task state machine
src/agent_core/vision.py     clean camera/minimal-body-state boundary
src/agent_core/cli.py        --session plus neutral --roam wiring
config.yaml                  metadata_to_llm=false and embodied defaults
```

The proven low-level execution stack remains in place:

```text
thor_adapter.py
executor.py
navigation.py
recovery.py
world_state.py
goals.py
goal_verifier.py
```

## Recommended WSL filesystem layout

Keep the active project and Python environment on the Linux filesystem:

```text
/home/davei/projects/author
/home/davei/.venvs/author
```

Do not build the virtual environment under `/mnt/c` or `/mnt/e`.

## Install/update

From WSL, after extracting the release into `~/projects/author`:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

## AI2-THOR / RX 6600

Use the already validated WSLg D3D12 path:

```bash
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD
```

Smoke test:

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_thor.py
```

Do not add `--headless` for the validated WSLg/OpenGL path.

## Qwen2-VL / llama.cpp

The Windows `llama-server` must be started with the Qwen2-VL GGUF and matching multimodal projector. v0.4's included PowerShell helper defaults to a 16,384-token context.

Example on Windows:

```powershell
cd E:\DaveInc\llama.cpp

.\llama-server.exe `
  -m "E:\DaveInc\OllamaModels\gguf_files\Qwen2-VL-2B-Instruct-Q4_K_M.gguf" `
  --mmproj "E:\DaveInc\OllamaModels\gguf_files\mmproj-Qwen2-VL-2B-Instruct-f16.gguf" `
  --alias qwen2-vl-2b `
  -ngl 999 `
  -c 16384 `
  --host 0.0.0.0 `
  --port 8080
```

Then in WSL:

```bash
WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')
export AGENT_LLM_BASE_URL="http://$WINDOWS_HOST:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"
```

## Recommended persistent run

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1 \
  --session \
  --llm \
  --vision \
  --interact \
  --vision-every 1 \
  --step-delay 0.08 \
  --roam-decisions 500 \
  --max-steps 3000
```

Or:

```bash
./scripts/run_session_wsl.sh
```

`--step-delay` is only the base physical animation delay. Qwen can select `slow`, `normal`, or `fast` per decision, scaling the base delay by 2.0, 1.0, or 0.5.

## Legacy one-shot task mode

The pre-v0.4 `--goal` mode is retained for regression compatibility:

```bash
python -m agent_core.cli --scene FloorPlan1 --goal "Pick up the Apple and put it on the CounterTop."
```

For the clean perception boundary plus user-assigned tasks, use **`--session`**, not the legacy LLM goal planner.

## Logs

```text
runs/actions.jsonl
runs/memory.jsonl
```

Internal logs may contain grounded simulator IDs for debugging. Those internal records are not automatically serialized back into the autonomous LLM prompt.


## v0.4.1 — embodiment mechanics without behavioral injection

v0.4.1 keeps the neutral-autonomy boundary from v0.4.0 and improves only the
body/controller interface exposed to the model.

### What changed

- `Observe` / `Wait` and even an empty `actions` list are valid autonomous choices.
  Remaining still is no longer treated as an LLM failure.
- Primitive movement actions no longer semantically point at objects. If a model
  emits legacy output such as `MoveAhead("microwave")`, `microwave` is retained as
  model-selected **attention**, while `MoveAhead` remains an ordinary forward step.
- Added semantic `Face("...")` in addition to `Approach("...")`.
- Added persistent `semantic_focus`, but only after the model itself refers to an
  object. The prompt receives the model's own phrase plus neutral physical facts:
  current viewport presence, distance, approach status, bearing error, and physical
  affordances for that selected focus.
- A repeated `Approach` after the focus is already within approach distance is
  rejected as mechanically satisfied and the model gets one neutral corrective
  retry. The controller does not choose the next behavior.
- Every decision now produces proprioceptive feedback: movement distance, yaw
  change, camera-pitch change, success, and primitive step count.
- Hidden grounding now uses AI2-THOR's actual rendered instance detections when
  available. This reduces cases where Qwen visually sees a microwave but the generic
  metadata `visible` flag disagrees. Instance IDs remain internal and are never sent
  to the LLM.

AI2-THOR instance segmentation is enabled internally by default. This has some
rendering overhead, but it aligns semantic grounding with the actual viewport.

### Run

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1 \
  --session \
  --llm \
  --vision \
  --interact \
  --vision-every 1 \
  --step-delay 0.08 \
  --roam-decisions 500 \
  --max-steps 3000
```

During the same persistent session, type a task normally, for example:

```text
Pick up the apple and put it in the sink.
```

Use `/roam` to cancel the task and return to autonomous mode.
