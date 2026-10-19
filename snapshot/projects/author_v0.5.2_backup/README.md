# AI2-THOR Agent v0.5.2 — Social Observer + Peer Perception Telemetry

v0.5.2 keeps the two independent Qwen contexts and read-only dual-camera dashboard from
v0.5.1, while addressing an important social-perception failure mode: an agent can have
the other embodied body in its camera view yet describe that body as an ordinary scene
object such as a metal rack or coffee machine.

The controller does **not** overwrite the model's visual report. Instead, it adds a
separate neutral body sensor and exposes the disagreement to the human observer.

## What changed in v0.5.2

- AI2-THOR shared-world initialization now explicitly sets `makeAgentsVisible=True`.
- Added a neutral `peer_camera_cone_sensor` to each agent's social sensory context.
- The peer sensor is pose geometry only: it reports that another embodied body's center
  lies inside the camera cone, plus distance and signed bearing.
- The sensor does **not** identify ordinary scene objects, does not copy the peer's
  camera/memory/task/focus, and does not require social behavior.
- The attached RGB image remains the source for appearance and possible occlusion.
- The social prompt states the simulator fact that visible peer bodies are rendered as
  simple colored capsule-like bodies. It does not tell the model to greet or approach them.
- The Observer Dashboard now shows per-agent peer distance, bearing, camera-cone status,
  and hearing-range status. A camera badge appears when the other body is in the cone.
- Observer-only `peer/report mismatch` telemetry flags cases where the body is in the camera
  cone but the model report contains no peer-like wording (for example, it says “metal rack”).
- Added neutral recent-progress proprioception over an 8-decision window: distance moved,
  absolute yaw change, repeated action signature/count, and a mechanical `stagnant` flag.
- Improved local-LLM response repair. Valid JSON before trailing text is accepted;
  clear one-line actions such as `Approach the door` are mechanically structured; plain
  descriptive/no-op text becomes a valid Observe rather than an artificial fallback.
- The dashboard remains read-only and still does not expose hidden chain-of-thought.

### Important interpretation

`another_embodied_agent_in_camera_cone=true` is deliberately **not** called
`peer_visible=true`. The sensor uses body positions and camera heading/FOV; a wall or other
occluder can still block the RGB pixels. The model receives a note telling it to use its
actual attached image for appearance/occlusion.

## Install

After downloading the ZIP to Windows Downloads:

```bash
cd ~/projects

mv author author_v0.5.1_backup

unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.5.2_observer_dashboard.zip \
  -d ~/projects

cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected shipped-archive result:

```text
71 passed
```

## Multi-agent smoke test

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_multi_thor.py
```

## Launch v0.5.2 with Observer Dashboard

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

Open:

```text
http://localhost:8765
```

The dashboard shows both current RGB views plus each agent's explicit vision/status report,
focus, chosen action, proprioception, task, speech/hearing, fallbacks, and the new peer
geometry telemetry.

Example observer interpretation:

```text
Agent A model report: "I am observing the metal rack."
Other body telemetry: B · 1.6 m · bearing +8° · camera cone=true · audible=true
```

That is intentionally displayed as a disagreement. The software does not rewrite A's
report to "I see Agent B." On the next turn Qwen can decide for itself what the image and
neutral peer sensor mean.

## Neutrality and privacy boundary

Each agent retains its own:

- RGB camera frame
- recent self-context/action history
- semantic focus
- episodic memory
- active user task
- proprioceptive feedback

Cross-agent channels are limited to:

1. physical coexistence in one AI2-THOR scene,
2. the neutral peer camera-cone body sensor,
3. explicit `Speak` utterances that pass the hearing-range test.

The agents do not receive one another's private prompts, private memory, hidden reasoning,
scene-object catalog, or camera frame.

The social prompt still does **not** tell either model to greet, cooperate, compete, follow,
trust, help, or approach the other agent.

## Configuration

```yaml
social:
  enabled: true
  agent_count: 2
  labels: ["A", "B"]
  speech_range_m: 4.0
  peer_camera_cone_sensor: true
  peer_camera_cone_margin_degrees: 3.0
  serialized_inference: true

observer:
  enabled: false
  host: 127.0.0.1
  port: 8765
  refresh_ms: 350
  image_width: 640
  jpeg_quality: 75
```

Turning `peer_camera_cone_sensor: false` disables that sensor in the LLM input. The
read-only dashboard can still display peer geometry for the human observer.

## User task routing

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

Plain text without a route remains an explicit shared user task for both agents.

## Speech

Only an explicit `Speak` action crosses agent boundaries, and only inside the configured
hearing radius. Human-facing `say`/vision-status output is not sent to the peer.

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
