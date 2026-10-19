# AI2-THOR Agent v0.5.1 — Social Embodied Agents + Observer Dashboard


## v0.5.1 Observer Dashboard

The social session can now expose a **read-only local browser dashboard** for the human observer. It does not become part of either agent's prompt or memory.

Run the two-agent session with:

```bash
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

The terminal prints:

```text
Observer dashboard: http://localhost:8765
```

Open that address in the Windows browser. WSL localhost forwarding normally makes the WSL service available at the same Windows `localhost` URL.

The page shows, simultaneously:

- Agent A current RGB camera viewport
- Agent B current RGB camera viewport
- each agent's explicit `say` / vision-status report
- current model-selected semantic focus
- chosen semantic action(s)
- body feedback such as movement distance, yaw change, success, and primitive step count
- autonomous/task mode and current user task
- explicit speech sent and speech heard
- fallback/error badges plus a shared observer event log

The two image panels are refreshed together after every embodied turn from the current shared Unity multi-agent event, so the displayed A/B frames represent the same current world state as closely as the serialized A-then-B inference loop permits.

### What the dashboard does not show

It does **not** expose hidden model chain-of-thought. The `Vision/status` field is only the concise outward `say` text the model already produced. It also does not expose private model prompts, private memory, hidden simulator object catalogs, or one agent's private camera/state to the other agent.

The observer endpoint is read-only (`GET /`, `GET /api/state`, `GET /health`). It has no command endpoint and cannot steer either body.

Dashboard options:

```text
--dashboard
--dashboard-host 127.0.0.1
--dashboard-port 8765
--dashboard-refresh-ms 350
```

Or configure it in `config.yaml`:

```yaml
observer:
  enabled: false
  host: 127.0.0.1
  port: 8765
  refresh_ms: 350
  image_width: 640
  jpeg_quality: 75
```

Convenience launcher:

```bash
scripts/run_social_dashboard_wsl.sh
```


This release extends the neutral embodied v0.4.1 architecture to **two independently
contextualized agents inside one shared AI2-THOR scene**. AI2-THOR receives
`agentCount=2`; each body is addressed with its own `agentId` and receives its own
per-agent camera event. The agents share the physical world, but not private LLM state.

## Social neutrality boundary

The social capability prompt says only that other embodied agents *may* be present and
that the model *can* speak. It does **not** tell either model to greet, cooperate, compete,
follow, trust, help, or interact with the other. Each context is free to ignore the other
agent.

The following stay private per agent:

- RGB camera frame
- recent self-context / action history
- semantic focus and grounded object IDs
- episodic memory file
- active user task
- proprioceptive feedback

The only automatic cross-agent channel is an explicit `Speak` action. `say` is merely a
terminal status for the human operator and is **not** delivered to the peer.

Speech is range-limited. The default is 4 meters. No message is delivered outside that
radius. v0.5.0 models distance only; wall attenuation/occlusion is intentionally left for
a later optional acoustics layer.

## Shared model weights, separate agent contexts

Both agents can use the same Windows `llama-server`/Qwen2-VL process. Inference is
serialized A then B, so the model weights do not need to be loaded twice in VRAM.
Each request carries only that agent's own camera, state, memory-derived recent context,
user task, and speech it physically heard.

## Install

After downloading the ZIP into Windows Downloads:

```bash
cd ~/projects

mv author author_v0.4.1_backup

unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.5.0_social_agents.zip \
  -d ~/projects

cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected package test result for the shipped archive:

```text
57 passed
```

## Multi-agent simulator smoke test

Run this before involving Qwen:

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_multi_thor.py
```

It verifies that one Unity controller exposes two per-agent camera/event streams and that
an action sent with `agentId=1` affects Agent B's channel rather than being routed to A.

## Start Qwen social session

Keep the Windows llama-server running, then from WSL:

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
  --vision-every 1 \
  --step-delay 0.08 \
  --roam-decisions 500 \
  --max-steps 3000
```

`--max-steps` is a per-agent primitive action budget. `--roam-decisions` is the maximum
number of social rounds; normally both A and B get one decision turn per round.

## User task routing

While the scene remains alive:

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

Plain text without a route is treated as an explicit shared user task for both agents.
This does not alter autonomous preferences after the task is cleared/completed.

## Speech

The model may emit:

```json
{
  "say": "I notice another figure.",
  "focus": null,
  "pace": "normal",
  "task_complete": false,
  "actions": [
    {"kind": "Speak", "target": null, "params": {"text": "Hello."}}
  ]
}
```

Only `Hello.` is routed to the other agent, and only if that body is within hearing range.
The `say` field remains visible only in your terminal.

## What lets the agents see one another?

AI2-THOR's native multi-agent mode renders multiple agent bodies in the same Unity scene.
Each agent receives its own RGB camera frame from the shared multi-agent event. The project
does not inject a text label saying that the other body is visible. If Qwen recognizes the
other body in its viewport, that recognition comes through its own image input.

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
