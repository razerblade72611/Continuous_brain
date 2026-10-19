# AI2-THOR Phase 9+ Agent — v0.3 Free Roam

This build is the rewritten Phase 9+ core for the AI2-THOR agent. It is designed around the setup that has now been proven to work: AI2-THOR 4.3.0 in WSL2/Ubuntu with WSLg OpenGL acceleration through an AMD Radeon RX 6600, while local LLM/VLM inference can remain native on Windows through an OpenAI-compatible `llama.cpp` server.

## What changed in v0.2

The original v0.1 implementation had two correctness problems for a goal such as:

```text
Pick up the Apple and put it on the CounterTop.
```

First, the goal verifier could declare the composite goal complete immediately after pickup. Second, `PutObject` verification only checked that the hand became empty, which could falsely count a failed or wrong placement as success.

v0.2 fixes both and also replaces blind fallback movement with deterministic navigation based on AI2-THOR `GetReachablePositions`.

### Core behavior

1. Parse a goal into explicit pickup / placement / open / close subgoals.
2. Observe authoritative AI2-THOR object metadata.
3. Navigate through reachable grid positions to an interaction target.
4. Pick up, open, close, or place the target object.
5. Verify the exact state transition.
6. Recover and replan after failure.
7. Stop only when the full goal is provably satisfied or a budget is exhausted.

## Architecture

```text
Natural-language goal
        |
        v
   Goal parser
        |
        v
Planner (LLM optional, deterministic fallback always available)
        |
        v
 NavigateTo / interaction action
        |
        v
AI2-THOR adapter
   |          |
   |          +--> GetReachablePositions --> shortest grid route
   v
Unity action
   |
   v
Verifier --> Recovery --> Replan
   |
   v
WorldState + episodic JSONL memory
```

`NavigateTo` is an agent-level pseudo-action. It is translated into ordinary `RotateLeft`, `RotateRight`, and `MoveAhead` actions before anything is sent to Unity.

## Recommended filesystem layout in WSL

Keep the active Linux project and virtual environment in the Linux filesystem, not under `/mnt/e`:

```text
/home/<user>/projects/author       project
/home/<user>/.venvs/author         Python environment
```

Windows model files can stay on the Windows drive.

## Install in WSL2 / Ubuntu 24.04

```bash
mkdir -p ~/.venvs
python3 -m venv ~/.venvs/author
source ~/.venvs/author/bin/activate

cd ~/projects/author
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

The package pins AI2-THOR to:

```text
>=4.3.0,<5.0.0
```

This avoids the unavailable 5.0.0 Unity build encountered during setup.

## RX 6600 / WSLg environment

For the tested WSLg hardware path:

```bash
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD
```

Verify:

```bash
glxinfo -B | grep -E 'Device:|Accelerated:|OpenGL renderer'
```

The desired result contains:

```text
Device: D3D12 (AMD Radeon RX 6600)
Accelerated: yes
OpenGL renderer string: D3D12 (AMD Radeon RX 6600)
```

Do **not** add `--headless` for this tested setup. The normal Linux/GLX AI2-THOR path is the one that has been validated.

## Smoke-test AI2-THOR

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_thor.py
```

A healthy run should report a started controller, a scene such as `FloorPlan1_physics`, an object count, `RotateRight success: True`, and a clean shutdown.

## Run tests

```bash
pytest -q
```

## Run the deterministic agent

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1 \
  --goal "Pick up the Apple and put it on the CounterTop."
```

Or:

```bash
./scripts/run_agent_wsl.sh "Pick up the Apple and put it on the CounterTop." FloorPlan1
```

## What success means now

For the composite Apple → CounterTop task, success requires AI2-THOR metadata to show that the Apple has the requested CounterTop object ID in its `parentReceptacles`. Merely dropping the Apple or emptying the agent's hand is not enough.

## Local LLM / Qwen2-VL on Windows

The code accepts an OpenAI-compatible server. A native Windows Vulkan build of `llama.cpp` is a practical way to use an RX 6600 while AI2-THOR remains in WSL.

Environment variables used by the agent:

```bash
export AGENT_LLM_BASE_URL="http://127.0.0.1:8080/v1"
export AGENT_LLM_MODEL="local-model"
```

Then run with:

```bash
python -m agent_core.cli \
  --scene FloorPlan1 \
  --goal "Pick up the Apple and put it on the CounterTop." \
  --llm
```

If Windows and WSL do not share localhost in the current networking mode, use the reachable Windows host address instead of `127.0.0.1`.

For the existing Qwen2-VL pair, the included PowerShell script accepts both the GGUF and the optional multimodal projector:

```powershell
.\scripts\start_llama_server.ps1 `
  -LlamaServer "C:\path\to\llama-server.exe" `
  -Model "E:\DaveInc\OllamaModels\gguf_files\Qwen2-VL-2B-Instruct-Q4_K_M.gguf" `
  -Mmproj "E:\DaveInc\OllamaModels\gguf_files\mmproj-Qwen2-VL-2B-Instruct-f16.gguf"
```

## Logs

Runtime logs are created automatically in:

```text
runs/actions.jsonl
runs/memory.jsonl
```

They are intentionally excluded from the clean source archive.

## Tests included

The test suite covers:

- AI2-THOR event normalization
- object lookup normalization
- goal parsing
- composite-goal correctness
- planner pickup → placement ordering
- deterministic navigation path generation
- exact `PutObject` receptacle verification
- recovery behavior

## Known boundary

The source code and unit tests can verify planning, goal semantics, routing logic, action verification, and recovery. The final integration truth remains the actual AI2-THOR scene, because visibility, object geometry, and receptacle placement are determined by Unity at runtime.

# v0.3 free-roam mode

Task mode intentionally stops as soon as the requested world state is already true. For example, if an Apple starts on a CounterTop, the goal `Pick up the Apple and put it on the CounterTop.` can correctly complete at step 0 because the final state already holds.

Free-roam mode is different: it has **no completion goal**. The agent repeatedly observes the current AI2-THOR world, asks the local LLM for one next exploration action, executes it, verifies/recoveries, remembers the result, and continues until its explicit budget is exhausted.

## Metadata-driven free roaming

With the Windows `llama-server` reachable from WSL:

```bash
export AGENT_LLM_BASE_URL="http://<windows-host>:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"

GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1 \
  --roam \
  --llm \
  --roam-decisions 100 \
  --max-steps 500
```

## Vision-driven free roaming with Qwen2-VL

If `llama-server` was started with the matching `--mmproj`, add `--vision`:

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1 \
  --roam \
  --llm \
  --vision \
  --roam-decisions 100 \
  --max-steps 500
```

The camera frame is resized to at most 512 pixels wide and sent to the same OpenAI-compatible `/chat/completions` endpoint as an `image_url` data URI. This lets a Qwen2-VL server use both AI2-THOR metadata and the current visual frame for each exploration decision.

To reduce vision latency, send an image only every 3 decisions:

```bash
python -m agent_core.cli --scene FloorPlan1 --roam --llm --vision --vision-every 3 --max-steps 500
```

## Allow object interaction while roaming

By default, roam mode only navigates and looks around. Add `--interact` if the LLM should also be allowed to pick up objects, open/close objects, and place held objects:

```bash
python -m agent_core.cli \
  --scene FloorPlan1 \
  --roam \
  --llm \
  --vision \
  --interact \
  --roam-decisions 200 \
  --max-steps 1000
```

`--roam-decisions` counts LLM decisions. `--max-steps` counts actual Unity primitive movement/rotation/interaction actions. A single `NavigateTo` decision can consume several primitive steps.

During execution the console prints lines like:

```text
[roam 001 | step 0000] NavigateTo -> Fridge [Fridge] vision: Inspect a new part of the kitchen.
[roam 002 | step 0012] RotateRight vision: Survey the new viewpoint.
```

Roam decisions are also appended to `runs/memory.jsonl` and primitive outcomes to `runs/actions.jsonl`.


## Slowing down free roam

v0.3.1 adds a real-time delay after every primitive AI2-THOR action.

Default:

```yaml
simulation:
  step_delay_seconds: 0.15
```

Override it per run:

```bash
python -m agent_core.cli \
  --scene FloorPlan1 \
  --roam --llm \
  --step-delay 0.25 \
  --roam-decisions 100 \
  --max-steps 500
```

Useful values:

- `0.08` = fast but watchable
- `0.15` = default
- `0.25` = relaxed
- `0.40` = slow demonstration pace

The delay is applied to primitive Unity actions, including each step generated by
`NavigateTo`, so long paths no longer happen instantly.

## llama.cpp HTTP 400 diagnostics

v0.3.1 retries once without `response_format` if a llama.cpp server rejects the
structured-output parameter. If the request still fails, the actual server
response body is printed in the fallback reason so model/alias/template problems
are visible instead of only `HTTPStatusError: 400`.


## v0.3.2 autonomous movement

Free-roam is no longer primarily object-to-object navigation.

The LLM may now choose a short burst of up to six actions, for example:

```json
{
  "reason": "I want to look down the other side of the kitchen.",
  "pace": "fast",
  "actions": [
    {"kind": "RotateRight", "params": {}},
    {"kind": "MoveAhead", "params": {}},
    {"kind": "MoveAhead", "params": {}},
    {"kind": "LookUp", "params": {}}
  ]
}
```

It can still use `NavigateTo`, but object targets are supplied to the model as
compact aliases (`c00`, `c01`, ...) rather than raw AI2-THOR IDs. This avoids
hallucinated object IDs.

A target cooldown prevents repeated `NavigateTo` loops. If the model returns an
invalid alias, malformed JSON, unsupported action, or a recently visited target,
the controller gives the LLM one corrective retry before using fallback.

The LLM can also choose its pace per decision:

- `slow` = 2x the base `--step-delay`
- `normal` = base delay
- `fast` = 0.5x the base delay

For a natural watchable baseline:

```bash
python -m agent_core.cli \
  --scene FloorPlan1 \
  --roam --llm \
  --step-delay 0.10 \
  --roam-decisions 300 \
  --max-steps 1500
```

With vision and interactions:

```bash
python -m agent_core.cli \
  --scene FloorPlan1 \
  --roam --llm --vision --interact \
  --vision-every 2 \
  --step-delay 0.10 \
  --roam-decisions 300 \
  --max-steps 1500
```
