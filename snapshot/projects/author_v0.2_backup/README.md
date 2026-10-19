# AI2-THOR Phase 9+ Agent — v0.2

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
