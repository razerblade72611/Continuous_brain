# AUTHOR v0.9.1 Operator README

## What v0.9.1 changes

v0.9.1 keeps the v0.9.0 two-room Discovery world and adds verified task execution. A task no longer ends just because the LLM returns `task_complete=true`. Texting must actually deliver a `TextPeer` event, manipulation must be verified in AI2-THOR state, and `Escape` in the Discovery scene must physically cross into a different connected region.

The task motor handoff is active only for an explicit user task. It never creates a normal autonomous preference. For manipulation, the model must already have grounded/focused the requested target and the simulator must report the requested interaction as currently possible. The motor layer then finishes the mechanical transition instead of allowing endless `Approach`/`Observe` loops.

## Upgrade from v0.9.0 and preserve learned state

WSL:

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

Expected: `117 passed`.

## Existing single-room player

The original player is still valid:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
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
  --step-delay 0 \
  --roam-decisions 500 \
  --max-steps 3000 \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-Linux64/thor-humanoid-Linux64
```

## Apply the corrected Discovery-world Unity source

WSL:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python unity_world/apply_discovery_world_patch.py ~/projects/ai2thor-humanoid
python unity_world/verify_discovery_world_patch.py ~/projects/ai2thor-humanoid
./scripts/sync_discovery_unity_to_windows.sh
cp scripts/build_discovery_world.ps1 /mnt/c/Users/davei/a2h/build_discovery_world.ps1
```

The verifier should include:

```text
OK: Unity SceneManager is fully qualified (v0.9.1 compiler fix)
```

Windows PowerShell:

```powershell
& C:\Users\davei\a2h\build_discovery_world.ps1
```

The previous CS0117 collision is fixed in the packaged `DiscoveryWorldBuilder.cs`.

## Copy the Discovery player back to WSL

```bash
rm -rf ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64
mkdir -p ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64
cp -a /mnt/c/Users/davei/a2h/builds/thor-discovery-Linux64/. \
  ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/
chmod +x ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

## Discovery-world smoke test

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_discovery_world.py \
  --scene FloorPlan1Discovery_physics \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

## Run the Discovery session

```bash
cd ~/projects/author
./scripts/run_discovery_session_wsl.sh
```

Or explicitly:

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
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

## What to look for in v0.9.1

For a messaging task such as:

```text
/a Text B and ask B to help with the door.
```

completion requires a console event like:

```text
[A texts B] Can you help with the door?
```

and B will subsequently receive:

```text
[B text from A] Can you help with the door?
```

For a pickup task, a model that has already focused the reachable apple may show:

```text
[A-task ...] PickupObject("apple") motor-handoff: ...
```

The native action is still grounded to the actual visible object and verified after execution.

For an unsupported/free-form task, an unsupported completion claim is now visible rather than silently accepted:

```text
[A task-claim ignored] ... (no verifiable completion evidence yet)
```

Use `/a roam`, `/b roam`, or `/both roam` to cancel a task that has no deterministic completion contract.

For `Escape` in `FloorPlan1Discovery_physics`, completion requires the body to physically cross the north or east doorway. The motor layer may open a reachable door only after that agent itself has selected/grounded a door; it does not choose which door to take.
