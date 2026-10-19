# AUTHOR v0.9.3 Operator README

## Purpose

v0.9.3 repairs the physical Discovery doorways and reduces peer-text echo loops while preserving the visible, auditable A↔B channel.

## Upgrade from v0.9.2 without losing state

WSL:

```bash
cd ~/projects
mv author author_v0.9.2_backup
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.9.3_physical_doorway_anti_echo.zip -d ~/projects
cp -a ~/projects/author_v0.9.2_backup/state ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected: `127 passed`.

## Reapply/sync the Discovery Unity patch

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
OK: v0.9.3 real hinge swing configured
OK: v0.9.3 physical portal clearance configured
OK: runtime door static flags cleared
```

## Rebuild on Windows

Keep llama.cpp in its own PowerShell window. In another Windows PowerShell:

```powershell
& C:\Users\davei\a2h\build_discovery_world.ps1
```

Required result:

```text
Discovery scene generation exit code: 0
Discovery player build exit code: 0
Executable: True
Scene generated: True
```

## Copy the rebuilt player back to WSL

```bash
rm -rf ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64
mkdir -p ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64
cp -a /mnt/c/Users/davei/a2h/builds/thor-discovery-Linux64/. ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/
chmod +x ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

## Mechanical Discovery smoke test

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_discovery_world.py \
  --scene FloorPlan1Discovery_physics \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

Expected shape:

```text
Opened: Door|... isOpen= True
Opened: Door|... isOpen= True
Reachable positions: ... north-extension: <nonzero> east-extension: <nonzero>
PASS: two openable exits, both room reachability extensions, and interactive content are present.
```

## llama.cpp

Windows PowerShell:

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

WSL connection check:

```bash
WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')
export AGENT_LLM_BASE_URL="http://$WINDOWS_HOST:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"
curl -sS "$AGENT_LLM_BASE_URL/models" | python -m json.tool
```

## Recommended two-agent Discovery run

Use a text opportunity every 3 turns. v0.9.3 suppresses echoes even at cadence 1, but cadence 3 gives the agents more new embodied information between messages.

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
  --roam-decisions 500 \
  --max-steps 3000 \
  --peer-text-every 3 \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

Dashboard: `http://localhost:8765`

## Watch only the peer text log

```bash
tail -f ~/projects/author/state/messages/peer_texts.jsonl
```

Console messages remain visible as:

```text
[A texts B] ...
[B text from A] ...
[B texts A] ...
[A text from B] ...
```

Periodic peer messages may intentionally be silent when the composer returns `NO_MESSAGE` or when a near-duplicate would otherwise be sent. Explicit user tasks such as `/a Text B and ask what B can see.` still trigger a real delivery attempt and are not considered complete without delivery.
