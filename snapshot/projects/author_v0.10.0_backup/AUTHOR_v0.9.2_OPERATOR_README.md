# AUTHOR v0.9.2 Operator README

## What this fixes

v0.9.2 fixes the exact failure `north-extension: 0 east-extension: 0`. AI2-THOR 4.3.0's reachable-position flood-fill treats only layer-8 colliders named exactly `Floor` as walkable. The v0.9.1 generated floor/threshold collider names were therefore blocking both exits. v0.9.2 uses descriptive marker parents with actual collider children named `Floor`.

It also makes A/B text communication visible and reliable. By default each agent gets a dedicated LLM peer-text opportunity every 3 turns. Text is printed in the terminal, shown in the dashboard event stream, persisted in `state/messages/peer_texts.jsonl`, and provided to the recipient's next LLM turn. No private peer state is copied.

## Upgrade while preserving state

WSL:

```bash
cd ~/projects
mv author author_v0.9.1_backup
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.9.2_walkable_discovery_visible_peer_link.zip -d ~/projects
cp -a ~/projects/author_v0.9.1_backup/state ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

## Reapply/rebuild the Discovery world

WSL:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python unity_world/apply_discovery_world_patch.py ~/projects/ai2thor-humanoid
python unity_world/verify_discovery_world_patch.py ~/projects/ai2thor-humanoid
./scripts/sync_discovery_unity_to_windows.sh
cp scripts/build_discovery_world.ps1 /mnt/c/Users/davei/a2h/build_discovery_world.ps1
```

Windows PowerShell:

```powershell
& C:\Users\davei\a2h\build_discovery_world.ps1
```

Expected:

```text
Discovery scene generation exit code: 0
Discovery player build exit code: 0
Executable: True
Scene generated: True
```

WSL copy-back:

```bash
rm -rf ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64
mkdir -p ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64
cp -a /mnt/c/Users/davei/a2h/builds/thor-discovery-Linux64/. ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/
chmod +x ~/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

## Mechanical world test

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_discovery_world.py \
  --scene FloorPlan1Discovery_physics \
  --thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

Success requires nonzero `north-extension` and `east-extension` and ends with:

```text
PASS: two openable exits, both room reachability extensions, and interactive content are present.
```

## Run A + B with visible peer text

Start llama.cpp on Windows, then WSL:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')
export AGENT_LLM_BASE_URL="http://$WINDOWS_HOST:8080/v1"
export AGENT_LLM_MODEL="qwen2-vl-2b"
curl -s "$AGENT_LLM_BASE_URL/models" | python -m json.tool

./scripts/run_discovery_session_wsl.sh
```

Default peer text cadence is every 3 turns. For more communication:

```bash
PEER_TEXT_EVERY=1 ./scripts/run_discovery_session_wsl.sh
```

Or with the full CLI, add:

```text
--peer-text-every 1
```

Disable periodic peer text with `--peer-text-every 0`. Explicit user texting tasks still use the dedicated text recovery path.

Visible output looks like:

```text
[A texts B] I found a closed door near the refrigerator. Can you check the other exit?
[B text from A] I found a closed door near the refrigerator. Can you check the other exit?
[B texts A] I can see another doorway on my side. I'll inspect it.
```

Persistent transcript:

```bash
tail -f state/messages/peer_texts.jsonl
```

Dashboard:

```text
http://localhost:8765
```

If dedicated LLM message composition fails, the transport can emit a clearly marked fallback such as `[status relay] ...`; this is deliberately labeled so it is never mistaken for LLM-authored dialogue.
