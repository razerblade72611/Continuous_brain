# AUTHOR v0.10.9 — Runtime Stool Sanitizer

## Why this release exists

The v0.10.8 Discovery geometry smoke test proved the new maneuver-clear room geometry was present, but two stock FloorPlan1 stools still appeared at runtime:

- `Stool|+00.70|+00.00|-00.51`
- `Stool|+00.74|+00.00|+00.56`

The editor-only stool removal pass therefore was not authoritative for the runtime `SimObjPhysics` representation used by AI2-THOR.

v0.10.9 keeps all v0.10.8 room-clearance geometry and adds a runtime sanitizer attached to the generated Discovery root. It runs in `Awake()` with early execution order and removes every runtime `SimObjPhysics` that identifies as a stool by type, object ID, asset ID, or object name. The editor removal pass is also hardened to scan `Resources.FindObjectsOfTypeAll<SimObjPhysics>()`.

## Validation

- 183 Python/source regression tests pass.
- Runtime sanitizer source is shipped under `unity_world/DiscoveryRuntimeSanitizer.cs`.
- Patch installer copies it to `unity/Assets/Scripts/DiscoveryRuntimeSanitizer.cs`.
- Windows sync copies it to the Windows Unity project.
- Patch verifier checks that the sanitizer exists and is attached by the builder.
- The existing runtime smoke test remains authoritative: it must report `Stool objects: []` before training resumes.

## Upgrade from v0.10.8

Preserve learned state; do not use `--reset-memory`.

```bash
cd ~/projects
BACKUP=~/projects/author_v0.10.8_backup_$(date +%Y%m%d-%H%M%S)
mv ~/projects/author "$BACKUP"
unzip /mnt/c/Users/davei/Downloads/author_phase9_v0.10.9_runtime_stool_sanitizer.zip -d ~/projects
cp -a "$BACKUP/state" ~/projects/author/
cd ~/projects/author
source ~/.venvs/author/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected: `183 passed`.

## Install/sync the Unity patch

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python unity_world/apply_discovery_world_patch.py ~/projects/ai2thor-humanoid
python unity_world/verify_discovery_world_patch.py ~/projects/ai2thor-humanoid
./scripts/sync_discovery_unity_to_windows.sh
```

The verifier must include OK lines for `DiscoveryRuntimeSanitizer.cs` and the runtime sanitizer attachment.

## Rebuild in Windows PowerShell

```powershell
& C:\Users\davei\a2h\build_discovery_world.ps1
```

Expected:

- Discovery scene generation exit code: 0
- Discovery player build exit code: 0
- Executable: True
- Scene generated: True

## Sync the rebuilt player back to WSL

```bash
cd ~/projects/author
./scripts/sync_discovery_build_to_wsl.sh
```

## Mandatory smoke test

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate
python scripts/smoke_test_discovery_world.py \
  --thor-executable \
  /home/davei/projects/ai2thor-humanoid/builds/thor-discovery-Linux64/thor-discovery-Linux64
```

Do not resume Level 3 until it reports:

```text
Stool objects: []
PASS: exits open, both rooms are reachable, stools are absent, maneuver zones are open, and interactive content is present.
```
