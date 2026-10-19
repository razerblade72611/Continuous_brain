# v0.6.1 Humanoid AI2-THOR Renderer

This directory contains the v0.6.1 visual hotfix for the two-agent AI2-THOR build.

Runtime logs from v0.6.0 proved `HumanoidPeerAvatar.ApplyAll(...)` executed for both agents,
so v0.6.1 does not move the lifecycle hook. It fixes the renderer hierarchy itself.

## What changed from v0.6.0

AI2-THOR 4.3 default/tall mode uses `TallVisCap` as `VisibilityCapsule`. v0.6.1 therefore:

- keeps `TallVisCap` as `VisibilityCapsule`;
- creates `V061HumanoidPeerAvatar` as a child of `TallVisCap`;
- suppresses every pre-existing stock TallVisCap renderer using
  `Renderer.forceRenderingOff = true`;
- puts every humanoid primitive on the stock visual layer;
- uses `ShadowCastingMode.Off` on humanoid renderers;
- calls `FirstPersonCharacterCull.SwitchRenderersToHide(TallVisCap)` after creation;
- centers the visual around the tall controller origin instead of treating the agent origin
  as floor level;
- leaves all generated primitive colliders disabled.

It does **not** modify CharacterController/CapsuleCollider dimensions, camera/FOV, navigation,
hand mechanics, interaction reach, AI2-THOR object metadata, or the Python social runtime.

## Upgrade an existing v0.6.0 AI2-THOR checkout

From WSL:

```bash
cd ~/projects/author
source ~/.venvs/author/bin/activate

python unity_humanoid/apply_humanoid_patch.py ~/projects/ai2thor-humanoid
python unity_humanoid/verify_patch.py ~/projects/ai2thor-humanoid
```

On a v0.6.0-patched checkout, the patcher upgrades the existing hook in place rather than
adding a duplicate. It creates:

```text
unity/Assets/Scripts/AgentManager.cs.v061.bak
unity/Assets/Scripts/HumanoidPeerAvatar.cs.v061.bak
```

The second file is created when an older avatar source already exists.

To return exactly to the pre-v0.6.1 state:

```bash
python unity_humanoid/apply_humanoid_patch.py ~/projects/ai2thor-humanoid --revert
```

## Verify source before building

```bash
python unity_humanoid/verify_patch.py ~/projects/ai2thor-humanoid
```

Expected checks include:

```text
OK: native TallVisCap host used
OK: stock renderer suppression present
OK: culler-compatible shadow mode present
OK: runtime verification marker present
```

## Copy the patched Unity project to Windows

The working setup for this project uses Unity 2020.3.48f1 on Windows with Linux Build
Support (Mono), then copies the Linux player back to WSL.

From WSL:

```bash
rm -rf /mnt/c/Users/davei/a2h/unity-2020.3.48-v061
mkdir -p /mnt/c/Users/davei/a2h/unity-2020.3.48-v061

rsync -rltv --copy-links \
  --delete \
  --no-perms --no-owner --no-group \
  --exclude Library --exclude Temp --exclude Logs --exclude obj \
  ~/projects/ai2thor-humanoid/unity/ \
  /mnt/c/Users/davei/a2h/unity-2020.3.48-v061/
```

## Windows build

PowerShell:

```powershell
$Unity48 = "C:\Program Files\Unity\Hub\Editor\2020.3.48f1\Editor\Unity.exe"
$Project = "C:\Users\davei\a2h\unity-2020.3.48-v061"
$OutDir  = "C:\Users\davei\a2h\builds\thor-humanoid-v061-Linux64"
$BuildLog = "C:\Users\davei\a2h\thor-humanoid-v061-build.log"

Remove-Item $OutDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $OutDir | Out-Null

$env:UNITY_BUILD_NAME = "$OutDir\thor-humanoid-v061-Linux64"
$env:BUILD_SCENES = "FloorPlan1_physics"

$BuildArgs = @(
  "-quit",
  "-batchmode",
  "-projectPath", "`"$Project`"",
  "-buildTarget", "Linux64",
  "-executeMethod", "Build.Linux64",
  "-logFile", "`"$BuildLog`""
)

$p = Start-Process -FilePath $Unity48 -ArgumentList $BuildArgs -Wait -PassThru
Write-Host "REAL Unity build exit code: $($p.ExitCode)"

Test-Path "$OutDir\thor-humanoid-v061-Linux64"
Test-Path "$OutDir\thor-humanoid-v061-Linux64_Data"
Test-Path "$OutDir\UnityPlayer.so"
```

## Copy back to WSL

```bash
rm -rf ~/projects/ai2thor-humanoid/builds/thor-humanoid-v061-Linux64
mkdir -p ~/projects/ai2thor-humanoid/builds/thor-humanoid-v061-Linux64
cp -a /mnt/c/Users/davei/a2h/builds/thor-humanoid-v061-Linux64/. \
  ~/projects/ai2thor-humanoid/builds/thor-humanoid-v061-Linux64/
chmod +x ~/projects/ai2thor-humanoid/builds/thor-humanoid-v061-Linux64/thor-humanoid-v061-Linux64
```

## Runtime verification

After starting a two-agent session, verify both replacement markers:

```bash
grep -nF 'v0.6.1 humanoid visual replacement applied' \
  "$HOME/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR/Player.log"
```

Each line also reports the host name, number of stock renderers suppressed, number of
humanoid renderers, and selected visual layer.

Then launch the `author` runtime with:

```bash
--thor-executable /home/davei/projects/ai2thor-humanoid/builds/thor-humanoid-v061-Linux64/thor-humanoid-v061-Linux64
```
