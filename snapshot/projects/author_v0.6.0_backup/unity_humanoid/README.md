# v0.6 Humanoid AI2-THOR Renderer

This directory contains the **source patch** needed to make the two AI2-THOR agents look
like humanoid robots to each other.

The Python package works without this patch, but stock AI2-THOR will continue rendering
its normal peer body. The custom Unity player must be compiled separately.

## What the patch changes

It adds `HumanoidPeerAvatar.cs` and inserts one hook immediately after AI2-THOR creates
all requested agents:

```csharp
HumanoidPeerAvatar.ApplyAll(this.agents, this.agentColors);
```

The helper disables the old peer-visible capsule renderer, creates a humanoid visual root,
assigns that root as `VisibilityCapsule`, and updates first-person self-culling.

It does **not** modify:

- `CharacterController`
- agent camera position or FOV
- navigation/grid movement
- pickup/put/open/close actions
- interaction distance
- agent hand mechanics
- AI2-THOR object metadata

The humanoid parts have their primitive colliders disabled.

## Recommended AI2-THOR source

The Python package is pinned to AI2-THOR 4.3.x. Release 4.3.0 corresponds to commit
`a634e28`.

Create a source checkout outside the `author` repository, for example:

```bash
cd ~/projects
git clone https://github.com/allenai/ai2thor.git ai2thor-humanoid
cd ai2thor-humanoid
git checkout a634e28
```

## Apply the patch

From the installed `author` project:

```bash
cd ~/projects/author
python unity_humanoid/apply_humanoid_patch.py ~/projects/ai2thor-humanoid
python unity_humanoid/verify_patch.py ~/projects/ai2thor-humanoid
```

The patcher creates:

```text
unity/Assets/Scripts/AgentManager.cs.v060.bak
```

before editing `AgentManager.cs`.

To revert:

```bash
python unity_humanoid/apply_humanoid_patch.py ~/projects/ai2thor-humanoid --revert
```

## Unity editor/build

For AI2-THOR 4.3.0 / commit `a634e28`, the recorded editor is:

```text
Unity 2020.3.25f1
revision 9b9180224418
```

You can verify this directly in:

```text
unity/ProjectSettings/ProjectVersion.txt
```

AI2-THOR's build tooling provides an `invoke local-build` task. Check the exact task
options in your checkout with:

```bash
cd ~/projects/ai2thor-humanoid
python -m pip install invoke
invoke --help local-build
```

AI2-THOR 4.3.0 defines `local-build` with `prefix`, `arch`, and iterable `scenes`, so for a Linux player containing only FloorPlan1 use:

```bash
invoke local-build --prefix humanoid --arch Linux64 --scenes FloorPlan1
```

The expected build directory is under:

```text
unity/builds/thor-humanoid-Linux64/
```

Still run `invoke --help local-build` first on your checkout so any local Invoke syntax differences are visible before a long Unity build.

After the build succeeds, locate the generated Linux executable under the checkout's
`builds/` directory and test it from `author`:

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python scripts/smoke_test_multi_thor.py \
  --thor-executable /absolute/path/to/custom/player
```

The shipped smoke script may also be run through the main CLI using `--thor-executable`.

## Runtime

Once the player path is known:

```bash
GALLIUM_DRIVER=d3d12 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
python -m agent_core.cli \
  --scene FloorPlan1 \
  --session \
  --agents 2 \
  --llm --vision --interact --dashboard \
  --thor-executable /absolute/path/to/custom/player \
  --vision-every 1 \
  --step-delay 0.08 \
  --roam-decisions 500 \
  --max-steps 3000
```

## Important limitation

This ZIP contains the patch source and tested patching tools, **not a precompiled Unity
binary**. A Unity editor is not available in the build environment used to assemble this
package, so compilation must be performed on a machine with the matching Unity editor and
Linux build support installed.
