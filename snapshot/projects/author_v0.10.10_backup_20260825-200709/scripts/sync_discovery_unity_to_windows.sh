#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
AI2THOR_ROOT="${AI2THOR_ROOT:-$HOME/projects/ai2thor-humanoid}"
WIN_PROJECT="${WIN_PROJECT:-/mnt/c/Users/davei/a2h/unity-2020.3.48}"
WIN_TOOLS_DIR="${WIN_TOOLS_DIR:-/mnt/c/Users/davei/a2h}"

cp "$AI2THOR_ROOT/unity/Assets/Scripts/SimObjType.cs" "$WIN_PROJECT/Assets/Scripts/SimObjType.cs"
cp "$AI2THOR_ROOT/unity/Assets/Editor/DiscoveryWorldBuilder.cs" "$WIN_PROJECT/Assets/Editor/DiscoveryWorldBuilder.cs"
cp "$AI2THOR_ROOT/unity/Assets/Scripts/DiscoveryRuntimeSanitizer.cs" "$WIN_PROJECT/Assets/Scripts/DiscoveryRuntimeSanitizer.cs"
# Keep the already-working humanoid visual patch synchronized too.
cp "$AI2THOR_ROOT/unity/Assets/Scripts/AgentManager.cs" "$WIN_PROJECT/Assets/Scripts/AgentManager.cs"
cp "$AI2THOR_ROOT/unity/Assets/Scripts/HumanoidPeerAvatar.cs" "$WIN_PROJECT/Assets/Scripts/HumanoidPeerAvatar.cs"
# Keep the Windows batch build entry point synchronized with this release.
mkdir -p "$WIN_TOOLS_DIR"
cp "scripts/build_discovery_world.ps1" "$WIN_TOOLS_DIR/build_discovery_world.ps1"

echo "Synced v0.9.4 Discovery World + humanoid source into: $WIN_PROJECT"
echo "Synced build script into: $WIN_TOOLS_DIR/build_discovery_world.ps1"
