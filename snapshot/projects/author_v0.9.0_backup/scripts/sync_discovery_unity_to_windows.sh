#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
AI2THOR_ROOT="${AI2THOR_ROOT:-$HOME/projects/ai2thor-humanoid}"
WIN_PROJECT="${WIN_PROJECT:-/mnt/c/Users/davei/a2h/unity-2020.3.48}"

cp "$AI2THOR_ROOT/unity/Assets/Scripts/SimObjType.cs" "$WIN_PROJECT/Assets/Scripts/SimObjType.cs"
cp "$AI2THOR_ROOT/unity/Assets/Editor/DiscoveryWorldBuilder.cs" "$WIN_PROJECT/Assets/Editor/DiscoveryWorldBuilder.cs"
# Keep the already-working humanoid visual patch synchronized too.
cp "$AI2THOR_ROOT/unity/Assets/Scripts/AgentManager.cs" "$WIN_PROJECT/Assets/Scripts/AgentManager.cs"
cp "$AI2THOR_ROOT/unity/Assets/Scripts/HumanoidPeerAvatar.cs" "$WIN_PROJECT/Assets/Scripts/HumanoidPeerAvatar.cs"

echo "Synced v0.9.0 Discovery World + humanoid source into: $WIN_PROJECT"
