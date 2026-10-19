#!/usr/bin/env bash
set -euo pipefail

WIN_BUILD="${WIN_BUILD:-/mnt/c/Users/davei/a2h/builds/thor-discovery-Linux64}"
WSL_BUILD="${WSL_BUILD:-$HOME/projects/ai2thor-humanoid/builds/thor-discovery-Linux64}"

if [[ ! -f "$WIN_BUILD/thor-discovery-Linux64" ]]; then
  echo "ERROR: Windows Discovery build not found: $WIN_BUILD/thor-discovery-Linux64" >&2
  exit 2
fi

mkdir -p "$(dirname "$WSL_BUILD")"
rm -rf "$WSL_BUILD"
cp -a "$WIN_BUILD" "$WSL_BUILD"
chmod +x "$WSL_BUILD/thor-discovery-Linux64"

echo "Synced rebuilt Discovery player:"
echo "  from: $WIN_BUILD"
echo "  to:   $WSL_BUILD"
ls -lh "$WSL_BUILD/thor-discovery-Linux64"
