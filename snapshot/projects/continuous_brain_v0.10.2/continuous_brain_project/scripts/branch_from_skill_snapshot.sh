#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 /path/to/L0_mastered.pt state/new_branch" >&2
  exit 2
fi
SNAPSHOT="$1"
DEST="$2"
[[ -f "$SNAPSHOT" ]] || { echo "snapshot not found: $SNAPSHOT" >&2; exit 2; }
[[ ! -e "$DEST/brain.pt" ]] || { echo "refusing to overwrite existing $DEST/brain.pt" >&2; exit 2; }
mkdir -p "$DEST"
cp -a "$SNAPSHOT" "$DEST/brain.pt"
echo "Created skill branch: $DEST"
echo "The snapshot restores neural/optimizer/curriculum/replay learning state; it does not clone episodic.sqlite3 autobiographical memory."
