#!/usr/bin/env bash
# Timestamped, checksummed snapshot. Never overwrites an existing snapshot.
set -euo pipefail
SRC="${1:?usage: snapshot.sh <path-to-snapshot> <label>}"
LABEL="${2:?usage: snapshot.sh <path-to-snapshot> <label>}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS=$(date -u +%Y%m%dT%H%M%SZ)
DEST="${BASE_DIR}/checkpoints/${TS}_${LABEL}"

if [ -e "$DEST" ]; then
  echo "ERROR: $DEST already exists — refusing to overwrite." >&2
  exit 1
fi

mkdir -p "$DEST"
cp -a "$SRC" "$DEST/"
find "$DEST" -type f ! -name checksums.sha256 -exec sha256sum {} \; > "$DEST/checksums.sha256"
echo "${TS} ${LABEL} ${DEST}" >> "${BASE_DIR}/logs/version_log.txt"
echo "Snapshot written: $DEST"
