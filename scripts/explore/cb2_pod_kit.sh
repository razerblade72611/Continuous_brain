#!/usr/bin/env bash
# On David's WSL: pack everything a GPU pod needs for one CB2 training round into cb_exchange.
#   bash scripts/explore/cb2_pod_kit.sh r1
# The kit holds: the code (a git bundle of this checkout's branch), the vision organ's weights, the canonical
# individual's home and its lived archive.  Practice is not packed: the pod regenerates it from the same seeds
# (run number = the round's number), which the archive's manifests make checkable.
# Nothing here changes the individual; stop the viewer first only so its last moments are in the archive.
set -euo pipefail
ROUND=${1:?round label, e.g. r1}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=/mnt/c/Users/davei/cb_exchange/cb2_kit_$ROUND.tgz
STAGE=$(mktemp -d)
cd "$REPO"
git bundle create "$STAGE/code.bundle" HEAD 2>/dev/null
git rev-parse HEAD > "$STAGE/code.rev"
git rev-parse --abbrev-ref HEAD > "$STAGE/code.branch"
cp runs/explore/vision/tile_decoder_v1.pt "$STAGE/"
mkdir -p "$STAGE/cb2/archive"
cp -r runs/explore/cb2/individuals "$STAGE/cb2/"
for d in runs/explore/cb2/archive/lived-*; do cp -r "$d" "$STAGE/cb2/archive/"; done
cp scripts/explore/cb2_pod_run.sh "$STAGE/"
tar -C "$STAGE" -czf "$OUT" .
rm -rf "$STAGE"
echo "kit: $OUT ($(du -h "$OUT" | cut -f1)), sha256 $(sha256sum "$OUT" | cut -c1-16)"
