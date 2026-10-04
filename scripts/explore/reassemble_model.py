"""Rebuild model weights that fetch_models.py split into chunks, verifying every chunk and every file.

    python scripts/explore/reassemble_model.py SPLIT_DIR [SPLIT_DIR ...] --models-dir DIR

``SPLIT_DIR`` is a folder written by ``fetch_models.py --split-mb N`` (it holds ``split_manifest.json`` and the
``*.partNNNN`` chunks, e.g. ``~/cb_models/_split/Qwen--Qwen2.5-0.5B-Instruct``).  Each chunk's SHA-256 and size
are checked against the manifest, the files are concatenated and their SHA-256 checked, and the model directory
``DIR/<org>--<name>/`` is then verified as a whole.  The model's lock entry (revision + per-file SHA-256, as
pinned on the machine that downloaded it) is added to ``DIR/models.lock.json``; if that lock already pins the
model differently, nothing is written and the script exits 2.  No network.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from continuous_brain_organs import model_store as ms  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reassemble and verify models split by fetch_models.py --split-mb.")
    ap.add_argument("split_dirs", nargs="+", type=Path, help="folders containing split_manifest.json")
    ap.add_argument("--models-dir", required=True, type=Path, help="where to rebuild the model directories")
    ap.add_argument("--lock", type=Path, default=None, help="lock file (default: <models-dir>/models.lock.json)")
    args = ap.parse_args(argv)
    models_dir = args.models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for d in args.split_dirs:
        d = d.expanduser().resolve()
        try:
            entry = ms.reassemble_model(d, models_dir, args.lock)
        except (ValueError, FileNotFoundError) as e:
            print(f"[{d.name}] REFUSED: {e}", file=sys.stderr)
            rc = 2
            continue
        total = sum(f["size"] for f in entry["files"].values())
        print(f"[{d.name}] ok: {len(entry['files'])} files, {total / 1e9:.2f} GB, revision {entry['revision']} "
              f"-> {models_dir / entry['dir']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
