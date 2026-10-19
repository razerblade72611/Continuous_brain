"""Verify the shipped source/starter against the release inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def verify(root):
    root = Path(root).resolve()
    manifest = root / "RELEASE_MANIFEST.json"
    if not manifest.is_file():
        raise FileNotFoundError("RELEASE_MANIFEST.json is missing. Extract the complete ZIP first.")
    data = json.loads(manifest.read_text())
    failures = []
    for name, expected in data["files"].items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            failures.append(name + " (missing/invalid path)")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name + " (modified)")
    if failures:
        raise ValueError("Release verification failed:\n" + "\n".join(failures[:12]))
    return {"build": data["build"], "verified_files": len(data["files"])}


if __name__ == "__main__":
    try:
        result = verify(Path(__file__).resolve().parent)
        print(f"Release verified: {result['build']} ({result['verified_files']} files).")
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
