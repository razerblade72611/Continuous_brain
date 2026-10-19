#!/usr/bin/env python3
"""Build a deterministic source-only release, with an embedded SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output path; release archives are not overwritten")
    excluded = {"__pycache__", ".pytest_cache", ".git", ".venv", "state", "dist", "build"}
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in excluded or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Unexpected source symlink: {rel}")
        if path.is_file() and path.suffix not in {".pyc", ".zip", ".pt", ".sqlite3"}:
            files.append((rel.as_posix(), path.read_bytes()))
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files}
    files.append(("MANIFEST_SHA256.json", json.dumps(manifest, indent=2, sort_keys=True).encode()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in files:
            info = zipfile.ZipInfo("continuous_brain_project/" + name, (2026,9,3,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            z.writestr(info, data)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for name, digest in manifest.items():
            assert hashlib.sha256(z.read("continuous_brain_project/" + name)).hexdigest() == digest
    return {"file": str(output), "files": len(files), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
