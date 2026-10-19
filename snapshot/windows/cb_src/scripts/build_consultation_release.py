"""Rebuild the complete local release using the already-trained starter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

BUILD = "v120-consultation-consolidated-r1"
ROOT = Path(__file__).resolve().parent.parent


def build(destination):
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("Choose a new ZIP output path")
    excluded = {"__pycache__", ".pytest_cache", ".git", ".venv_dml", ".venv_cpu",
                "state", "results", "dist", "build", "node_modules"}
    files = [p for p in sorted(ROOT.rglob("*")) if p.is_file() and
             not any(part in excluded or part.endswith(".egg-info") for part in p.relative_to(ROOT).parts) and
             p.name not in ("RELEASE_MANIFEST.json", "MANIFEST_SHA256.json") and
             p.suffix not in (".pyc", ".partial") and p.resolve() != destination]
    if not (ROOT / "assets/development_starter.pt").is_file():
        raise FileNotFoundError("The trained starter is missing")
    inventory = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    old_manifest = ROOT / "MANIFEST_SHA256.json"
    old_manifest.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files.append(old_manifest)
    inventory[old_manifest.name] = hashlib.sha256(old_manifest.read_bytes()).hexdigest()
    manifest = ROOT / "RELEASE_MANIFEST.json"
    manifest.write_text(json.dumps({"build": BUILD, "files": inventory}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files.append(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo("continuous_brain_consolidated/" + path.relative_to(ROOT).as_posix(),
                                   date_time=(2026, 9, 12, 0, 0, 0))
            info.external_attr = ((0o755 if path.name.endswith(".sh") else 0o644) | 0o100000) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    with zipfile.ZipFile(destination) as archive:
        prefix = "continuous_brain_consolidated/"
        expected = {prefix + p.relative_to(ROOT).as_posix() for p in files}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise RuntimeError("Archive inventory mismatch")
        for name, digest in inventory.items():
            if hashlib.sha256(archive.read(prefix + name)).hexdigest() != digest:
                raise RuntimeError("Archive content hash mismatch: " + name)
    return {"path": str(destination), "size_bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "files": len(files)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    print(json.dumps(build(p.parse_args().output), indent=2))
