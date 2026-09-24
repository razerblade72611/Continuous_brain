"""Never-overwrite, checksummed snapshots of all learned state.

Policy (spec: "Hardware plan & state preservation"):
  * Every snapshot is a new timestamped directory. Nothing is overwritten in place.
  * A snapshot directory is claimed atomically (os.mkdir fails if it exists).
  * Every file gets a SHA-256 checksum; files are made read-only once written.
  * A COMPLETE marker is written last, so a crash mid-write leaves an
    incomplete snapshot that is ignored by `latest()` and fails `verify()`.
  * Every snapshot appends a line to an append-only version log.
  * Restoring copies into a directory that must not already exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

CHECKSUM_FILE = "checksums.sha256"
MANIFEST_FILE = "manifest.json"
COMPLETE_MARKER = "COMPLETE"


class SnapshotExistsError(FileExistsError):
    """Raised instead of ever overwriting an existing snapshot."""


class SnapshotIntegrityError(RuntimeError):
    """Raised when a snapshot fails verification."""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _make_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _utc_stamp() -> str:
    # Microsecond resolution; uniqueness is still enforced by atomic mkdir.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _safe_label(label: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in label)
    if not cleaned:
        raise ValueError("snapshot label must contain at least one safe character")
    return cleaned


@dataclass(frozen=True)
class SnapshotInfo:
    path: Path
    label: str
    stamp: str
    complete: bool


class SnapshotStore:
    """Owns `<root>/checkpoints/` and `<root>/logs/version_log.jsonl`."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.checkpoints = self.root / "checkpoints"
        self.logs = self.root / "logs"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.version_log = self.logs / "version_log.jsonl"

    # ------------------------------------------------------------------ writing

    def _claim(self, label: str) -> Path:
        label = _safe_label(label)
        for _ in range(10):
            dest = self.checkpoints / f"{_utc_stamp()}_{label}"
            try:
                os.mkdir(dest)  # atomic claim; never exist_ok
                return dest
            except FileExistsError:
                time.sleep(0.001)
        raise SnapshotExistsError(f"could not claim a unique snapshot directory for {label!r}")

    def _finalize(self, dest: Path, label: str, meta: dict[str, Any]) -> Path:
        files = sorted(p for p in dest.rglob("*") if p.is_file())
        sums = {str(p.relative_to(dest)): _sha256(p) for p in files}
        manifest = {
            "label": label,
            "created_utc": dest.name.split("_", 1)[0],
            "files": sums,
            "meta": meta,
        }
        (dest / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True))
        with open(dest / CHECKSUM_FILE, "w") as f:
            for rel, digest in sums.items():
                f.write(f"{digest}  {rel}\n")
        for p in files + [dest / MANIFEST_FILE, dest / CHECKSUM_FILE]:
            _make_read_only(p)
        marker = dest / COMPLETE_MARKER
        marker.write_text(json.dumps({"n_files": len(sums)}))
        _make_read_only(marker)
        with open(self.version_log, "a") as log:
            log.write(json.dumps({"stamp": manifest["created_utc"], "label": label,
                                  "path": str(dest), "n_files": len(sums), "meta": meta}) + "\n")
            log.flush()
            os.fsync(log.fileno())
        return dest

    def snapshot_files(self, sources: Iterable[str | os.PathLike], label: str,
                       meta: dict[str, Any] | None = None) -> Path:
        """Copy existing files/directories into a new snapshot."""
        sources = [Path(s) for s in sources]
        for s in sources:
            if not s.exists():
                raise FileNotFoundError(s)
        dest = self._claim(label)
        for s in sources:
            target = dest / s.name
            if s.is_dir():
                shutil.copytree(s, target)
            else:
                shutil.copy2(s, target)
        return self._finalize(dest, label, meta or {})

    def save_state(self, states: dict[str, Any], label: str,
                   meta: dict[str, Any] | None = None) -> Path:
        """Save named state dicts (torch-serializable) into a new snapshot."""
        if not states:
            raise ValueError("nothing to snapshot")
        dest = self._claim(label)
        for name, state in states.items():
            torch.save(state, dest / f"{_safe_label(name)}.pt")
        return self._finalize(dest, label, meta or {})

    # ------------------------------------------------------------------ reading

    def verify(self, snapshot: str | os.PathLike) -> bool:
        dest = Path(snapshot)
        if not (dest / COMPLETE_MARKER).exists():
            return False
        manifest = json.loads((dest / MANIFEST_FILE).read_text())
        for rel, digest in manifest["files"].items():
            p = dest / rel
            if not p.is_file() or _sha256(p) != digest:
                return False
        return True

    def list(self) -> list[SnapshotInfo]:
        out = []
        for d in sorted(self.checkpoints.iterdir()):
            if not d.is_dir() or "_" not in d.name:
                continue
            stamp, label = d.name.split("_", 1)
            out.append(SnapshotInfo(d, label, stamp, (d / COMPLETE_MARKER).exists()))
        return out

    def latest(self, label: str | None = None) -> Path | None:
        candidates = [s for s in self.list() if s.complete and (label is None or s.label == label)]
        return candidates[-1].path if candidates else None

    def load_state(self, snapshot: str | os.PathLike, name: str,
                   map_location: str | torch.device = "cpu", verify: bool = True) -> Any:
        dest = Path(snapshot)
        if verify and not self.verify(dest):
            raise SnapshotIntegrityError(f"snapshot failed verification: {dest}")
        return torch.load(dest / f"{_safe_label(name)}.pt", map_location=map_location,
                          weights_only=False)

    def restore_to(self, snapshot: str | os.PathLike, dest: str | os.PathLike) -> Path:
        """Copy a verified snapshot to a NEW directory (never overwrites)."""
        src = Path(snapshot)
        dest = Path(dest)
        if dest.exists():
            raise SnapshotExistsError(f"refusing to restore over existing path: {dest}")
        if not self.verify(src):
            raise SnapshotIntegrityError(f"snapshot failed verification: {src}")
        shutil.copytree(src, dest)
        for p in dest.rglob("*"):
            if p.is_file():
                p.chmod(p.stat().st_mode | stat.S_IWUSR)
        return dest
