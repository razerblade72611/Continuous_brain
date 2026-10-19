"""Run provenance (DAV-76 evidence rule).

Every run can be traced from code revision + config + seed + anatomy +
checkpoint + Linear issue. A claim that a run passed or failed must cite a
primary artifact by path and SHA-256; `RunRecord.status` is "UNVERIFIED" until
an artifact is attached.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def code_revision(start: str | os.PathLike | None = None) -> dict:
    """Git revision of the working tree containing this package, and whether it is dirty."""
    cwd = Path(start or Path(__file__).resolve().parent)
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
        return {"git_rev": rev or None, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_rev": None, "dirty": None}


def environment() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


@dataclass
class RunRecord:
    run_id: str
    kind: str                           # e.g. "e0_gate", "smoke", "experiment"
    linear_issue: str | None
    seed: int
    config_hash: str
    anatomy_hash: str
    device: str
    code: dict = field(default_factory=code_revision)
    env: dict = field(default_factory=environment)
    started_utc: float = field(default_factory=time.time)
    artifacts: dict[str, str] = field(default_factory=dict)   # path -> sha256
    checkpoint: str | None = None
    status: str = "UNVERIFIED"
    notes: str = ""

    def attach(self, path: str | os.PathLike) -> str:
        digest = sha256_file(path)
        self.artifacts[str(path)] = digest
        return digest


class RunRegistry:
    """Append-only registry at <root>/logs/run_registry.jsonl."""

    def __init__(self, root: str | os.PathLike):
        self.path = Path(root) / "logs" / "run_registry.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, record: RunRecord) -> None:
        if record.status in ("PASS", "FAIL") and not record.artifacts:
            raise ValueError("a PASS/FAIL claim needs at least one primary artifact (path + SHA-256)")
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
