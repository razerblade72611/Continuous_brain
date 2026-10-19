"""Content-addressed run directories and the append-only result registry.

  runs/<run_id>/spec.json       the full run specification (config hashes, seeds,
                                substrate, decision rule, code revision)
  runs/<run_id>/result.json     the ResultRecord
  runs/<run_id>/manifest.json   SHA-256 of every file in the run directory
  logs/results.jsonl            one line per finalized run (append-only, fsync'd)

run_id is the first 20 hex digits of the SHA-256 of the canonical spec, so an
identical spec maps to the same directory: re-running it is refused unless a
replicate label is part of the spec. That makes accidental re-opening of a
sealed TEST configuration a hard error rather than a silent overwrite.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from continuous_brain_protocols import ResultStatus, canonical_json, sha256_file, sha256_json
from continuous_brain_unified.prereg import PreregistrationRegistry

from .results import ResultRecord


class RunExistsError(FileExistsError):
    pass


@dataclass
class RunDir:
    run_id: str
    path: Path
    spec: dict

    def artifact(self, name: str) -> Path:
        p = (self.path / name).resolve()
        if self.path.resolve() not in p.parents:
            raise ValueError("artifacts live inside the run directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


class RunStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        self.registry = self.root / "logs" / "results.jsonl"

    @staticmethod
    def run_id(spec: dict) -> str:
        return sha256_json(spec)[:20]

    def create(self, spec: dict[str, Any]) -> RunDir:
        rid = self.run_id(spec)
        path = self.root / "runs" / rid
        try:
            os.mkdir(path)
        except FileExistsError:
            raise RunExistsError(f"run {rid} already exists for this exact spec; add a replicate label "
                                 f"to the spec to run it again") from None
        (path / "spec.json").write_text(canonical_json(spec) + "\n")
        return RunDir(rid, path, spec)

    def finalize(self, run: RunDir, record: ResultRecord, registry: PreregistrationRegistry | None = None) -> Path:
        """Attach every file as an artifact, turn a pending gate verdict into the status, validate, seal.

        A run is finalized once: an existing result.json is never overwritten."""
        if (run.path / "result.json").exists() or (run.path / "manifest.json").exists():
            raise RunExistsError(f"run {run.run_id} is already finalized")
        artifacts = {}
        for p in sorted(run.path.rglob("*")):
            if p.is_file():
                artifacts[str(p.relative_to(run.path))] = sha256_file(p)
        record.provenance.setdefault("artifacts", {}).update(artifacts)
        record.provenance["run_id"] = run.run_id
        verdict = record.metrics.get("verdict")
        if verdict in ("PASS", "FAIL") and record.status == ResultStatus.UNVERIFIED:
            record.status = ResultStatus.PASS if verdict == "PASS" else ResultStatus.FAIL
        record.validate(registry)
        (run.path / "result.json").write_text(json.dumps(record.as_dict(), indent=1, sort_keys=True, default=str) + "\n")
        files = {str(p.relative_to(run.path)): sha256_file(p) for p in sorted(run.path.rglob("*"))
                 if p.is_file() and p.name != "manifest.json"}
        manifest = run.path / "manifest.json"
        manifest.write_text(json.dumps({"run_id": run.run_id, "files": files}, indent=1, sort_keys=True) + "\n")
        for p in run.path.rglob("*"):
            if p.is_file():
                p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        entry = {"run_id": run.run_id, "item": record.item, "status": record.status.value,
                 "substrate": record.substrate, "manifest_sha256": sha256_file(manifest),
                 "result_sha256": files["result.json"], "utc": datetime.now(timezone.utc).isoformat()}
        with open(self.registry, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return manifest

    def entries(self) -> list[dict]:
        if not self.registry.exists():
            return []
        return [json.loads(line) for line in self.registry.read_text().splitlines() if line.strip()]

    def verify(self, run_id: str) -> bool:
        """True iff every manifested file is present and unchanged and no file was added since."""
        path = self.root / "runs" / run_id
        try:
            manifest = json.loads((path / "manifest.json").read_text())
        except (OSError, ValueError):
            return False
        present = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and p.name != "manifest.json"}
        if present != set(manifest["files"]):
            return False
        return all(sha256_file(path / rel) == digest for rel, digest in manifest["files"].items())
