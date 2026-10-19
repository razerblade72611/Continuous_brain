"""Legacy manifest: the hash-pinned list of mechanisms the runtime may bind (C1).

Only VERIFIED entries may appear in `legacy/manifest.json`. An entry becomes
VERIFIED when the DAV-74 audit has identified the mechanism's module,
evaluator, primary result, and checkpoints *in the canonical legacy tree* and
recorded their SHA-256. Candidates found elsewhere (for example the Windows
source mirror) live in `legacy/candidates.json` as UNVERIFIED and are never
bound.

The legacy root is never written into the manifest (no absolute paths are
committed). It comes from `--legacy-root` or CONTINUOUS_BRAIN_LEGACY_ROOT.
Before binding, every file of an entry is re-hashed under that root; any
mismatch refuses the binding unless an explicit developer override is given,
which is logged and marks the runtime non-scientific.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from continuous_brain_protocols import sha256_file, sha256_json

MANIFEST_FORMAT = "cb-legacy-manifest/1"
ENV_ROOT = "CONTINUOUS_BRAIN_LEGACY_ROOT"
FILE_ROLES = ("module", "evaluator", "runner", "protocol", "result", "result_bundle", "checkpoint", "support")
REQUIRED_ROLES = ("module", "evaluator", "result")   # checkpoints are verified whenever listed (role "checkpoint")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    pass


class LegacyHashMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class LegacyFile:
    relpath: str
    sha256: str
    role: str

    def as_dict(self) -> dict:
        return {"relpath": self.relpath, "sha256": self.sha256, "role": self.role}


@dataclass(frozen=True)
class ManifestEntry:
    mechanism_id: str
    slot: str
    lineage_version: str
    evidence_status: str              # dossier classification, e.g. "ADMITTED/CLOSED PASS"
    status: str                       # must be "VERIFIED" in the manifest
    files: tuple[LegacyFile, ...]
    entry_points: dict[str, str]      # e.g. {"module": "continuous_brain.x", "load": "load_reader"}
    audit_ref: str                    # link to the DAV-74 audit record
    verified_by: str
    verified_utc: str
    notes: str = ""

    def as_dict(self) -> dict:
        return {"mechanism_id": self.mechanism_id, "slot": self.slot, "lineage_version": self.lineage_version,
                "evidence_status": self.evidence_status, "status": self.status,
                "files": [f.as_dict() for f in self.files], "entry_points": dict(self.entry_points),
                "audit_ref": self.audit_ref, "verified_by": self.verified_by, "verified_utc": self.verified_utc,
                "notes": self.notes}

    def digest(self) -> str:
        return sha256_json(self.as_dict())

    def file(self, role: str) -> LegacyFile:
        for f in self.files:
            if f.role == role:
                return f
        raise KeyError(role)


def check_relpath(relpath: str) -> str:
    p = PurePosixPath(relpath)
    if p.is_absolute() or ".." in p.parts or "\\" in relpath or re.match(r"^[A-Za-z]:", relpath):
        raise ManifestError(f"legacy paths must be relative POSIX paths inside the root: {relpath!r}")
    return str(p)


def parse_entry(d: dict[str, Any]) -> ManifestEntry:
    from ..slots import MECHANISM_SLOTS
    for key in ("mechanism_id", "slot", "lineage_version", "evidence_status", "status", "files", "entry_points",
                "audit_ref", "verified_by", "verified_utc"):
        if key not in d:
            raise ManifestError(f"entry missing {key!r}: {d.get('mechanism_id')}")
    if d["status"] != "VERIFIED":
        raise ManifestError(f"{d['mechanism_id']}: only VERIFIED entries belong in the manifest "
                            f"(got {d['status']!r}); candidates go in legacy/candidates.json")
    if d["slot"] not in MECHANISM_SLOTS:
        raise ManifestError(f"{d['mechanism_id']}: {d['slot']!r} is not a mechanism slot")
    if not str(d["audit_ref"]).strip():
        raise ManifestError(f"{d['mechanism_id']}: a VERIFIED entry needs its DAV-74 audit reference")
    files = []
    for f in d["files"]:
        if f["role"] not in FILE_ROLES:
            raise ManifestError(f"{d['mechanism_id']}: unknown file role {f['role']!r}")
        if not _HEX64.match(f["sha256"]):
            raise ManifestError(f"{d['mechanism_id']}: {f['relpath']} has no valid SHA-256")
        files.append(LegacyFile(check_relpath(f["relpath"]), f["sha256"], f["role"]))
    roles = {f.role for f in files}
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    if missing:
        raise ManifestError(f"{d['mechanism_id']}: missing required file roles {missing}")
    datetime.fromisoformat(d["verified_utc"].replace("Z", "+00:00"))
    return ManifestEntry(d["mechanism_id"], d["slot"], d["lineage_version"], d["evidence_status"], d["status"],
                         tuple(files), dict(d["entry_points"]), d["audit_ref"], d["verified_by"],
                         d["verified_utc"], d.get("notes", ""))


@dataclass
class LegacyManifest:
    entries: tuple[ManifestEntry, ...] = ()
    source_sha256: str | None = None

    @classmethod
    def load(cls, path: str | os.PathLike) -> "LegacyManifest":
        raw = json.loads(Path(path).read_text())
        if raw.get("format") != MANIFEST_FORMAT:
            raise ManifestError(f"{path}: format must be {MANIFEST_FORMAT!r}")
        entries = tuple(parse_entry(e) for e in raw.get("entries", []))
        ids = [e.mechanism_id for e in entries]
        if len(ids) != len(set(ids)):
            raise ManifestError("duplicate mechanism ids")
        return cls(entries, sha256_file(path))

    def get(self, mechanism_id: str) -> ManifestEntry:
        for e in self.entries:
            if e.mechanism_id == mechanism_id:
                return e
        raise KeyError(mechanism_id)


def resolve_legacy_root(cli_root: str | os.PathLike | None = None) -> Path:
    raw = cli_root if cli_root is not None else os.environ.get(ENV_ROOT)
    if not raw:
        raise ManifestError(f"no legacy root: pass --legacy-root or set {ENV_ROOT}")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"legacy root {root} is not a directory")
    return root


@dataclass
class VerificationResult:
    mechanism_id: str
    entry_digest: str
    ok: bool
    mismatches: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    overridden: bool = False


def verify_entry(entry: ManifestEntry, root: Path, dev_override: bool = False,
                 override_log: str | os.PathLike | None = None) -> VerificationResult:
    """Re-hash every file the entry lists (module, evaluator, result, and any checkpoints/support files)."""
    if dev_override and override_log is None:
        raise ValueError("a developer override must name the log it is recorded in")
    res = VerificationResult(entry.mechanism_id, entry.digest(), True)
    for f in entry.files:
        p = (root / f.relpath).resolve()
        if root not in p.parents and p != root:
            raise ManifestError(f"{f.relpath} resolves outside the legacy root")
        if not p.is_file():
            res.missing.append(f.relpath)
            continue
        actual = sha256_file(p)
        if actual != f.sha256:
            res.mismatches.append({"relpath": f.relpath, "role": f.role, "expected": f.sha256, "actual": actual})
    res.ok = not res.mismatches and not res.missing
    if not res.ok:
        if not dev_override:
            raise LegacyHashMismatch(f"{entry.mechanism_id}: {len(res.mismatches)} hash mismatches, "
                                     f"{len(res.missing)} missing files; refusing to bind")
        res.overridden = True
        if override_log is not None:
            Path(override_log).parent.mkdir(parents=True, exist_ok=True)
            with open(override_log, "a") as fh:
                fh.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "mechanism_id": entry.mechanism_id,
                                     "entry_digest": res.entry_digest, "mismatches": res.mismatches,
                                     "missing": res.missing}) + "\n")
    return res
