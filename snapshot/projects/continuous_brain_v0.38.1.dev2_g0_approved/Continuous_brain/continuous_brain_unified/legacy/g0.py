"""G0 isolated legacy requalification evidence plan.

G0 is a prerequisite to composition, not an admission gate. It answers a
narrower question: is the historical mechanism's *identity* (source,
evaluator, result and learned state) sufficiently recovered that we can write
an isolated fresh-DEV adapter without guessing?

This module never opens historical TEST ranges and never turns a dossier claim
into a PASS. It verifies the immutable evidence plan built from David's WSL
inventory and reports which mechanisms are ready for a future G0 runner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from continuous_brain_protocols import sha256_file

PLAN_FORMAT = "cb-g0-plan/1"


class G0Disposition(str, Enum):
    READY_FOR_G0 = "READY_FOR_G0"
    READY_FOR_G0_SHARED_STATE = "READY_FOR_G0_SHARED_STATE"
    CONDITIONAL_EXTERNAL_ENV = "CONDITIONAL_EXTERNAL_ENV"
    NEEDS_REFRESHED_RESULT_PARSE = "NEEDS_REFRESHED_RESULT_PARSE"
    NEEDS_ADJUDICATION = "NEEDS_ADJUDICATION"
    SUPPORT_ONLY = "SUPPORT_ONLY"
    CONTROL_ONLY = "CONTROL_ONLY"


@dataclass(frozen=True)
class G0File:
    root: str
    role: str
    relpath: str
    sha256: str


@dataclass(frozen=True)
class G0Entry:
    mechanism_id: str
    slot_options: tuple[str, ...]
    lineage_version: str
    dossier_status: str
    disposition: G0Disposition
    reason: str
    files: tuple[G0File, ...]

    @property
    def can_prepare_adapter(self) -> bool:
        return self.disposition in {
            G0Disposition.READY_FOR_G0,
            G0Disposition.READY_FOR_G0_SHARED_STATE,
            G0Disposition.CONDITIONAL_EXTERNAL_ENV,
        }


@dataclass(frozen=True)
class G0Verification:
    mechanism_id: str
    disposition: str
    ok: bool
    missing: tuple[str, ...]
    mismatches: tuple[dict, ...]
    files_checked: int


@dataclass(frozen=True)
class G0Plan:
    source_bundle_sha256: str
    source_inventory_sha256: str
    source_repo_commit: str
    entries: tuple[G0Entry, ...]
    protected_checkpoint: dict
    counts: dict
    rules: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "G0Plan":
        raw=json.loads(Path(path).read_text())
        if raw.get("format") != PLAN_FORMAT:
            raise ValueError(f"G0 plan format must be {PLAN_FORMAT!r}")
        entries=[]
        for e in raw["entries"]:
            files=tuple(G0File(**f) for f in e.get("files", []))
            entries.append(G0Entry(
                e["mechanism_id"], tuple(e.get("slot_options", ())), e["lineage_version"],
                e["dossier_status"], G0Disposition(e["disposition"]), e["reason"], files))
        return cls(raw["source_bundle_sha256"], raw["source_inventory_sha256"], raw["source_repo_commit"],
                   tuple(entries), dict(raw["protected_checkpoint"]), dict(raw["counts"]), tuple(raw["rules"]))

    def get(self, mechanism_id: str) -> G0Entry:
        for e in self.entries:
            if e.mechanism_id == mechanism_id:
                return e
        raise KeyError(mechanism_id)

    def verify(self, roots: Mapping[str, str | Path], mechanism_id: str | None = None) -> list[G0Verification]:
        chosen=[self.get(mechanism_id)] if mechanism_id else list(self.entries)
        out=[]
        for e in chosen:
            missing=[]; mismatches=[]; checked=0
            for f in e.files:
                if f.root not in roots:
                    missing.append(f"{f.root}:{f.relpath} (root not configured)")
                    continue
                root=Path(roots[f.root]).expanduser().resolve()
                p=(root/f.relpath).resolve()
                if root not in p.parents and p != root:
                    raise ValueError(f"{f.root}:{f.relpath} resolves outside configured root")
                if not p.is_file():
                    missing.append(f"{f.root}:{f.relpath}")
                    continue
                checked += 1
                got=sha256_file(p)
                if got != f.sha256:
                    mismatches.append({"root":f.root,"relpath":f.relpath,"role":f.role,
                                       "expected":f.sha256,"actual":got})
            out.append(G0Verification(e.mechanism_id,e.disposition.value,not missing and not mismatches,
                                      tuple(missing),tuple(mismatches),checked))
        return out
