"""Preregistration registry (registry/preregistrations.json).

Anything that would produce a scientific PASS/FAIL, or put a new mechanism
into the runtime (C4), must name a preregistration whose status is APPROVED
(or FROZEN, once its sealed TEST has been defined). Drafts and proposals are
listed so code can refer to them, but they unlock nothing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_REGISTRY = "CONTINUOUS_BRAIN_REGISTRY"
UNLOCKING = frozenset({"APPROVED", "FROZEN"})
STATUSES = frozenset({"DRAFT", "PROPOSED", "APPROVED", "FROZEN", "WITHDRAWN", "REJECTED"})


class PreregistrationError(RuntimeError):
    pass


def registry_dir(explicit: str | os.PathLike | None = None) -> Path:
    """The repository's registry/ directory (env override, else found above this package)."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(ENV_REGISTRY)
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "registry" / "preregistrations.json").is_file():
            return parent / "registry"
    raise PreregistrationError(f"registry/ not found; set {ENV_REGISTRY} or run from a repository checkout")


@dataclass(frozen=True)
class Preregistration:
    id: str
    title: str
    status: str
    linear_doc: str | None = None
    allowed_slots: tuple[str, ...] = ()
    allowed_items: tuple[str, ...] = ()        # result items (e.g. "G1", "B4") this preregistration may decide
    approved_by: str | None = None
    approved_utc: str | None = None
    notes: str = ""

    @property
    def unlocks(self) -> bool:
        return self.status in UNLOCKING and bool(self.approved_by)


@dataclass
class PreregistrationRegistry:
    entries: dict[str, Preregistration] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: str | os.PathLike | None = None) -> "PreregistrationRegistry":
        path = registry_dir(directory) / "preregistrations.json"
        raw = json.loads(path.read_text())
        entries = {}
        for e in raw["preregistrations"]:
            if e["status"] not in STATUSES:
                raise PreregistrationError(f"{e['id']}: unknown status {e['status']!r}")
            if e["status"] in UNLOCKING and not e.get("approved_by"):
                raise PreregistrationError(f"{e['id']}: {e['status']} without approved_by")
            entries[e["id"]] = Preregistration(e["id"], e["title"], e["status"], e.get("linear_doc"),
                                               tuple(e.get("allowed_slots", ())), tuple(e.get("allowed_items", ())),
                                               e.get("approved_by"),
                                               e.get("approved_utc"), e.get("notes", ""))
        return cls(entries)

    def require(self, prereg_id: str | None, slot: str | None = None, item: str | None = None) -> Preregistration:
        if not prereg_id:
            raise PreregistrationError("a preregistration id is required")
        p = self.entries.get(prereg_id)
        if p is None:
            raise PreregistrationError(f"unknown preregistration {prereg_id!r}")
        if not p.unlocks:
            raise PreregistrationError(f"{prereg_id} is {p.status}; only APPROVED/FROZEN preregistrations unlock this")
        if slot is not None and slot not in p.allowed_slots:
            raise PreregistrationError(f"{prereg_id} does not cover slot {slot!r}")
        if item is not None and item not in p.allowed_items:
            raise PreregistrationError(f"{prereg_id} does not cover result item {item!r}")
        return p
