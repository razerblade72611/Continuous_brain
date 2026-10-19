"""Consciousness-indicator evidence registry (Joint Contract §7): metadata only.

A row can only be marked admitted when every link of the evidence chain
(mechanism -> internal variable -> intervention -> ablation -> behavioural
consequence -> persistence) cites a primary artifact with its SHA-256 and the
row names an APPROVED preregistration. The loader enforces that, so a row can
never be turned green by editing JSON. Admission is still not evidence of
sentience; nothing in this module says otherwise.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from continuous_brain_unified.prereg import PreregistrationRegistry, registry_dir

CANONICAL_IDS = ("RPT-1", "RPT-2", "GWT-1", "GWT-2", "GWT-3", "GWT-4", "HOT-1", "HOT-2", "HOT-3", "HOT-4",
                 "AST-1", "PP-1", "AE-1", "AE-2")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_CHAIN = ("mechanism", "internal_variable", "intervention", "ablation", "behavioural_consequence",
                  "persistence")


class IndicatorError(ValueError):
    pass


@dataclass(frozen=True)
class Indicator:
    id: str
    theory: str
    wording: str
    dossier_mechanism: str
    dossier_status: str
    admitted: bool
    evidence: tuple[dict, ...] = field(default_factory=tuple)
    prereg_id: str | None = None


def load_indicators(directory: str | os.PathLike | None = None, preregs: PreregistrationRegistry | None = None,
                    artifact_root: str | os.PathLike | None = None) -> tuple[list[Indicator], tuple[str, ...]]:
    """Load and check the registry. With `artifact_root`, every cited evidence file must exist there and
    match its SHA-256."""
    raw = json.loads((registry_dir(directory) / "indicators.json").read_text())
    chain = tuple(raw["evidence_chain"])
    if chain != EVIDENCE_CHAIN:
        raise IndicatorError(f"the evidence chain is fixed in code as {EVIDENCE_CHAIN}; the JSON says {chain}")
    rows = [Indicator(r["id"], r["theory"], r["wording"], r["dossier_mechanism"], r["dossier_status"],
                      bool(r["admitted"]), tuple(r.get("evidence", ())), r.get("prereg_id")) for r in raw["indicators"]]
    ids = tuple(r.id for r in rows)
    if ids != CANONICAL_IDS:
        raise IndicatorError(f"indicator ids must be exactly the 14 canonical ids in order; got {ids}")
    for r in rows:
        if r.admitted:
            links = {e.get("link") for e in r.evidence if _HEX64.match(str(e.get("sha256", ""))) and e.get("path")}
            missing = [c for c in chain if c not in links]
            if missing:
                raise IndicatorError(f"{r.id} is marked admitted but lacks cited evidence for {missing}")
            (preregs or PreregistrationRegistry.load(directory)).require(r.prereg_id, item=r.id)
        if artifact_root is not None:
            from pathlib import Path
            from continuous_brain_protocols import sha256_file
            for e in r.evidence:
                p = Path(artifact_root) / str(e.get("path", ""))
                if not p.is_file() or sha256_file(p) != e.get("sha256"):
                    raise IndicatorError(f"{r.id}: cited evidence {e.get('path')} is missing or changed")
    return rows, chain
