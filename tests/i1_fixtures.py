"""Synthetic I1 preflights for tests and the lifecycle suite (no arm is run).

Everything a real preflight leaves behind is produced through the production code paths, in a temporary
root that holds a copy of the registry: the family is spent in the copy's ledger by runner.open_family()
(so seed_opened.json carries the real registry binding), the manifest is this machine's, and the sealed
report and its value-free view are written by evaluate.seal_report(). Only the report's contents are
synthetic (all 14 checks true, all 12 arms bound, no primary or secondary values)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from continuous_brain_unified.integration import runner
from continuous_brain_unified.integration.evaluate import ARMS, VALIDITY_KEYS, seal_report


def as_proposed(prereg: dict, central: dict) -> tuple[dict, dict]:
    """The PROPOSED form of a preregistration and central registry (approval fields removed)."""
    p = json.loads(json.dumps(prereg))
    p["status"], p["approved_by"] = "PROPOSED", None
    for k in runner.APPROVAL_FIELDS:
        p.pop(k, None)
    c = json.loads(json.dumps(central))
    for e in c["preregistrations"]:
        if e["id"] == runner.PREREG_ID:
            e["status"] = "PROPOSED"
            e.pop("approved_by", None)
            e.pop("approved_utc", None)
    return p, c


def proposed_root(tmp: Path, src: Path = runner.REPO) -> Path:
    """A temporary root whose registry/ is a PROPOSED copy of src's, with both I1 families never opened."""
    root = Path(tmp) / "root"
    (root / "registry").mkdir(parents=True)
    prereg = json.loads((src / runner.PREREG_FILE).read_text())
    p, c = as_proposed(prereg, json.loads((src / "registry" / "preregistrations.json").read_text()))
    (root / runner.PREREG_FILE).write_text(json.dumps(p, indent=2) + "\n")
    (root / "registry" / "preregistrations.json").write_text(json.dumps(c, indent=2) + "\n")
    shutil.copy(src / "registry" / "seed_ranges.json", root / "registry" / "seed_ranges.json")
    names = {p["families"][k]["name"] for k in ("engineering_preflight_proposed", "scientific_dev_proposed")}
    local = src / "registry" / "seed_ranges.local.jsonl"
    lines = local.read_text().splitlines() if local.exists() else []
    keep = [x for x in lines if x.strip() and json.loads(x).get("name") not in names]
    (root / "registry" / "seed_ranges.local.jsonl").write_text("".join(x + "\n" for x in keep))
    return root


def synthetic_preflight(root: Path, label: str = "pf", *, valid: bool = True, threads: int = 2, n: int = 64) -> Path:
    """runs/i1/<label> of root: a preflight opened by runner.open_family() with a synthetic sealed report."""
    root = Path(root)
    prereg = json.loads((root / runner.PREREG_FILE).read_text())
    d = root / "runs" / "i1" / label
    d.mkdir(parents=True)
    man = runner.manifest(threads)
    (d / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    runner.open_family("preflight", label, prereg=prereg, man=man, run_dir=d, root=root)
    sha = man["manifest_sha256"]
    report = {"format": "cb-i1-report/2", "status": "ENGINEERING_ONLY", "run": label, "n": n,
              "validity": {k: valid for k in VALIDITY_KEYS},
              "validity_diagnostics": {"arm_manifests": {a: sha for a in ARMS}, "expected_manifest_sha256": sha,
                                       "driver_side_model_forwards": 0, "cross_slot_nested_calls": 0,
                                       "host_blocked_calls": 0, "unbound_calls": 0},
              "primary": {}, "secondary": {}, "synthetic": "no arm was run"}
    seal_report(d, report, valid)
    return d
