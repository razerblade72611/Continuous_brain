#!/usr/bin/env python3
"""End-to-end check of the approval gate on a REAL run (no synthetic validity block).

usage: python scripts/dev/i1_approval_gate_check.py RUN_DIR [OUT_JSON]

RUN_DIR must be a finished I1 run made with this checkout's code on this machine (so its manifest
equals the one this machine computes). The script copies the run, produces its genuine sealed
(preflight) view with evaluate(sealed=True), and builds a preflight directory from it. It then calls
runner.approve() on a temporary copy of the two registry files, never the checkout's own:
  - the genuine view must be ACCEPTED;
  - the view with the two dev7 checks removed (the old 12-check schema) must be REFUSED;
  - the view with any single one of the 14 checks removed must be REFUSED;
  - the view with an extra check, or with the manifest binding removed or altered, must be REFUSED;
  - the view with any one of the I1 arms unbound, with only NORMAL bound, or with an unknown arm, must be
    REFUSED.
Exit 0 only if every case behaves as stated."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from continuous_brain_unified.integration import runner  # noqa: E402
from continuous_brain_unified.integration.evaluate import ARMS, VALIDITY_KEYS, evaluate  # noqa: E402


def _registry_copy(root: Path) -> None:
    (root / "registry").mkdir(parents=True)
    p = json.loads((REPO / runner.PREREG_FILE).read_text())
    c = json.loads((REPO / "registry" / "preregistrations.json").read_text())
    p["status"], p["approved_by"] = "PROPOSED", None
    for k in ("approved_utc", "preflight_manifest_sha256", "approved_design_sha256", "preflight_run"):
        p.pop(k, None)
    for e in c["preregistrations"]:
        if e["id"] == runner.PREREG_ID:
            e["status"] = "PROPOSED"; e.pop("approved_by", None); e.pop("approved_utc", None)
    (root / runner.PREREG_FILE).write_text(json.dumps(p, indent=2))
    (root / "registry" / "preregistrations.json").write_text(json.dumps(c, indent=2))
    return p


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    results = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        run = tmp / src.name
        shutil.copytree(src, run, ignore=shutil.ignore_patterns("historical_core", "persist"))
        view = evaluate(run, sealed=True)                     # the genuine sealed preflight view
        prereg = json.loads((REPO / runner.PREREG_FILE).read_text())
        pf = prereg["families"]["engineering_preflight_proposed"]

        def attempt(name: str, mutate, expect_ok: bool):
            case = tmp / f"case{len(results)}"
            root = case / "root"
            _registry_copy(root)
            d = case / "pf"
            d.mkdir()
            v = json.loads(json.dumps(view))
            mutate(v)
            (d / "i1_validity.json").write_text(json.dumps(v))
            shutil.copy(run / "manifest.json", d / "manifest.json")
            (d / "seed_opened.json").write_text(json.dumps({"mode": "preflight", "family": {**pf, "spent": True}}))
            try:
                runner.approve(preflight_dir=str(d), approver="David", confirm=runner.PREREG_ID, root=root)
                ok, why = True, "accepted"
            except runner.I1Refused as e:
                ok, why = False, str(e)[:160]
            results.append({"case": name, "accepted": ok, "expected_accept": expect_ok, "pass": ok == expect_ok,
                            "detail": why})
            print(("ok   " if ok == expect_ok else "FAIL ") + f"{name}: {'accepted' if ok else 'refused'}")

        attempt("genuine sealed view", lambda v: None, True)
        attempt("old 12-check schema", lambda v: [v["validity"].pop(k) for k in
                                                   ("no_hidden_mutable_state", "donor_pairing_fixed_by_ids")], False)
        for k in VALIDITY_KEYS:
            attempt(f"missing {k}", lambda v, k=k: v["validity"].pop(k), False)
        attempt("unexpected extra check", lambda v: v["validity"].__setitem__("extra_check", True), False)
        attempt("manifest binding removed", lambda v: v["validity_diagnostics"].pop("expected_manifest_sha256"), False)
        attempt("manifest binding altered",
                lambda v: v["validity_diagnostics"].__setitem__("expected_manifest_sha256", "0" * 64), False)
        for a in ARMS:
            attempt(f"arm {a} unbound", lambda v, a=a: v["validity_diagnostics"]["arm_manifests"].pop(a), False)
        attempt("only NORMAL bound", lambda v: v["validity_diagnostics"].__setitem__(
            "arm_manifests", {"NORMAL": v["validity_diagnostics"]["arm_manifests"]["NORMAL"]}), False)
        attempt("unexpected arm", lambda v: v["validity_diagnostics"]["arm_manifests"].__setitem__(
            "SOME_OTHER_ARM", v["validity_diagnostics"]["arm_manifests"]["NORMAL"]), False)
        attempt("one arm under another manifest",
                lambda v: v["validity_diagnostics"]["arm_manifests"].__setitem__(
                    sorted(v["validity_diagnostics"]["arm_manifests"])[0], "0" * 64), False)
    ok = all(r["pass"] for r in results)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps({"format": "cb-i1-approval-gate/1", "run": src.name,
                                                 "cases": len(results), "all_pass": ok, "results": results},
                                                indent=2) + "\n")
    print(f"{sum(r['pass'] for r in results)}/{len(results)} cases as expected: " + ("GATE OK" if ok else "GATE FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
