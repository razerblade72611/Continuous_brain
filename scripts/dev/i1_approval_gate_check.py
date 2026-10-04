#!/usr/bin/env python3
"""End-to-end check of the approval gate on a REAL run (no synthetic validity block).

usage: python scripts/dev/i1_approval_gate_check.py RUN_DIR [OUT_JSON]

RUN_DIR must be a finished I1 run made with this checkout's code on this machine (so its manifest
equals the one this machine computes). The script copies the run and produces its genuine sealed
(preflight) view with evaluate(sealed=True). For every case it builds a temporary root holding a PROPOSED
copy of the registry (both I1 families never opened), opens the preflight family there through
runner.open_family() (the production path: a real spend in the temporary ledger, and seed_opened.json
with the real registry binding), places the genuine view, sealed report and manifest in
runs/i1/<run name>, applies the case's fault, and calls runner.approve() on that root, never on the
checkout's own registry:
  - the genuine chain must be ACCEPTED;
  - the old 12-check schema, any one of the 14 checks removed, an extra check, the manifest binding
    removed or altered, any one of the 12 arms unbound, only NORMAL bound, an unknown arm, or an arm under
    another manifest must be REFUSED;
  - every break in the chain of custody must be REFUSED: design or registry files changed after opening,
    no ledger spend (a fabricated "spent": true), a spend by another source or run, a double spend, a later
    row superseding the spend, the family marked spent by hand, the sealed report missing, altered, or
    recommitted with other contents, the run opened under another status or with approval fields, the
    opening record without its binding or of another run or manifest, the directory renamed or outside
    runs/i1, an interrupted approval journal, CONTINUOUS_BRAIN_REGISTRY pointing elsewhere, an approver who is
    not exactly the frozen approval_authority, any seed-ledger change after opening (even an unrelated row), and
    a preflight that carries a BLOCKED report.
Exit 0 only if every case behaves as stated."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from continuous_brain_unified.integration import runner  # noqa: E402
from continuous_brain_unified.integration.evaluate import ARMS, VALIDITY_KEYS, evaluate  # noqa: E402


def _proposed_root(root: Path) -> dict:
    (root / "registry").mkdir(parents=True)
    p = json.loads((REPO / runner.PREREG_FILE).read_text())
    c = json.loads((REPO / "registry" / "preregistrations.json").read_text())
    p["status"], p["approved_by"] = "PROPOSED", None
    for k in runner.APPROVAL_FIELDS:
        p.pop(k, None)
    for e in c["preregistrations"]:
        if e["id"] == runner.PREREG_ID:
            e["status"] = "PROPOSED"; e.pop("approved_by", None); e.pop("approved_utc", None)
    (root / runner.PREREG_FILE).write_text(json.dumps(p, indent=2) + "\n")
    (root / "registry" / "preregistrations.json").write_text(json.dumps(c, indent=2) + "\n")
    shutil.copy(REPO / "registry" / "seed_ranges.json", root / "registry" / "seed_ranges.json")
    names = {p["families"][k]["name"] for k in ("engineering_preflight_proposed", "scientific_dev_proposed")}
    local = REPO / "registry" / "seed_ranges.local.jsonl"
    lines = local.read_text().splitlines() if local.exists() else []
    (root / "registry" / "seed_ranges.local.jsonl").write_text(
        "".join(x + "\n" for x in lines if x.strip() and json.loads(x).get("name") not in names))
    return p


def _edit(path: Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _ledger(root: Path, fn) -> None:
    f = root / "registry" / "seed_ranges.local.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    f.write_text("".join(json.dumps(r) + "\n" for r in fn(rows)))


def _pf_spend(root: Path, row: dict) -> bool:
    name = json.loads((root / runner.PREREG_FILE).read_text())["families"]["engineering_preflight_proposed"]["name"]
    return bool(row.get("spent")) and row.get("name") == name


def _reseal(d: Path, fn) -> None:
    rep = json.loads((d / "sealed" / "i1_report.json").read_text())
    fn(rep)
    data = json.dumps(rep, indent=2, sort_keys=True) + "\n"
    (d / "sealed" / "i1_report.json").write_text(data)
    _edit(d / "i1_validity.json", lambda v: v.__setitem__("sealed_report_sha256", hashlib.sha256(data.encode()).hexdigest()))


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    results = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        run = tmp / "staging" / src.name
        shutil.copytree(src, run, ignore=shutil.ignore_patterns("historical_core", "persist"))
        evaluate(run, sealed=True)                      # the genuine sealed preflight view and sealed report

        def attempt(name: str, *, view=None, chain=None, expect_ok: bool = False, approver: str = "David"):
            case = tmp / f"case{len(results)}"
            root = case / "root"
            prereg = _proposed_root(root)
            d = root / "runs" / "i1" / src.name
            d.mkdir(parents=True)
            shutil.copy(run / "manifest.json", d / "manifest.json")
            man = json.loads((d / "manifest.json").read_text())
            runner.open_family("preflight", src.name, prereg=prereg, man=man, run_dir=d, root=root)
            (d / "sealed").mkdir()
            shutil.copy(run / "sealed" / "i1_report.json", d / "sealed" / "i1_report.json")
            v = json.loads((run / "i1_validity.json").read_text())
            if view is not None:
                view(v)
            (d / "i1_validity.json").write_text(json.dumps(v, indent=2, sort_keys=True) + "\n")
            env_before = os.environ.get("CONTINUOUS_BRAIN_REGISTRY")
            target = d
            if chain is not None:
                moved = chain(root, d)
                target = moved if isinstance(moved, Path) else d
            try:
                runner.approve(preflight_dir=str(target), approver=approver, confirm=runner.PREREG_ID, root=root)
                ok, why = True, "accepted"
            except runner.I1Refused as e:
                ok, why = False, str(e)[:200]
            finally:
                if env_before is None:
                    os.environ.pop("CONTINUOUS_BRAIN_REGISTRY", None)
                else:
                    os.environ["CONTINUOUS_BRAIN_REGISTRY"] = env_before
            results.append({"case": name, "accepted": ok, "expected_accept": expect_ok, "pass": ok == expect_ok,
                            "detail": why})
            print(("ok   " if ok == expect_ok else "FAIL ") + f"{name}: {'accepted' if ok else 'refused'}"
                  + ("" if ok else f"  [{why[:90]}]"))

        attempt("genuine chain", expect_ok=True)
        # the validity schema and the arm bindings (rounds 64e-64g)
        attempt("old 12-check schema", view=lambda v: [v["validity"].pop(k) for k in
                                                        ("no_hidden_mutable_state", "donor_pairing_fixed_by_ids")])
        for k in VALIDITY_KEYS:
            attempt(f"missing {k}", view=lambda v, k=k: v["validity"].pop(k))
        attempt("unexpected extra check", view=lambda v: v["validity"].__setitem__("extra_check", True))
        attempt("manifest binding removed", view=lambda v: v["validity_diagnostics"].pop("expected_manifest_sha256"))
        attempt("manifest binding altered",
                view=lambda v: v["validity_diagnostics"].__setitem__("expected_manifest_sha256", "0" * 64))
        for a in ARMS:
            attempt(f"arm {a} unbound", view=lambda v, a=a: v["validity_diagnostics"]["arm_manifests"].pop(a))
        attempt("only NORMAL bound", view=lambda v: v["validity_diagnostics"].__setitem__(
            "arm_manifests", {"NORMAL": v["validity_diagnostics"]["arm_manifests"]["NORMAL"]}))
        attempt("unexpected arm", view=lambda v: v["validity_diagnostics"]["arm_manifests"].__setitem__(
            "SOME_OTHER_ARM", v["validity_diagnostics"]["arm_manifests"]["NORMAL"]))
        attempt("one arm under another manifest", view=lambda v: v["validity_diagnostics"]["arm_manifests"].__setitem__(
            sorted(v["validity_diagnostics"]["arm_manifests"])[0], "0" * 64))
        # the chain of custody (round 64g findings and hardening)
        attempt("design changed after opening", chain=lambda r, d: _edit(r / runner.PREREG_FILE, lambda p: p[
            "decision_rule"]["primary"]["C2_first_goal_completes"].__setitem__("threshold", "0.5")))
        attempt("prereg file changed after opening (design unchanged)", chain=lambda r, d: _edit(
            r / runner.PREREG_FILE, lambda p: p.setdefault("deviation_log", []).append("late edit")))
        attempt("central registry changed after opening", chain=lambda r, d: _edit(
            r / "registry" / "preregistrations.json", lambda c: c.__setitem__("touched", 1)))
        attempt("no ledger spend (fabricated spent:true)", chain=lambda r, d: _ledger(
            r, lambda rows: [x for x in rows if not _pf_spend(r, x)]))
        attempt("ledger spend by another source", chain=lambda r, d: _ledger(
            r, lambda rows: [{**x, "source": "lifecycle-suite:simulated"} if _pf_spend(r, x) else x for x in rows]))
        attempt("ledger spend recorded twice", chain=lambda r, d: _ledger(
            r, lambda rows: rows + [x for x in rows if _pf_spend(r, x)]))
        attempt("ledger row superseding the spend", chain=lambda r, d: _ledger(
            r, lambda rows: rows + [{**[x for x in rows if _pf_spend(r, x)][0], "spent": False}]))
        attempt("family marked spent by hand in seed_ranges.json", chain=lambda r, d: _edit(
            r / "registry" / "seed_ranges.json", lambda j: j["ranges"].append(
                {**json.loads((d / "seed_opened.json").read_text())["family"], "source": "hand", "note": ""})))
        attempt("sealed report missing", chain=lambda r, d: (d / "sealed" / "i1_report.json").unlink())
        attempt("sealed report altered", chain=lambda r, d: (d / "sealed" / "i1_report.json").write_text(
            (d / "sealed" / "i1_report.json").read_text() + " "))
        attempt("sealed report recommitted with another validity block", chain=lambda r, d: _reseal(
            d, lambda rep: rep["validity"].__setitem__("no_errors", False)))
        attempt("sealed report recommitted with other arm bindings", chain=lambda r, d: _reseal(
            d, lambda rep: rep["validity_diagnostics"]["arm_manifests"].__setitem__("DONOR", "0" * 64)))
        attempt("sealed report of another run", chain=lambda r, d: _reseal(d, lambda rep: rep.__setitem__("run", "x")))
        attempt("opened under another status", chain=lambda r, d: _edit(
            d / "seed_opened.json", lambda o: o.__setitem__("prereg_status", "APPROVED")))
        attempt("approval fields at opening", chain=lambda r, d: _edit(
            d / "seed_opened.json", lambda o: o.__setitem__("approved_design_sha256", "0" * 64)))
        attempt("opening record without its binding", chain=lambda r, d: _edit(
            d / "seed_opened.json", lambda o: o.pop("seed_ledger_sha256")))
        attempt("opening record of another run", chain=lambda r, d: _edit(
            d / "seed_opened.json", lambda o: o.__setitem__("run", "other")))
        attempt("opening record under another manifest", chain=lambda r, d: _edit(
            d / "seed_opened.json", lambda o: o.__setitem__("manifest_sha256", "0" * 64)))
        attempt("directory renamed", chain=lambda r, d: Path(shutil.copytree(d, d.parent / "renamed")))
        attempt("directory outside runs/i1", chain=lambda r, d: Path(shutil.copytree(d, r / "elsewhere" / d.name)))
        attempt("interrupted approval journal present", chain=lambda r, d: (r / runner.JOURNAL).write_text("{}"))
        # ChatGPT's fourth audit: the approval authority and the frozen seed ledger
        attempt("approved by Claude", approver="Claude")
        attempt("approver not exactly the authority (case)", approver="david")
        attempt("unrelated ledger row appended after opening", chain=lambda r, d: _ledger(
            r, lambda rows: rows + [{"name": "unrelated-later-family", "start": 990000, "stop": 990001,
                                     "purpose": "ENGINEERING", "source": "someone", "note": ""}]))
        attempt("preflight has a BLOCKED report", chain=lambda r, d: runner._blocked(d, "preflight", "injected", None))
        attempt("CONTINUOUS_BRAIN_REGISTRY points elsewhere", chain=lambda r, d: os.environ.__setitem__(
            "CONTINUOUS_BRAIN_REGISTRY", str(r / "other_registry")))
    ok = all(r["pass"] for r in results)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps({"format": "cb-i1-approval-gate/2", "run": src.name,
                                                 "cases": len(results), "all_pass": ok, "results": results},
                                                indent=2) + "\n")
    print(f"{sum(r['pass'] for r in results)}/{len(results)} cases as expected: " + ("GATE OK" if ok else "GATE FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
