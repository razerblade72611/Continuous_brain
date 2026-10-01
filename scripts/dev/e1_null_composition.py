#!/usr/bin/env python3
"""E1 (engineering, preregistered in the G1 v2 review): null-composition equivalence.

For each of the six mechanisms, on the same cloud-dev seed slice:
  G0      the unmodified G0 route (scripts/dev/g0_route_dev.py), run twice (self-reproducibility)
  FULL    composition route, all six hosts bound
  USED    composition route, only the hosts FULL actually sent requests to
  FOCAL   composition route, only the focal mechanism's host bound (informative: a nested-lineage
          evaluator is expected to be BLOCKED here, which shows which mechanisms it depends on)

E1 PASS for a mechanism (equivalence): both G0 runs, FULL and USED produce byte-identical
result.json, with equal return codes, and neither composed run was blocked or failed.
Faithfulness (also required): every shared-model acquisition was verified bit-identical to a
fresh load, every constructed stateless instance has the canonical default configuration, and
no call reached an unbound mechanism.
Mediation (reported, not gating): how much of the route's model computation went through hosts
versus direct driver calls on runtime-owned models (STORE_MEDIATED / PARTLY_DRIVER_DIRECT /
DRIVER_DIRECT).
Only hashes, return codes and attestation structure are reported; no metric or pass field is read.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "scripts" / "dev"
MECHS = ["recurrent_self_trajectory_v031", "autobiographical_discrepancy_v0322",
         "event_consistent_self_development_v0341", "endogenous_goal_genesis_v0353",
         "recursive_autobiographical_curriculum_v0361", "consequence_replanning_v0252"]


def run(script: str, args: list[str]) -> dict:
    out = subprocess.run([sys.executable, str(DEV / script), *args], cwd=REPO, text=True, capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"{script} failed: {out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-root", required=True)
    ap.add_argument("--historical-core-archive", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mechanisms", default=",".join(MECHS))
    a = ap.parse_args()
    common = ["--legacy-root", a.legacy_root, "--historical-core-archive", a.historical_core_archive, "--n", str(a.n)]
    rows = []
    for m in a.mechanisms.split(","):
        g0a = run("g0_route_dev.py", ["--mechanism", m, "--label", f"{a.tag}_g0a", *common])
        g0b = run("g0_route_dev.py", ["--mechanism", m, "--label", f"{a.tag}_g0b", *common])
        full = run("composition_route_dev.py", ["--mechanism", m, "--bind", "all", "--label", f"{a.tag}_full", *common])
        att = json.loads((REPO / "runs/dev" / f"{a.tag}_full" / m / "composition_attestation.json").read_text())
        used = sorted(h["mechanism_id"] for h in att.get("hosts", {}).values() if h["requests"] > 0)
        usedr = run("composition_route_dev.py", ["--mechanism", m, "--bind", ",".join(used) or m,
                                                  "--label", f"{a.tag}_used", *common])
        focal = run("composition_route_dev.py", ["--mechanism", m, "--bind", m, "--label", f"{a.tag}_focal", *common])
        ref = g0a["raw_result_sha256"]
        row = {"mechanism": m, "seeds": g0a["seeds"], "g0_sha": ref,
               "g0_self_reproducible": ref is not None and g0b["raw_result_sha256"] == ref,
               "full_equal": full["raw_result_sha256"] == ref, "used_equal": usedr["raw_result_sha256"] == ref,
               "returncodes": {"g0a": g0a["returncode"], "g0b": g0b["returncode"], "full": full["returncode"],
                               "used": usedr["returncode"], "focal": focal["returncode"]},
               "used_hosts": used, "full_ticks": full["ticks"],
               "focal_only": {"result_equal": focal["raw_result_sha256"] == ref, "route_blocked": focal["route_blocked"]},
               "content_flows": att.get("boundary", {}).get("content_flows", {}),
               "idle_hosts": sorted(h["mechanism_id"] for h in att.get("hosts", {}).values() if h["requests"] == 0),
               "private_instances": att.get("census", {}).get("private_instances", []),
               "nested_calls": att.get("boundary", {}).get("nested_calls", {})}
        uatt = json.loads((REPO / "runs/dev" / f"{a.tag}_used" / m / "composition_attestation.json").read_text())
        tree_ok = all(x.get("legacy_tree_unchanged") is True for x in (g0a, g0b, full, usedr, focal))
        rc_ok = (g0a["returncode"] == g0b["returncode"] == att.get("evaluator_exit_code") == uatt.get("evaluator_exit_code"))
        clean = all(x.get("route_blocked") is None and x.get("fatal_error") is None for x in (att, uatt))
        cen = att.get("census", {})
        faithful = (all(v["bit_identical"] for v in cen.get("acquisition_verifications", []))
                    and all(c["default_config"] for c in cen.get("constructed_instances", []))
                    and not att.get("boundary", {}).get("unbound_calls"))
        row.update({"legacy_tree_unchanged": tree_ok, "returncodes_equal": rc_ok, "composed_runs_clean": clean, "faithfulness_checks": faithful,
                    "mediation": att.get("mediation"), "model_forwards": att.get("model_forwards"),
                    "acquisition_verifications": cen.get("acquisition_verifications"),
                    "constructed_instances": len(cen.get("constructed_instances", []))})
        row["E1"] = "PASS" if (row["g0_self_reproducible"] and row["full_equal"] and row["used_equal"]
                               and rc_ok and clean and faithful and tree_ok) else "FAIL"
        rows.append(row)
        print(json.dumps({**{k: row[k] for k in ("mechanism", "E1", "full_equal", "used_equal", "returncodes_equal",
                                                 "faithfulness_checks", "used_hosts", "full_ticks")},
                          "mediation": (row["mediation"] or {}).get("class")}), flush=True)
    rep = {"format": "cb-e1-null-composition/1", "status": "ENGINEERING_ONLY", "tag": a.tag, "n": a.n,
           "family": "cloud-dev-claude-eng", "rows": rows,
           "E1_all": "PASS" if all(r["E1"] == "PASS" for r in rows) else "FAIL"}
    out = REPO / "runs/dev" / f"{a.tag}_e1_report.json"
    out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print("E1_all", rep["E1_all"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
