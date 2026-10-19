#!/usr/bin/env python3
"""E1b (engineering positive control): the frozen evaluator's results depend on what the
canonical store holds, API by API.

For each route and each (host slot, API) it used, take that API's first response as a donor and
REPLACE <slot>.response, at the store level, on up to 16 later ticks where that API's true
response differed from the donor (single-tick windows, so other APIs of the same host are
untouched). Outcomes:
  CHANGED                    result.json exists and differs from the G0 route
  ABORTED_AFTER_REPLACEMENT  the evaluator ran past the first replaced tick and then failed
  UNCHANGED                  byte-identical to the G0 route although consumed values differed
  NOT_TESTABLE               the API never returned a value different from its first response
  INFRA                      the control could not be applied (no application, blocked or failed
                             before the first replaced tick)
Gate (E1b PASS): every host a route used (at least one request) shows store authority:
  tier 1  at least one of its APIs is CHANGED or ABORTED_AFTER_REPLACEMENT under single-tick,
          same-API replacement; otherwise
  tier 2  replacing the host's whole response stream after its first response (all APIs) makes
          the evaluator's outcome CHANGED or ABORTED_AFTER_REPLACEMENT (an abort from a wrong-typed
          value still shows the evaluator consumed what the store held)
and no row is INFRA. UNCHANGED per-API rows are reported as robustness observations.
Reads only hashes, exit codes and attestation structure; no metric or pass field is interpreted.
"""
from __future__ import annotations

import argparse, base64, hashlib, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "scripts" / "dev"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-root", required=True)
    ap.add_argument("--historical-core-archive", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--e1-tag", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-windows", type=int, default=16)
    a = ap.parse_args()

    def run_control(m: str, label: str, ctl: Path) -> tuple[dict, dict]:
        prev = REPO / "runs/dev" / label / m / "dev_report.json"
        if prev.is_file():                         # reuse a completed control run (same label = same control)
            rep = json.loads(prev.read_text())
        else:
            out = subprocess.run([sys.executable, str(DEV / "composition_route_dev.py"), "--mechanism", m,
                                  "--bind", "all", "--n", str(a.n), "--label", label,
                                  "--legacy-root", a.legacy_root, "--historical-core-archive",
                                  a.historical_core_archive, "--store-control", str(ctl)],
                                 cwd=REPO, text=True, capture_output=True)
            rep = json.loads(out.stdout.strip().splitlines()[-1])
        catt_p = REPO / "runs/dev" / label / m / "composition_attestation.json"
        return rep, (json.loads(catt_p.read_text()) if catt_p.is_file() else {})

    def classify(rep: dict, catt: dict, first_tick: int, g0_sha: str) -> str:
        applied = catt.get("intervention_applications") or 0
        if applied == 0 or (catt.get("ticks") or 0) <= first_tick:
            return "INFRA"
        sha = rep["raw_result_sha256"]
        if sha == g0_sha:
            return "UNCHANGED"
        return "CHANGED" if sha is not None else "ABORTED_AFTER_REPLACEMENT"
    e1 = json.loads((REPO / "runs/dev" / f"{a.e1_tag}_e1_report.json").read_text())
    if e1["n"] != a.n:
        raise SystemExit("seed count must match the E1 run")
    rows, slot_ok = [], {}
    for row in e1["rows"]:
        m = row["mechanism"]
        att = json.loads((REPO / "runs/dev" / f"{a.e1_tag}_full" / m / "composition_attestation.json").read_text())
        for slot, apis in sorted(att.get("api_trace", {}).items()):
            if not apis:
                continue                            # idle host in this route: nothing to test
            slot_ok.setdefault((m, slot), False)
            for api, tr in sorted(apis.items()):
                base = {"mechanism": m, "slot": slot, "api": api}
                if tr["first_response_b64"] is None:
                    rows.append({**base, "outcome": "NOT_TESTABLE", "why": "first response > 64 KiB"}); continue
                donor = base64.b64decode(tr["first_response_b64"])
                dsha = hashlib.sha256(donor).hexdigest()
                ticks = [t for t, sha in tr["ticks"] if t != tr["first_response_tick"] and sha != dsha][:a.max_windows]
                if not ticks:
                    rows.append({**base, "outcome": "NOT_TESTABLE", "why": "no differing response among first 64 calls"})
                    continue
                ctl = REPO / "runs/dev" / f"{a.tag}_ctl" / f"{m}_{slot}_{api}.json"
                ctl.parent.mkdir(parents=True, exist_ok=True)
                ctl.write_text(json.dumps([{"kind": "replace", "target": f"{slot}.response", "start_tick": t,
                                            "end_tick": t + 1, "payload_b64": tr["first_response_b64"]} for t in ticks]))
                label = f"{a.tag}_{slot}_{api.replace('.', '-')}"
                rep, catt = run_control(m, label, ctl)
                outcome = classify(rep, catt, ticks[0], row["g0_sha"])
                applied = catt.get("intervention_applications") or 0
                r = {**base, "tier": 1, "outcome": outcome, "windows": len(ticks), "first_window_tick": ticks[0],
                     "intervention_applications": applied, "returncode": rep["returncode"],
                     "route_blocked": catt.get("route_blocked"), "fatal_error": catt.get("fatal_error")}
                rows.append(r)
                if outcome in ("CHANGED", "ABORTED_AFTER_REPLACEMENT"):
                    slot_ok[(m, slot)] = True
                print(json.dumps({k: r[k] for k in ("mechanism", "slot", "api", "outcome", "windows",
                                                    "intervention_applications")}), flush=True)
        # tier 2: whole-host replacement for hosts no single-API control could show
        for (mm, slot), ok in sorted(slot_ok.items()):
            if mm != m or ok:
                continue
            fr = att["first_responses"][slot]
            if fr["b64"] is None:
                rows.append({"mechanism": m, "slot": slot, "api": "*", "tier": 2, "outcome": "NOT_TESTABLE",
                             "why": "first response > 64 KiB"}); continue
            ctl = REPO / "runs/dev" / f"{a.tag}_ctl" / f"{m}_{slot}_ALL.json"
            ctl.write_text(json.dumps([{"kind": "replace", "target": f"{slot}.response",
                                        "start_tick": fr["tick"] + 1, "payload_b64": fr["b64"]}]))
            label = f"{a.tag}_{slot}_ALL"
            rep, catt = run_control(m, label, ctl)
            outcome = classify(rep, catt, fr["tick"] + 1, row["g0_sha"])
            r = {"mechanism": m, "slot": slot, "api": "*", "tier": 2, "outcome": outcome,
                 "first_window_tick": fr["tick"] + 1, "intervention_applications": catt.get("intervention_applications"),
                 "returncode": rep["returncode"], "route_blocked": catt.get("route_blocked"),
                 "fatal_error": catt.get("fatal_error")}
            rows.append(r)
            if outcome in ("CHANGED", "ABORTED_AFTER_REPLACEMENT"):
                slot_ok[(m, slot)] = True
            print(json.dumps({k: r[k] for k in ("mechanism", "slot", "api", "tier", "outcome")}), flush=True)
    infra = [r for r in rows if r["outcome"] == "INFRA"]
    res = {"format": "cb-e1b-store-authority/3", "status": "ENGINEERING_ONLY", "n": a.n, "e1_tag": a.e1_tag,
           "rows": rows, "hosts": [{"mechanism": m, "slot": s, "store_output_consumed": ok}
                                   for (m, s), ok in sorted(slot_ok.items())],
           "E1b_all": "PASS" if slot_ok and all(slot_ok.values()) and not infra else "FAIL"}
    p = REPO / "runs/dev" / f"{a.tag}_e1b_report.json"
    p.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print("E1b_all", res["E1b_all"], p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
