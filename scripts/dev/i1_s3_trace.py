#!/usr/bin/env python3
"""ENGINEERING_ONLY diagnostic for I1's S3 finding: trace, episode by episode, the goal commitment's
self-model reliability and the environment's true capability, under the developmental controller
(NORMAL) and under fixed SCAFFOLD (FIXED_SCAFFOLD).

usage: python scripts/dev/i1_s3_trace.py --legacy-root L --historical-core-archive A --start S --n N --label X

The arms run through the unchanged I1 code (integration/route.run_arm). The only addition is a read-only
tracer wrapped around the driver's client, which records what the driver already receives (reliability,
chosen strategy, target check) and reads the environment's capability after each episode. To show that
tracing changes nothing, NORMAL is also run untraced: every traced life must have the same final store
digest, final tick and life trace as its untraced twin, or the script stops.

Seeds: a slice of a registered ENGINEERING family, checked by the I1 dev guard (runner._dev_sids) and
logged in runs/dev/seed_usage.jsonl. Not part of any preregistration; reported as engineering data."""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from continuous_brain_eval.seeds import SeedRegistry  # noqa: E402
from continuous_brain_unified.integration import runner  # noqa: E402

TRACE_BOOT = '''import importlib.abc, importlib.util, json, os, sys
sys.dont_write_bytecode=True
import torch
torch.set_num_threads({threads})
OVERRIDES={overrides!r}
class _HistoricalCoreFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        p=OVERRIDES.get(fullname)
        if p is None: return None
        return importlib.util.spec_from_file_location(fullname,p)
sys.meta_path.insert(0,_HistoricalCoreFinder())
sys.path.insert(0,{root!r})
sys.path.insert(1,{repo!r})
from continuous_brain_unified.integration import route
TRACE = {trace!r}
if TRACE:
    base = route.Life
    class _Tracer:
        def __init__(self, inner, life):
            self.inner, self.life, self.phase, self.events = inner, life, 0, []
        def __call__(self, slot, api, *args):
            res = self.inner(slot, api, *args)
            ev = {{"phase": self.phase, "api": f"{{slot}}.{{api}}"}}
            if api == "reliability":
                ev["rel"] = [float(x) for x in res.reshape(-1).tolist()]
            elif api == "choose":
                ev["strategy"] = int(res)
            elif api == "check_target":
                ev["met"] = bool(res)
            elif api == "append" and self.life.env is not None:
                ev["cap"] = [float(self.life.env.capability_a), float(self.life.env.capability_b)]
            elif api == "generate_if_idle":
                ev["primary"] = None if res is None else int(res.primary)
                if res is not None:
                    self.phase += 1
            else:
                return res
            self.events.append(ev)
            return res
    class TracedLife(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.c = _Tracer(self.c, self)
        def finish(self):
            super().finish()
            with open(TRACE, "a") as f:
                f.write(json.dumps({{"sid": self.sid, "arm": self.arm, "events": self.c.events}}) + "\\n")
    route.Life = TracedLife
raise SystemExit(route.run_arm(arm={arm!r}, sids={sids!r}, out_dir={out!r}, legacy_root={root!r}, plan_path=None))
'''


def _arm(run_dir: Path, name: str, arm: str, sids, legacy, core, threads, trace: bool):
    out = run_dir / name
    boot = run_dir / f"bootstrap_{name}.py"
    boot.write_text(TRACE_BOOT.format(threads=threads, overrides=core, root=str(legacy), repo=str(REPO), arm=arm,
                                      sids=sids, out=str(out), trace=str(run_dir / f"trace_{name}.jsonl") if trace else ""))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    p = subprocess.run([sys.executable, "-B", str(boot)], cwd=legacy, env=env, text=True, capture_output=True)
    (run_dir / f"{name}.stderr.txt").write_text(p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"{name} failed: {p.stderr[-2000:]}")
    lives = {json.loads(x)["sid"]: json.loads(x) for x in (out / "lives.jsonl").read_text().splitlines() if x.strip()}
    bad = [s for s, r in lives.items() if "error" in r]
    if bad:
        raise SystemExit(f"{name}: errors in {bad[:5]}")
    print(json.dumps({"arm": name, "lives": len(lives)}), flush=True)
    return lives


def _goal2_series(trace_rows: dict, lives: dict) -> dict:
    """Per completer: per goal-2 episode k, the goal commitment's reliability before the episode and
    the true capability after it; plus the starting values."""
    out = {}
    for sid, tr in trace_rows.items():
        rec = lives[sid]["record"]
        if not rec.get("first_completed_store") or rec.get("second_goal") is None:
            continue
        p = int(rec["second_goal"]["primary"])
        i = 0 if p == 1 else 1
        ev = [e for e in tr["events"] if e["phase"] == 2]
        rels, caps, strategies = [], [], []
        for e in ev:
            if e["api"] == "autobiography.reliability":
                rels.append(e["rel"][i])
            elif e["api"] == "autobiography.append":
                caps.append(e["cap"][i])
            elif e["api"] == "skills.choose":
                strategies.append(e["strategy"])
        out[sid] = {"primary": p, "baseline": rec["second_goal"]["baseline_reliability"],
                    "target": rec["second_goal"]["target_reliability"], "cap0": rec["capability_after_first"][i],
                    "rel": rels, "cap": caps, "strategies": strategies, "completed": bool(rec["second_completed"])}
    return out


def _matched(series: dict, k: int):
    """Mean reliability gain and capability gain after k goal-2 episodes, over lives that ran >= k episodes
    and had not completed earlier (so both arms are compared on the same episode count)."""
    dr, dc = [], []
    for s in series.values():
        if len(s["cap"]) >= k and len(s["rel"]) > k:
            dr.append(s["rel"][k] - s["baseline"])
            dc.append(s["cap"][k - 1] - s["cap0"])
    return (st.mean(dr) if dr else None, st.mean(dc) if dc else None, len(dr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-root", required=True)
    ap.add_argument("--historical-core-archive", required=True)
    ap.add_argument("--family", default="cloud-dev-claude-eng")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    reg = SeedRegistry.load()
    prereg = json.loads((REPO / runner.PREREG_FILE).read_text())
    fam, sids = runner._dev_sids(reg, prereg, a.family, a.start, a.n)
    legacy = Path(a.legacy_root).resolve()
    problems = runner.verify_pins(legacy)
    if problems:
        raise SystemExit("legacy pins do not verify: " + "; ".join(problems[:5]))
    run_dir = (REPO / "runs" / "dev" / a.label).resolve()
    if run_dir.exists():
        raise SystemExit(f"output exists: {run_dir}")
    run_dir.mkdir(parents=True)
    core = runner._extract_historical_core(a.historical_core_archive, run_dir / "historical_core", str(runner.CORE_MANIFEST))
    with open(REPO / "runs/dev/seed_usage.jsonl", "a") as f:
        f.write(json.dumps({"family": fam.name, "start": sids[0], "stop": sids[-1], "mechanism": "I1-S3-trace",
                            "label": a.label, "purpose": "ENGINEERING_ONLY",
                            "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    before = runner._tree_metadata(legacy)
    plain = _arm(run_dir, "NORMAL_UNTRACED", "NORMAL", sids, legacy, core, a.threads, trace=False)
    normal = _arm(run_dir, "NORMAL", "NORMAL", sids, legacy, core, a.threads, trace=True)
    fixed = _arm(run_dir, "FIXED_SCAFFOLD", "FIXED_SCAFFOLD", sids, legacy, core, a.threads, trace=True)
    if runner._tree_metadata(legacy) != before:
        raise SystemExit("legacy tree metadata changed")
    for s in sids:                                  # the tracer must not change the individual
        x, y = plain[s]["record"], normal[s]["record"]
        if (x["final_store_digest"], x["final_tick"], x["life_trace_sha256"]) != \
           (y["final_store_digest"], y["final_tick"], y["life_trace_sha256"]):
            raise SystemExit(f"tracing changed individual {s}")
    tr = {arm: {json.loads(l)["sid"]: json.loads(l) for l in (run_dir / f"trace_{arm}.jsonl").read_text().splitlines()}
          for arm in ("NORMAL", "FIXED_SCAFFOLD")}
    S = {"NORMAL": _goal2_series(tr["NORMAL"], normal), "FIXED_SCAFFOLD": _goal2_series(tr["FIXED_SCAFFOLD"], fixed)}
    both = sorted(set(S["NORMAL"]) & set(S["FIXED_SCAFFOLD"]))
    rep = {"format": "cb-i1-s3-trace/1", "status": "ENGINEERING_ONLY", "label": a.label, "individuals": [sids[0], sids[-1]],
           "manifest_sha256": runner.manifest(a.threads)["manifest_sha256"], "tracing_changed_nothing": True,
           "completers_both_arms": len(both), "matched_by_episode": {}}
    for k in (1, 2, 4, 6, 8, 10, 12, 16):
        row = {}
        for arm in ("NORMAL", "FIXED_SCAFFOLD"):
            r, c, n = _matched({s: S[arm][s] for s in both}, k)
            row[arm] = {"mean_reliability_gain": r, "mean_capability_gain": c, "lives_still_pursuing": n}
        rep["matched_by_episode"][str(k)] = row
    # per-episode efficiency over all goal-2 episodes: reliability gain per unit of true capability gain
    for arm in ("NORMAL", "FIXED_SCAFFOLD"):
        dr = [s["rel"][j + 1] - s["rel"][j] for s in S[arm].values() for j in range(min(len(s["cap"]), len(s["rel"]) - 1))]
        caps = [(s["cap"][j] - (s["cap"][j - 1] if j else s["cap0"])) for s in S[arm].values()
                for j in range(min(len(s["cap"]), len(s["rel"]) - 1))]
        rep[f"per_episode_{arm}"] = {"episodes": len(dr), "mean_reliability_step": st.mean(dr) if dr else None,
                                     "mean_capability_step": st.mean(caps) if caps else None,
                                     "strategy_DIRECT_share": (sum(1 for s in S[arm].values() for x in s["strategies"] if x == 0) /
                                                               max(1, sum(len(s["strategies"]) for s in S[arm].values())))
                                     if arm == "NORMAL" else 0.0}
    (run_dir / "s3_trace_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
