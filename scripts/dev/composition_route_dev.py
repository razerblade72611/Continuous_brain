#!/usr/bin/env python3
"""ENGINEERING_ONLY: run a frozen evaluator through the store-mediated composition
route on a slice of cloud-dev-claude-eng [906000-906999], with a chosen set of
bound mechanism hosts, and write result + composition attestation.

  --bind all        all six hosts bound (full composition)
  --bind <ids,...>  only the listed mechanisms bound (other APIs report BLOCKED if called)

Same bootstrap as G0 (historical-core overlay, seed-symbol substitution,
redirected --state-dir). Never scientific evidence; seed use is appended to
runs/dev/seed_usage.jsonl.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from continuous_brain_protocols import sha256_file  # noqa: E402
from continuous_brain_eval.seeds import SeedRegistry  # noqa: E402
from continuous_brain_unified.composition.spec import SPECS_BY_ID  # noqa: E402
from continuous_brain_unified.legacy.g0_runner import (  # noqa: E402
    _extract_historical_core, _historical_core_digest, _tree_metadata, load_recipes)

sys.path.insert(0, str(REPO / "scripts" / "dev"))
from g0_route_dev import FAMILY, SLICES  # noqa: E402

BOOT = '''import importlib.abc, importlib.util, pathlib, sys
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
from continuous_brain_unified.composition.route import run_focal
raise SystemExit(run_focal(evaluator={evaluator!r}, seed_symbol={seed_symbol!r}, start={start}, stop={stop},
    container={container!r}, state_dir={state_dir!r}, bound={bound!r}, focal={focal!r},
    attestation_path={attestation!r}, legacy_root={root!r}, evaluator_sha256={evaluator_sha!r},
    historical_core_digest={core_digest!r}, store_interventions={interventions!r}))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", required=True, choices=sorted(SLICES))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--bind", default="all")
    ap.add_argument("--legacy-root", required=True)
    ap.add_argument("--historical-core-archive", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--store-control", default=None, help="JSON file: list of store-level positive-control interventions")
    a = ap.parse_args()
    if a.n < 2 or a.n > 100 or a.n % 2:
        raise SystemExit("--n must be even and in 2..100")
    bound = sorted(SPECS_BY_ID) if a.bind == "all" else sorted(x for x in a.bind.split(",") if x)
    reg = SeedRegistry.load()
    fam = reg.get(FAMILY)
    start = SLICES[a.mechanism]; stop = start + a.n - 1
    reg.check_seeds(range(start, stop + 1), purpose_allowed=(fam.purpose,))
    recipes, _ = load_recipes()
    r = recipes[a.mechanism]
    legacy = Path(a.legacy_root).resolve()
    evaluator = legacy / r.evaluator
    run_dir = (Path("runs/dev") / a.label / a.mechanism).resolve()
    if run_dir.exists():
        raise SystemExit(f"output exists: {run_dir}")
    run_dir.mkdir(parents=True)
    raw = run_dir / "raw_evaluator"; raw.mkdir()
    overrides = _extract_historical_core(a.historical_core_archive, run_dir / "historical_core")
    src = BOOT.format(threads=a.threads, overrides=overrides, root=str(legacy), repo=str(REPO),
                      evaluator=str(evaluator), seed_symbol=r.seed_symbol, start=start, stop=stop,
                      container=r.seed_container, state_dir=str(raw), bound=bound, focal=a.mechanism,
                      attestation=str(run_dir / "composition_attestation.json"),
                      evaluator_sha=sha256_file(evaluator), core_digest=_historical_core_digest(),
                      interventions=json.loads(Path(a.store_control).read_text()) if a.store_control else None)
    (run_dir / "bootstrap.py").write_text(src)
    usage = Path("runs/dev/seed_usage.jsonl"); usage.parent.mkdir(parents=True, exist_ok=True)
    with open(usage, "a") as f:
        f.write(json.dumps({"family": FAMILY, "start": start, "stop": stop, "mechanism": a.mechanism,
                            "label": a.label, "route": "composition", "bound": bound,
                            "purpose": "ENGINEERING_ONLY", "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    before_meta = _tree_metadata(legacy)
    t = time.time()
    proc = subprocess.run([sys.executable, "-B", str(run_dir / "bootstrap.py")], cwd=legacy, env=env,
                          text=True, capture_output=True)
    secs = time.time() - t
    (run_dir / "stdout.txt").write_text(proc.stdout); (run_dir / "stderr.txt").write_text(proc.stderr)
    tree_unchanged = _tree_metadata(legacy) == before_meta
    res = raw / "result.json"
    att_p = run_dir / "composition_attestation.json"
    att = json.loads(att_p.read_text()) if att_p.is_file() else {}
    report = {"status": "ENGINEERING_ONLY", "route": "composition", "mechanism": a.mechanism, "bound": bound,
              "seeds": [start, stop], "returncode": proc.returncode, "seconds": round(secs, 1),
              "legacy_tree_unchanged": tree_unchanged,
              "raw_result_sha256": sha256_file(res) if res.is_file() else None,
              "route_blocked": att.get("route_blocked"), "ticks": att.get("ticks"),
              "stderr_tail": proc.stderr.strip().splitlines()[-4:] if proc.returncode not in (0, 4) else []}
    (run_dir / "dev_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
