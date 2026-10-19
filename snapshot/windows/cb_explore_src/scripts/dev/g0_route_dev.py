#!/usr/bin/env python3
"""ENGINEERING_ONLY development harness: run a frozen evaluator through the exact
G0 bootstrap (historical-core overlay, seed-symbol substitution, redirected
--state-dir) on a slice of the registered cloud-development family
cloud-dev-claude-eng [906000-906999].

Never scientific evidence. Every invocation appends to runs/dev/seed_usage.jsonl
so the canonical registry owner can see exactly which engineering seeds were used.
Outputs are written under runs/dev/<label>/ and never into the legacy root.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

from continuous_brain_protocols import sha256_file
from continuous_brain_eval.seeds import SeedRegistry
from continuous_brain_unified.legacy.g0_runner import (
    _bootstrap_source, _extract_historical_core, _tree_metadata, load_recipes,
)

FAMILY = "cloud-dev-claude-eng"
SLICES = {  # fixed per-mechanism development slices inside 906000-906999
    "consequence_replanning_v0252": 906000,
    "recurrent_self_trajectory_v031": 906100,
    "autobiographical_discrepancy_v0322": 906200,
    "event_consistent_self_development_v0341": 906300,
    "endogenous_goal_genesis_v0353": 906400,
    "recursive_autobiographical_curriculum_v0361": 906500,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", required=True, choices=sorted(SLICES))
    ap.add_argument("--n", type=int, default=8, help="seeds from the mechanism's slice (even; max 100)")
    ap.add_argument("--legacy-root", required=True)
    ap.add_argument("--historical-core-archive", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if a.n < 2 or a.n > 100 or a.n % 2:
        raise SystemExit("--n must be even and in 2..100")
    reg = SeedRegistry.load()
    fam = reg.get(FAMILY)
    start = SLICES[a.mechanism]; stop = start + a.n - 1
    if not (fam.start <= start and stop <= fam.stop) or fam.purpose.value != "ENGINEERING":
        raise SystemExit("slice outside the registered cloud-development family")
    reg.check_seeds(range(start, stop + 1), purpose_allowed=(fam.purpose,))
    recipes, _ = load_recipes()
    r = recipes[a.mechanism]
    legacy = Path(a.legacy_root).resolve()
    evaluator = legacy / r.evaluator
    run_dir = Path("runs/dev") / a.label / a.mechanism
    if run_dir.exists():
        raise SystemExit(f"output exists: {run_dir}")
    run_dir.mkdir(parents=True)
    raw = run_dir / "raw_evaluator"; raw.mkdir()
    overrides = _extract_historical_core(a.historical_core_archive, run_dir / "historical_core")
    src = _bootstrap_source(evaluator, r.seed_symbol, start, stop, r.seed_container, raw.resolve(), overrides)
    src = src.replace("sys.dont_write_bytecode=True\n",
                      f"sys.dont_write_bytecode=True\nimport torch\ntorch.set_num_threads({a.threads})\n", 1)
    (run_dir / "bootstrap.py").write_text(src)
    usage = Path("runs/dev/seed_usage.jsonl"); usage.parent.mkdir(parents=True, exist_ok=True)
    with open(usage, "a") as f:
        f.write(json.dumps({"family": FAMILY, "start": start, "stop": stop, "mechanism": a.mechanism,
                            "label": a.label, "purpose": "ENGINEERING_ONLY",
                            "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    before_meta = _tree_metadata(legacy)
    t = time.time()
    proc = subprocess.run([sys.executable, "-B", str((run_dir / "bootstrap.py").resolve())],
                          cwd=legacy, env=env, text=True, capture_output=True)
    secs = time.time() - t
    (run_dir / "stdout.txt").write_text(proc.stdout); (run_dir / "stderr.txt").write_text(proc.stderr)
    tree_unchanged = _tree_metadata(legacy) == before_meta
    res = raw / "result.json"
    report = {"status": "ENGINEERING_ONLY", "mechanism": a.mechanism, "seeds": [start, stop],
              "returncode": proc.returncode, "seconds": round(secs, 1),
              "legacy_tree_unchanged": tree_unchanged,
              "raw_result_sha256": sha256_file(res) if res.is_file() else None,
              "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.returncode not in (0, 4) else []}
    (run_dir / "dev_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
