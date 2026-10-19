"""Run Gate v2.

  python scripts/run_v2.py dev    --config configs/v2.json --out results/v2_dev.json      # seeds 0-9, for tuning
  python scripts/run_v2.py sealed --config configs/v2.json                                 # seeds 1000-1019, ONCE

The sealed run refuses to start unless PREREG_v2.md is committed, the config's SHA-256 matches the one written in
PREREG_v2.md, the working tree is clean, and results/v2_sealed.json does not already exist.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from being.backbone import load_backbone  # noqa: E402
from being.gate_v2 import V2Config, run  # noqa: E402

DEV = list(range(0, 10))
SEALED = list(range(1000, 1020))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["dev", "sealed"])
    ap.add_argument("--config", default="configs/v2.json")
    ap.add_argument("--out")
    ap.add_argument("--seeds", help="comma list, dev only")
    a = ap.parse_args()
    cfg_path = ROOT / a.config
    cfg = V2Config(**json.loads(cfg_path.read_text()))
    if a.which == "sealed":
        out = ROOT / "results/v2_sealed.json"
        prereg = ROOT / "PREREG_v2.md"
        if out.exists():
            sys.exit("sealed result already exists; the sealed gate runs once")
        if not prereg.exists() or sha(cfg_path) not in prereg.read_text():
            sys.exit("config SHA-256 is not recorded in PREREG_v2.md")
        if git("status", "--porcelain"):
            sys.exit("working tree not clean; commit first")
        if git("log", "-1", "--format=%H", "--", "PREREG_v2.md") == "":
            sys.exit("PREREG_v2.md is not committed")
        seeds = SEALED
    else:
        out = ROOT / (a.out or "results/v2_dev.json")
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else DEV
    model, tok = load_backbone()
    t0 = time.time()
    res = run(model, tok, cfg, seeds)
    res["which"] = a.which
    res["commit"] = git("rev-parse", "HEAD")
    res["config_sha256"] = sha(cfg_path)
    res["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
