"""Run Gate v2.

  python scripts/run_v2.py dev    [--seeds 0,1] --out results/v2_dev.json   # dev seeds 0-9 only, for tuning
  python scripts/run_v2.py sealed                                           # seeds 1000-1019, once ever

Sealed protections (Sol review, DAV-104):
  * dev accepts only seeds 0-9; sealed seeds cannot be reached through dev.
  * PREREG_v2.md must be committed, contain the SHA-256 of configs/v2.json, and the working tree must be clean.
  * Before any evaluation, results/v2_sealed.OPENED is created atomically (O_EXCL) with commit and time.
    If it already exists the run refuses, so an interrupted or concurrent attempt cannot reopen the family.
  * Every completed seed is written to results/v2_sealed.json as it finishes, so a failed attempt is retained.
"""
import argparse
import hashlib
import json
import os
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
CONFIG = ROOT / "configs/v2.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["dev", "sealed"])
    ap.add_argument("--out")
    ap.add_argument("--seeds", help="comma list, dev only, subset of 0-9")
    ap.add_argument("--config", default=str(CONFIG), help="dev only")
    a = ap.parse_args()
    commit = git("rev-parse", "HEAD")

    if a.which == "sealed":
        cfg_path = CONFIG
        prereg = ROOT / "PREREG_v2.md"
        if not prereg.exists() or sha(cfg_path) not in prereg.read_text():
            sys.exit("SHA-256 of configs/v2.json is not recorded in PREREG_v2.md")
        if git("status", "--porcelain"):
            sys.exit("working tree not clean; commit first")
        if not git("log", "-1", "--format=%H", "--", "PREREG_v2.md"):
            sys.exit("PREREG_v2.md is not committed")
        marker = ROOT / "results/v2_sealed.OPENED"
        marker.parent.mkdir(exist_ok=True)
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            sys.exit("sealed family already opened (results/v2_sealed.OPENED); it runs once")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "commit": commit, "config_sha256": sha(cfg_path)}) + "\n")
        seeds, out, wdir = SEALED, ROOT / "results/v2_sealed.json", ROOT / "results/weights/sealed"
    else:
        cfg_path = Path(a.config).resolve()
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else DEV
        if not set(seeds) <= set(DEV):
            sys.exit("dev runs accept only seeds 0-9")
        out = ROOT / (a.out or "results/v2_dev.json")
        wdir = ROOT / "results/weights/dev"

    cfg = V2Config(**json.loads(cfg_path.read_text()))
    meta = {"which": a.which, "commit": commit, "config_sha256": sha(cfg_path), "config_path": str(cfg_path.relative_to(ROOT))}
    out.parent.mkdir(exist_ok=True)

    def partial(rows):
        out.write_text(json.dumps({**meta, "status": "in_progress", "per_seed": rows}, indent=2))

    model, tok = load_backbone(str(ROOT / "models/distilgpt2"))
    t0 = time.time()
    res = run(model, tok, cfg, seeds, wdir, cfg_path, on_row=partial)
    res.update(meta, status="complete", seconds=round(time.time() - t0, 1))
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
