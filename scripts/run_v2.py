"""Run Gate v2.

  python scripts/run_v2.py dev    [--seeds 0,1] --out results/v2_dev.json   # dev seeds 0-9 only, for tuning
  python scripts/run_v2.py sealed [--max-seeds N]                           # seeds 1000-1019, family opened once
  python scripts/run_v2.py sealed --resume [--max-seeds N]                  # continue an opened, unfinished family

Sealed protections (Sol review, DAV-104):
  * dev accepts only seeds 0-9; sealed seeds cannot be reached through dev.
  * PREREG_v2.md must be committed, contain the SHA-256 of configs/v2.json, and the working tree must be clean.
  * Before any evaluation, results/v2_sealed.OPENED is created atomically (O_EXCL) with commit and time.
    If it already exists a fresh open refuses, so the family cannot be reopened or restarted from seed 1000.
  * Competence is gated on the sealed-only confirmation sets (data/confirm_*.txt), never read in development.
  * Every completed seed is written (atomically) to results/v2_sealed.json as it finishes, so nothing is lost.

Resume (PREREG_v2 r4): a workspace restart kills the process mid-family. `--resume` continues it only if
  * the marker exists and results/v2_sealed.json is an unfinished sealed run with the same config SHA-256;
  * the code, configs, data and PREREG are byte-identical to the commit recorded in the marker
    (git diff against that commit is empty; only results/ may differ) and the tree is otherwise clean;
  * the recorded rows are exactly seeds 1000..1000+k-1 in order. They are kept verbatim; no recorded seed is
    ever rerun or replaced. Only a seed that was interrupted before its row was written runs (from scratch).
  Every segment (start, end, seeds, backbone hash before/after) is logged in the results file and in the marker.
  `--max-seeds N` ends a segment cleanly after N new seeds.
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
from being.competence import DEV_MONITOR_SETS, SEALED_SETS  # noqa: E402
from being.gate_v2 import V2Config, run, validate_prior  # noqa: E402

DEV = list(range(0, 10))
SEALED = list(range(1000, 1020))
CONFIG = ROOT / "configs/v2.json"
FROZEN_PATHS = ["being", "scripts", "configs", "data", "tests", "PREREG_v2.md"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a, check=True) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=check).stdout.strip()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def sealed_preconditions(cfg_path: Path) -> None:
    prereg = ROOT / "PREREG_v2.md"
    if not prereg.exists() or sha(cfg_path) not in prereg.read_text():
        sys.exit("SHA-256 of configs/v2.json is not recorded in PREREG_v2.md")
    if not git("log", "-1", "--format=%H", "--", "PREREG_v2.md"):
        sys.exit("PREREG_v2.md is not committed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["dev", "sealed"])
    ap.add_argument("--out")
    ap.add_argument("--seeds", help="comma list, dev only, subset of 0-9")
    ap.add_argument("--config", default=str(CONFIG), help="dev only")
    ap.add_argument("--resume", action="store_true", help="sealed only: continue an opened, unfinished family")
    ap.add_argument("--max-seeds", type=int, help="sealed only: end this segment after N new seeds")
    a = ap.parse_args()
    commit = git("rev-parse", "HEAD")
    prior_rows, segments, max_new = [], [], None

    if a.which == "sealed":
        cfg_path = CONFIG
        sealed_preconditions(cfg_path)
        marker = ROOT / "results/v2_sealed.OPENED"
        out, wdir, comp_sets, seeds = ROOT / "results/v2_sealed.json", ROOT / "results/weights/sealed", SEALED_SETS, SEALED
        max_new = a.max_seeds
        if not a.resume:
            if git("status", "--porcelain"):
                sys.exit("working tree not clean; commit first")
            marker.parent.mkdir(exist_ok=True)
            try:
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                sys.exit("sealed family already opened (results/v2_sealed.OPENED); use --resume to continue it")
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"opened_utc": now(), "commit": commit, "config_sha256": sha(cfg_path)}) + "\n")
            opened_commit = commit
        else:
            if not marker.exists() or not out.exists():
                sys.exit("--resume needs an opened family (marker and results/v2_sealed.json)")
            opened = json.loads(marker.read_text().splitlines()[0])
            opened_commit = opened["commit"]
            prev = json.loads(out.read_text())
            if prev.get("status") == "complete":
                sys.exit("sealed family already complete")
            if opened["config_sha256"] != sha(cfg_path) or prev["config_sha256"] != sha(cfg_path):
                sys.exit("config differs from the opened family")
            if prev.get("opened_commit", prev.get("commit")) != opened_commit:
                sys.exit("results file does not belong to the opened family")
            if subprocess.run(["git", "diff", "--quiet", opened_commit, "HEAD", "--", *FROZEN_PATHS], cwd=ROOT).returncode:
                sys.exit(f"code/config/data/PREREG differ from the opened commit {opened_commit}")
            dirty = [ln for ln in git("status", "--porcelain").splitlines() if not ln[3:].startswith("results/")]
            if dirty:
                sys.exit(f"working tree not clean outside results/: {dirty}")
            prior_rows, segments = prev["per_seed"], prev.get("segments", [])
            try:
                validate_prior(prior_rows, seeds)
            except ValueError as e:
                sys.exit(str(e))
    else:
        if a.resume or a.max_seeds is not None:
            sys.exit("--resume/--max-seeds are sealed-only")
        cfg_path = Path(a.config).resolve()
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else DEV
        if not set(seeds) <= set(DEV):
            sys.exit("dev runs accept only seeds 0-9")
        out = ROOT / (a.out or "results/v2_dev.json")
        wdir = ROOT / "results/weights/dev"
        comp_sets = DEV_MONITOR_SETS
        opened_commit = commit

    cfg = V2Config(**json.loads(cfg_path.read_text()))
    meta = {"which": a.which, "commit": commit, "opened_commit": opened_commit, "config_sha256": sha(cfg_path),
            "config_path": str(cfg_path.relative_to(ROOT))}
    out.parent.mkdir(exist_ok=True)
    seg = {"started_utc": now(), "commit": commit, "resumed_after_seeds": [r["seed"] for r in prior_rows],
           "backbone_hash_before": None, "backbone_hash_after": None, "ended_utc": None}
    if a.which == "sealed":
        with open(marker, "a") as f:
            f.write(json.dumps({"segment_started_utc": seg["started_utc"], "commit": commit,
                                "resume": a.resume, "rows_already_recorded": len(prior_rows)}) + "\n")

    def partial(rows):
        write_atomic(out, {**meta, "status": "in_progress", "segments": segments + [seg], "per_seed": rows})

    model, tok = load_backbone(str(ROOT / "models/distilgpt2"))
    t0 = time.time()
    prior_ok = all(s.get("backbone_hash_after") is None or s["backbone_hash_before"] == s["backbone_hash_after"]
                   for s in segments)
    res = run(model, tok, cfg, seeds, wdir, cfg_path, on_row=partial, competence_sets=comp_sets,
              prior_rows=prior_rows, max_new=max_new, prior_segments_ok=prior_ok)
    seg.update(backbone_hash_before=res["backbone_hash_before"], backbone_hash_after=res["backbone_hash_after"],
               ended_utc=now(), seconds=round(time.time() - t0, 1),
               seeds_run=[r["seed"] for r in res["per_seed"][len(prior_rows):]])
    res.update(meta, status="complete" if res["complete"] else "in_progress", segments=segments + [seg])
    write_atomic(out, res)
    if a.which == "sealed":
        with open(marker, "a") as f:
            f.write(json.dumps({"segment_ended_utc": seg["ended_utc"], "seeds_run": seg["seeds_run"],
                                "complete": res["complete"]}) + "\n")
    print(json.dumps(res.get("summary", {"status": res["status"], "seeds_done": len(res["per_seed"])}), indent=2))


if __name__ == "__main__":
    main()
