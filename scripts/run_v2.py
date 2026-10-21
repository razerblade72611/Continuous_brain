"""Run Gate v2.

  python scripts/run_v2.py dev    [--seeds 0,1] --out results/v2_dev.json   # dev seeds 0-9 only, for tuning
  python scripts/run_v2.py sealed [--max-seeds N]                           # seeds 1000-1019, family opened once
  python scripts/run_v2.py sealed --resume [--max-seeds N]                  # continue an opened, unfinished family

Sealed protections (Sol review, DAV-104):
  * dev accepts only seeds 0-9; sealed seeds cannot be reached through dev.
  * PREREG_v2.md must be committed, contain the SHA-256 of configs/v2.json, and the working tree must be clean.
  * Every sealed process (open or resume) first takes an OS-held exclusive lock (flock on results/v2_sealed.lock),
    held until exit; a second process refuses. Process death releases it automatically.
  * Before any evaluation, results/v2_sealed.OPENED is created atomically (O_EXCL) with commit and time, and an
    empty resumable state is written to results/v2_sealed.json before the model is loaded.
    If the marker already exists a fresh open refuses, so the family cannot be restarted from seed 1000.
  * Competence is gated on the sealed-only confirmation sets (data/confirm_*.txt), never read in development.
  * Every completed seed is written atomically to results/v2_sealed.json, then its row SHA-256 is appended to the
    marker (append-only log). A deleted or edited results file is therefore detectable.

Resume (PREREG_v2 r5): `--resume` continues an opened family only if
  * the marker exists with the same config SHA-256; the results file (if present) is unfinished and belongs to it;
  * the code, configs, data, tests and PREREG are byte-identical to the commit recorded in the marker and the tree
    is clean outside results/;
  * the recorded rows are exactly seeds 1000..1000+k-1 in order, every row logged in the marker matches the
    results file by SHA-256, and at most one trailing row is unlogged (a kill between the two writes).
  If the results file is missing it resumes from zero ONLY if the marker logs no rows (a kill before the initial
  write). Recorded rows are kept verbatim; no recorded seed is ever rerun or replaced.
  `--max-seeds N` ends a segment cleanly.
"""
import argparse
import fcntl
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
MARKER = ROOT / "results/v2_sealed.OPENED"
LOCK = ROOT / "results/v2_sealed.lock"
_lock_fd = None


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def row_sha(row) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def git(*a, check=True) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=check).stdout.strip()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def log_marker(entry: dict) -> None:
    with open(MARKER, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def take_lock() -> None:
    """OS-held exclusive writer lock for the whole process lifetime; released automatically on death."""
    global _lock_fd
    LOCK.parent.mkdir(exist_ok=True)
    _lock_fd = os.open(LOCK, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another sealed process holds results/v2_sealed.lock")


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
    prior_rows, segments, max_new, logged = [], [], None, 0
    sealed = a.which == "sealed"

    if sealed:
        cfg_path = CONFIG
        sealed_preconditions(cfg_path)
        out, wdir, comp_sets, seeds = ROOT / "results/v2_sealed.json", ROOT / "results/weights/sealed", SEALED_SETS, SEALED
        max_new = a.max_seeds
        if not a.resume:
            if git("status", "--porcelain"):
                sys.exit("working tree not clean; commit first")
            take_lock()
            try:
                fd = os.open(MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                sys.exit("sealed family already opened (results/v2_sealed.OPENED); use --resume to continue it")
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"opened_utc": now(), "commit": commit, "config_sha256": sha(cfg_path)}) + "\n")
                f.flush()
                os.fsync(f.fileno())
            opened_commit = commit
        else:
            take_lock()
            if not MARKER.exists():
                sys.exit("--resume needs an opened family (results/v2_sealed.OPENED)")
            lines = [json.loads(x) for x in MARKER.read_text().splitlines() if x.strip()]
            opened, recorded = lines[0], [(e["row_recorded"], e["row_sha256"]) for e in lines if "row_recorded" in e]
            opened_commit = opened["commit"]
            if opened["config_sha256"] != sha(cfg_path):
                sys.exit("config differs from the opened family")
            if out.exists():
                prev = json.loads(out.read_text())
                if prev.get("status") == "complete":
                    sys.exit("sealed family already complete")
                if prev["config_sha256"] != sha(cfg_path) or prev.get("opened_commit") != opened_commit:
                    sys.exit("results file does not belong to the opened family")
                prior_rows, segments = prev["per_seed"], prev.get("segments", [])
            elif recorded:
                sys.exit("results/v2_sealed.json is missing but the marker logs recorded rows; refusing")
            if subprocess.run(["git", "diff", "--quiet", opened_commit, "HEAD", "--", *FROZEN_PATHS], cwd=ROOT).returncode:
                sys.exit(f"code/config/data/PREREG differ from the opened commit {opened_commit}")
            dirty = [ln for ln in git("status", "--porcelain").splitlines() if not ln[3:].startswith("results/")]
            if dirty:
                sys.exit(f"working tree not clean outside results/: {dirty}")
            try:
                validate_prior(prior_rows, seeds)
            except ValueError as e:
                sys.exit(str(e))
            if len(prior_rows) < len(recorded) or len(prior_rows) > len(recorded) + 1:
                sys.exit(f"results hold {len(prior_rows)} rows but the marker logs {len(recorded)}; refusing")
            for (s, h), r in zip(recorded, prior_rows):
                if r["seed"] != s or row_sha(r) != h:
                    sys.exit(f"recorded row for seed {s} does not match the marker log; refusing")
            logged = len(recorded)
            for r in prior_rows[logged:]:  # a kill between the results write and its marker entry
                log_marker({"row_recorded": r["seed"], "row_sha256": row_sha(r), "logged_on_resume": True})
            logged = len(prior_rows)
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
    if sealed:
        log_marker({"segment_started_utc": seg["started_utc"], "commit": commit, "resume": a.resume,
                    "rows_already_recorded": len(prior_rows)})

    def partial(rows):
        nonlocal logged
        write_atomic(out, {**meta, "status": "in_progress", "segments": segments + [seg], "per_seed": rows})
        if sealed:
            for r in rows[logged:]:
                log_marker({"row_recorded": r["seed"], "row_sha256": row_sha(r)})
            logged = len(rows)

    partial(prior_rows)  # resumable state exists before the model is even loaded
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
    if sealed:
        log_marker({"segment_ended_utc": seg["ended_utc"], "seeds_run": seg["seeds_run"], "complete": res["complete"]})
    print(json.dumps(res.get("summary", {"status": res["status"], "seeds_done": len(res["per_seed"])}), indent=2))


if __name__ == "__main__":
    main()
