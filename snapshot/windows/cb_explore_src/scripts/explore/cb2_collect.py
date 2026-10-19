#!/usr/bin/env python3
"""CB2-S1 experience collection (exploration lane, ENGINEERING_ONLY).

  wake      the one individual lives, every tick archived as LIVED; saved after every life
            python scripts/explore/cb2_collect.py wake --lives 20 --seed0 75101
  practice  N worker processes, each a routine teacher in fresh worlds, archived as PRACTICE x TEACHER
            python scripts/explore/cb2_collect.py practice --workers 2 --steps 200000
  bench     practice throughput for 1, 2, ... workers (nothing is kept unless --keep)
            python scripts/explore/cb2_collect.py bench --workers 1,2 --seconds 60

Seeds: wake from --seed0 (default 75101); practice worker k of run r uses 80,000,000 + r*1,000,000 + k*100,000 + i.
75001-75040 are CB2 engineering evaluation and 90001-90200 are untouched; neither is ever collected here.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"
ARCHIVE = REPO / "runs/explore/cb2/archive"
INDIVIDUALS = REPO / "runs/explore/cb2/individuals"
RESERVED = [(75001, 75040), (90001, 90200)]
PRACTICE_BASE = 80_000_000


def reserved(seed: int) -> bool:
    return any(a <= seed <= b for a, b in RESERVED)


def git_rev() -> dict:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True,
                                capture_output=True).stdout.strip())
    return {"git": rev, "git_dirty": dirty}


def vision():
    import torch
    torch.set_num_threads(1)
    from continuous_brain_organs.vision import VisionOrgan
    return VisionOrgan(WEIGHTS, CARD)


def snapshot_path(home: Path, meta: dict) -> Path:
    """Saves are named relative to the individual's home, so a copied or moved home still finds its own."""
    p = Path(meta["snapshot"])
    return p if p.is_absolute() and p.exists() else home / "checkpoints" / p.name


def provenance(v, role: str) -> dict:
    return {"role": role, **git_rev(), "vision_weights_sha256": v.weights_sha256, "language": "development_parser",
            "skills": "routine"}


# ------------------------------------------------------------------------------------------------------ wake
def wake(a) -> int:
    from continuous_brain_cb2.archive import StreamWriter
    from continuous_brain_cb2.record import record_life
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_snn.snapshot import SnapshotStore
    from continuous_brain_world.env import W1Config, W1Env
    v = vision()
    home = Path(a.individuals) / a.individual
    store = SnapshotStore(home)
    start = 0
    meta_path = home / "individual.json"
    prior = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    skills = prior.get("skills", "routine")          # "cb2" once a consolidation gave it its learned skill
    ind = Individual(IndividualConfig(lifelong=True, skills=skills), v, LanguageOrgan.from_config({"backend": "dev"}))
    if prior.get("in_life"):
        raise SystemExit("it was stopped in the middle of a life in the viewer: finish that life there first "
                         "(python scripts/explore/w1_viewer.py --individual %s)" % a.individual)
    if prior:
        meta = prior
        ind.rt.restore(store, snapshot_path(home, meta))
        ind.life, start = int(meta["life"]), int(meta["lives_lived"])
        if ind.state_hash() != meta["state_hash"]:
            raise SystemExit("the saved individual does not restore to its recorded state")
    w = StreamWriter(a.archive, f"lived-{a.individual}", "LIVED", a.individual, provenance(v, "wake"))
    for k in range(start, start + a.lives):
        seed = a.seed0 + k
        if reserved(seed):
            raise SystemExit(f"seed {seed} is reserved for evaluation")
        t = time.time()
        rec = record_life(ind, W1Env(W1Config()), seed, w, steps=a.steps)
        w.close()                                   # each life ends a shard: a crash never loses a sealed life
        ind.next_life()
        snap = ind.rt.save(store, label=f"lives{k + 1}")
        meta = {**prior, "individual": a.individual, "snapshot": snap.name, "life": ind.life, "lives_lived": k + 1,
                "skills": skills, "state_hash": ind.state_hash(), **git_rev()}
        Path(str(snap) + ".meta.json").write_text(json.dumps(meta) + "\n")
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps({"life": k, "seed": seed, **rec, "ticks_per_s": round(rec["steps"] / (time.time() - t), 1)}),
              flush=True)
    return 0


# -------------------------------------------------------------------------------------------------- practice
def _worker(args) -> dict:
    k, run, archive, steps, seconds, keep = args
    import tempfile
    from continuous_brain_cb2.archive import StreamWriter
    from continuous_brain_cb2.record import record_life
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_world.env import W1Config, W1Env
    v = vision()
    root = archive if keep else tempfile.mkdtemp(prefix="cb2bench")
    w = StreamWriter(root, f"practice-r{run:03d}-w{k:02d}", "PRACTICE", "teacher-routine", provenance(v, "practice"))
    t0, n, i = time.time(), 0, 0
    lives = []
    while (steps is None or n < steps) and (seconds is None or time.time() - t0 < seconds):
        seed = PRACTICE_BASE + run * 1_000_000 + k * 100_000 + i
        teacher = Individual(IndividualConfig(), v, LanguageOrgan.from_config({"backend": "dev"}))
        budget = 10000 if steps is None else min(10000, steps - n)
        rec = record_life(teacher, W1Env(W1Config()), seed, w, steps=budget, who="TEACHER", lived=False,
                          stop=None if seconds is None else (lambda: time.time() - t0 >= seconds))
        n += rec["steps"]
        lives.append(rec)
        i += 1
    w.close()
    return {"worker": k, "steps": n, "seconds": time.time() - t0, "lives": len(lives)}


def _run_workers(n_workers, run, archive, steps_each, seconds, keep):
    ctx = mp.get_context("fork")
    t = time.time()
    with ctx.Pool(n_workers) as pool:
        res = pool.map(_worker, [(k, run, archive, steps_each, seconds, keep) for k in range(n_workers)])
    wall = time.time() - t
    total = sum(r["steps"] for r in res)
    return {"workers": n_workers, "steps": total, "wall_s": round(wall, 1),
            "steps_per_s": round(total / max(r["seconds"] for r in res), 1), "per_worker": res}


def practice(a) -> int:
    out = _run_workers(a.workers, a.run, a.archive, a.steps // a.workers, None, True)
    print(json.dumps(out), flush=True)
    return 0


def bench(a) -> int:
    rows = []
    for n in [int(x) for x in a.workers.split(",")]:
        r = _run_workers(n, 999, a.archive, None, a.seconds, a.keep)
        rows.append({k: r[k] for k in ("workers", "steps", "steps_per_s")})
        print(json.dumps(rows[-1]), flush=True)
    base = rows[0]["steps_per_s"] / rows[0]["workers"]
    for r in rows:
        r["efficiency"] = round(r["steps_per_s"] / (base * r["workers"]), 3)
    out = {"cpus": os.cpu_count(), "rows": rows, **git_rev()}
    print(json.dumps(out), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wake")
    w.add_argument("--lives", type=int, default=20)
    w.add_argument("--seed0", type=int, default=75101)
    w.add_argument("--steps", type=int, default=10000)
    w.add_argument("--individual", default="cb2-ind-0001")
    w.add_argument("--archive", default=str(ARCHIVE))
    w.add_argument("--individuals", default=str(INDIVIDUALS))
    p = sub.add_parser("practice")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=200000, help="total, split across workers")
    p.add_argument("--run", type=int, default=0, help="practice run number (its seeds)")
    p.add_argument("--archive", default=str(ARCHIVE))
    b = sub.add_parser("bench")
    b.add_argument("--workers", default="1,2")
    b.add_argument("--seconds", type=float, default=60)
    b.add_argument("--keep", action="store_true")
    b.add_argument("--archive", default=str(ARCHIVE))
    b.add_argument("--out", default=None)
    a = ap.parse_args()
    return {"wake": wake, "practice": practice, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
