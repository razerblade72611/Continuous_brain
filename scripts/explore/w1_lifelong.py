#!/usr/bin/env python3
"""One individual living life after life in W1, learning from them (exploration lane, ENGINEERING_ONLY).

  train     python scripts/explore/w1_lifelong.py train --lives 150 --out runs/explore/lifelong/run1
            One lifelong individual with learned skills lives on training worlds (seeds 73001, 73002, ...).
            After each life: one line in <out>/lives.jsonl (W1's grading of the life plus what it learned).
            Every --save-every lives: the whole individual is saved (never-overwrite SnapshotStore) under
            <out>/individual/checkpoints/, with <snapshot>.meta.json beside it.

  evaluate  python scripts/explore/w1_lifelong.py evaluate --snapshot <dir> [--seeds 71001-71020]
            The individual as saved, frozen (it does not learn while being tested), living each test world
            once, starting from the same saved state each time.  Writes <snapshot>.eval.json.  The test
            worlds are the 20 of INDIVIDUAL_W1.md section 3, which it never trained on.

  untrained python scripts/explore/w1_lifelong.py evaluate --untrained [--seeds ...] --out <file>
            The same evaluation for a newborn learned individual (no lives yet): the starting point.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # small matrices: threads only cost
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from continuous_brain_snn.snapshot import SnapshotStore  # noqa: E402
from continuous_brain_world.env import W1Config, W1Env  # noqa: E402
from continuous_brain_world.scoring import ACHIEVEMENTS, crafter_score  # noqa: E402
from continuous_brain_organs.language import LanguageOrgan  # noqa: E402
from continuous_brain_individual.individual import Individual, IndividualConfig, first_info  # noqa: E402

WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"
TRAIN_SEED0 = 73001


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


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


def build(v, learn: bool) -> Individual:
    return Individual(IndividualConfig(skills="learned", learn=learn, lifelong=True), v,
                      LanguageOrgan.from_config({"backend": "dev"}))


def live_one(ind: Individual, seed: int, steps: int) -> dict:
    env = W1Env(W1Config())
    obs = env.reset(seed)
    info = first_info(env)
    n, done, t = 0, False, time.time()
    while n < steps and not done:
        a, u = ind.decide(obs, info)
        obs, _, done, info = env.step(a, u)
        n += 1
    if done:
        ind.die(obs, info)
    s = env.episode_summary()
    return {"seed": seed, "steps": n, "died": bool(done), "seconds": round(time.time() - t, 1),
            "unlocked": sorted(s["crafter"]["unlocked"]), "n_unlocked": s["crafter"]["n_unlocked"],
            "fulfilled": s["obligations"]["fulfilled"], "failed": s["obligations"]["failed"],
            "memory": [s["memory"]["correct"], s["memory"]["asked"]]}


def learning_stats(ind: Individual) -> dict:
    sk = ind.read_state("skills")["ctl"]
    wm = ind.read_state("world_model")
    auto = ind.read_state("autobiography") or {}
    last = (auto.get("lives") or [{}])[-1]
    return {"decisions": sk["decisions"], "q_updates": sk["updates"], "q_loss": sk["loss"],
            "options": sk["life_options"], "life_return": round(sk["returns"][-1], 3) if sk["returns"] else None,
            "foresight_samples": wm["samples"], "foresight_loss": wm["loss"],
            "foresight_brier": [None if b is None else round(b, 4) for b in wm["brier"]],
            "felt_at_death": last.get("felt")}


def summarize(eps: list[dict]) -> dict:
    unlocked = [set(e["unlocked"]) for e in eps]
    succ = [100.0 * sum(a in u for u in unlocked) / len(unlocked) for a in ACHIEVEMENTS]
    ful, fail = sum(e["fulfilled"] for e in eps), sum(e["failed"] for e in eps)
    return {"episodes": len(eps), "crafter_score": crafter_score(succ),
            "achievements_per_episode": float(np.mean([e["n_unlocked"] for e in eps])),
            "survival_steps": float(np.mean([e["steps"] for e in eps])), "deaths": sum(e["died"] for e in eps),
            "obligations_on_time": {"fulfilled": ful, "failed": fail, "rate": ful / (ful + fail) if ful + fail else None},
            "memory": [sum(e["memory"][0] for e in eps), sum(e["memory"][1] for e in eps)]}


# ------------------------------------------------------------------------------------------------- train
def train(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = vision()
    ind = build(v, learn=True)
    store = SnapshotStore(out / "individual")
    start = 0
    if a.resume:
        snap = Path(a.resume)
        meta = json.loads(Path(str(snap) + ".meta.json").read_text())
        ind.rt.restore(store, snap)
        ind.life = int(meta["life"])
        start = int(meta["lives_lived"])
    log = out / "lives.jsonl"
    header = {"format": "cb-w1-lifelong/1", "status": "ENGINEERING_ONLY", **git_rev(),
              "vision_weights_sha256": v.weights_sha256, "language": "development_parser", "steps": a.steps,
              "train_seed0": TRAIN_SEED0}
    (out / "run.json").write_text(json.dumps(header, indent=1) + "\n") if not (out / "run.json").exists() else None
    for k in range(start, start + a.lives):
        rec = live_one(ind, TRAIN_SEED0 + k, a.steps)
        rec.update(life=k, **learning_stats(ind))
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({x: rec[x] for x in ("life", "seed", "steps", "n_unlocked", "fulfilled", "failed",
                                                "decisions", "q_updates", "life_return", "felt_at_death")}),
              flush=True)
        ind.next_life()
        if (k + 1) % a.save_every == 0 or k + 1 == start + a.lives:
            snap = ind.rt.save(store, label=f"lives{k + 1}")
            Path(str(snap) + ".meta.json").write_text(json.dumps(
                {"life": ind.life, "lives_lived": k + 1, "state_hash": ind.state_hash(), **git_rev()}) + "\n")
            print(f"saved {snap}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- evaluate
def evaluate(a) -> int:
    v = vision()
    seeds = parse_seeds(a.seeds)
    trained = build(v, learn=True)
    if a.untrained:
        lives_lived, life0, label = 0, 0, "untrained"
        out = Path(a.out)
    else:
        snap = Path(a.snapshot)
        meta = json.loads(Path(str(snap) + ".meta.json").read_text())
        trained.rt.restore(SnapshotStore(snap.parent.parent), snap)
        lives_lived, life0, label = int(meta["lives_lived"]), int(meta["life"]), snap.name
        out = Path(a.out) if a.out else Path(str(snap) + ".eval.json")
    saved = copy.deepcopy(trained.rt.store.state_dict())
    eps = []
    for i, seed in enumerate(seeds):
        test = build(v, learn=False)                       # the same individual, frozen: it does not learn here
        test.rt.store.load_state_dict(copy.deepcopy(saved))
        test.rt.tick = int(saved["tick"])
        test.life = life0 + 1 + i                          # a new world for it (its world-specific memory resets)
        e = live_one(test, seed, a.steps)
        e["options"] = test.read_state("skills")["ctl"]["life_options"]
        eps.append(e)
        print(json.dumps({k: e[k] for k in ("seed", "steps", "n_unlocked", "fulfilled", "failed", "options")}),
              flush=True)
    rep = {"format": "cb-w1-lifelong-eval/1", "status": "ENGINEERING_ONLY", "label": label,
           "lives_lived": lives_lived, "seeds": [seeds[0], seeds[-1], len(seeds)], "steps": a.steps, **git_rev(),
           "summary": summarize(eps), "episodes": eps}
    out.write_text(json.dumps(rep, indent=1) + "\n")
    s = rep["summary"]
    print(f"{label}: lives lived {lives_lived}, Crafter score {s['crafter_score']:.1f}, "
          f"achievements {s['achievements_per_episode']:.1f}, survival {s['survival_steps']:.0f}, "
          f"on time {s['obligations_on_time']['rate']}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--lives", type=int, default=150)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--save-every", type=int, default=25)
    t.add_argument("--out", default=str(REPO / "runs/explore/lifelong/run1"))
    t.add_argument("--resume", default=None, help="a snapshot directory written by train")
    e = sub.add_parser("evaluate")
    e.add_argument("--snapshot", default=None)
    e.add_argument("--untrained", action="store_true")
    e.add_argument("--seeds", default="71001-71020")
    e.add_argument("--steps", type=int, default=3000)
    e.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "evaluate" and not (a.untrained or a.snapshot):
        ap.error("evaluate needs --snapshot or --untrained")
    if a.cmd == "evaluate" and a.untrained and not a.out:
        ap.error("--untrained needs --out")
    return train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    raise SystemExit(main())
