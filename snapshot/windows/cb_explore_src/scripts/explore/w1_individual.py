#!/usr/bin/env python3
"""The W1 individual against the references, with one-faculty ablations (exploration lane, ENGINEERING_ONLY).

usage: python scripts/explore/w1_individual.py [--seeds 71001-71020] [--steps 3000] [--arms ...] [--out DIR]

Arms (same seeds, same W1 config, same step budget):
  random                      W1's uniform random baseline
  privileged_scripted         W1's privileged reference: the true map, objects, structured obligations, the
                              visibility log -- an upper reference of what the skill routines can do
  individual                  the individual: vision organ (pixels), its own memory, the language organ
                              (development parser stand-in on this CPU), the skill routines on its own beliefs
  individual[perception=oracle]   the same, but perceiving the current view's true tiles (isolates the vision
                              organ's errors; reads privileged state for the current view only)
  individual[memory=none]     no memory beyond the current view (what memory is worth)
  individual[language=none]   ignores the caretaker: no commitments, no answers (what language is worth)

Writes <out>/report.json (per arm: per-episode summaries and means) and prints a table.  Deterministic: the
same arguments give the same report (wall times aside).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # small matrices: threads only cost
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from continuous_brain_world.baselines.random_agent import RandomAgent  # noqa: E402
from continuous_brain_world.baselines.scripted import ScriptedAgent  # noqa: E402
from continuous_brain_world.env import W1Config, W1Env  # noqa: E402
from continuous_brain_organs.language import LanguageOrgan  # noqa: E402
from continuous_brain_organs.vision import VisionOrgan  # noqa: E402
from continuous_brain_individual.individual import Individual, IndividualConfig, first_info  # noqa: E402

ARMS = ("random", "privileged_scripted", "individual", "individual[perception=oracle]", "individual[memory=none]",
        "individual[language=none]")
WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def individual_config(arm: str) -> IndividualConfig:
    cfg = IndividualConfig()
    if "[" in arm:
        for kv in arm[arm.index("[") + 1:-1].split(","):
            k, v = kv.split("=")
            setattr(cfg, k, v)
    return cfg


LANGUAGE = {"backend": "dev"}


def run_episode(arm: str, seed: int, steps: int, vision) -> dict:
    env = W1Env(W1Config())
    obs = env.reset(seed)
    info = first_info(env)
    t = time.time()
    if arm.startswith("individual"):
        ind = Individual(individual_config(arm), vision, LanguageOrgan.from_config(dict(LANGUAGE)))
        life = ind.live(env, obs, info, steps)
        n = life["steps"]
        extra = {"memory_known_fraction": float((ind.read_state("episodic_memory")["map"] >= 0).mean()),
                 "days_narrated": len((ind.read_state("autobiography") or {}).get("days", [])),
                 "state_hash": ind.state_hash()}
    else:
        agent = RandomAgent(seed) if arm == "random" else ScriptedAgent(seed)
        agent.reset(obs, info)
        n, done = 0, False
        while n < steps and not done:
            out = agent.act(obs, info)
            action, utter = out if isinstance(out, tuple) else (out, None)
            obs, _, done, info = env.step(action, utter)
            n += 1
        extra = {}
    s = env.episode_summary()
    return {"seed": seed, "steps": n, "seconds": round(time.time() - t, 2), "summary": s, **extra}


def means(episodes: list[dict]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None
    fam = {}
    for f in ("collect", "craft", "place", "maintain", "avoid", "visit"):
        ful = sum(e["summary"]["obligations"]["per_family"][f]["fulfilled"] for e in episodes)
        fail = sum(e["summary"]["obligations"]["per_family"][f]["failed"] for e in episodes)
        fam[f] = {"fulfilled": ful, "failed": fail, "rate": ful / (ful + fail) if ful + fail else None}
    ful = sum(e["summary"]["obligations"]["fulfilled"] for e in episodes)
    fail = sum(e["summary"]["obligations"]["failed"] for e in episodes)
    asked = sum(e["summary"]["memory"]["asked"] for e in episodes)
    correct = sum(e["summary"]["memory"]["correct"] for e in episodes)
    unlocked = [set(e["summary"]["crafter"]["unlocked"]) for e in episodes]
    names = sorted(set().union(*unlocked)) if unlocked else []
    rates = {n: sum(n in u for u in unlocked) / len(unlocked) for n in names}
    # the official Crafter score over these episodes: success rate (%) per achievement (all 22)
    from continuous_brain_world.scoring import crafter_score, ACHIEVEMENTS
    succ = [100.0 * sum(a in u for u in unlocked) / len(unlocked) for a in ACHIEVEMENTS]
    return {"episodes": len(episodes), "crafter_score": crafter_score(succ),
            "achievements_per_episode": mean([e["summary"]["crafter"]["n_unlocked"] for e in episodes]),
            "obligations_on_time": {"fulfilled": ful, "failed": fail, "rate": ful / (ful + fail) if ful + fail else None},
            "per_family": fam, "memory": {"asked": asked, "correct": correct, "accuracy": correct / asked if asked else None},
            "survival_steps": mean([e["steps"] for e in episodes]),
            "deaths": sum(1 for e in episodes if e["summary"]["died"]),
            "achievement_success": rates, "seconds": sum(e["seconds"] for e in episodes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="71001-71020")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=str(REPO / "runs/explore/w1_individual"))
    ap.add_argument("--language", choices=("dev", "hf"), default="dev",
                    help="dev: the template stand-in (no weights); hf: an open-weight language model")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--local-dir", default=None, help="the model's directory (from scripts/explore/fetch_models.py)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None)
    a = ap.parse_args()
    global LANGUAGE
    if a.language == "hf":
        LANGUAGE = {"backend": "hf", "model_id": a.model, "local_dir": a.local_dir, "device": a.device,
                    **({"dtype": a.dtype} if a.dtype else {})}
    import torch
    torch.set_num_threads(1)
    seeds, arms = parse_seeds(a.seeds), a.arms.split(",")
    vision = VisionOrgan(WEIGHTS, CARD)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"format": "cb-w1-individual/1", "status": "ENGINEERING_ONLY", "seeds": [seeds[0], seeds[-1], len(seeds)],
              "steps": a.steps, "vision_weights_sha256": vision.weights_sha256,
              "language_backend": ("development_parser (a stand-in that knows W1's templates; not a language model)"
                                   if a.language == "dev" else LANGUAGE),
              "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True).stdout.strip(),
              "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                               text=True, capture_output=True).stdout.strip()),
              "arms": {}}
    for arm in arms:
        eps = []
        for s in seeds:
            e = run_episode(arm, s, a.steps, vision)
            eps.append(e)
            print(json.dumps({"arm": arm, "seed": s, "steps": e["steps"], "died": e["summary"]["died"],
                              "unlocked": e["summary"]["crafter"]["n_unlocked"],
                              "oblig": [e["summary"]["obligations"]["fulfilled"], e["summary"]["obligations"]["failed"]],
                              "mem": [e["summary"]["memory"]["correct"], e["summary"]["memory"]["asked"]],
                              "s": e["seconds"]}), flush=True)
        report["arms"][arm] = {"means": means(eps), "episodes": eps}
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"\n{'arm':32s} {'score':>6s} {'ach/ep':>6s} {'oblig':>12s} {'memory':>9s} {'survival':>8s} {'deaths':>6s}")
    for arm, r in report["arms"].items():
        m = r["means"]
        ob, me = m["obligations_on_time"], m["memory"]
        print(f"{arm:32s} {m['crafter_score']:6.1f} {m['achievements_per_episode']:6.1f} "
              f"{ob['fulfilled']:4d}/{ob['fulfilled'] + ob['failed']:<4d}{(ob['rate'] or 0):4.2f} "
              f"{me['correct']:3d}/{me['asked']:<3d}  {m['survival_steps']:8.0f} {m['deaths']:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
