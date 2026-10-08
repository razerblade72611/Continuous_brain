#!/usr/bin/env python3
"""r3-H arm A (exploration lane, ENGINEERING_ONLY, CPU): routine intent -> scripted executor organ.

Can the organ's atomic options (organ.COMMANDS) reproduce the routine when told, every step, which option the
routine's current intent maps to (organ.BRANCH_TO_OPTION)?  If not, a learned selector over those options cannot
reach the routine either, whatever it learns.  Each development world is lived by a routine individual whose own
action is replaced by the organ's; its beliefs, speech and ledger are its own.  Every delegated action is logged.

  python scripts/explore/cb2_organ_check.py --seeds 75001-75040 --workers 2 --reference evidence/.../eval-r3t-full.json
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))
from cb2_collect import git_rev  # noqa: E402
from cb2_evaluate import parse_seeds  # noqa: E402


def _vision_or_none():
    from cb2_collect import CARD, WEIGHTS
    if WEIGHTS.exists():
        from continuous_brain_organs.vision import VisionOrgan
        return VisionOrgan(WEIGHTS, CARD), "organ"
    return None, "oracle"


def _life(args):
    seed, steps, log_dir = args
    import torch
    torch.set_num_threads(1)
    from continuous_brain_cb2.intent import BRANCHES, IntentShadow
    from continuous_brain_cb2.organ import ExecutorOrgan, command_for
    from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
    from continuous_brain_individual.skills import ACTIONS
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_world.env import W1Config, W1Env
    v, perception = _vision_or_none()
    ind = Individual(IndividualConfig(perception=perception), v, LanguageOrgan.from_config({"backend": "dev"}))
    env = W1Env(W1Config())
    obs = env.reset(seed)
    info = first_info(env)
    shadow, organ = IntentShadow(0), ExecutorOrgan(0)
    command, n, done = "explore", 0, False
    status, commands, same = Counter(), Counter(), 0
    log = open(Path(log_dir) / f"organ-{seed}.jsonl", "w") if log_dir else None
    t = time.time()
    while n < steps and not done:
        routine_action, utter = ind.decide(obs, info)          # its beliefs, speech and ledger tick as usual
        beliefs = IntentShadow.read(ind)
        _, _, vec = shadow(*beliefs)
        branch = BRANCHES[int(np.argmax(vec[:len(BRANCHES)]))]
        command = command_for(branch, shadow.leaving, command)
        name, tel = organ(command, *beliefs)
        action = ACTIONS[name]
        same += int(action == routine_action)
        status[tel["status"]] += 1
        commands[command] += 1
        if log:
            log.write(json.dumps({"step": n, "branch": branch, **tel, "routine_action": int(routine_action)}) + "\n")
        obs, _, done, info = env.step(action, utter)
        n += 1
    if log:
        log.close()
    s = env.episode_summary()
    return {"seed": seed, "steps": n, "died": bool(done) and n < steps, "unlocked": sorted(s["crafter"]["unlocked"]),
            "n_unlocked": s["crafter"]["n_unlocked"], "fulfilled": s["obligations"]["fulfilled"],
            "failed": s["obligations"]["failed"], "agreement_with_routine": same / max(n, 1),
            "status": dict(status), "commands": dict(commands), "seconds": round(time.time() - t, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="75001-75040")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reference", default=None, help="an eval-*.json whose routine arm lived the same worlds")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--out", default=str(REPO / "evidence/explore/cb2/r3h-armA.json"))
    a = ap.parse_args()
    from continuous_brain_world.scoring import ACHIEVEMENTS, crafter_score
    seeds = parse_seeds(a.seeds)
    if a.log_dir:
        Path(a.log_dir).mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(a.workers) as p:
        res = []
        for r in p.imap(_life, [(s, a.steps, a.log_dir) for s in seeds]):
            res.append(r)
            print(json.dumps({k: r[k] for k in ("seed", "steps", "n_unlocked", "agreement_with_routine", "status")}),
                  flush=True)

    def summ(eps):
        unl = [set(e["unlocked"]) for e in eps]
        succ = [100.0 * sum(x in u for u in unl) / len(unl) for x in ACHIEVEMENTS]
        ful, fail = sum(e["fulfilled"] for e in eps), sum(e["failed"] for e in eps)
        return {"crafter_score": crafter_score(succ), "survival_steps": float(np.mean([e["steps"] for e in eps])),
                "achievements_per_episode": float(np.mean([e["n_unlocked"] for e in eps])),
                "obligations": ful / (ful + fail) if ful + fail else None, "deaths": sum(e["died"] for e in eps)}
    out = {"format": "cb2-r3h-armA/1", "status": "ENGINEERING_ONLY", **git_rev(), "seeds": [seeds[0], seeds[-1], len(seeds)],
           "organ": summ(res), "agreement_with_routine": float(np.mean([r["agreement_with_routine"] for r in res])),
           "status_counts": dict(sum((Counter(r["status"]) for r in res), Counter())),
           "command_counts": dict(sum((Counter(r["commands"]) for r in res), Counter())), "episodes": res}
    if a.reference:
        ref = json.load(open(a.reference))
        rt = [e for e in ref["episodes"]["routine"] if e["seed"] in set(seeds)]
        out["routine"] = summ(rt)
        out["ratio_to_routine"] = {k: round(out["organ"][k] / out["routine"][k], 3)
                                   for k in ("crafter_score", "survival_steps", "achievements_per_episode")}
        d = [r["steps"] - e["steps"] for r, e in zip(sorted(res, key=lambda r: r["seed"]), sorted(rt, key=lambda e: e["seed"]))]
        out["paired_steps"] = {"mean": float(np.mean(d)), "se": float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
