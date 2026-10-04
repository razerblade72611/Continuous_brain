"""Run W1's two reference baselines on a fixed seed set and write the report.

    /home/claude/venvs/explore/bin/python scripts/explore/w1_baselines.py

Agents (both are references, not learners):
  * uniform_random       -- uniform over the 17 Crafter actions, never speaks;
  * privileged_scripted  -- reads privileged ground truth from ``info`` (see continuous_brain_world/baselines).

Seeds: the exploration lane's engineering family ``w1-baselines-eng`` = 71001 .. 71000+N (default N = 20).  The
scripted agent was tuned on seeds 1-12 only.  Episodes are 3000 steps (10 W1 days).  Everything runs in one
process, one episode at a time.

Writes the full report (every episode summary) to runs/explore/w1_baselines/report.json (runs/ is gitignored)
and the summary to evidence/explore/w1/baselines.json (committed).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from continuous_brain_world import W1Config, W1Env, aggregate  # noqa: E402
from continuous_brain_world.baselines import RandomAgent, ScriptedAgent  # noqa: E402
from continuous_brain_world.caretaker import language_inventory  # noqa: E402
from continuous_brain_world.crafter_bridge import crafter_provenance  # noqa: E402

SEED_FAMILY = "w1-baselines-eng"
SEED_BASE = 71000


def git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def run_episode(env: W1Env, agent, seed: int) -> dict:
    obs = env.reset(seed)
    agent.reset(obs)
    info = env.privileged_now()
    t_agent = 0.0
    env.timing = {"crafter_s": 0.0, "layer_s": 0.0, "steps": 0}
    t0 = time.perf_counter()
    done = False
    while not done:
        t = time.perf_counter()
        action, utter = agent.act(obs, info)
        t_agent += time.perf_counter() - t
        obs, reward, done, info = env.step(action, utter)
    wall = time.perf_counter() - t0
    s = env.episode_summary()
    n = max(1, env.timing["steps"])
    s["seed"] = seed
    s["timing_ms_per_step"] = {"crafter": 1e3 * env.timing["crafter_s"] / n, "layer": 1e3 * env.timing["layer_s"] / n,
                               "agent": 1e3 * t_agent / n, "wall": 1e3 * wall / n}
    return s


def compact(ag: dict) -> dict:
    fam = {f: {k: v[k] for k in ("issued", "fulfilled", "failed", "cancelled", "unresolved", "amended",
                                 "on_time_rate", "value_weighted_rate", "fail_reasons")}
           for f, v in ag["obligations"]["per_family"].items()}
    return {
        "episodes": ag["episodes"],
        "crafter_score": ag["crafter"]["score"],
        "crafter_mean_achievements": ag["crafter"]["mean_unlocked"],
        "crafter_success_rates": ag["crafter"]["success_rates"],
        "obligations_on_time_rate": ag["obligations"]["on_time_rate"],
        "obligations_value_weighted_rate": ag["obligations"]["value_weighted_rate"],
        "obligations_fulfilled": ag["obligations"]["fulfilled"], "obligations_failed": ag["obligations"]["failed"],
        "per_family": fam,
        "memory_accuracy": ag["memory"]["accuracy"], "memory_asked": ag["memory"]["asked"],
        "memory_per_question": ag["memory"]["per_question"],
        "survival_mean_steps": ag["survival"]["mean_steps"], "survival_mean_fraction": ag["survival"]["mean_fraction"],
        "deaths": ag["survival"]["deaths"],
        "w1_index": ag["w1_index"], "w1_index_components": ag["index_components"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--length", type=int, default=3000)
    ap.add_argument("--out", default=str(REPO / "runs/explore/w1_baselines/report.json"))
    ap.add_argument("--evidence", default=str(REPO / "evidence/explore/w1/baselines.json"))
    args = ap.parse_args(argv)

    cfg = W1Config(episode_length=args.length)
    env = W1Env(cfg)
    seeds = [SEED_BASE + i for i in range(1, args.seeds + 1)]
    agents = {"uniform_random": lambda s: RandomAgent(seed=s), "privileged_scripted": lambda s: ScriptedAgent(seed=s)}
    started = time.time()
    episodes, summary = {}, {}
    for name, make in agents.items():
        eps = []
        for seed in seeds:
            eps.append(run_episode(env, make(seed), seed))
            print(f"{name:20s} seed {seed} steps {eps[-1]['steps']:5d} died {eps[-1]['died']!s:5s} "
                  f"achievements {eps[-1]['crafter']['n_unlocked']:2d} "
                  f"obligations {eps[-1]['obligations']['fulfilled']}/{eps[-1]['obligations']['fulfilled'] + eps[-1]['obligations']['failed']} "
                  f"memory {eps[-1]['memory']['correct']}/{eps[-1]['memory']['asked']}", flush=True)
        ag = aggregate(eps)
        steps = sum(e["steps"] for e in eps)
        timing = {k: sum(e["timing_ms_per_step"][k] * e["steps"] for e in eps) / steps
                  for k in ("crafter", "layer", "agent", "wall")}
        episodes[name] = eps
        summary[name] = dict(compact(ag), timing_ms_per_step=timing, total_steps=steps)
    meta = {
        "format": "w1-baselines/1",
        "what": "W1 reference baselines (engineering result, exploration lane; not preregistered)",
        "agents": {"uniform_random": "uniform over 17 actions, no utterances",
                   "privileged_scripted": "PRIVILEGED heuristic reading info['privileged'] ground truth; an upper "
                                          "reference for what the world permits, not a learner; tuned on seeds 1-12"},
        "seed_family": SEED_FAMILY, "seeds": seeds, "episode_length": args.length, "config": cfg.to_dict(),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "crafter": crafter_provenance(), "python": platform.python_version(), "numpy": np.__version__,
        "machine": platform.machine(), "language_inventory": language_inventory(),
        "wall_seconds": round(time.time() - started, 1),
        "index_note": "w1_index = 100 * mean(crafter_score/100, value-weighted on-time rate, memory accuracy, "
                      "survival fraction); equal weights, untuned. Read the components, not the index.",
    }
    report = dict(meta, summary=summary, episodes=episodes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    ev = Path(args.evidence)
    ev.parent.mkdir(parents=True, exist_ok=True)
    per_episode = {name: [{k: e[k] for k in ("seed", "steps", "died")} | {
        "achievements": e["crafter"]["n_unlocked"], "obligations_fulfilled": e["obligations"]["fulfilled"],
        "obligations_failed": e["obligations"]["failed"], "memory_correct": e["memory"]["correct"],
        "memory_asked": e["memory"]["asked"]} for e in eps] for name, eps in episodes.items()}
    ev.write_text(json.dumps(dict(meta, summary=summary, per_episode=per_episode), indent=1, sort_keys=True) + "\n")
    print(json.dumps({n: {k: s[k] for k in ("crafter_score", "obligations_on_time_rate", "memory_accuracy",
                                           "survival_mean_steps", "w1_index", "timing_ms_per_step")}
                      for n, s in summary.items()}, indent=1))
    print(f"wrote {out} and {ev} in {meta['wall_seconds']} s")


if __name__ == "__main__":
    main()
