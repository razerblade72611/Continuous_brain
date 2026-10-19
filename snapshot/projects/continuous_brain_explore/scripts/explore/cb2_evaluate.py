#!/usr/bin/env python3
"""CB2-S1: consolidate what sleep learned into the individual, and test it on fresh worlds (exploration lane,
ENGINEERING_ONLY).

  install    the one recorded event that gives the individual its learned skill (core + policy):
             python scripts/explore/cb2_evaluate.py install --core <core.pt> --policy <policy.pt>
  mark-stable  mark the skill it has now as its last stable one
  rollback     return its skill to the last stable one (a new recorded consolidation; its memories and lives kept)
  evaluate   copies of the individual as it is now, each living one fresh world (75001-75040 by default) from the
             same saved state; the routine individual lives the same worlds:
             python scripts/explore/cb2_evaluate.py evaluate --label pa-round1

Each evaluation life is archived as PRACTICE x SELF (stream ``eval-<label>``, individual ``copy-of-<id>``), with
the routine's COUNTERFACTUAL action at every step: the data for the next imitation round (DAgger).  They are not
the individual's own lives; its chronology continues only through ``cb2_collect.py wake``.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))

from cb2_collect import ARCHIVE, INDIVIDUALS, git_rev, provenance, snapshot_path, vision  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def organ_vision(home: Path, meta: dict, v):
    """The vision organ this individual lives with: the default one, unless an organ swap (Gate 7,
    cb2_organ_swap.py) gave it another, kept by hash in its home (organs/<sha256>.pt and .card.json)."""
    o = (meta.get("organs") or {}).get("vision")
    if not o or (v is not None and o["weights_sha256"] == getattr(v, "weights_sha256", None)):
        return v
    from continuous_brain_organs.vision import VisionOrgan
    d = home / "organs"
    return VisionOrgan(d / f"{o['weights_sha256']}.pt", d / f"{o['weights_sha256']}.card.json")


def load_individual(home: Path, v):
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_snn.snapshot import SnapshotStore
    meta = json.loads((home / "individual.json").read_text())
    v = organ_vision(home, meta, v)
    ind = Individual(IndividualConfig(lifelong=True, skills=meta.get("skills", "routine")), v,
                     LanguageOrgan.from_config({"backend": "dev"}))
    ind.rt.restore(SnapshotStore(home), snapshot_path(home, meta))
    ind.life = int(meta["life"])
    if ind.state_hash() != meta["state_hash"]:
        raise SystemExit("the saved individual does not restore to its recorded state")
    return ind, meta


def install_cmd(a) -> int:
    import torch
    from continuous_brain_cb2.consolidation import ConsolidationLog, install
    from continuous_brain_cb2.policy import SkillPolicy
    from continuous_brain_cb2.skill_host import pack_weights
    from continuous_brain_cb2.world_model import PredictiveCore
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_snn.snapshot import SnapshotStore
    v = vision()
    home = Path(a.individuals) / a.individual
    src, meta = load_individual(home, v)
    core = PredictiveCore.from_state(torch.load(a.core, map_location="cpu", weights_only=False))
    pol = SkillPolicy.from_state(torch.load(a.policy, map_location="cpu", weights_only=False))
    if pol.feat_size != core.feat_size:
        raise SystemExit("the policy was trained on a different core")
    lineage = {"core": {"path": str(a.core), "sha256": _sha(a.core),
                        "run": json.loads((Path(a.core).parent / "run.json").read_text())},
               "policy": {"path": str(a.policy), "sha256": _sha(a.policy),
                          "run": json.loads((Path(a.policy).parent / "run.json").read_text())},
               "code": git_rev(), "note": a.note}
    for part in ("core", "policy"):
        lineage[part]["run"].pop("train_shards", None)          # kept in the run.json the path names
        lineage[part]["run"].pop("heldout_shards", None)
    return _consolidate(home, src, meta, v, pack_weights(core, pol), lineage, candidate=a.candidate)


def _consolidate(home: Path, src, meta: dict, v, weights: bytes, lineage: dict, candidate: bool = False) -> int:
    """One consolidation event: install, log it, keep the weights (write-once, by hash), save the individual."""
    from continuous_brain_cb2.consolidation import ConsolidationLog, install
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_snn.snapshot import SnapshotStore
    if meta.get("in_life") and not candidate:
        raise SystemExit("it is in the middle of a life (stopped in the viewer): finish that life first")
    dst = Individual(IndividualConfig(lifelong=True, skills="cb2"), src.vision,   # its own organ (Gate 7)
                     LanguageOrgan.from_config({"backend": "dev"}))
    event = install(src, dst, weights, lineage)
    wdir = home / "weights"
    wdir.mkdir(exist_ok=True)
    wfile = wdir / f"{event['weights_sha256']}.pt"
    if not wfile.exists():
        wfile.write_bytes(weights)
    rec = ConsolidationLog(home / "consolidations.jsonl").append(event)
    snap = dst.rt.save(SnapshotStore(home), label=f"consolidation{rec['seq']}")
    new = {**meta, "snapshot": snap.name, "skills": "cb2", "state_hash": dst.state_hash(), "life": dst.life,
           "consolidations": rec["seq"] + 1, "parent_snapshot": Path(meta["snapshot"]).name,
           "weights_sha256": event["weights_sha256"], "event_sha": event["event_sha"], **git_rev()}
    Path(str(snap) + ".meta.json").write_text(json.dumps(new) + "\n")
    (home / "individual.json").write_text(json.dumps(new, indent=1) + "\n")
    print(json.dumps({"event": event["kind"], "event_sha": event["event_sha"], "parent": event["parent_state_hash"],
                      "state_hash": event["state_hash"], "weights_sha256": event["weights_sha256"],
                      "snapshot": str(snap)}, indent=1))
    return 0


def mark_stable_cmd(a) -> int:
    """Mark the skill it has now as its last stable one: what a rollback returns it to."""
    home = Path(a.individuals) / a.individual
    meta = json.loads((home / "individual.json").read_text())
    if not meta.get("weights_sha256"):
        raise SystemExit("it has no learned skill installed yet")
    meta["stable"] = {"weights_sha256": meta["weights_sha256"], "event_sha": meta.get("event_sha"),
                      "snapshot": meta["snapshot"], "note": a.note,
                      "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (home / "individual.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta["stable"], indent=1))
    return 0


def rollback_cmd(a) -> int:
    """Return its skill to the last stable one, as a new recorded consolidation.  Its memories, diary and lives
    are kept: only the learned faculty goes back."""
    v = vision()
    home = Path(a.individuals) / a.individual
    src, meta = load_individual(home, v)
    st = meta.get("stable")
    if not st:
        raise SystemExit("no stable skill marked (cb2_evaluate.py mark-stable)")
    weights = (home / "weights" / f"{st['weights_sha256']}.pt").read_bytes()
    lineage = {"rollback": True, "to_event_sha": st["event_sha"], "to_weights_sha256": st["weights_sha256"],
               "from_weights_sha256": meta.get("weights_sha256"), "code": git_rev(), "note": a.note}
    return _consolidate(home, src, meta, v, weights, lineage)


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def evaluate_cmd(a) -> int:
    from continuous_brain_cb2.archive import StreamWriter
    from continuous_brain_cb2.record import record_life
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_world.env import W1Config, W1Env
    from continuous_brain_world.scoring import ACHIEVEMENTS, crafter_score
    v = vision()
    seeds = parse_seeds(a.seeds)
    home = Path(a.individuals) / a.individual
    ind, meta = load_individual(home, v)
    saved = copy.deepcopy(ind.save())
    v_ind = ind.vision                                  # its own organ (it may have been swapped, Gate 7)

    def summarize(eps):
        unl = [set(e["unlocked"]) for e in eps]
        succ = [100.0 * sum(x in u for u in unl) / len(unl) for x in ACHIEVEMENTS]
        ful, fail = sum(e["fulfilled"] for e in eps), sum(e["failed"] for e in eps)
        return {"episodes": len(eps), "crafter_score": crafter_score(succ),
                "achievements_per_episode": float(np.mean([e["n_unlocked"] for e in eps])),
                "survival_steps": float(np.mean([e["steps"] for e in eps])), "deaths": sum(e["died"] for e in eps),
                "obligations_on_time": {"fulfilled": ful, "failed": fail,
                                        "rate": ful / (ful + fail) if ful + fail else None}}

    def one(actor, seed, writer, who):
        env = W1Env(W1Config())
        t = time.time()
        on_step, tlog = None, None
        if getattr(a, "telemetry", None) and who == "SELF":   # r3-H: every delegated action, with provenance
            from continuous_brain_individual import codec
            Path(a.telemetry).mkdir(parents=True, exist_ok=True)
            tlog = open(Path(a.telemetry) / f"delegation-{seed}.jsonl", "w")

            def on_step(n, action, _u, _o, _r, _i):
                st = codec.unblob(actor.rt.store.read("skills.state")) or {}
                if "delegation" in st:
                    tlog.write(json.dumps({"n": n, "taken": int(action), **st["delegation"]}) + "\n")
        rec = record_life(actor, env, seed, writer, steps=a.steps, who=who, lived=False, on_step=on_step)
        if tlog:
            tlog.close()
        s = env.episode_summary()
        return {"seed": seed, "steps": rec["steps"], "died": rec["done"] and rec["steps"] < a.steps,
                "unlocked": sorted(s["crafter"]["unlocked"]), "n_unlocked": s["crafter"]["n_unlocked"],
                "fulfilled": s["obligations"]["fulfilled"], "failed": s["obligations"]["failed"],
                "seconds": round(time.time() - t, 1)}

    arms = {"individual": [], "routine": []}
    w_ind = StreamWriter(a.archive, f"eval-{a.label}", "PRACTICE", f"copy-of-{a.individual}",
                         {**provenance(v, "evaluate"), "individual_state_hash": meta["state_hash"],
                          "skills": meta.get("skills", "routine")})
    w_rt = StreamWriter(a.archive, f"eval-{a.label}-routine", "PRACTICE", "teacher-routine", provenance(v, "evaluate"))
    for i, seed in enumerate(seeds):
        test = Individual(IndividualConfig(lifelong=True, skills=meta.get("skills", "routine")), v_ind,
                          LanguageOrgan.from_config({"backend": "dev"}))
        test.restore(copy.deepcopy(saved))
        test.life = int(meta["life"]) + 1 + i             # a fresh world, and a new life number, for this copy
        e = one(test, seed, w_ind, "SELF")
        w_ind.close()
        r = one(Individual(IndividualConfig(), v, LanguageOrgan.from_config({"backend": "dev"})), seed, w_rt, "TEACHER")
        w_rt.close()
        arms["individual"].append(e)
        arms["routine"].append(r)
        print(json.dumps({"seed": seed, "it": [e["steps"], e["n_unlocked"]], "routine": [r["steps"], r["n_unlocked"]]}),
              flush=True)
    diff = {k: [x[k] - y[k] for x, y in zip(arms["individual"], arms["routine"])] for k in ("steps", "n_unlocked")}
    paired = {k: {"mean": float(np.mean(v_)), "se": float(np.std(v_, ddof=1) / np.sqrt(len(v_)))} for k, v_ in diff.items()}
    rep = {"format": "cb2-eval/1", "status": "ENGINEERING_ONLY", "label": a.label, "individual": a.individual,
           "individual_state_hash": meta["state_hash"], "skills": meta.get("skills", "routine"),
           "consolidations": meta.get("consolidations", 0), "seeds": [seeds[0], seeds[-1], len(seeds)],
           "steps": a.steps, **git_rev(), "summary": {k: summarize(v_) for k, v_ in arms.items()},
           "paired_difference": paired, "episodes": arms}
    out = Path(a.out) if a.out else REPO / f"evidence/explore/cb2/eval-{a.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"summary": rep["summary"], "paired_difference": paired}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install")
    i.add_argument("--core", required=True)
    i.add_argument("--policy", required=True)
    i.add_argument("--note", default="")
    i.add_argument("--candidate", action="store_true",
                   help="a candidate copy (evaluation only): it may have been saved mid-life")
    e = sub.add_parser("evaluate")
    e.add_argument("--label", required=True)
    e.add_argument("--seeds", default="75001-75040")
    e.add_argument("--steps", type=int, default=10000)
    e.add_argument("--out", default=None)
    e.add_argument("--telemetry", default=None, help="directory for per-life delegation logs (r3-H organ)")
    for p in (i, e):
        p.add_argument("--individual", default="cb2-ind-0001")
        p.add_argument("--individuals", default=str(INDIVIDUALS))
        p.add_argument("--archive", default=str(ARCHIVE))
    m = sub.add_parser("mark-stable")
    m.add_argument("--note", required=True)
    r = sub.add_parser("rollback")
    r.add_argument("--note", default="")
    for p in (m, r):
        p.add_argument("--individual", default="cb2-ind-0001")
        p.add_argument("--individuals", default=str(INDIVIDUALS))
    a = ap.parse_args()
    return {"install": install_cmd, "evaluate": evaluate_cmd, "mark-stable": mark_stable_cmd,
            "rollback": rollback_cmd}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
