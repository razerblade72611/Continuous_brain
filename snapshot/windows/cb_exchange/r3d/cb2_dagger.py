"""DAgger for the CB2 skill: on-policy corrective labels, with the predictive core frozen (exploration lane,
ENGINEERING_ONLY).

Why: in r2 the imitation skill agreed with the routine 0.79 of the time on held-out lived states but only 0.42 in
its own lives (0.28 by steps 50-99), while the routine's counterfactual labels are fully self-consistent (teacher
consistency 1.0 on 20,027 steps).  That is compounding error: the student visits states the routine never showed it.

  collect  A candidate (a copy of the individual with a learned skill installed) lives fresh worlds.  Every step is
           archived as PRACTICE x SELF with the routine's COUNTERFACTUAL action (ShadowRoutine): the states its own
           policy visits, labelled with what the routine would have done.  Worker k writes stream <stream>-wKK into
           its own archive root (--archive), never into the individual's canonical archive.
  train    Continues a policy (warm start) on a fixed mixture: --frac of every batch from the DAgger streams
           (--dagger-archive, --dagger-prefix), the rest from the base data (LIVED + PRACTICE of --archive, minus
           any stream starting with --exclude).  Plain size-proportional sampling would leave a few thousand
           corrective rows under 1% of 1.36M; the fixed fraction is the point.  --frac 0 is the no-DAgger control
           (same warm start, same updates, base data only).

Seeds: DAgger collection uses 89,000,000 + iteration*100,000 + i (practice run 9, never used); 75001-75040 are
development evaluation only; 90001-90200 stay sealed for a preregistered acceptance test.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))
from cb2_collect import ARCHIVE, INDIVIDUALS, git_rev, provenance, vision  # noqa: E402
from cb2_evaluate import load_individual, parse_seeds  # noqa: E402
import cb2_sleep as sl  # noqa: E402

DAGGER_ARCHIVE = REPO / "runs/explore/cb2/dagger_archive"


# ---------------------------------------------------------------------------------------------------- collect
def _collect_worker(args):
    k, items, a = args
    import torch
    torch.set_num_threads(1)
    from continuous_brain_cb2.archive import StreamWriter
    from continuous_brain_cb2.record import record_life
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_world.env import W1Config, W1Env
    v = vision()
    ind, meta = load_individual(Path(a.individuals) / a.individual, v)
    if meta.get("skills") != "cb2":
        raise SystemExit("collect needs a candidate with a learned skill installed (cb2_evaluate.py install --candidate)")
    saved = copy.deepcopy(ind.save())
    w = StreamWriter(a.archive, f"{a.stream}-w{k:02d}", "PRACTICE", f"copy-of-{a.individual}",
                     {**provenance(v, "dagger-collect"), "individual_state_hash": meta["state_hash"],
                      "weights_sha256": meta.get("weights_sha256"), "skills": "cb2"})
    out = []
    for idx, seed in items:
        test = Individual(IndividualConfig(lifelong=True, skills="cb2"), v, LanguageOrgan.from_config({"backend": "dev"}))
        test.restore(copy.deepcopy(saved))
        test.life = int(meta["life"]) + 1 + idx               # a fresh world and a new life number for this copy
        env = W1Env(W1Config())
        rec = record_life(test, env, seed, w, steps=a.steps, who="SELF", lived=False)
        w.close()                                             # a life ends a shard
        s = env.episode_summary()
        out.append({"seed": seed, "steps": rec["steps"], "n_unlocked": s["crafter"]["n_unlocked"]})
        print(json.dumps({"worker": k, **out[-1]}), flush=True)
    return out


def collect(a) -> int:
    seeds = parse_seeds(a.seeds)
    n = min(a.workers, len(seeds))
    chunks = [(k, [(i, s) for i, s in enumerate(seeds)][k::n], a) for k in range(n)]
    t = time.time()
    with get_context("fork").Pool(n) as p:
        res = sorted((r for part in p.map(_collect_worker, chunks) for r in part), key=lambda r: r["seed"])
    rep = {"format": "cb2-dagger-collect/1", "status": "ENGINEERING_ONLY", "stream": a.stream, "archive": str(a.archive),
           "individuals": str(a.individuals), "seeds": [seeds[0], seeds[-1], len(seeds)], "workers": n,
           "lives": len(res), "steps": int(sum(r["steps"] for r in res)),
           "mean_steps": float(np.mean([r["steps"] for r in res])),
           "mean_unlocked": float(np.mean([r["n_unlocked"] for r in res])),
           "seconds": round(time.time() - t, 1), **git_rev(), "lives_detail": res}
    out = Path(a.out) if a.out else REPO / f"evidence/explore/cb2/dagger-collect-{a.stream}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "lives_detail"}), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------ train
def train(a) -> int:
    import torch
    from continuous_brain_cb2.archive import Archive
    from continuous_brain_cb2.data import SequenceSampler
    from continuous_brain_cb2.policy import SkillPolicy
    from continuous_brain_cb2.world_model import PredictiveCore
    torch.manual_seed(a.seed)
    dev = sl.device(a.device)
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    out = sl.new_out(a.out)
    core = PredictiveCore.from_state(torch.load(a.core, map_location=dev, weights_only=False), dev)
    core.eval()
    pol = SkillPolicy.from_state(torch.load(a.init_policy, map_location=dev, weights_only=False), dev)
    if pol.cfg.arm != a.arm or pol.feat_size != core.feat_size:
        raise SystemExit(f"the initial policy ({pol.cfg.arm}, feat {pol.feat_size}) does not match --arm {a.arm} / the core")
    arch, base_all, test, held = sl.split(a.archive, a.heldout_lives)
    excl = tuple(x for x in a.exclude.split(",") if x)
    base = [r for r in base_all if not (excl and r.stream.startswith(excl)) and not r.stream.startswith(a.dagger_prefix)]
    dag = []
    if a.frac > 0:
        darch = Archive(a.dagger_archive)
        dag = [r for r in darch.shards(("PRACTICE",)) if r.stream.startswith(a.dagger_prefix)]
        if not dag:
            raise SystemExit(f"--frac {a.frac} but no stream starting with {a.dagger_prefix!r} in {a.dagger_archive}")
    ns = ("student", "teacher", "organ")
    pa = a.arm == "P-A"
    tr = SequenceSampler(arch, base, T=a.T, seed=a.seed, namespaces=ns, map_crop=pa, preload=a.preload)
    te = SequenceSampler(arch, test, T=a.T, seed=a.seed + 1, namespaces=ns, map_crop=pa, preload=a.preload)
    dg = SequenceSampler(darch, dag, T=a.T, seed=a.seed + 2, namespaces=ns, map_crop=pa, preload=a.preload) if dag else None
    nd = int(round(a.frac * a.batch)) if dg is not None else 0
    run = {"format": "cb2-dagger-policy/1", "status": "ENGINEERING_ONLY", **git_rev(), "device": dev, "arm": a.arm,
           "core": str(a.core), "core_sha256": sl.file_sha(Path(a.core)),
           "init_policy": str(a.init_policy), "init_policy_sha256": sl.file_sha(Path(a.init_policy)),
           "base_rows": tr.rows(), "base_streams": len(tr.streams), "excluded_prefixes": list(excl),
           "dagger_archive": str(a.dagger_archive) if dg else None, "dagger_streams": sorted({r.stream for r in dag}),
           "dagger_rows": dg.rows() if dg else 0, "per_batch": {"base": a.batch - nd, "dagger": nd},
           "heldout_lives": held, "args": {k: str(v) for k, v in vars(a).items() if k != "func"}}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(json.dumps({k: run[k] for k in ("base_rows", "dagger_rows", "per_batch", "dagger_streams")}), flush=True)
    t0 = time.time()
    for u in range(1, a.updates + 1):
        b = tr.batch(a.batch - nd)
        if nd:
            d = dg.batch(nd)
            b = {k: np.concatenate([b[k], d[k]]) for k in b.keys() & d.keys()}
        m = pol.train_step(sl._feats(core, b), b)
        if u % a.log_every == 0 or u == a.updates:
            rec = {"update": u, "s": round(time.time() - t0, 1), "train": m}
            with torch.no_grad():
                ag = [pol.losses(sl._feats(core, eb), eb)[1]["agreement"] for eb in (te.batch(a.batch) for _ in range(a.eval_batches))]
                rec["heldout_lived_agreement"] = round(float(np.mean(ag)), 4)
                if dg is not None:
                    ag = [pol.losses(sl._feats(core, eb), eb)[1]["agreement"] for eb in (dg.batch(a.batch) for _ in range(a.eval_batches))]
                    rec["onpolicy_agreement"] = round(float(np.mean(ag)), 4)   # on DAgger states (also trained on)
            sl.log(out, rec)
    torch.save(pol.state_dict_full(), out / "policy.pt")
    (out / "policy.pt.sha256").write_text(sl.file_sha(out / "policy.pt") + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--individuals", required=True, help="the candidate copy's directory (never the canonical one)")
    c.add_argument("--individual", default="cb2-ind-0001")
    c.add_argument("--stream", required=True)
    c.add_argument("--seeds", required=True)
    c.add_argument("--steps", type=int, default=10000)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--archive", default=str(DAGGER_ARCHIVE))
    c.add_argument("--out", default=None)
    c.set_defaults(func=collect)
    t = sub.add_parser("train")
    t.add_argument("--core", required=True)
    t.add_argument("--init-policy", required=True)
    t.add_argument("--arm", default="P-A")
    t.add_argument("--out", required=True)
    t.add_argument("--archive", default=str(ARCHIVE))
    t.add_argument("--dagger-archive", default=str(DAGGER_ARCHIVE))
    t.add_argument("--dagger-prefix", default="dagger-r3d-")
    t.add_argument("--exclude", default="eval-", help="comma-separated stream prefixes left out of the base data")
    t.add_argument("--frac", type=float, default=0.5)
    t.add_argument("--updates", type=int, default=4000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--T", type=int, default=64)
    t.add_argument("--heldout-lives", type=int, default=3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--preload", action="store_true")
    t.add_argument("--log-every", type=int, default=250)
    t.add_argument("--eval-batches", type=int, default=4)
    t.set_defaults(func=train)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
