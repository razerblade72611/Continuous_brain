#!/usr/bin/env python3
"""CB2 r3-T: the target-given diagnostic (exploration lane, ENGINEERING_ONLY, DIAGNOSTIC: privileged input).

Question (Sol, DAV-88/89, 2026-10-08): is the learned skill's closed-loop collapse (r2 0.45x the routine's
survival; DAgger did not help, r3-D) caused by not knowing what the routine is trying to do (its hidden intent),
or by not being able to do it?  Three policies, all warm-started from r2's P-A policy on the frozen r2 core, all
trained on the same batches, differ only in what part of the routine's intent (``continuous_brain_cb2/intent.py``)
they are given as extra input:

  none     nothing (the new input columns stay zero: r2's architecture, trained on).  The control.
  branch   which routine produced the action (drink, eat, shelter, obligations, tech tree, explore, ...).
  full     that, and where it was walking (goal offset and path cost).

  practice  routine teachers in fresh worlds (practice run --run, default 4: seeds 84,000,000 + k*100,000 + i),
            archived as PRACTICE x TEACHER into --archive (never the canonical archive), with the routine's intent
            at every row in a sidecar (--sidecar/<stream>.npy, float16 [rows, INTENT_DIM]).  The intent comes from
            an IntentShadow (the routine library on the teacher's own beliefs, seed 0, as the routine host runs
            it); every step its action is checked against the routine's actual action.
  train     the three arms on the same batches; held out: the last worker's stream.
  summary   reads the three development evaluations (75001-75040) and the eval streams' closed-loop agreement.

In the closed loop the candidate's host computes the intent itself, on its own beliefs (skill_host.py), so a
policy that uses it is told, every step, what the routine would be trying to do from where it actually is.
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
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))
from cb2_collect import PRACTICE_BASE, git_rev, provenance, vision  # noqa: E402
import cb2_sleep as sl  # noqa: E402

INTENT_ARCHIVE = REPO / "runs/explore/cb2/intent_archive"
SIDECAR = REPO / "runs/explore/cb2/intent_sidecar"


# ------------------------------------------------------------------------------------------------- practice
def _worker(args) -> dict:
    k, a = args
    import torch
    torch.set_num_threads(1)
    from continuous_brain_cb2.archive import StreamWriter
    from continuous_brain_cb2.intent import BRANCHES, IntentShadow
    from continuous_brain_cb2.organ import COMMANDS, command_for
    from continuous_brain_cb2.record import record_life
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_individual.skills import ACTIONS
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_world.env import W1Config, W1Env
    v = vision()
    stream = f"practice-r{a.run:03d}-w{k:02d}"
    w = StreamWriter(a.archive, stream, "PRACTICE", "teacher-routine", {**provenance(v, "practice-intent")})
    steps_each = a.steps // a.workers
    t0, n, i = time.time(), 0, 0
    rows, cmds, mismatch, lives = [], [], 0, 0
    while n < steps_each:
        seed = PRACTICE_BASE + a.run * 1_000_000 + k * 100_000 + i
        teacher = Individual(IndividualConfig(), v, LanguageOrgan.from_config({"backend": "dev"}))
        shadow = IntentShadow(0)
        prev = ["explore"]

        def on_step(_n, action, _u, _o, _r, _i, teacher=teacher, shadow=shadow, prev=prev):
            nonlocal mismatch
            name, _opt, vec = shadow(*IntentShadow.read(teacher))
            mismatch += int(ACTIONS[name] != int(action))
            rows.append(vec.astype(np.float16))
            prev[0] = command_for(BRANCHES[int(np.argmax(vec[:len(BRANCHES)]))], shadow.leaving, prev[0])
            cmds.append(COMMANDS.index(prev[0]))

        rec = record_life(teacher, W1Env(W1Config()), seed, w, steps=min(10000, steps_each - n), who="TEACHER",
                          lived=False, on_step=on_step)
        n += rec["steps"]
        lives += 1
        i += 1
    w.close()
    arr = np.stack(rows)
    Path(a.sidecar).mkdir(parents=True, exist_ok=True)
    np.save(Path(a.sidecar) / f"{stream}.npy", arr)
    np.save(Path(a.sidecar) / f"{stream}.cmd.npy", np.asarray(cmds, np.int8))   # r3-H organ-command labels
    return {"worker": k, "stream": stream, "steps": n, "rows": len(arr), "lives": lives,
            "intent_action_mismatch": mismatch, "seconds": round(time.time() - t0, 1)}


def practice(a) -> int:
    t = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_worker, [(k, a) for k in range(a.workers)])
    steps = sum(r["steps"] for r in res)
    mism = sum(r["intent_action_mismatch"] for r in res)
    rep = {"format": "cb2-intent-practice/1", "status": "ENGINEERING_ONLY", "run": a.run, "workers": a.workers,
           "steps": steps, "wall_s": round(time.time() - t, 1), "intent_action_mismatch": mism,
           "mismatch_rate": mism / max(steps, 1), **git_rev(), "per_worker": res}
    out = REPO / f"evidence/explore/cb2/intent-practice-r{a.run:03d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_worker"}), flush=True)
    if rep["mismatch_rate"] > 0.01:
        print(f"STOP: the intent shadow disagrees with the routine on {rep['mismatch_rate']:.2%} of steps", flush=True)
        return 1
    return 0


# ---------------------------------------------------------------------------------------------------- train
def _attach(sampler, sidecar: Path) -> None:
    for s in sampler.streams:
        arr = np.load(sidecar / f"{s.stream}.npy")
        if len(arr) != len(s):
            raise SystemExit(f"intent sidecar {s.stream}: {len(arr)} rows, stream has {len(s)}")
        s.cols["intent"] = arr
        cmd = sidecar / f"{s.stream}.cmd.npy"
        if cmd.exists():
            s.cols["command"] = np.load(cmd)


def train(a) -> int:
    import torch
    from continuous_brain_cb2.archive import Archive
    from continuous_brain_cb2.data import SequenceSampler
    from continuous_brain_cb2.intent import MASKS
    from continuous_brain_cb2.policy import SkillPolicy
    from continuous_brain_cb2.world_model import PredictiveCore
    torch.manual_seed(a.seed)
    dev = sl.device(a.device)
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    out = sl.new_out(a.out)
    core = PredictiveCore.from_state(torch.load(a.core, map_location=dev, weights_only=False), dev)
    core.eval()
    base = SkillPolicy.from_state(torch.load(a.init_policy, map_location=dev, weights_only=False), dev)
    if base.cfg.arm != "P-A" or base.feat_size != core.feat_size:
        raise SystemExit("the initial policy must be a P-A policy trained on this core")
    arms = [m for m in a.arms.split(",") if m]
    if not arms or any(m not in MASKS for m in arms):
        raise SystemExit(f"--arms must be from {MASKS}")
    pols = {m: SkillPolicy.with_intent(base, m, dev) for m in arms}
    arch = Archive(a.archive)
    refs = [r for r in arch.shards(("PRACTICE",)) if r.stream.startswith(a.prefix)]
    streams = sorted({r.stream for r in refs})
    held = streams[-a.heldout_streams:]
    tr_refs = [r for r in refs if r.stream not in held]
    te_refs = [r for r in refs if r.stream in held]
    ns = ("student", "teacher", "organ")
    tr = SequenceSampler(arch, tr_refs, T=a.T, seed=a.seed, namespaces=ns, map_crop=True, preload=a.preload)
    te = SequenceSampler(arch, te_refs, T=a.T, seed=a.seed + 1, namespaces=ns, map_crop=True, preload=a.preload)
    _attach(tr, Path(a.sidecar))
    _attach(te, Path(a.sidecar))
    common = {"format": "cb2-intent-policy/1", "status": "ENGINEERING_ONLY (DIAGNOSTIC: privileged intent input)",
              **git_rev(), "device": dev, "core": str(a.core), "core_sha256": sl.file_sha(Path(a.core)),
              "init_policy": str(a.init_policy), "init_policy_sha256": sl.file_sha(Path(a.init_policy)),
              "archive": str(a.archive), "train_rows": tr.rows(), "train_streams": len(tr.streams),
              "heldout_streams": held, "heldout_rows": te.rows(), "arms": arms,
              "args": {k: str(v) for k, v in vars(a).items() if k != "func"}}
    for m in arms:
        (out / m).mkdir()
        (out / m / "run.json").write_text(json.dumps({**common, "intent_mask": m}, indent=1) + "\n")
    print(json.dumps({k: common[k] for k in ("train_rows", "heldout_rows", "heldout_streams", "arms")}), flush=True)

    def heldout():
        bs = [te.batch(a.batch) for _ in range(a.eval_batches)]
        fs = [sl._feats(core, b) for b in bs]
        with torch.no_grad():
            return {m: round(float(np.mean([p.losses(f, b)[1]["agreement"] for f, b in zip(fs, bs)])), 4)
                    for m, p in pols.items()}

    sl.log(out, {"update": 0, "heldout_agreement": heldout()})        # all arms equal here: r2 P-A itself
    t0 = time.time()
    for u in range(1, a.updates + 1):
        b = tr.batch(a.batch)
        f = sl._feats(core, b)
        ms = {m: p.train_step(f, b) for m, p in pols.items()}
        if u % a.log_every == 0 or u == a.updates:
            sl.log(out, {"update": u, "s": round(time.time() - t0, 1), "train": ms, "heldout_agreement": heldout()})
    for m, p in pols.items():
        torch.save(p.state_dict_full(), out / m / "policy.pt")
        (out / m / "policy.pt.sha256").write_text(sl.file_sha(out / m / "policy.pt") + "\n")
    return 0


# ------------------------------------------------------------------------------------------- r3-H selector
def selector(a) -> int:
    """r3-H arm B: a learned organ-command selector on legal inputs only (core state, body sense, its memory
    organ's readout), imitating the command that serves the routine's intent (organ.command_for), r2 core frozen,
    warm-started from r2 P-A's trunk.  The routine's intent itself is never an input."""
    import torch
    from continuous_brain_cb2.archive import Archive
    from continuous_brain_cb2.data import SequenceSampler
    from continuous_brain_cb2.organ import COMMANDS
    from continuous_brain_cb2.policy import SkillPolicy
    from continuous_brain_cb2.world_model import PredictiveCore
    torch.manual_seed(a.seed)
    dev = sl.device(a.device)
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    out = sl.new_out(a.out)
    core = PredictiveCore.from_state(torch.load(a.core, map_location=dev, weights_only=False), dev)
    core.eval()
    base = SkillPolicy.from_state(torch.load(a.init_policy, map_location=dev, weights_only=False), dev)
    pol = SkillPolicy.selector_from(base, dev)
    arch = Archive(a.archive)
    refs = [r for r in arch.shards(("PRACTICE",)) if r.stream.startswith(a.prefix)]
    streams = sorted({r.stream for r in refs})
    held = streams[-a.heldout_streams:]
    ns = ("student", "teacher", "organ")
    tr = SequenceSampler(arch, [r for r in refs if r.stream not in held], T=a.T, seed=a.seed, namespaces=ns,
                         map_crop=True, preload=a.preload)
    te = SequenceSampler(arch, [r for r in refs if r.stream in held], T=a.T, seed=a.seed + 1, namespaces=ns,
                         map_crop=True, preload=a.preload)
    _attach(tr, Path(a.sidecar))
    _attach(te, Path(a.sidecar))
    if any("command" not in s.cols for s in tr.streams + te.streams):
        raise SystemExit("no organ-command labels (<stream>.cmd.npy) in the sidecar: re-run practice with this code")
    counts = np.bincount(np.concatenate([s.cols["command"] for s in tr.streams]), minlength=len(COMMANDS))
    run = {"format": "cb2-r3h-selector/1", "status": "ENGINEERING_ONLY", **git_rev(), "device": dev,
           "core": str(a.core), "core_sha256": sl.file_sha(Path(a.core)), "init_policy": str(a.init_policy),
           "init_policy_sha256": sl.file_sha(Path(a.init_policy)), "archive": str(a.archive),
           "train_rows": tr.rows(), "heldout_streams": held, "heldout_rows": te.rows(),
           "commands": list(COMMANDS), "label_share": {c: round(float(n / counts.sum()), 4) for c, n in zip(COMMANDS, counts)},
           "inputs": "core state + body sense + memory organ readout (legal; no routine intent)",
           "args": {k: str(v) for k, v in vars(a).items() if k != "func"}}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(json.dumps({k: run[k] for k in ("train_rows", "heldout_rows", "label_share")}), flush=True)

    def heldout():
        bs = [te.batch(a.batch) for _ in range(a.eval_batches)]
        with torch.no_grad():
            return round(float(np.mean([pol.losses(sl._feats(core, b), b)[1]["agreement"] for b in bs])), 4)

    t0 = time.time()
    for u in range(1, a.updates + 1):
        b = tr.batch(a.batch)
        m = pol.train_step(sl._feats(core, b), b)
        if u % a.log_every == 0 or u == a.updates:
            sl.log(out, {"update": u, "s": round(time.time() - t0, 1), "train": m, "heldout_command_agreement": heldout()})
    torch.save(pol.state_dict_full(), out / "policy.pt")
    (out / "policy.pt.sha256").write_text(sl.file_sha(out / "policy.pt") + "\n")
    return 0


# -------------------------------------------------------------------------------------------------- summary
def _closed_loop_agreement(archive: Path, stream: str) -> dict:
    from continuous_brain_cb2.archive import Archive
    arch = Archive(archive)
    acts, tacts = [], []
    for r in arch.shards(("PRACTICE",)):
        if r.stream != stream:
            continue
        t = arch.read(r, ("student", "teacher"), {"student": ["action"], "teacher": ["teacher_action"]})
        acts.append(t["student"]["action"].to_numpy())
        tacts.append(t["teacher"]["teacher_action"].to_numpy())
    if not acts:
        return {"steps": 0, "agreement": None}
    a_, t_ = np.concatenate(acts), np.concatenate(tacts)
    return {"steps": int(len(a_)), "agreement": round(float((a_ == t_).mean()), 4)}


def summary(a) -> int:
    ev = Path(a.evidence)
    r2 = {"crafter_score": 10.729, "survival_steps": 226.275, "achievements_per_episode": 7.225,
          "obligations": 4 / 56, "closed_loop_agreement": 0.421}
    rows, rt = {"r2 P-A (reference)": r2}, None
    for m in a.arms.split(","):
        e = json.load(open(ev / f"eval-{a.label}-{m}.json"))
        s, rt = e["summary"]["individual"], e["summary"]["routine"]
        rows[m] = {"crafter_score": s["crafter_score"], "survival_steps": s["survival_steps"],
                   "achievements_per_episode": s["achievements_per_episode"],
                   "obligations": s["obligations_on_time"]["rate"],
                   "closed_loop_agreement": _closed_loop_agreement(Path(a.archive), f"eval-{a.label}-{m}")["agreement"],
                   "paired_vs_routine": e["paired_difference"]}
    rtv = {k: rt[k] for k in ("crafter_score", "survival_steps", "achievements_per_episode")}
    rtv["obligations"] = rt["obligations_on_time"]["rate"]
    ratio = {k: {m: round(v[m] / rtv[m], 3) for m in ("crafter_score", "survival_steps")} for k, v in rows.items()}
    out = {"format": "cb2-r3t-summary/1", "status": "ENGINEERING_ONLY (DIAGNOSTIC)", "label": a.label,
           "seeds": "75001-75040 (development)", "routine": rtv, "arms": rows, "ratio_to_routine": ratio}
    if "none" in rows:
        for m in ("branch", "full"):
            if m in rows:
                out[f"intent_effect_{m}_minus_none"] = {k: round(ratio[m][k] - ratio["none"][k], 3)
                                                        for k in ("crafter_score", "survival_steps")}
    (ev / f"{a.label}-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("practice")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--steps", type=int, default=1_000_000, help="total, split across workers")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--archive", default=str(INTENT_ARCHIVE))
    p.add_argument("--sidecar", default=str(SIDECAR))
    p.set_defaults(func=practice)
    t = sub.add_parser("train")
    t.add_argument("--core", required=True)
    t.add_argument("--init-policy", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--arms", default="none,branch,full")
    t.add_argument("--archive", default=str(INTENT_ARCHIVE))
    t.add_argument("--sidecar", default=str(SIDECAR))
    t.add_argument("--prefix", default="practice-r004-")
    t.add_argument("--heldout-streams", type=int, default=1)
    t.add_argument("--updates", type=int, default=8000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--T", type=int, default=64)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--preload", action="store_true")
    t.add_argument("--log-every", type=int, default=250)
    t.add_argument("--eval-batches", type=int, default=4)
    t.set_defaults(func=train)
    h = sub.add_parser("selector")
    h.add_argument("--core", required=True)
    h.add_argument("--init-policy", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--archive", default=str(INTENT_ARCHIVE))
    h.add_argument("--sidecar", default=str(SIDECAR))
    h.add_argument("--prefix", default="practice-r004-")
    h.add_argument("--heldout-streams", type=int, default=1)
    h.add_argument("--updates", type=int, default=8000)
    h.add_argument("--batch", type=int, default=16)
    h.add_argument("--T", type=int, default=64)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--device", default="auto")
    h.add_argument("--threads", type=int, default=8)
    h.add_argument("--preload", action="store_true")
    h.add_argument("--log-every", type=int, default=250)
    h.add_argument("--eval-batches", type=int, default=4)
    h.set_defaults(func=selector)
    s = sub.add_parser("summary")
    s.add_argument("--label", default="r3t")
    s.add_argument("--arms", default="none,branch,full")
    s.add_argument("--evidence", default=str(REPO / "evidence/explore/cb2"))
    s.add_argument("--archive", default=str(REPO / "runs/explore/cb2/archive"))
    s.set_defaults(func=summary)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
