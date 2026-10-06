#!/usr/bin/env python3
"""CB2-S1 sleep: train the predictive core and the first learned skill from the archive (exploration lane,
ENGINEERING_ONLY).  Runs on a GPU when there is one (``--device auto``), on the CPU otherwise.

  core     python scripts/explore/cb2_sleep.py core --size S --updates 20000 --out runs/explore/cb2/sleep/core-S
  policy   python scripts/explore/cb2_sleep.py policy --core runs/explore/cb2/sleep/core-S/core.pt --arm P-A \
               --updates 5000 --out runs/explore/cb2/sleep/policy-PA
  probe    python scripts/explore/cb2_sleep.py probe --core runs/explore/cb2/sleep/core-S/core.pt

Split: the lived stream's last ``--heldout-lives`` lives (default 3) are held out; everything else the archive
holds (LIVED and PRACTICE, never DREAMED) is training data.  Checkpoints are never overwritten: each save is a new
file, and ``core.pt`` / ``policy.pt`` are written once, at the end, beside ``metrics.jsonl`` and ``run.json``.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
ARCHIVE = REPO / "runs/explore/cb2/archive"


def git_rev() -> dict:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True,
                                capture_output=True).stdout.strip())
    return {"git": rev, "git_dirty": dirty}


def device(name: str) -> str:
    import torch
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def split(archive, heldout_lives: int):
    from continuous_brain_cb2.archive import Archive
    arch = Archive(archive)
    lived = arch.shards(("LIVED",))
    lives = sorted({l for r in lived for l in r.manifest["lives"]})
    held = set(lives[-heldout_lives:]) if heldout_lives else set()
    train = [r for r in arch.shards(("LIVED", "PRACTICE")) if not set(r.manifest["lives"]) & held or r.where != "LIVED"]
    test = [r for r in lived if set(r.manifest["lives"]) & held]
    return arch, train, test, sorted(held)


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def new_out(path: str) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: never overwritten; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    return out


def log(out: Path, rec: dict) -> None:
    with open(out / "metrics.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


# --------------------------------------------------------------------------------------------------- core
def train_core(a) -> int:
    import torch
    from continuous_brain_cb2.data import SequenceSampler
    from continuous_brain_cb2.world_model import CoreConfig, PredictiveCore
    torch.manual_seed(a.seed)
    dev = device(a.device)
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    out = new_out(a.out)
    arch, train, test, held = split(a.archive, a.heldout_lives)
    tr = SequenceSampler(arch, train, T=a.T, seed=a.seed)
    te = SequenceSampler(arch, test, T=a.T, seed=a.seed + 1)
    core = PredictiveCore(CoreConfig.sized(a.size), dev)
    run = {"format": "cb2-sleep-core/1", "status": "ENGINEERING_ONLY", **git_rev(), "device": dev, **core.describe(),
           "train_rows": tr.rows(), "heldout_rows": te.rows(), "heldout_lives": held, "args": vars(a),
           "train_shards": [[r.stream, r.manifest["shard"], r.manifest["manifest_sha"]] for r in train],
           "heldout_shards": [[r.stream, r.manifest["shard"], r.manifest["manifest_sha"]] for r in test]}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(json.dumps({k: run[k] for k in ("device", "params", "train_rows", "heldout_rows")}), flush=True)
    t0 = time.time()
    for u in range(1, a.updates + 1):
        m = core.train_step(tr.batch(a.batch))
        if u % a.log_every == 0 or u == a.updates:
            ev = core.evaluate(te.batch(a.batch))
            log(out, {"update": u, "s": round(time.time() - t0, 1),
                      "train": {k: round(v, 4) for k, v in m.items()},
                      "heldout": {k: round(v, 5) for k, v in ev.items()}})
        if a.save_every and u % a.save_every == 0 and u != a.updates:
            torch.save(core.state_dict_full(), out / f"core-u{u:07d}.pt")
    torch.save(core.state_dict_full(), out / "core.pt")
    (out / "core.pt.sha256").write_text(file_sha(out / "core.pt") + "\n")
    return 0


# ------------------------------------------------------------------------------------------------- policy
def _feats(core, b):
    import torch
    post, _ = core.scan(b["image"], b["body"], b["prev_action"], b["is_first"])
    return core.feat(post).detach()


def train_policy(a) -> int:
    import torch
    from continuous_brain_cb2.data import SequenceSampler
    from continuous_brain_cb2.policy import PolicyConfig, SkillPolicy
    from continuous_brain_cb2.world_model import PredictiveCore
    torch.manual_seed(a.seed)
    dev = device(a.device)
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    out = new_out(a.out)
    core = PredictiveCore.from_state(torch.load(a.core, map_location=dev, weights_only=False), dev)
    core.eval()
    arch, train, test, held = split(a.archive, a.heldout_lives)
    ns = ("student", "teacher", "organ")
    pa = a.arm == "P-A"
    tr = SequenceSampler(arch, train, T=a.T, seed=a.seed, namespaces=ns, map_crop=pa)
    te = SequenceSampler(arch, test, T=a.T, seed=a.seed + 1, namespaces=ns, map_crop=pa)
    pol = SkillPolicy(PolicyConfig(arm=a.arm), core.feat_size, dev)
    run = {"format": "cb2-sleep-policy/1", "status": "ENGINEERING_ONLY", **git_rev(), "device": dev, "arm": a.arm,
           "core": str(a.core), "core_sha256": file_sha(Path(a.core)),
           "params": sum(p.numel() for p in pol.parameters()), "heldout_lives": held, "args": vars(a)}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    majority = int(np.bincount(np.concatenate([s.cols["teacher_action"] for s in tr.streams]), minlength=17).argmax())
    t0 = time.time()
    for u in range(1, a.updates + 1):
        b = tr.batch(a.batch)
        m = pol.train_step(_feats(core, b), b)
        if u % a.log_every == 0 or u == a.updates:
            agree, maj = [], []
            for _ in range(a.eval_batches):
                eb = te.batch(a.batch)
                with torch.no_grad():
                    _, em = pol.losses(_feats(core, eb), eb)
                agree.append(em["agreement"])
                maj.append(np.mean(eb["teacher_action"] == majority))
            log(out, {"update": u, "s": round(time.time() - t0, 1), "train": m,
                      "heldout_agreement": round(float(np.mean(agree)), 4),
                      "majority_action_baseline": round(float(np.mean(maj)), 4)})
    torch.save(pol.state_dict_full(), out / "policy.pt")
    (out / "policy.pt.sha256").write_text(file_sha(out / "policy.pt") + "\n")
    return 0


# -------------------------------------------------------------------------------------------------- probes
def probe(a) -> int:
    """Ridge probes from the core's state to what the hand-built faculties knew: the 62 belief features and the
    remembered map around it (which tiles it knows, out of view included).  Baselines: the same probe on an
    untrained core of the same size, and on the frame encoder alone (no recurrence: the present frame only)."""
    import torch
    from continuous_brain_cb2.data import SequenceSampler
    from continuous_brain_cb2.world_model import PredictiveCore
    dev = device(a.device)
    trained = PredictiveCore.from_state(torch.load(a.core, map_location=dev, weights_only=False), dev)
    torch.manual_seed(12345)
    untrained = PredictiveCore(trained.cfg, dev)
    arch, train, test, held = split(a.archive, a.heldout_lives)
    ns = ("student", "organ")
    samplers = {"fit": SequenceSampler(arch, train, T=a.T, seed=7, namespaces=ns, map_crop=True),
                "test": SequenceSampler(arch, test, T=a.T, seed=8, namespaces=ns, map_crop=True)}

    def collect(core, which, n):
        X = {"core": [], "frame": []}
        Y = {"features": [], "known": []}
        for _ in range(n):
            b = samplers[which].batch(a.batch)
            with torch.no_grad():
                post, _ = core.scan(b["image"], b["body"], b["prev_action"], b["is_first"])
                f = core.feat(post)[:, a.T // 2:].reshape(-1, core.feat_size).cpu().numpy()
                e = core.embed(b["image"], b["body"])[:, a.T // 2:].reshape(-1, core.embed_size).cpu().numpy()
            X["core"].append(f)
            X["frame"].append(e)
            Y["features"].append(b["features"][:, a.T // 2:].reshape(-1, b["features"].shape[-1]))
            crop = b["map_crop"][:, a.T // 2:].reshape(-1, 33, 33)
            out_of_view = np.ones((33, 33), bool)
            out_of_view[16 - 3:16 + 4, 16 - 4:16 + 5] = False                # the 7 x 9 view around it
            Y["known"].append((crop[:, out_of_view] >= 0).astype(np.float32))
        return {k: np.concatenate(v) for k, v in X.items()}, {k: np.concatenate(v) for k, v in Y.items()}

    def fit(X, Y, lam):
        Xs = np.c_[X, np.ones(len(X))]
        reg = lam * len(X) * np.eye(Xs.shape[1])
        reg[-1, -1] = 0
        return np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ Y)

    def r2(W, X, Y):
        P = np.c_[X, np.ones(len(X))] @ W
        keep = Y.std(0) > 1e-6
        ss_res, ss_tot = ((Y - P) ** 2).sum(0), ((Y - Y.mean(0)) ** 2).sum(0)
        return float(np.mean(1 - ss_res[keep] / ss_tot[keep]))

    def ridge(X, Y, Xt, Yt):
        """Standardise on the fit set, choose the penalty on a fifth of it, refit, score on held-out lives."""
        mu, sd = X.mean(0), X.std(0) + 1e-6
        X, Xt = (X - mu) / sd, (Xt - mu) / sd
        k = int(len(X) * 0.8)
        lam = max((1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0), key=lambda l: r2(fit(X[:k], Y[:k], l), X[k:], Y[k:]))
        return r2(fit(X, Y, lam), Xt, Yt), lam

    res = {}
    for name, core in (("trained", trained), ("untrained", untrained)):
        Xf, Yf = collect(core, "fit", a.fit_batches)
        Xt, Yt = collect(core, "test", a.test_batches)
        for x in ("core", "frame"):
            for y in ("features", "known"):
                score, lam = ridge(Xf[x], Yf[y], Xt[x], Yt[y])
                res[f"{name}/{x}->{y}"] = {"r2": round(score, 4), "penalty": lam}
    out = {"format": "cb2-probe/1", "status": "ENGINEERING_ONLY", **git_rev(), "core": str(a.core),
           "core_sha256": file_sha(Path(a.core)), "heldout_lives": held, "r2": res,
           "reading": "held-out R^2 of ridge probes; 'known' = which out-of-view tiles of its remembered 33x33 map it "
                      "knows; 'frame' = the frame and body encoder alone (no recurrence)"}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------------------------------- dreams
def dream_cmd(a) -> int:
    """Imagined continuations from moments it lived, archived as DREAMED (they can never be read as lived)."""
    import torch
    from continuous_brain_cb2.archive import StreamWriter
    from continuous_brain_cb2.data import SequenceSampler
    from continuous_brain_cb2.dream import dream
    from continuous_brain_cb2.policy import SkillPolicy
    from continuous_brain_cb2.world_model import PredictiveCore
    dev = device(a.device)
    core = PredictiveCore.from_state(torch.load(a.core, map_location=dev, weights_only=False), dev)
    pol = SkillPolicy.from_state(torch.load(a.policy, map_location=dev, weights_only=False), dev)
    arch, train, _, _ = split(a.archive, a.heldout_lives)
    lived = [r for r in train if r.where == "LIVED"]
    sampler = SequenceSampler(arch, lived, T=a.T, seed=a.seed)
    w = StreamWriter(a.archive, a.stream, "DREAMED", a.individual,
                     {"role": "dream", **git_rev(), "core_sha256": file_sha(Path(a.core)),
                      "policy_sha256": file_sha(Path(a.policy)), "horizon": a.horizon})
    n = dream(core, pol, sampler, w, a.n, a.horizon, batch=a.batch)
    print(json.dumps({"stream": a.stream, "dreamed_rows": n, "lived_rows_after": sum(
        len(t) for t in arch.lived_rows(a.individual))}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("core", "policy", "probe", "dream"):
        p = sub.add_parser(name)
        p.add_argument("--archive", default=str(ARCHIVE))
        p.add_argument("--heldout-lives", type=int, default=3)
        p.add_argument("--device", default="auto")
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--T", type=int, default=64)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default=None, required=name in ("core", "policy"))
        if name in ("core", "policy"):
            p.add_argument("--updates", type=int, default=20000)
            p.add_argument("--log-every", type=int, default=250)
        if name == "core":
            p.add_argument("--size", default="S", choices=("XS", "S", "M", "L"))
            p.add_argument("--save-every", type=int, default=5000)
        else:
            p.add_argument("--core", required=True)
        if name == "policy":
            p.add_argument("--arm", default="P-A", choices=("P-A", "P-B"))
            p.add_argument("--eval-batches", type=int, default=4)
        if name == "dream":
            p.add_argument("--policy", required=True, help="a P-B policy (a dream has no memory readout)")
            p.add_argument("--n", type=int, default=64, help="how many dreams")
            p.add_argument("--horizon", type=int, default=32)
            p.add_argument("--stream", required=True)
            p.add_argument("--individual", default="cb2-ind-0001")
        if name == "probe":
            p.add_argument("--fit-batches", type=int, default=24)
            p.add_argument("--test-batches", type=int, default=12)
    a = ap.parse_args()
    return {"core": train_core, "policy": train_policy, "probe": probe, "dream": dream_cmd}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
