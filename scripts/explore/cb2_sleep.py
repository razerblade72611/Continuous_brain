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


def split(archive, heldout_lives: int, test_archive=None):
    from continuous_brain_cb2.archive import Archive
    arch = Archive(archive)
    if test_archive:                    # lives reserved before training, in their own archive: a true holdout
        test = Archive(test_archive).shards(("LIVED", "PRACTICE"))
        held = sorted({l for r in test for l in r.manifest["lives"]})
        return arch, arch.shards(("LIVED", "PRACTICE")), test, held
    lived = arch.shards(("LIVED",))
    if not lived:                       # no lived stream (an engineering archive): hold out the last practice lives
        lived = [r for r in arch.shards(("PRACTICE",)) if not r.stream.endswith("-routine")]
    lives = sorted({l for r in lived for l in r.manifest["lives"]})
    held = set(lives[-heldout_lives:]) if heldout_lives else set()
    test_streams = {r.stream for r in lived}
    train = [r for r in arch.shards(("LIVED", "PRACTICE"))
             if not (set(r.manifest["lives"]) & held and r.stream in test_streams)]
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
    mc = bool(a.map_radius)
    tr = SequenceSampler(arch, train, T=a.T, seed=a.seed, preload=a.preload, map_crop=mc)
    te = SequenceSampler(arch, test, T=a.T, seed=a.seed + 1, preload=a.preload, map_crop=mc)
    if a.init:                          # sleep continues the individual's own core (L-A), never a fresh one
        saved = torch.load(a.init, map_location=dev, weights_only=False)
        add = {"map_radius": a.map_radius} if a.map_radius and not saved["config"].get("map_radius") else {}
        core = PredictiveCore.from_state(saved, dev, **add)
    else:
        from dataclasses import replace
        core = PredictiveCore(replace(CoreConfig.sized(a.size), map_radius=a.map_radius), dev)
    run = {"format": "cb2-sleep-core/1", "status": "ENGINEERING_ONLY", **git_rev(), "device": dev, **core.describe(),
           "init": a.init, "init_sha256": file_sha(Path(a.init)) if a.init else None,
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
    tr = SequenceSampler(arch, train, T=a.T, seed=a.seed, namespaces=ns, map_crop=pa, preload=a.preload)
    te = SequenceSampler(arch, test, T=a.T, seed=a.seed + 1, namespaces=ns, map_crop=pa, preload=a.preload)
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


# ------------------------------------------------------------------------------------------ L-A: probe a life
def life_arrays(sampler, s, start: int, end: int) -> dict:
    """One life's rows [start, end) of a stream as arrays [1, L, ...], with the core's inputs as in training."""
    c = s.cols
    act = c["action"][start:end].astype(np.int64)
    rew = c["rew_total"][start:end]
    out = {"image": sampler._column(s, start, end - start, "student", "frame")[None], "body": c["body"][start:end][None],
           "prev_action": np.r_[0, act[:-1]][None], "is_first": np.r_[1.0, np.zeros(end - start - 1)][None].astype(np.float32),
           "reward": np.r_[0.0, rew[:-1]][None]}
    if sampler.map_crop:
        out["map_crop"] = sampler._column(s, start, end - start, "organ", "map_crop")[None]
    return out


def scan_life(core, b, chunk: int = 256):
    """The core lives the whole life from its first step, carrying its state across chunks (no gradient)."""
    import torch
    feats, st, L = [], None, b["image"].shape[1]
    for i in range(0, L, chunk):
        sl = slice(i, min(L, i + chunk))
        post, _ = core.scan(b["image"][:, sl], b["body"][:, sl], b["prev_action"][:, sl], b["is_first"][:, sl], state=st)
        feats.append(core.feat(post))
        st = {k: v[:, -1] for k, v in post.items()}
    return torch.cat(feats, 1)


def probe_life(a) -> int:
    """Gate 4 with long context: on held-out lives, the core lives each life from its first step; from step
    ``--skip`` on, (i) its remembered-map head's out-of-view accuracy, (ii) ridge probes from its state to which
    out-of-view tiles it remembers, against the same probe on the frame encoder alone and on an untrained core."""
    import torch
    from continuous_brain_cb2.data import SequenceSampler
    from continuous_brain_cb2.world_model import PredictiveCore
    dev = device(a.device)
    trained = PredictiveCore.from_state(torch.load(a.core, map_location=dev, weights_only=False), dev)
    trained.eval()
    torch.manual_seed(12345)
    untrained = PredictiveCore(trained.cfg, dev)
    arch, train, test, held = split(a.archive, a.heldout_lives, getattr(a, "test_archive", None))
    out_of_view = np.ones((33, 33), bool)
    out_of_view[16 - 3:16 + 4, 16 - 4:16 + 5] = False

    def lives_of(refs, limit):
        smp = SequenceSampler(arch, refs, T=1, seed=0, map_crop=True)
        for s in smp.streams:
            starts = np.flatnonzero(s.cols["is_first"]).tolist() + [len(s)]
            for i0, i1 in zip(starts[:-1], starts[1:]):
                if i1 - i0 > a.skip + 8:
                    yield smp, s, i0, i1
                    limit -= 1
                    if limit <= 0:
                        return

    def collect(refs, limit):
        X = {"trained": [], "untrained": [], "frame": []}
        Y, M = [], {"acc": [], "const": [], "bal": [], "balc": [], "n": 0}
        for smp, s, i0, i1 in lives_of(refs, limit):
            b = life_arrays(smp, s, i0, i1)
            with torch.no_grad():
                ft = scan_life(trained, b)
                fu = scan_life(untrained, b)
                fe = trained.embed(b["image"], b["body"])
                if "map" in trained.heads:
                    m = trained.map_metrics(ft, b["map_crop"], skip=a.skip)
                    if m:
                        M["acc"].append(m["map_oov_seen_acc"]); M["const"].append(m["map_oov_seen_const"])
                        M["bal"].append(m["map_oov_seen_balanced"]); M["balc"].append(m["map_oov_seen_balanced_const"])
                        M["n"] += 1
            idx = np.arange(a.skip, i1 - i0, a.every)
            X["trained"].append(ft[0, idx].cpu().numpy()); X["untrained"].append(fu[0, idx].cpu().numpy())
            X["frame"].append(fe[0, idx].cpu().numpy())
            Y.append((b["map_crop"][0, idx].reshape(-1, 33, 33)[:, out_of_view] >= 0).astype(np.float32))
        return {k: np.concatenate(v) for k, v in X.items()}, np.concatenate(Y), M

    def ridge(X, Y, Xt, Yt):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        X, Xt = (X - mu) / sd, (Xt - mu) / sd
        def fit(X, Y, lam):
            Xs = np.c_[X, np.ones(len(X))]
            reg = lam * len(X) * np.eye(Xs.shape[1]); reg[-1, -1] = 0
            return np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ Y)
        def r2(W, X, Y):
            P = np.c_[X, np.ones(len(X))] @ W
            keep = Y.std(0) > 1e-6
            return float(np.mean(1 - ((Y - P) ** 2).sum(0)[keep] / ((Y - Y.mean(0)) ** 2).sum(0)[keep]))
        k = int(len(X) * 0.8)
        lam = max((1e-2, 1e-1, 1.0, 10.0, 100.0), key=lambda l: r2(fit(X[:k], Y[:k], l), X[k:], Y[k:]))
        return round(r2(fit(X, Y, lam), Xt, Yt), 4), lam

    Xf, Yf, _ = collect(train, a.fit_lives)
    Xt, Yt, M = collect(test, a.test_lives)
    res = {k: dict(zip(("r2", "penalty"), ridge(Xf[k], Yf, Xt[k], Yt))) for k in Xf}
    out = {"format": "cb2-probe-life/1", "status": "ENGINEERING_ONLY", **git_rev(), "core": str(a.core),
           "core_sha256": file_sha(Path(a.core)), "heldout_lives": held, "skip": a.skip, "every": a.every,
           "known_out_of_view_r2": res,
           "map_head_oov": ({"seen_acc": float(np.mean(M["acc"])), "const_baseline": float(np.mean(M["const"])),
                             "balanced_acc": float(np.mean(M["bal"])), "balanced_chance": float(np.mean(M["balc"])),
                             "lives": M["n"]} if M["n"] else None),
           "gate4_reading": "persistent state shown if trained > frame and trained > untrained on known-out-of-view R^2, "
                            "and the map head beats the constant guess out of view"}
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


# ------------------------------------------------------------------------------------------- gate 1
def gate_foresight(a) -> int:
    """Gate 1 (capability ledger): the predictive core's 100-step risk predictions against the old foresight
    network (62 belief features -> 48 -> 48 -> 5, logistic loss), trained here on the same training rows (every
    5th, as it sampled), both scored by Brier on the held-out lived lives.  The core passes only if it is at least
    as good for every risk."""
    import torch
    from continuous_brain_cb2.data import RISK_H, RISKS, StreamIndex
    from continuous_brain_cb2.world_model import PredictiveCore
    torch.manual_seed(a.seed)
    dev = device(a.device)
    arch, train, test, held = split(a.archive, a.heldout_lives)
    j100 = RISK_H.index(100)
    cols = [r * len(RISK_H) + j100 for r in range(len(RISKS))]
    by = lambda refs: {s: sorted([r for r in refs if r.stream == s], key=lambda r: r.manifest["shard"])
                       for s in sorted({r.stream for r in refs})}
    Xtr, Ytr = [], []
    for refs in by(train).values():
        si = StreamIndex.load(arch, refs, ("student", "organ"))
        Xtr.append(si.cols["features"][::5])
        Ytr.append(si.cols["risk"][::5][:, cols])
    Xtr, Ytr = np.concatenate(Xtr).astype(np.float32), np.concatenate(Ytr).astype(np.float32)
    old = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 48), torch.nn.ReLU(), torch.nn.Linear(48, 48),
                              torch.nn.ReLU(), torch.nn.Linear(48, len(RISKS)))
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(old(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    core = PredictiveCore.from_state(torch.load(a.core, map_location=dev, weights_only=False), dev)
    core.eval()
    P_core, P_old, Y = [], [], []
    for refs in by(test).values():
        si = StreamIndex.load(arch, refs, ("student", "organ"))
        from continuous_brain_cb2.data import FrameCache
        cache = FrameCache(arch, 4)
        n = len(si)
        frames_all = []
        for ref in si.shards:
            for g in range(arch.row_groups(ref, "student")):
                frames_all.append(cache.group(ref, g))
        img = np.concatenate(frames_all)
        c = si.cols
        prev = np.r_[0, c["action"][:-1]].astype(np.int64)
        prev[c["is_first"]] = 0
        state, out = None, []
        for i in range(0, n, a.T):                         # the whole held-out stream, its state carried along
            sl = slice(i, min(n, i + a.T))
            post, _ = core.scan(img[sl][None], c["body"][sl][None], prev[sl][None], c["is_first"][sl][None].astype(np.float32),
                                state=state)
            state = {k: v[:, -1] for k, v in post.items()}
            with torch.no_grad():
                out.append(core.heads["risk"](core.feat(post)).mean[0].cpu().numpy())
        P_core.append(np.concatenate(out)[:, cols])
        with torch.no_grad():
            P_old.append(torch.sigmoid(old(torch.from_numpy(c["features"].astype(np.float32)))).numpy())
        Y.append(c["risk"][:, cols])
    P_core, P_old, Y = np.concatenate(P_core), np.concatenate(P_old), np.concatenate(Y)
    base = Ytr.mean(0)
    rows = []
    for r, name in enumerate(RISKS):
        b = lambda p: float(((p - Y[:, r]) ** 2).mean())
        rows.append({"risk": name, "rate_heldout": round(float(Y[:, r].mean()), 4), "core": round(b(P_core[:, r]), 5),
                     "old_foresight": round(b(P_old[:, r]), 5), "base_rate": round(b(np.full(len(Y), base[r])), 5)})
    passed = all(x["core"] <= x["old_foresight"] for x in rows)
    out = {"format": "cb2-gate/1", "gate": "foresight", "status": "ENGINEERING_ONLY", **git_rev(),
           "core": str(a.core), "core_sha256": file_sha(Path(a.core)), "heldout_lives": held,
           "heldout_rows": int(len(Y)), "train_rows_old": int(len(Xtr)), "horizon": 100, "risks": rows,
           "passed": passed,
           "verdict": ("core at least as good for every risk" if passed else "core worse for: " +
                       ", ".join(x["risk"] for x in rows if x["core"] > x["old_foresight"])),
           "note": "The old network's two routine-state features (hurt-place near, in shelter) are 0 here: the "
                   "archive keeps no routine plan state in the organ namespace."}
    print(json.dumps(out, indent=1))
    dest = Path(a.out) if a.out else REPO / f"evidence/explore/cb2/gate-foresight-{a.label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("core", "policy", "probe", "probe-life", "dream", "gate-foresight"):
        p = sub.add_parser(name)
        p.add_argument("--archive", default=str(ARCHIVE))
        p.add_argument("--heldout-lives", type=int, default=3)
        p.add_argument("--device", default="auto")
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--T", type=int, default=64)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default=None, required=name in ("core", "policy"))
        p.add_argument("--preload", action="store_true", help="hold all frames in memory (for a GPU)")
        if name in ("core", "policy"):
            p.add_argument("--updates", type=int, default=20000)
            p.add_argument("--log-every", type=int, default=250)
        if name == "core":
            p.add_argument("--size", default="S", choices=("XS", "S", "M", "L"))
            p.add_argument("--save-every", type=int, default=5000)
            p.add_argument("--init", default=None, help="continue this core (L-A sleep), never overwritten")
            p.add_argument("--map-radius", type=int, default=0, help="L-A remembered-map head radius (0 = none)")
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
        if name == "gate-foresight":
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument("--label", required=True)
        if name == "probe-life":
            p.add_argument("--fit-lives", type=int, default=24)
            p.add_argument("--test-archive", default=None, help="test on this archive's lives (reserved before training)")
            p.add_argument("--test-lives", type=int, default=12)
            p.add_argument("--skip", type=int, default=64, help="probe from this step of each life on")
            p.add_argument("--every", type=int, default=4)
        if name == "probe":
            p.add_argument("--fit-batches", type=int, default=24)
            p.add_argument("--test-batches", type=int, default=12)
    a = ap.parse_args()
    return {"core": train_core, "policy": train_policy, "probe": probe, "probe-life": probe_life, "dream": dream_cmd,
            "gate-foresight": gate_foresight}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
