#!/usr/bin/env python3
"""CB2-S2 (r4): outcome-based improvement of the accepted high-level controller (exploration lane, ENGINEERING_ONLY).

DRAFT pending the preregistration agreed on DAV-91.  The question: with the scripted executor organ, the r2 core and
the vision organ fixed, can PPO on the environment's own reward make the frozen r3h-B selector better than the
routine?  r3h-B is the initial policy, the KL anchor and the rollback point; nothing else changes.

  train     iterations of: practice lives on fresh worlds (run 5, seeds 85,000,000 + it*10,000 + n) lived in
            parallel by copies of r3h-B with the current selector installed, commands sampled at temperature 1
            (reproducibly: a pure function of (salt, life, step)); then PPO over the recorded selector inputs with a
            KL(pi || pi_r3h-B) penalty (adaptive beta), a value head on the detached trunk, GAE.  A checkpoint per
            iteration; every metric the anti-hacking checks need is logged per iteration.
  dev       a checkpoint, installed in a copy of r3h-B, on the development worlds (76001-76040 by default) with the
            unchanged cb2_evaluate protocol.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
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
from cb2_collect import PRACTICE_BASE, git_rev, vision  # noqa: E402
from cb2_evaluate import load_individual  # noqa: E402

RUN = 5


def _death_cause(inv: dict, died: bool) -> str:
    if not died:
        return "alive"
    if inv["food"] == 0:
        return "food"
    if inv["drink"] == 0:
        return "drink"
    if inv["energy"] == 0:
        return "energy"
    return "damage"


def _rollout(args):
    k, items, weights, home, max_steps = args[:5]
    record = args[5] if len(args) > 5 else True
    import torch
    torch.set_num_threads(1)
    from continuous_brain_cb2.consolidation import install
    from continuous_brain_cb2.organ import COMMANDS
    from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_world.env import W1Config, W1Env
    src, meta = load_individual(Path(home), vision())
    new = lambda: Individual(IndividualConfig(lifelong=True, skills="cb2"), src.vision, LanguageOrgan.from_config({"backend": "dev"}))
    dst = new()
    install(src, dst, weights, {"note": "S2 practice copy (never saved)"})
    base = dst.save()
    rec = {k_: [] for k_ in ("feat", "body", "features", "map_crop", "logits", "command", "reward", "r_crafter",
                             "r_care", "done", "life")}
    lives = []
    for gidx, seed in items:
        ind = new()
        ind.restore(copy.deepcopy(base))
        ind.life = int(meta["life"]) + 1 + gidx
        host = ind.rt.components["skills"]
        env = W1Env(W1Config())
        obs = env.reset(seed)
        info = first_info(env)
        n, done, cmds, cannot = 0, False, Counter(), 0
        while n < max_steps and not done:
            a, u = ind.decide(obs, info)
            last = host.last
            st = ind.read_state("skills") or {}
            cannot += int(str(st.get("delegation", {}).get("status", "")).startswith("cannot"))
            obs, rew, done, info = env.step(a, u)
            if not record:
                cmds[COMMANDS[last["command"]]] += 1
                n += 1
                continue
            for key in ("feat", "body", "features", "map_crop", "logits", "command"):
                rec[key].append(last[key])
            rec["reward"].append(float(rew["total"]))
            rec["r_crafter"].append(float(rew["crafter"]))
            rec["r_care"].append(float(rew["obligations"]))
            rec["done"].append(bool(done))
            rec["life"].append(gidx)
            cmds[COMMANDS[last["command"]]] += 1
            n += 1
        if not done and record and rec["done"]:
            rec["done"][-1] = False                                  # truncated, not terminal
        died = bool(done) and n < max_steps
        s = env.episode_summary()
        lives.append({"seed": seed, "steps": n, "died": died, "cause": _death_cause(info["privileged"]["inventory"], died),
                      "n_unlocked": s["crafter"]["n_unlocked"], "fulfilled": s["obligations"]["fulfilled"],
                      "failed": s["obligations"]["failed"], "commands": dict(cmds), "cannot": cannot,
                      "unlocked": sorted(s["crafter"]["unlocked"]),
                      "return": float(np.sum(rec["reward"][-n:])) if record else None})
    if not record:
        return None, lives
    arr = {"feat": np.asarray(rec["feat"], np.float32), "body": np.asarray(rec["body"], np.float32),
           "features": np.asarray(rec["features"], np.float32), "map_crop": np.asarray(rec["map_crop"], np.int8),
           "logits": np.asarray(rec["logits"], np.float32), "command": np.asarray(rec["command"], np.int64),
           "reward": np.asarray(rec["reward"], np.float32), "r_crafter": np.asarray(rec["r_crafter"], np.float32),
           "r_care": np.asarray(rec["r_care"], np.float32), "done": np.asarray(rec["done"], bool),
           "life": np.asarray(rec["life"], np.int64)}
    return arr, lives


def _routine(args):
    k, seeds, max_steps = args
    import torch
    torch.set_num_threads(1)
    from continuous_brain_cb2.archive import StreamWriter  # noqa: F401  (import check only)
    from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_world.env import W1Config, W1Env
    v = vision()
    out = []
    for seed in seeds:
        ind = Individual(IndividualConfig(), v, LanguageOrgan.from_config({"backend": "dev"}))
        env = W1Env(W1Config())
        obs, n, done = env.reset(seed), 0, False
        info = first_info(env)
        while n < max_steps and not done:
            a, u = ind.decide(obs, info)
            obs, _, done, info = env.step(a, u)
            n += 1
        s = env.episode_summary()
        out.append({"seed": seed, "steps": n, "unlocked": sorted(s["crafter"]["unlocked"]),
                    "n_unlocked": s["crafter"]["n_unlocked"], "fulfilled": s["obligations"]["fulfilled"],
                    "failed": s["obligations"]["failed"]})
    return out


def summarize(lives: list[dict]) -> dict:
    from continuous_brain_world.scoring import crafter_score, success_rates
    ful, fail = sum(l["fulfilled"] for l in lives), sum(l["failed"] for l in lives)
    return {"lives": len(lives), "crafter_score": crafter_score(list(success_rates([l["unlocked"] for l in lives]).values())),
            "survival_steps": float(np.mean([l["steps"] for l in lives])),
            "achievements_per_episode": float(np.mean([l["n_unlocked"] for l in lives])),
            "requests": {"fulfilled": ful, "failed": fail, "rate": ful / (ful + fail) if ful + fail else None}}


def dev_eval(weights: bytes, home: Path, seeds: list[int], workers: int, max_steps: int, base_life: int) -> dict:
    items = [(base_life + i, s) for i, s in enumerate(seeds)]
    chunks = [(k, items[k::workers], weights, str(home), max_steps, False) for k in range(workers) if items[k::workers]]
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        res = pool.map(_rollout, chunks)
    lives = sorted((l for r in res for l in r[1]), key=lambda l: l["seed"])
    return {"summary": summarize(lives), "lives": lives}


def routine_eval(seeds: list[int], workers: int, max_steps: int) -> dict:
    chunks = [(k, seeds[k::workers], max_steps) for k in range(workers) if seeds[k::workers]]
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        res = pool.map(_routine, chunks)
    lives = sorted((l for r in res for l in r), key=lambda l: l["seed"])
    return {"summary": summarize(lives), "lives": lives}


def _gae(rew, val, done, life, gamma, lam):
    """Per life, in order; a life's last step bootstraps 0 when terminal, its own value when truncated."""
    adv = np.zeros_like(rew)
    last = 0.0
    for i in range(len(rew) - 1, -1, -1):
        end = i == len(rew) - 1 or life[i + 1] != life[i]
        if end:
            nv = 0.0 if done[i] else val[i]
            last = 0.0
        else:
            nv = val[i + 1]
        delta = rew[i] + gamma * nv - val[i]
        last = delta + gamma * lam * (0.0 if end else last)
        adv[i] = last
    return adv, adv + val


def train(a) -> int:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from continuous_brain_cb2.organ import COMMANDS
    from continuous_brain_cb2.skill_host import pack_weights, unpack_weights
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True)
    home = Path(a.individuals) / a.individual
    meta = json.loads((home / "individual.json").read_text())
    core, ref = unpack_weights(np.frombuffer((home / "weights" / f"{meta['weights_sha256']}.pt").read_bytes(), np.uint8))
    if ref.cfg.head != "command":
        raise SystemExit("S2 improves an organ-command selector (r3-H)")
    pol = copy.deepcopy(ref)
    for p in ref.parameters():
        p.requires_grad_(False)
    value = nn.Sequential(nn.Linear(ref.cfg.units, 256), nn.SiLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(list(pol.parameters()) + list(value.parameters()), lr=a.lr)
    beta = a.beta
    run = {"format": "cb2-s2/1", "status": "ENGINEERING_ONLY (DRAFT, not preregistered)", **git_rev(),
           "reference": {"state_hash": meta["state_hash"], "weights_sha256": meta["weights_sha256"]},
           "args": {k: str(v) for k, v in vars(a).items() if k != "func"}}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    ctx = mp.get_context("fork")
    dev_seeds = list(range(a.dev_seeds[0], a.dev_seeds[1] + 1)) if a.dev_seeds[0] > 0 else []
    dev = {}
    if dev_seeds:
        rt = routine_eval(dev_seeds, a.workers, a.max_steps)
        (out / "dev-routine.json").write_text(json.dumps(rt, indent=1) + "\n")
        pol.cfg.temperature, pol.cfg.sample_salt = 0.0, 0
        dev[0] = dev_eval(pack_weights(core, pol), home, dev_seeds, a.workers, a.max_steps, 10_000_000)["summary"]
        dev[0]["ratio_to_routine"] = {m: dev[0][m] / rt["summary"][m] for m in ("crafter_score", "survival_steps")}
        print(json.dumps({"dev": 0, "r3h-B": dev[0], "routine": rt["summary"]}), flush=True)
        (out / "dev.json").write_text(json.dumps(dev, indent=1) + "\n")
    for it in range(1, a.iterations + 1):
        t0 = time.time()
        pol.cfg.temperature, pol.cfg.sample_salt = 1.0, it
        weights = pack_weights(core, pol)
        seeds = [PRACTICE_BASE + RUN * 1_000_000 + it * 10_000 + n for n in range(a.lives)]
        items = [(i + (it - 1) * a.lives, s) for i, s in enumerate(seeds)]
        chunks = [(k, items[k::a.workers], weights, str(home), a.max_steps) for k in range(a.workers) if items[k::a.workers]]
        with ctx.Pool(len(chunks)) as pool:
            res = pool.map(_rollout, chunks)
        batch = {key: np.concatenate([r[0][key] for r in res]) for key in res[0][0]}
        lives = [l for r in res for l in r[1]]
        roll_s = time.time() - t0
        # -- PPO update over the recorded inputs ------------------------------------------------------------------
        T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt)
        inp = (T(batch["feat"]), T(batch["body"]), T(batch["features"]), T(batch["map_crop"], torch.int64))
        cmd = T(batch["command"], torch.int64)
        logp_old = F.log_softmax(T(batch["logits"]), -1).gather(1, cmd[:, None])[:, 0]
        with torch.no_grad():
            ref_logp = torch.cat([F.log_softmax(ref(*[x[i:i + 4096] for x in inp])[0], -1) for i in range(0, len(cmd), 4096)])
            vals = torch.cat([value(pol.hidden(*[x[i:i + 4096] for x in inp])).squeeze(-1) for i in range(0, len(cmd), 4096)])
            now = torch.cat([F.log_softmax(pol(*[x[i:i + 4096] for x in inp])[0], -1) for i in range(0, len(cmd), 4096)])
            replay_gap = float((now.gather(1, cmd[:, None])[:, 0] - logp_old).abs().max())   # must be ~0
        adv, ret = _gae(batch["reward"], vals.numpy(), batch["done"], batch["life"], a.gamma, a.lam)
        adv_t, ret_t = T(adv), T(ret)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n = len(cmd)
        stats = Counter()
        stopped = False
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, a.minibatch):
                idx = perm[i:i + a.minibatch]
                h = pol.hidden(*[x[idx] for x in inp])
                logits = pol.action(h)
                logp_all = F.log_softmax(logits, -1)
                logp = logp_all.gather(1, cmd[idx][:, None])[:, 0]
                ratio = torch.exp(logp - logp_old[idx])
                pg = -torch.min(ratio * adv_t[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[idx]).mean()
                p_all = logp_all.exp()
                kl_ref = (p_all * (logp_all - ref_logp[idx])).sum(-1).mean()
                ent = -(p_all * logp_all).sum(-1).mean()
                v = value(h.detach()).squeeze(-1)
                vloss = F.mse_loss(v, ret_t[idx])
                loss = pg + beta * kl_ref - a.entropy * ent + 0.5 * vloss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(pol.parameters()) + list(value.parameters()), 1.0)
                opt.step()
                with torch.no_grad():
                    approx_kl_old = (logp_old[idx] - logp).mean().item()
                stats["pg"] += pg.item()
                stats["kl_ref"] += kl_ref.item()
                stats["entropy"] += ent.item()
                stats["vloss"] += vloss.item()
                stats["mb"] += 1
                if approx_kl_old > a.max_kl_old:
                    stopped = True
                    break
            if stopped:
                break
        mb = max(stats.pop("mb"), 1)
        upd = {k: round(v_ / mb, 5) for k, v_ in stats.items()}
        if upd["kl_ref"] > 1.5 * a.kl_target:
            beta *= 2.0
        elif upd["kl_ref"] < a.kl_target / 1.5:
            beta /= 2.0
        cmds = sum((Counter(l["commands"]) for l in lives), Counter())
        tot = max(sum(cmds.values()), 1)
        ful, fail = sum(l["fulfilled"] for l in lives), sum(l["failed"] for l in lives)
        rec = {"iteration": it, "rollout_s": round(roll_s, 1), "update_s": round(time.time() - t0 - roll_s, 1),
               "lives": len(lives), "steps": int(n), "mean_steps": float(np.mean([l["steps"] for l in lives])),
               "mean_unlocked": float(np.mean([l["n_unlocked"] for l in lives])),
               "mean_return": float(np.mean([l["return"] for l in lives])),
               "reward_crafter": float(batch["r_crafter"].sum() / len(lives)), "reward_caretaker": float(batch["r_care"].sum() / len(lives)),
               "requests": {"fulfilled": ful, "failed": fail}, "deaths": dict(Counter(l["cause"] for l in lives)),
               "command_share": {c: round(cmds.get(c, 0) / tot, 4) for c in COMMANDS},
               "cannot_rate": round(sum(l["cannot"] for l in lives) / tot, 4), "update": upd, "beta": beta,
               "early_stop": stopped, "logp_replay_gap": replay_gap,
               "red_flag_command_collapse": max(cmds.values()) / tot > 0.9}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        pol.cfg.temperature, pol.cfg.sample_salt = 0.0, 0
        ck = out / f"it{it:03d}"
        ck.mkdir()
        torch.save(pol.state_dict_full(), ck / "policy.pt")
        torch.save(value.state_dict(), ck / "value.pt")
        (ck / "run.json").write_text(json.dumps({**run, "iteration": it, "metrics": rec}, indent=1) + "\n")
        if dev_seeds and (it % a.dev_every == 0 or it == a.iterations):
            dev[it] = dev_eval(pack_weights(core, pol), home, dev_seeds, a.workers, a.max_steps, 10_000_000)["summary"]
            dev[it]["ratio_to_routine"] = {m: dev[it][m] / rt["summary"][m] for m in ("crafter_score", "survival_steps")}
            print(json.dumps({"dev": it, **dev[it]}), flush=True)
            (out / "dev.json").write_text(json.dumps(dev, indent=1) + "\n")
    # preregistered selection: best development survival among checkpoints whose development score >= r3h-B's
    if dev_seeds:
        ok = [i for i in dev if i > 0 and dev[i]["crafter_score"] >= dev[0]["crafter_score"]]
        best = max(ok, key=lambda i: dev[i]["survival_steps"]) if ok else None
        sel = {"rule": "best development survival among checkpoints with development score >= r3h-B's (iteration 0)",
               "eligible": ok, "selected": best,
               "selected_dev": dev.get(best), "r3h-B_dev": dev[0],
               "beats_r3hB_on_dev_survival": bool(best is not None and dev[best]["survival_steps"] > dev[0]["survival_steps"])}
        (out / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
        print(json.dumps({"selection": sel}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--individuals", required=True, help="a copy of the frozen r3h-B candidate")
    t.add_argument("--individual", default="cb2-ind-0001")
    t.add_argument("--out", required=True)
    t.add_argument("--iterations", type=int, default=50)
    t.add_argument("--lives", type=int, default=64)
    t.add_argument("--workers", type=int, default=32)
    t.add_argument("--max-steps", type=int, default=10000)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--minibatch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--lam", type=float, default=0.95)
    t.add_argument("--beta", type=float, default=0.1)
    t.add_argument("--kl-target", type=float, default=0.02)
    t.add_argument("--max-kl-old", type=float, default=0.05)
    t.add_argument("--entropy", type=float, default=0.001)
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--dev-seeds", type=int, nargs=2, default=[76001, 76040], help="first last (0 0: none)")
    t.add_argument("--dev-every", type=int, default=5)
    t.set_defaults(func=train)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
