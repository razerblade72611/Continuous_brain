"""Lived sequence replay, demonstration rehearsal, and paired promotion gates."""
from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from .memory import ExperienceMemory, BehaviorHistory
from .model import tensor, observation_tensors, finite_step, fingerprint
from .world import ResourceWorld, teacher_action

# These are reusable development sets, NOT a sealed generalization test.
# The v0.11 blind range 51001--51100 is never used here.
EVAL_SEEDS = {"open": tuple(range(700001, 700007)),
              "detour": tuple(range(710001, 710007))}


def experience(obs, memory, style, action, reward, nxt, info, done, teacher=-1):
    return {"grid": obs["grid"].astype(np.uint8), "sensors": obs["sensors"].copy(),
            "memory": memory.copy(), "style": style.copy(), "action": int(action),
            "teacher": int(teacher), "reward": float(reward),
            "next_grid": nxt["grid"].astype(np.uint8), "next_sensors": nxt["sensors"].copy(),
            "body": np.array(info["body_after"], np.float32),
            "collision": float(info["collision"]),
            "resource": {None: 0, "food": 1, "water": 2}[info["resource"]],
            "done": float(done)}


def demonstrations(episodes=96, horizon=64, seed=1000001):
    rng = random.Random(seed)
    traces = []
    for index in range(episodes):
        env = ResourceWorld(seed + index, kind="open" if index % 2 == 0 else "detour", horizon=horizon)
        mem, style = ExperienceMemory(), BehaviorHistory()
        obs = env.observe()
        rows = []
        for _ in range(horizon):
            label = teacher_action(env)
            action = rng.randrange(5) if rng.random() < .12 else label
            memory, traits = mem.retrieve(obs["sensors"]), style.vector()
            nxt, reward, done, info = env.step(action)
            rows.append(experience(obs, memory, traits, action, reward, nxt, info, done, label))
            mem.add(obs["sensors"], action, reward, info)
            style.observe(action, info)
            obs = nxt
            if done:
                break
        traces.append(rows)
    return traces


@torch.no_grad()
def correction_demonstrations(model, *, episodes=64, horizon=64, seed=1500001):
    """Training-only feedback on states visited by the learner's own policy.

    Teacher labels are recorded but never executed in this collection pass.
    This addresses the distribution shift of demonstration-only initialization.
    """
    device = next(model.parameters()).device
    rng = random.Random(seed)
    model.eval()
    traces = []
    for index in range(episodes):
        env = ResourceWorld(seed + index, kind="open" if index % 2 == 0 else "detour", horizon=horizon)
        obs, mem, style = env.observe(), ExperienceMemory(), BehaviorHistory()
        hidden = model.initial(1, device)
        rows = []
        for _ in range(horizon):
            label = teacher_action(env)
            memory, traits = mem.retrieve(obs["sensors"]), style.vector()
            result = model(*observation_tensors(obs, memory, traits, device), hidden)
            hidden = result["hidden"]
            action = int(model.choose(result).cpu().argmax(-1)[0])
            if rng.random() < .15:
                action = rng.randrange(5)
            nxt, reward, done, info = env.step(action)
            rows.append(experience(obs, memory, traits, action, reward, nxt, info, done, label))
            mem.add(obs["sensors"], action, reward, info)
            style.observe(action, info)
            obs = nxt
            if done:
                break
        traces.append(rows)
    return traces


def _batch(traces, batch_size, length, rng, device):
    eligible = [trace for trace in traces if len(trace) >= length]
    if not eligible:
        raise ValueError("No replay sequences long enough for this update")
    chosen = []
    for _ in range(batch_size):
        trace = rng.choice(eligible)
        start = rng.randrange(len(trace) - length + 1)
        chosen.append(trace[start:start + length])
    keys = chosen[0][0].keys()
    return {key: tensor(np.asarray([[row[key] for row in rows] for rows in chosen]), device)
            for key in keys}


def train_block(model, teacher_traces, live_traces, *, updates, batch_size=8,
                sequence=12, lr=2e-4, rng=None, reference=None, optimizer=None,
                progress=None):
    if updates < 1 or not 1 <= batch_size <= 64 or not 4 <= sequence <= 32:
        raise ValueError("Invalid update, batch, or sequence budget")
    if not 0 < lr <= .001:
        raise ValueError("Learning rate must be in (0, 0.001]")
    rng = rng or random.Random(17)
    device = next(model.parameters()).device
    model.train()
    if reference is not None:
        reference.eval()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
    losses = []
    for update in range(updates):
        # Rehearsal and lived learning receive separate, explicit route turns.
        pool = live_traces if update % 2 and any(len(x) >= sequence for x in live_traces) else teacher_traces
        batch = _batch(pool, batch_size, sequence, rng, device)
        hidden = model.initial(batch_size, device)
        ref_hidden = model.initial(batch_size, device)
        burn_in = min(3, sequence // 3)
        loss = torch.zeros((), device=device)
        for t in range(sequence):
            args = tuple(batch[k][:, t] for k in ("grid", "sensors", "memory", "style"))
            if t < burn_in:
                with torch.no_grad():
                    hidden = model(*args, hidden)["hidden"]
                    if reference is not None:
                        ref_hidden = reference(*args, ref_hidden)["hidden"]
                continue
            out = model(*args, hidden)
            hidden = out["hidden"]
            action = batch["action"][:, t].long()
            # One-hot construction remains CPU-side on DirectML.
            onehot = F.one_hot(action.detach().cpu(), 5).float().to(device)
            pred = model.predict(hidden, onehot)
            labels = batch["teacher"][:, t].long()
            mask = (labels >= 0).float()
            bc = F.cross_entropy(out["logits"], labels.clamp(min=0), reduction="none")
            loss = loss + (bc * mask).sum() / mask.sum().clamp(min=1.)
            loss = loss + .4 * F.mse_loss(pred["body"], batch["body"][:, t])
            loss = loss + .25 * F.smooth_l1_loss(pred["reward"], batch["reward"][:, t])
            loss = loss + .15 * F.binary_cross_entropy_with_logits(pred["collision_logit"], batch["collision"][:, t])
            loss = loss + .15 * F.cross_entropy(pred["resource_logits"], batch["resource"][:, t].long())
            with torch.no_grad():
                target_model = reference or model
                if reference is not None:
                    ref = reference(*args, ref_hidden)
                    ref_hidden = ref["hidden"]
                    target_hidden = ref_hidden
                else:
                    target_hidden = hidden.detach()
                nxt = target_model(batch["next_grid"][:, t], batch["next_sensors"][:, t],
                                   batch["memory"][:, t], batch["style"][:, t], target_hidden)
                td = batch["reward"][:, t] + .97 * (1. - batch["done"][:, t]) * nxt["q"].max(-1).values
            q_value = (out["q"] * onehot).sum(-1)
            loss = loss + .2 * F.smooth_l1_loss(q_value, td)
            if reference is not None:
                # Keep reference decisions on rehearsal; world prediction may improve.
                anchors = F.kl_div(F.log_softmax(out["logits"], -1),
                                   F.softmax(ref["logits"], -1), reduction="none").sum(-1)
                loss = loss + .5 * (anchors * mask).sum() / mask.sum().clamp(min=1.)
        losses.append(finite_step(loss / (sequence - burn_in), model, optimizer))
        if progress:
            progress(update + 1, losses[-1])
    model.eval()
    return {"updates": updates, "mean_loss": float(np.mean(losses)),
            "last_loss": losses[-1], "lr": lr}, optimizer


@torch.no_grad()
def evaluate(model, *, horizon=64, seed_sets=None):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    records = []
    try:
        for kind, seeds in (seed_sets or EVAL_SEEDS).items():
            for seed in seeds:
                env = ResourceWorld(seed, kind=kind, horizon=horizon)
                obs, mem, style = env.observe(), ExperienceMemory(), BehaviorHistory()
                hidden = model.initial(1, device)
                prediction_error = []
                actions = []
                for _ in range(horizon):
                    memory, traits = mem.retrieve(obs["sensors"]), style.vector()
                    out = model(*observation_tensors(obs, memory, traits, device), hidden)
                    hidden = out["hidden"]
                    action = int(model.choose(out).detach().cpu().argmax(-1)[0])
                    onehot = F.one_hot(torch.tensor([action]), 5).float().to(device)
                    pred = model.predict(hidden, onehot)
                    nxt, reward, done, info = env.step(action)
                    prediction_error.append(float(np.square(pred["body"][0].cpu().numpy() - env.body).mean()))
                    mem.add(obs["sensors"], action, reward, info)
                    style.observe(action, info)
                    actions.append(action)
                    obs = nxt
                    if done:
                        break
                records.append({"seed": seed, "kind": kind,
                                "success": bool(env.harvests >= 2 and min(env.body) > 0),
                                "harvests": env.harvests, "return": env.total_return,
                                "collisions": env.collisions,
                                "body_prediction_mse": float(np.mean(prediction_error)),
                                "actions": actions})
    finally:
        model.train(was_training)
    summary = {}
    for kind in ("open", "detour"):
        rows = [r for r in records if r["kind"] == kind]
        if not rows:
            continue
        summary[kind] = {"successes": sum(r["success"] for r in rows), "episodes": len(rows),
                         "mean_return": float(np.mean([r["return"] for r in rows])),
                         "mean_collisions": float(np.mean([r["collisions"] for r in rows])),
                         "body_prediction_mse": float(np.mean([r["body_prediction_mse"] for r in rows]))}
    return {"model_sha256": fingerprint(model), "horizon": horizon,
            "seeds": {k: list(v) for k, v in (seed_sets or EVAL_SEEDS).items()},
            "teacher_used": False, "persistent_writes": False,
            "suite_role": "reused development admission suite; not a sealed test",
            "summary": summary, "episodes": records}


def promotion_gate(reference, candidate, founder, *, retention=.95):
    if not .5 <= retention <= 1.:
        raise ValueError("retention must be a proportion in [0.5, 1]")
    for other in (reference, founder):
        if candidate["seeds"] != other["seeds"] or candidate["horizon"] != other["horizon"]:
            raise ValueError("Promotion evaluations must use identical seeds and horizon")
    reasons, paired = [], {}
    for kind, new in candidate["summary"].items():
        for label, source in (("active", reference), ("founder", founder)):
            old = source["summary"][kind]
            if new["episodes"] != old["episodes"]:
                raise ValueError("Promotion episode counts differ")
            required = math.ceil(retention * old["successes"] - 1e-12)
            if new["successes"] < required:
                reasons.append(f"{kind}: retained {new['successes']}, requires {required} against {label}")
            if new["mean_return"] < old["mean_return"] - .10:
                reasons.append(f"{kind}: mean return regressed against {label}")
            if new["mean_collisions"] > old["mean_collisions"] + .5:
                reasons.append(f"{kind}: collision regression against {label}")
        old_by_seed = {r["seed"]: r for r in reference["episodes"] if r["kind"] == kind}
        new_rows = [r for r in candidate["episodes"] if r["kind"] == kind]
        paired[kind] = {"lost": [r["seed"] for r in new_rows if old_by_seed[r["seed"]]["success"] and not r["success"]],
                        "gained": [r["seed"] for r in new_rows if r["success"] and not old_by_seed[r["seed"]]["success"]]}
    improvements = []
    for kind, new in candidate["summary"].items():
        old = reference["summary"][kind]
        if new["successes"] > old["successes"] or new["mean_return"] > old["mean_return"] + .02:
            improvements.append(f"{kind}: control improvement")
        if new["body_prediction_mse"] < old["body_prediction_mse"] * .99:
            improvements.append(f"{kind}: body prediction improved by more than 1%")
    if not improvements:
        reasons.append("No measured control or prediction acquisition")
    return {"accepted": not reasons, "reasons": reasons, "improvements": improvements,
            "paired": paired, "retention_fraction": retention,
            "interpretation": "Finite development-suite rule, not a confidence guarantee"}


def forecast_features(evaluation, *, live_steps, blocks, updates, lr):
    a, b = evaluation["summary"]["open"], evaluation["summary"]["detour"]
    return np.array([a["successes"] / a["episodes"], b["successes"] / b["episodes"],
                     np.tanh(a["mean_return"] / 10), np.tanh(b["mean_return"] / 10),
                     min(1., a["body_prediction_mse"]), min(1., b["body_prediction_mse"]),
                     min(1., live_steps / 10000), min(1., blocks / 100),
                     updates / 64, lr / .001], np.float32)


def measured_deltas(before, after):
    result = []
    for kind in ("open", "detour"):
        a, b = before["summary"][kind], after["summary"][kind]
        result.append(b["successes"] / b["episodes"] - a["successes"] / a["episodes"])
    ar = np.mean([v["mean_return"] for v in after["summary"].values()])
    br = np.mean([v["mean_return"] for v in before["summary"].values()])
    result.append(float(np.clip((ar - br) / 10, -1, 1)))
    return result
