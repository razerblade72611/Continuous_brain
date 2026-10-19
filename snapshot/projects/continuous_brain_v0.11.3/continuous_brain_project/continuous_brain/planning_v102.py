"""v0.10.2 child-owned temporal options / intentions.

The recurrent planning latent introduced in v0.10.0 and the conservative Bellman
Q repair from v0.10.1 are preserved.  v0.10.2 adds temporal abstraction learned
from successful lived trajectories without gap/wall labels or hand-coded phase
names.

Successful sequences are described by short future displacement/action statistics
and clustered into anonymous latent option IDs.  The new OptionCortex learns option
selection, continuation/termination, an option-conditioned primitive policy, and
option-conditioned action values.  During proof training every pre-v0.10.2 tensor
is frozen bit-for-bit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .environment import DetourWorld
from .infant import InfantBrain, one_hot_action
from .parented_data import NurseryDataset
from .parented_train import load_checkpoint, save_checkpoint
from .planning_v100 import (
    _complete_episode_rows,
    _split_episodes,
    _extract_episode_features,
    _env_image_tensor,
    _env_context,
)


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(statistics.fmean(xs)) if xs else 0.0


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _configure_options_only(model: InfantBrain) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.options.parameters():
        p.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _successful(ep: dict) -> bool:
    return bool(float(ep["success"].max().item()) > 0.5)


def _future_descriptor(ep: dict, t: int, lookahead: int, action_dim: int) -> np.ndarray:
    """Anonymous short-horizon motion descriptor used only for option discovery.

    No wall/gap coordinate or semantic phase label appears here.  The descriptor
    summarizes what changed over the following lived chunk and what primitive
    actions occurred in that chunk.
    """
    n = len(ep["action"])
    end = min(n - 1, int(t) + max(1, int(lookahead)) - 1)
    start_ctx = ep["context"][t].numpy()
    end_ctx = ep["next_context"][end].numpy()
    delta = end_ctx[:4] - start_ctx[:4]
    actions = ep["action"][t:end + 1].numpy().astype(np.int64)
    hist = np.bincount(actions, minlength=action_dim).astype(np.float32)
    hist /= max(1.0, float(hist.sum()))
    rewards = ep["reward"][t:end + 1].numpy()
    duration = np.array([(end - t + 1) / max(1.0, float(lookahead))], dtype=np.float32)
    reward_sum = np.array([float(rewards.sum())], dtype=np.float32)
    return np.concatenate((delta.astype(np.float32), hist, duration, reward_sum), axis=0)


def _kmeans(x: np.ndarray, k: int, seed: int, iterations: int = 60):
    if x.ndim != 2 or len(x) < k:
        raise RuntimeError(f"Option discovery needs at least {k} successful sequence states; got {len(x)}")
    rng = np.random.default_rng(int(seed))
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xn = (x - mean) / std
    # Deterministic far-spread initialization: random first point, then farthest.
    first = int(rng.integers(0, len(xn)))
    centers = [xn[first].copy()]
    while len(centers) < int(k):
        d2 = np.min(np.stack([((xn - c) ** 2).sum(axis=1) for c in centers], axis=1), axis=1)
        centers.append(xn[int(np.argmax(d2))].copy())
    centers = np.stack(centers)
    labels = np.zeros(len(xn), dtype=np.int64)
    for _ in range(int(iterations)):
        dist = ((xn[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(int(k)):
            members = xn[new_labels == j]
            if len(members):
                new_centers[j] = members.mean(axis=0)
            else:
                far = np.max(np.min(dist, axis=1))
                idx = int(np.argmax(np.min(dist, axis=1))) if np.isfinite(far) else int(rng.integers(0, len(xn)))
                new_centers[j] = xn[idx]
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers, atol=1e-6):
            labels = new_labels
            centers = new_centers
            break
        labels, centers = new_labels, new_centers
    return centers.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _assign_descriptor(desc: np.ndarray, centers: np.ndarray, mean: np.ndarray, std: np.ndarray) -> int:
    z = (desc - mean) / std
    return int(np.argmin(((centers - z[None]) ** 2).sum(axis=1)))


def _smooth_labels(labels: list[int], radius: int = 2) -> list[int]:
    if len(labels) <= 2:
        return list(labels)
    out = []
    for i in range(len(labels)):
        lo, hi = max(0, i - radius), min(len(labels), i + radius + 1)
        window = labels[lo:hi]
        counts = {v: window.count(v) for v in set(window)}
        out.append(max(counts, key=lambda v: (counts[v], -abs(v - labels[i]))))
    return out


def _discover_options(train_features: list[dict], option_count: int, lookahead: int,
                      action_dim: int, seed: int):
    descs = []
    for ep in train_features:
        if not _successful(ep):
            continue
        for t in range(len(ep["action"])):
            descs.append(_future_descriptor(ep, t, lookahead, action_dim))
    if not descs:
        raise RuntimeError("Option discovery requires successful lived training episodes")
    x = np.stack(descs)
    centers, mean, std = _kmeans(x, int(option_count), int(seed))
    return centers, mean, std


def _label_features(features: list[dict], centers, mean, std, lookahead: int, action_dim: int):
    labeled = []
    for ep in features:
        if not _successful(ep):
            continue
        raw = [
            _assign_descriptor(_future_descriptor(ep, t, lookahead, action_dim), centers, mean, std)
            for t in range(len(ep["action"]))
        ]
        labels = _smooth_labels(raw, radius=2)
        item = dict(ep)
        item["option"] = torch.tensor(labels, dtype=torch.long)
        terminate = []
        for t, label in enumerate(labels):
            terminate.append(float(t == len(labels) - 1 or labels[min(t + 1, len(labels) - 1)] != label))
        item["terminate"] = torch.tensor(terminate, dtype=torch.float32)
        labeled.append(item)
    return labeled


@torch.no_grad()
def _precompute_latents(model: InfantBrain, features: list[dict], device: str):
    out = []
    for ep in features:
        item = dict(ep)
        item["latent"] = model.planning.encode_state(ep["spatial"].to(device), ep["context"].to(device)).cpu()
        item["next_latent"] = model.planning.encode_state(ep["next_spatial"].to(device), ep["next_context"].to(device)).cpu()
        out.append(item)
    return out


def _flatten(features: list[dict]):
    rows = []
    for ep in features:
        n = len(ep["action"])
        for t in range(n):
            rows.append({
                "z": ep["latent"][t],
                "next_z": ep["next_latent"][t],
                "option": ep["option"][t],
                "action": ep["action"][t],
                "terminate": ep["terminate"][t],
                "return": ep["return"][t],
                "reward": ep["reward"][t],
            })
    return rows


def _batch(rows: list[dict], indices: list[int], device: str):
    keys = ("z", "next_z", "option", "action", "terminate", "return", "reward")
    out = {}
    for key in keys:
        out[key] = torch.stack([rows[i][key] for i in indices]).to(device)
    return out


def _option_q_all(model: InfantBrain, z: torch.Tensor, option: torch.Tensor) -> torch.Tensor:
    b = z.shape[0]
    a = int(model.cfg.action_dim)
    zrep = z[:, None, :].expand(b, a, z.shape[-1]).reshape(b * a, -1)
    orep = option[:, None].expand(b, a).reshape(-1)
    actions = torch.arange(a, device=z.device)[None, :].expand(b, a).reshape(-1)
    aoh = one_hot_action(actions, a)
    return model.options.action_value(zrep, orep, aoh).reshape(b, a)


def _loss(model: InfantBrain, b: dict, *, q_margin: float = 0.08):
    opt = model.options
    selector = opt.selector_logits(b["z"])
    selector_loss = F.cross_entropy(selector, b["option"].long())
    term = opt.termination_logit(b["z"], b["option"].long())
    pos_weight = torch.tensor(4.0, device=term.device)
    termination_loss = F.binary_cross_entropy_with_logits(term, b["terminate"], pos_weight=pos_weight)
    action_logits = opt.action_logits(b["z"], b["option"].long())
    action_loss = F.cross_entropy(action_logits, b["action"].long())
    qall = _option_q_all(model, b["z"], b["option"].long())
    qtaken = qall.gather(1, b["action"].long()[:, None]).squeeze(1)
    q_loss = F.smooth_l1_loss(qtaken, b["return"])
    mask = F.one_hot(b["action"].long(), num_classes=model.cfg.action_dim).bool()
    alt = qall.masked_fill(mask, -1e9).max(dim=1).values
    rank_loss = F.relu(float(q_margin) + alt - qtaken).mean()
    total = selector_loss + 0.70 * termination_loss + action_loss + 0.70 * q_loss + 0.70 * rank_loss
    metrics = {
        "loss": float(total.detach()),
        "selector": float(selector_loss.detach()),
        "termination": float(termination_loss.detach()),
        "action": float(action_loss.detach()),
        "q": float(q_loss.detach()),
        "rank": float(rank_loss.detach()),
    }
    return total, metrics


@torch.no_grad()
def _validation(model: InfantBrain, rows: list[dict], batch_size: int, device: str) -> dict:
    model.options.eval()
    selector_ok, action_ok, term_tp, term_fp, term_fn, q_abs, rank_ok = [], [], 0, 0, 0, [], []
    option_counts = {str(i): 0 for i in range(model.cfg.option_count)}
    for start in range(0, len(rows), max(1, int(batch_size))):
        inds = list(range(start, min(len(rows), start + max(1, int(batch_size)))))
        b = _batch(rows, inds, device)
        sl = model.options.selector_logits(b["z"])
        selector_ok.extend(sl.argmax(dim=1).eq(b["option"].long()).float().cpu().tolist())
        al = model.options.action_logits(b["z"], b["option"].long())
        action_ok.extend(al.argmax(dim=1).eq(b["action"].long()).float().cpu().tolist())
        pred_t = torch.sigmoid(model.options.termination_logit(b["z"], b["option"].long())).ge(0.5)
        true_t = b["terminate"].ge(0.5)
        term_tp += int((pred_t & true_t).sum().item())
        term_fp += int((pred_t & ~true_t).sum().item())
        term_fn += int((~pred_t & true_t).sum().item())
        qall = _option_q_all(model, b["z"], b["option"].long())
        qt = qall.gather(1, b["action"].long()[:, None]).squeeze(1)
        q_abs.extend((qt - b["return"]).abs().cpu().tolist())
        rank_ok.extend(qall.argmax(dim=1).eq(b["action"].long()).float().cpu().tolist())
        for o in b["option"].cpu().tolist(): option_counts[str(int(o))] += 1
    precision = term_tp / max(1, term_tp + term_fp)
    recall = term_tp / max(1, term_tp + term_fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "selector_accuracy": _mean(selector_ok),
        "action_accuracy": _mean(action_ok),
        "termination_f1": float(f1),
        "termination_precision": float(precision),
        "termination_recall": float(recall),
        "option_q_mae": _mean(q_abs),
        "executed_action_rank_accuracy": _mean(rank_ok),
        "option_counts": option_counts,
        "samples": len(rows),
    }


def train_temporal_options(dataset_path: str | Path, state_dir: str | Path, *,
                           updates=3000, batch_size=256, lookahead=6,
                           gamma=0.95, goal_gamma=0.97, learning_rate=3e-4,
                           q_margin=0.08, seed=10210, device="cpu") -> dict:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    dataset = NurseryDataset(dataset_path)
    episodes, dropped = _complete_episode_rows(dataset.records())
    train_eps, val_eps = _split_episodes(episodes, int(seed))
    state_dir = Path(state_dir)
    checkpoint = state_dir / "infant_training.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Option training requires an existing v0.10.1 child checkpoint: {checkpoint}")
    before_hash = _sha(checkpoint)
    model, saved = load_checkpoint(checkpoint, device)
    metadata = dict(saved.get("metadata") or {})
    if metadata.get("planning_value_version") != "conservative-bellman-q-v1":
        raise RuntimeError("v0.10.2 option training requires the v0.10.1 conservative Bellman Q checkpoint")
    source_updates = int(saved.get("updates", metadata.get("updates", 0)))
    missing_options = list(saved.get("_v102_missing_keys") or [])

    before_non_option = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        if not k.startswith("options.")
    }
    trainable = _configure_options_only(model)
    model.eval(); model.options.train()

    print("Extracting lived successful sequences for temporal option discovery...")
    train_features = _extract_episode_features(model, dataset, train_eps, device, float(gamma), float(goal_gamma))
    val_features = _extract_episode_features(model, dataset, val_eps, device, float(gamma), float(goal_gamma))
    centers, desc_mean, desc_std = _discover_options(
        train_features, model.cfg.option_count, int(lookahead), model.cfg.action_dim, int(seed)
    )
    train_labeled = _label_features(train_features, centers, desc_mean, desc_std, int(lookahead), model.cfg.action_dim)
    val_labeled = _label_features(val_features, centers, desc_mean, desc_std, int(lookahead), model.cfg.action_dim)
    train_labeled = _precompute_latents(model, train_labeled, device)
    val_labeled = _precompute_latents(model, val_labeled, device)
    train_rows = _flatten(train_labeled)
    val_rows = _flatten(val_labeled)
    if not train_rows or not val_rows:
        raise RuntimeError("Option training needs successful episodes in both train and validation splits")

    optimizer = torch.optim.AdamW(model.options.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    rng = random.Random(int(seed) + 31)
    best_state = None
    best_score = float("inf")
    best_update = 0
    latest = {}
    for step in range(1, int(updates) + 1):
        model.options.train()
        inds = [rng.randrange(len(train_rows)) for _ in range(int(batch_size))]
        b = _batch(train_rows, inds, device)
        optimizer.zero_grad(set_to_none=True)
        loss, latest = _loss(model, b, q_margin=float(q_margin))
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.options.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == int(updates):
            val = _validation(model, val_rows, int(batch_size), device)
            # Lower is better; all terms are dimensionless or return-unit errors.
            score = (
                (1.0 - val["selector_accuracy"]) +
                (1.0 - val["action_accuracy"]) +
                (1.0 - val["termination_f1"]) +
                0.30 * val["option_q_mae"] +
                0.50 * (1.0 - val["executed_action_rank_accuracy"])
            )
            if score < best_score:
                best_score = float(score)
                best_update = int(step)
                best_state = {k: v.detach().cpu().clone() for k, v in model.options.state_dict().items()}
            print(
                f"option_update={step}/{updates} loss={latest['loss']:.5f} "
                f"sel={latest['selector']:.4f} term={latest['termination']:.4f} "
                f"act={latest['action']:.4f} q={latest['q']:.4f} rank={latest['rank']:.4f} grad={grad:.3f} "
                f"val_sel={val['selector_accuracy']:.3f} val_act={val['action_accuracy']:.3f} "
                f"val_term_f1={val['termination_f1']:.3f} val_rank={val['executed_action_rank_accuracy']:.3f}"
            )
    if best_state is not None:
        model.options.load_state_dict(best_state)
    model.eval()
    validation = _validation(model, val_rows, int(batch_size), device)

    changed_non_option = [
        k for k, v in before_non_option.items()
        if not torch.equal(v, model.state_dict()[k].detach().cpu())
    ]
    if changed_non_option:
        raise RuntimeError(f"Option isolation failure; non-option tensors changed: {changed_non_option[:20]}")

    for p in model.parameters(): p.requires_grad_(True)
    for p in model.target_vision.parameters(): p.requires_grad_(False)
    for p in model.target_spatial.parameters(): p.requires_grad_(False)

    previous = int(metadata.get("option_updates", 0))
    metadata.update({
        "option_cortex_version": "latent-temporal-options-v1",
        "option_updates": previous + int(updates),
        "option_best_update": int(best_update),
        "option_count": int(model.cfg.option_count),
        "option_descriptor_lookahead": int(lookahead),
        "option_discovery": "unsupervised-kmeans-on-successful-lived-future-motion-descriptors",
        "option_training_dataset_digest": dataset.digest(),
        "planning_cortex_version": "recurrent-latent-v1+conservative-bellman-q-v1+temporal-options-v1",
    })
    parent_id = str(metadata.get("parent_id", "vision.dinov3.base"))
    save_checkpoint(
        checkpoint, model, updates=source_updates, parent_id=parent_id,
        dataset_digest=str(metadata.get("dataset_digest", dataset.digest())), optimizer=None, metadata=metadata,
    )
    after_hash = _sha(checkpoint)
    return {
        "format": "continuous_brain.temporal_option_training.v102",
        "checkpoint": str(checkpoint),
        "source_training_updates": source_updates,
        "planning_updates": int(metadata.get("planning_updates", 0)),
        "planning_bellman_q_updates": int(metadata.get("planning_bellman_q_updates", 0)),
        "option_updates": previous + int(updates),
        "option_cortex_version": metadata["option_cortex_version"],
        "option_only": True,
        "option_only_trainable_parameters": trainable,
        "option_parameters": sum(p.numel() for p in model.options.parameters()),
        "changed_non_option_tensors": len(changed_non_option),
        "option_migrated": bool(missing_options),
        "option_migration_keys": missing_options,
        "episodes_complete": len(episodes),
        "successful_train_episodes": len(train_labeled),
        "successful_validation_episodes": len(val_labeled),
        "dropped_incomplete_episode_ids": dropped,
        "samples_train": len(train_rows),
        "samples_validation": len(val_rows),
        "best_update": int(best_update),
        "training": {
            "updates": int(updates), "batch": int(batch_size), "lookahead": int(lookahead),
            "gamma": float(gamma), "goal_gamma": float(goal_gamma),
            "learning_rate": float(learning_rate), "q_margin": float(q_margin), "seed": int(seed),
            "option_labels": "anonymous clusters discovered from successful lived sequences",
            "detour_geometry_labels_used": False,
        },
        "validation": validation,
        "source_checkpoint_sha256": before_hash,
        "trained_checkpoint_sha256": after_hash,
    }


def _resolve_option(model: InfantBrain, latent: torch.Tensor, current: int | None,
                    age: int, min_commit: int, terminate_threshold: float):
    opt = model.options
    if current is None:
        selected = int(opt.selector_logits(latent).argmax(dim=-1)[0].item())
        return selected, 0, True
    o = torch.tensor([int(current)], dtype=torch.long, device=latent.device)
    should = bool(torch.sigmoid(opt.termination_logit(latent, o))[0].item() >= float(terminate_threshold))
    if int(age) >= int(min_commit) and should:
        selected = int(opt.selector_logits(latent).argmax(dim=-1)[0].item())
        return selected, 0, selected != int(current)
    return int(current), int(age), False


@torch.no_grad()
def _option_plan_action(model: InfantBrain, spatial: torch.Tensor, context7: torch.Tensor,
                        current_option: int | None, option_age: int, *, horizon: int,
                        beam_width: int, discount: float, min_commit: int,
                        terminate_threshold: float, device: str):
    planner, options = model.planning, model.options
    action_dim = int(model.cfg.action_dim)
    root = planner.encode_state(spatial, context7)
    root_option, root_age, switched = _resolve_option(
        model, root, current_option, option_age, min_commit, terminate_threshold
    )
    beam = [{"latent": root, "score": 0.0, "seq": [], "option": root_option, "age": root_age}]
    for depth in range(int(horizon)):
        expanded = []
        for node in beam:
            z = node["latent"]
            option = int(node["option"])
            actions = torch.arange(action_dim, device=device)
            aoh = one_hot_action(actions, action_dim)
            zrep = z.expand(action_dim, -1)
            pred = planner.step(zrep, aoh)
            for action in range(action_dim):
                nz = pred["latent"][action:action+1]
                nage = int(node["age"]) + 1
                noption, nage2, _ = _resolve_option(
                    model, nz, option, nage, min_commit, terminate_threshold
                )
                score = float(node["score"]) + (float(discount) ** depth) * float(pred["reward"][action].item())
                expanded.append({
                    "latent": nz, "score": score, "seq": node["seq"] + [int(action)],
                    "option": int(noption), "age": int(nage2),
                })
        for node in expanded:
            z = node["latent"]
            orepeat = torch.full((action_dim,), int(node["option"]), dtype=torch.long, device=device)
            zrep = z.expand(action_dim, -1)
            all_a = one_hot_action(torch.arange(action_dim, device=device), action_dim)
            tail = options.action_value(zrep, orepeat, all_a).max().item()
            node["rank"] = float(node["score"]) + (float(discount) ** (depth + 1)) * float(tail)
        expanded.sort(key=lambda x: x["rank"], reverse=True)
        beam = expanded[:max(1, int(beam_width))]
    best = beam[0]
    return int(best["seq"][0]), int(root_option), int(root_age), bool(switched), float(best["rank"]), list(best["seq"])


@torch.no_grad()
def _run_option_episode(model: InfantBrain, seed: int, mode: str, *, horizon: int,
                        beam_width: int, max_steps: int, discount: float,
                        min_commit: int, terminate_threshold: float, device: str) -> dict:
    env = DetourWorld(seed=int(seed), max_episode_steps=int(max_steps))
    obs = env.reset()
    hidden = model.nucleus.initial_state(1, device)
    memory = torch.zeros(1, model.cfg.memory_dim, dtype=torch.float32, device=device)
    previous_action = torch.zeros(1, dtype=torch.long, device=device)
    current_option = None
    option_age = 0
    option_switches = 0
    option_counts = {str(i): 0 for i in range(model.cfg.option_count)}
    action_counts = {str(i): 0 for i in range(model.cfg.action_dim)}
    total_reward = 0.0
    completed = False
    steps_used = int(max_steps)
    planner_overrides = 0
    for step in range(1, int(max_steps) + 1):
        image = _env_image_tensor(env, model.cfg.image_size, device)
        context, context7 = _env_context(obs, device)
        semantic = model.vision(image)
        spatial_online = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial_online)
        nuc = model.nucleus(fused, context, memory, previous_action, hidden)
        hidden = nuc["hidden"]
        reactive = int(torch.argmax(model.motor(nuc["workspace"]), dim=-1)[0].item())
        stable_spatial = model.target_spatial_bus(image)
        root = model.planning.encode_state(stable_spatial, context7)
        if mode == "option_direct":
            current_option, option_age, switched = _resolve_option(
                model, root, current_option, option_age, min_commit, terminate_threshold
            )
            o = torch.tensor([current_option], dtype=torch.long, device=device)
            action = int(model.options.action_logits(root, o).argmax(dim=-1)[0].item())
        elif mode == "option_planning":
            action, current_option, option_age, switched, _, _ = _option_plan_action(
                model, stable_spatial, context7, current_option, option_age,
                horizon=int(horizon), beam_width=int(beam_width), discount=float(discount),
                min_commit=int(min_commit), terminate_threshold=float(terminate_threshold), device=device,
            )
        elif mode == "reactive":
            action = reactive
            switched = False
        else:
            raise ValueError(mode)
        if mode != "reactive":
            option_counts[str(int(current_option))] += 1
            option_switches += int(switched)
            planner_overrides += int(action != reactive)
            option_age += 1
        action_counts[str(int(action))] += 1
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        previous_action = torch.tensor([action + 1], dtype=torch.long, device=device)
        if bool(info.get("reached")):
            completed = True
            steps_used = step
        if done:
            break
    return {
        "seed": int(seed), "mode": mode, "horizon": int(horizon if mode == "option_planning" else 0),
        "reward": float(total_reward), "completed": bool(completed), "steps": int(steps_used),
        "collisions": int(env.collisions), "final_energy": float(env.energy),
        "planner_overrides": int(planner_overrides), "option_switches": int(option_switches),
        "option_counts": option_counts, "action_counts": action_counts,
    }


def _summary(rows, max_steps):
    option_switches = [r.get("option_switches", 0) for r in rows]
    return {
        "episodes": len(rows),
        "completion_rate": _mean(float(r["completed"]) for r in rows),
        "mean_reward": _mean(r["reward"] for r in rows),
        "mean_steps_penalized": _mean(r["steps"] if r["completed"] else max_steps + 1 for r in rows),
        "mean_collisions": _mean(r["collisions"] for r in rows),
        "mean_planner_overrides": _mean(r.get("planner_overrides", 0) for r in rows),
        "mean_option_switches": _mean(option_switches),
    }


def evaluate_temporal_options(checkpoint: str | Path, *, episodes=10, steps=120,
                              seed_start=9001, horizons=(1, 3, 6, 12), beam_width=32,
                              discount=0.95, min_commit=3, terminate_threshold=0.50,
                              device="cpu", output=None) -> dict:
    model, saved = load_checkpoint(checkpoint, device)
    model.eval()
    metadata = dict(saved.get("metadata") or {})
    if metadata.get("option_cortex_version") != "latent-temporal-options-v1":
        raise RuntimeError("Checkpoint has not completed v0.10.2 temporal-option training")
    reactive, direct = [], []
    planned = {int(h): [] for h in horizons}
    for i in range(int(episodes)):
        seed = int(seed_start) + i
        reactive.append(_run_option_episode(
            model, seed, "reactive", horizon=0, beam_width=beam_width, max_steps=steps,
            discount=discount, min_commit=min_commit, terminate_threshold=terminate_threshold, device=device,
        ))
        direct.append(_run_option_episode(
            model, seed, "option_direct", horizon=0, beam_width=beam_width, max_steps=steps,
            discount=discount, min_commit=min_commit, terminate_threshold=terminate_threshold, device=device,
        ))
        for h in horizons:
            planned[int(h)].append(_run_option_episode(
                model, seed, "option_planning", horizon=int(h), beam_width=beam_width, max_steps=steps,
                discount=discount, min_commit=min_commit, terminate_threshold=terminate_threshold, device=device,
            ))
        if (i + 1) % 10 == 0:
            print(f"paired_v102_option_episodes={i+1}/{episodes}")
    r = _summary(reactive, int(steps))
    d = _summary(direct, int(steps))
    horizon_results = {}
    any_pass = False
    any_completion = False
    for h, rows in planned.items():
        m = _summary(rows, int(steps))
        completion_adv = m["completion_rate"] - r["completion_rate"]
        collision_regression = m["mean_collisions"] - r["mean_collisions"]
        passed = bool(m["completion_rate"] >= 0.20 and completion_adv > 0.05 and collision_regression <= 2.0)
        any_pass = any_pass or passed
        any_completion = any_completion or m["completion_rate"] > 0.0
        horizon_results[str(h)] = {
            "option_planning": m,
            "advantages_vs_reactive": {
                "completion_percentage_points": 100.0 * completion_adv,
                "reward_absolute": m["mean_reward"] - r["mean_reward"],
                "collisions_reduced": r["mean_collisions"] - m["mean_collisions"],
            },
            "p4b_passed": passed,
        }
    result = {
        "format": "continuous_brain.p4b_temporal_options_gate.v102",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "source_training_updates": int(saved.get("updates", metadata.get("updates", 0))),
        "planning_updates": int(metadata.get("planning_updates", 0)),
        "planning_bellman_q_updates": int(metadata.get("planning_bellman_q_updates", 0)),
        "option_updates": int(metadata.get("option_updates", 0)),
        "option_cortex_version": metadata.get("option_cortex_version"),
        "experiment": {
            "paired_episodes": int(episodes), "steps_per_episode": int(steps),
            "seed_start": int(seed_start), "seed_end": int(seed_start) + int(episodes) - 1,
            "horizons": [int(h) for h in horizons], "beam_width": int(beam_width),
            "discount": float(discount), "min_commit": int(min_commit),
            "terminate_threshold": float(terminate_threshold),
            "parents_used": False, "training_performed_during_evaluation": False,
            "memory_written": False,
            "planner": "dedicated recurrent imagination conditioned by learned temporal option",
            "option_semantics": "anonymous clusters learned from successful lived sequences; no gap/wall labels",
        },
        "reactive": r,
        "option_direct_diagnostic": d,
        "horizons": horizon_results,
        "thresholds": {
            "absolute_completion": 0.20,
            "completion_advantage": 0.05,
            "max_collision_regression_per_episode": 2.0,
        },
        "any_option_planning_completion": bool(any_completion),
        "p4b_passed": bool(any_pass),
        "episodes": {
            "reactive": reactive, "option_direct": direct,
            **{f"h{h}": rows for h, rows in planned.items()},
        },
    }
    if output:
        out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
