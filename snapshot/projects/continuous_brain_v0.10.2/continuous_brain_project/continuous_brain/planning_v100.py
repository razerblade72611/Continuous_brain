"""v0.10 dedicated child-owned planning cortex training and qualification.

The planning cortex is deliberately separate from the ordinary recurrent nucleus.
It is trained from contiguous lived transitions and learns a latent state that can
be advanced repeatedly under hypothetical actions without requiring a new real
image at every imagined step.

No external parent chooses actions here.  Training can be run in planning-only
mode so all pre-v0.10 child parameters remain bit-for-bit frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .environment import DetourWorld
from .infant import InfantBrain, one_hot_action
from .parented_data import NurseryDataset
from .parented_train import load_checkpoint, save_checkpoint


@dataclass
class PlanningTrainingConfig:
    updates: int = 4000
    batch_size: int = 32
    unroll: int = 12
    gamma: float = 0.95
    goal_gamma: float = 0.97
    learning_rate: float = 3e-4
    seed: int = 10010


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def _checkpoint_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_tensor(path: Path, size: int) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("planning training requires Pillow") from exc
    image = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.tensor(np.transpose(arr, (2, 0, 1)), dtype=torch.float32)


def _context7(row: dict, key: str = "context") -> np.ndarray:
    value = np.asarray(row[key], dtype=np.float32).reshape(-1)
    if value.size < 7:
        raise ValueError(f"{key} must contain at least 7 structured values")
    return value[:7].copy()


def _complete_episode_rows(rows: list[dict]) -> tuple[list[list[dict]], list[int]]:
    """Return complete contiguous episodes and IDs dropped as incomplete.

    Nursery collection resets after a terminal transition.  Therefore every
    episode except possibly the final ID is known complete because a later ID
    exists.  The final ID is retained only if it visibly reached the goal or
    reached the DetourWorld step limit.
    """
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if "episode_id" not in row or "episode_step" not in row:
            raise ValueError("v0.10 planning training requires episode_id and episode_step")
        grouped.setdefault(int(row["episode_id"]), []).append(row)
    if not grouped:
        return [], []
    ids = sorted(grouped)
    last_id = ids[-1]
    complete: list[list[dict]] = []
    dropped: list[int] = []
    for eid in ids:
        episode = sorted(grouped[eid], key=lambda r: int(r["episode_step"]))
        # Require contiguous recorded steps so multi-step training never bridges a gap.
        expected = list(range(int(episode[0]["episode_step"]), int(episode[0]["episode_step"]) + len(episode)))
        actual = [int(r["episode_step"]) for r in episode]
        if actual != expected:
            dropped.append(eid)
            continue
        final = episode[-1]
        terminal_visible = bool(float(final.get("success", 0.0)) > 0.5) or int(final["episode_step"]) >= 119
        if eid != last_id or terminal_visible:
            complete.append(episode)
        else:
            dropped.append(eid)
    return complete, dropped


def _split_episodes(episodes: list[list[dict]], seed: int, val_fraction: float = 0.20):
    """Episode-level stratified train/validation split."""
    rng = random.Random(int(seed))
    success = [ep for ep in episodes if any(float(r.get("success", 0.0)) > 0.5 for r in ep)]
    failed = [ep for ep in episodes if ep not in success]
    rng.shuffle(success); rng.shuffle(failed)

    def split_group(group):
        if not group:
            return [], []
        n_val = max(1, int(round(len(group) * float(val_fraction)))) if len(group) > 1 else 0
        n_val = min(n_val, max(0, len(group) - 1))
        return group[n_val:], group[:n_val]

    train_s, val_s = split_group(success)
    train_f, val_f = split_group(failed)
    train = train_s + train_f
    val = val_s + val_f
    rng.shuffle(train); rng.shuffle(val)
    if not val and len(train) > 1:
        val.append(train.pop())
    return train, val


def _compute_returns(episode: list[dict], gamma: float, goal_gamma: float) -> tuple[list[float], list[float]]:
    rewards = [float(r.get("reward", 0.0)) for r in episode]
    returns = [0.0] * len(rewards)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + float(gamma) * acc
        returns[i] = float(acc)

    success_index = None
    for i, row in enumerate(episode):
        if float(row.get("success", 0.0)) > 0.5:
            success_index = i
            break
    goal = [0.0] * len(episode)
    if success_index is not None:
        for i in range(success_index + 1):
            remaining = max(0, success_index - i)
            goal[i] = float(2.0 * (float(goal_gamma) ** remaining))
    return returns, goal


@torch.no_grad()
def _extract_episode_features(model: InfantBrain, dataset: NurseryDataset,
                              episodes: list[list[dict]], device: str,
                              gamma: float, goal_gamma: float) -> list[dict]:
    """Cache child-owned dorsal targets and structured sequence targets."""
    model.eval()
    out = []
    count = 0
    for episode in episodes:
        returns, goals = _compute_returns(episode, gamma, goal_gamma)
        cur_images = []
        next_images = []
        for row in episode:
            cur_images.append(_image_tensor(dataset.directory / row["image"], model.cfg.image_size))
            next_images.append(_image_tensor(dataset.directory / row["next_image"], model.cfg.image_size))
        cur = torch.stack(cur_images).to(device)
        nxt = torch.stack(next_images).to(device)
        # Stable child-owned planning coordinates; no DINO/foundation model participates.
        cur_spatial = model.target_spatial_bus(cur).cpu()
        next_spatial = model.target_spatial_bus(nxt).cpu()
        out.append({
            "episode_id": int(episode[0]["episode_id"]),
            "spatial": cur_spatial,
            "next_spatial": next_spatial,
            "context": torch.tensor(np.stack([_context7(r, "context") for r in episode]), dtype=torch.float32),
            "next_context": torch.tensor(np.stack([_context7(r, "next_context") for r in episode]), dtype=torch.float32),
            "action": torch.tensor([int(r.get("executed_action", r.get("action", 0))) for r in episode], dtype=torch.long),
            "reward": torch.tensor([float(r.get("reward", 0.0)) for r in episode], dtype=torch.float32),
            "success": torch.tensor([float(r.get("success", 0.0)) for r in episode], dtype=torch.float32),
            "collision": torch.tensor([float(bool(r.get("collision", False))) for r in episode], dtype=torch.float32),
            "return": torch.tensor(returns, dtype=torch.float32),
            "goal": torch.tensor(goals, dtype=torch.float32),
        })
        count += len(episode)
        if count // 500 != (count - len(episode)) // 500:
            print(f"planning_features={count}")
    return out


def _configure_planning_only(model: InfantBrain) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.planning.parameters():
        p.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _sample_windows(features: list[dict], batch_size: int, unroll: int,
                    rng: random.Random) -> list[tuple[dict, int, int]]:
    windows = []
    eligible = [ep for ep in features if len(ep["action"]) >= 1]
    if not eligible:
        raise RuntimeError("No usable planning sequences")
    for _ in range(int(batch_size)):
        ep = rng.choice(eligible)
        start = rng.randrange(len(ep["action"]))
        length = min(int(unroll), len(ep["action"]) - start)
        windows.append((ep, start, length))
    return windows


def _planner_loss(model: InfantBrain, windows, device: str) -> tuple[torch.Tensor, dict]:
    """Vectorized multi-step self-rollout loss from lived trajectories.

    A whole minibatch of sequence windows advances through one recurrent planning
    step per depth.  Ground-truth next encodings supervise the predicted latent,
    but the next rollout depth always starts from the model's own predicted latent.
    """
    planner = model.planning
    batch = len(windows)
    if batch == 0:
        raise RuntimeError("Empty planning minibatch")

    sp0 = torch.cat([ep["spatial"][st:st+1] for ep, st, _ in windows], dim=0).to(device)
    ctx0 = torch.cat([ep["context"][st:st+1] for ep, st, _ in windows], dim=0).to(device)
    current_return = torch.stack([ep["return"][st] for ep, st, _ in windows]).to(device)
    current_goal = torch.stack([ep["goal"][st] for ep, st, _ in windows]).to(device)
    latent = planner.encode_state(sp0, ctx0)

    losses = {k: [] for k in (
        "latent", "context", "spatial", "reward", "success", "collision",
        "value", "q", "goal", "reconstruct"
    )}
    recon = planner.decode_state(latent)
    losses["reconstruct"].append(
        F.smooth_l1_loss(recon["spatial"], sp0) + F.smooth_l1_loss(recon["context"], ctx0)
    )
    losses["value"].append(F.smooth_l1_loss(planner.value(latent), current_return))
    losses["goal"].append(F.smooth_l1_loss(planner.goal_value(latent), current_goal))

    max_depth = max(length for _, _, length in windows)
    for depth in range(max_depth):
        active = [j for j, (_, _, length) in enumerate(windows) if depth < length]
        if not active:
            break
        index = torch.tensor(active, dtype=torch.long, device=device)
        z = latent.index_select(0, index)

        actions = torch.stack([
            windows[j][0]["action"][windows[j][1] + depth] for j in active
        ]).to(device)
        action_oh = one_hot_action(actions, model.cfg.action_dim)
        target_return = torch.stack([
            windows[j][0]["return"][windows[j][1] + depth] for j in active
        ]).to(device)
        losses["q"].append(F.smooth_l1_loss(planner.action_value(z, action_oh), target_return))

        pred = planner.step(z, action_oh)
        target_sp = torch.stack([
            windows[j][0]["next_spatial"][windows[j][1] + depth] for j in active
        ]).to(device)
        target_ctx = torch.stack([
            windows[j][0]["next_context"][windows[j][1] + depth] for j in active
        ]).to(device)
        rewards = torch.stack([
            windows[j][0]["reward"][windows[j][1] + depth] for j in active
        ]).to(device)
        successes = torch.stack([
            windows[j][0]["success"][windows[j][1] + depth] for j in active
        ]).to(device)
        collisions = torch.stack([
            windows[j][0]["collision"][windows[j][1] + depth] for j in active
        ]).to(device)
        next_returns = torch.stack([
            windows[j][0]["return"][min(windows[j][1] + depth + 1, len(windows[j][0]["return"]) - 1)]
            for j in active
        ]).to(device)
        next_goals = torch.stack([
            windows[j][0]["goal"][min(windows[j][1] + depth + 1, len(windows[j][0]["goal"]) - 1)]
            for j in active
        ]).to(device)

        with torch.no_grad():
            target_latent = planner.encode_state(target_sp, target_ctx)
        losses["latent"].append(F.smooth_l1_loss(pred["latent"], target_latent))
        losses["spatial"].append(F.smooth_l1_loss(pred["spatial"], target_sp))
        losses["context"].append(F.smooth_l1_loss(pred["context"], target_ctx))
        losses["reward"].append(F.smooth_l1_loss(pred["reward"], rewards))
        losses["success"].append(F.binary_cross_entropy_with_logits(pred["success_logit"], successes))
        losses["collision"].append(F.binary_cross_entropy_with_logits(pred["collision_logit"], collisions))
        losses["value"].append(F.smooth_l1_loss(planner.value(pred["latent"]), next_returns))
        losses["goal"].append(F.smooth_l1_loss(planner.goal_value(pred["latent"]), next_goals))

        # Copy predicted active rows into the next batched rollout state while
        # preserving autograd through the recurrent predictions.
        latent = torch.index_copy(latent, 0, index, pred["latent"])

    means = {k: torch.stack(v).mean() if v else torch.zeros((), device=device) for k, v in losses.items()}
    total = (
        1.50 * means["latent"]
        + 1.00 * means["context"]
        + 0.50 * means["spatial"]
        + 1.00 * means["reward"]
        + 0.30 * means["success"]
        + 0.40 * means["collision"]
        + 0.60 * means["value"]
        + 0.80 * means["q"]
        + 0.40 * means["goal"]
        + 0.20 * means["reconstruct"]
    )
    metrics = {k: float(v.detach()) for k, v in means.items()}
    metrics["loss"] = float(total.detach())
    return total, metrics


@torch.no_grad()
def _validation_metrics(model: InfantBrain, features: list[dict], device: str,
                        max_unroll: int = 12) -> dict:
    planner = model.planning
    planner.eval()
    ctx_errors = {1: [], 3: [], 6: [], 12: []}
    latent_errors = {1: [], 3: [], 6: [], 12: []}
    reward_abs, collision_ok, q_abs, value_abs = [], [], [], []
    for ep in features:
        n = len(ep["action"])
        for start in range(n):
            latent = planner.encode_state(
                ep["spatial"][start:start+1].to(device), ep["context"][start:start+1].to(device)
            )
            action0 = ep["action"][start:start+1].to(device)
            q = planner.action_value(latent, one_hot_action(action0, model.cfg.action_dim))
            q_abs.append(float((q - ep["return"][start:start+1].to(device)).abs().item()))
            value_abs.append(float((planner.value(latent) - ep["return"][start:start+1].to(device)).abs().item()))
            for depth in range(1, min(int(max_unroll), n - start) + 1):
                i = start + depth - 1
                action = ep["action"][i:i+1].to(device)
                pred = planner.step(latent, one_hot_action(action, model.cfg.action_dim))
                if depth == 1:
                    reward_abs.append(float((pred["reward"] - ep["reward"][i:i+1].to(device)).abs().item()))
                    collision_pred = torch.sigmoid(pred["collision_logit"]).ge(0.5)
                    collision_true = ep["collision"][i:i+1].to(device).ge(0.5)
                    collision_ok.append(float(collision_pred.eq(collision_true).float().item()))
                if depth in ctx_errors:
                    target_ctx = ep["next_context"][i:i+1].to(device)
                    target_sp = ep["next_spatial"][i:i+1].to(device)
                    ctx_errors[depth].append(float(F.smooth_l1_loss(pred["context"], target_ctx).item()))
                    target_latent = planner.encode_state(target_sp, target_ctx)
                    latent_errors[depth].append(float(F.smooth_l1_loss(pred["latent"], target_latent).item()))
                latent = pred["latent"]
    return {
        "reward_mae": _mean(reward_abs),
        "collision_accuracy": _mean(collision_ok),
        "value_mae": _mean(value_abs),
        "action_value_mae": _mean(q_abs),
        "context_error_by_horizon": {str(k): _mean(v) for k, v in ctx_errors.items() if v},
        "latent_error_by_horizon": {str(k): _mean(v) for k, v in latent_errors.items() if v},
    }


def train_planning_cortex(dataset_path: str | Path, state_dir: str | Path, *,
                          updates=4000, batch_size=32, unroll=12, gamma=0.95,
                          goal_gamma=0.97, learning_rate=3e-4, seed=10010,
                          device="cpu") -> dict:
    dataset = NurseryDataset(dataset_path)
    rows = dataset.records()
    episodes, dropped = _complete_episode_rows(rows)
    if len(episodes) < 4:
        raise RuntimeError("Need at least four complete lived episodes for planning training")
    train_eps, val_eps = _split_episodes(episodes, seed)
    state_dir = Path(state_dir)
    checkpoint = state_dir / "infant_training.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Planning training requires an existing child checkpoint: {checkpoint}")

    before_hash = _checkpoint_sha256(checkpoint)
    model, saved = load_checkpoint(checkpoint, device)
    source_updates = int(saved.get("updates", (saved.get("metadata") or {}).get("updates", 0)))
    migration_keys = list(saved.get("_v100_missing_keys") or [])
    trainable = _configure_planning_only(model)
    model.eval(); model.planning.train()

    print("Extracting child-owned dorsal planning features...")
    train_features = _extract_episode_features(model, dataset, train_eps, device, gamma, goal_gamma)
    val_features = _extract_episode_features(model, dataset, val_eps, device, gamma, goal_gamma)

    optimizer = torch.optim.AdamW(model.planning.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    rng = random.Random(int(seed) + 7)
    latest = {}
    best_state = None
    best_val = float("inf")
    best_update = 0
    for step in range(1, int(updates) + 1):
        model.planning.train()
        windows = _sample_windows(train_features, int(batch_size), int(unroll), rng)
        optimizer.zero_grad(set_to_none=True)
        loss, latest = _planner_loss(model, windows, device)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.planning.parameters(), 1.0))
        optimizer.step()
        if step % 250 == 0 or step == 1 or step == int(updates):
            model.planning.eval()
            val = _validation_metrics(model, val_features, device, max_unroll=min(12, int(unroll)))
            # Context rollout + action-value accuracy determines best checkpoint.
            hkey = str(min([h for h in (12, 6, 3, 1) if h <= int(unroll)]))
            selection = float(val["context_error_by_horizon"].get(hkey, 1e9)) + 0.10 * float(val["action_value_mae"])
            if selection < best_val:
                best_val = selection
                best_update = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.planning.state_dict().items()}
            print(
                f"planning_update={step}/{updates} loss={latest['loss']:.5f} "
                f"latent={latest['latent']:.5f} ctx={latest['context']:.5f} "
                f"reward={latest['reward']:.5f} q={latest['q']:.5f} grad={grad:.3f} "
                f"val_h{hkey}_ctx={val['context_error_by_horizon'].get(hkey, 0.0):.6f} "
                f"val_q_mae={val['action_value_mae']:.4f}"
            )
    if best_state is not None:
        model.planning.load_state_dict(best_state)
    model.eval()
    validation = _validation_metrics(model, val_features, device, max_unroll=min(12, int(unroll)))

    # Planning-only optimization changed requires_grad flags only in memory. Restore
    # the child's normal trainability before reporting/saving; EMA target cortices
    # remain frozen exactly as in ordinary parented development.
    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.target_vision.parameters():
        p.requires_grad_(False)
    for p in model.target_spatial.parameters():
        p.requires_grad_(False)

    # Save a new v0.10 checkpoint.  Do not carry the legacy whole-brain optimizer;
    # the next integrated training phase must intentionally initialize an optimizer
    # that includes the newly learned planning cortex.
    metadata = dict(saved.get("metadata") or {})
    previous_planning_updates = int(metadata.get("planning_updates", 0))
    metadata.update({
        "planning_cortex_version": "recurrent-latent-v1",
        "planning_updates": previous_planning_updates + int(updates),
        "planning_best_update": int(best_update),
        "planning_dataset_digest": dataset.digest(),
        "planning_gamma": float(gamma),
        "planning_goal_gamma": float(goal_gamma),
        "planning_unroll": int(unroll),
        "planning_migration_from_pre_v100": bool(migration_keys),
    })
    parent_id = str(metadata.get("parent_id", saved.get("metadata", {}).get("parent_id", "vision.dinov3.base")))
    save_checkpoint(
        checkpoint, model, updates=source_updates, parent_id=parent_id,
        dataset_digest=str(saved.get("metadata", {}).get("dataset_digest", dataset.digest())),
        optimizer=None, metadata=metadata,
    )
    after_hash = _checkpoint_sha256(checkpoint)

    return {
        "format": "continuous_brain.planning_training.v100",
        "checkpoint": str(checkpoint),
        "source_training_updates": source_updates,
        "planning_updates": previous_planning_updates + int(updates),
        "planning_parameters": sum(p.numel() for p in model.planning.parameters()),
        "total_trainable_child_parameters": model.parameter_count,
        "planning_migrated": bool(migration_keys),
        "planning_migration_keys": migration_keys,
        "planning_only": True,
        "planning_only_trainable_parameters": trainable,
        "episodes_complete": len(episodes),
        "episodes_train": len(train_eps),
        "episodes_validation": len(val_eps),
        "dropped_incomplete_episode_ids": dropped,
        "best_update": int(best_update),
        "training": {
            "updates": int(updates), "batch": int(batch_size), "unroll": int(unroll),
            "gamma": float(gamma), "goal_gamma": float(goal_gamma),
            "learning_rate": float(learning_rate), "seed": int(seed),
        },
        "validation": validation,
        "source_checkpoint_sha256": before_hash,
        "trained_checkpoint_sha256": after_hash,
        "legacy_optimizer_carried_forward": False,
    }


@torch.no_grad()
def _plan_action(model: InfantBrain, spatial: torch.Tensor, context7: torch.Tensor, *,
                 horizon: int, beam_width: int, discount: float, device: str):
    """Beam search entirely inside the dedicated v0.10 planning state."""
    planner = model.planning
    action_dim = int(model.cfg.action_dim)
    root = planner.encode_state(spatial, context7)
    beam = [{"latent": root, "score": 0.0, "seq": []}]
    for depth in range(int(horizon)):
        parent_indices = []
        actions_list = []
        for pi in range(len(beam)):
            for action in range(action_dim):
                parent_indices.append(pi)
                actions_list.append(action)
        actions = torch.tensor(actions_list, dtype=torch.long, device=device)
        action_oh = one_hot_action(actions, action_dim)
        latents = torch.cat([beam[i]["latent"] for i in parent_indices], dim=0)
        pred = planner.step(latents, action_oh)
        expanded = []
        for j, (pi, action) in enumerate(zip(parent_indices, actions_list)):
            parent = beam[pi]
            score = parent["score"] + (float(discount) ** depth) * float(pred["reward"][j].item())
            expanded.append({
                "latent": pred["latent"][j:j+1],
                "score": score,
                "seq": parent["seq"] + [int(action)],
            })
        # At intermediate pruning, use learned max action-value as a terminal heuristic.
        if depth + 1 < int(horizon):
            for node in expanded:
                z = node["latent"]
                zrep = z.expand(action_dim, -1)
                all_a = one_hot_action(torch.arange(action_dim, device=device), action_dim)
                tail = planner.action_value(zrep, all_a).max().item()
                node["rank"] = node["score"] + (float(discount) ** (depth + 1)) * float(tail)
            expanded.sort(key=lambda x: x["rank"], reverse=True)
        else:
            for node in expanded:
                z = node["latent"]
                zrep = z.expand(action_dim, -1)
                all_a = one_hot_action(torch.arange(action_dim, device=device), action_dim)
                qtail = planner.action_value(zrep, all_a).max().item()
                # Terminal Q is in the same discounted lived-return units as the
                # imagined reward prefix. The auxiliary goal head is intentionally
                # excluded from action scoring so qualification has no hand-tuned
                # goal-weight coefficient.
                node["rank"] = node["score"] + (float(discount) ** int(horizon)) * float(qtail)
            expanded.sort(key=lambda x: x["rank"], reverse=True)
        beam = expanded[:max(1, int(beam_width))]
    best = beam[0]
    return int(best["seq"][0]), float(best["rank"]), list(best["seq"])


def _env_image_tensor(env: DetourWorld, size: int, device: str) -> torch.Tensor:
    arr = env.render_rgb(size).astype(np.float32) / 255.0
    return torch.tensor(np.transpose(arr, (2, 0, 1))[None], dtype=torch.float32, device=device)


def _env_context(obs: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    full = np.zeros(48, dtype=np.float32)
    full[:6] = obs[:6]
    full[6] = float(obs[4])
    full[30] = 1.0
    full_t = torch.tensor(full[None], dtype=torch.float32, device=device)
    return full_t, full_t[:, :7]


@torch.no_grad()
def run_planning_episode(model: InfantBrain, seed: int, mode: str, *, horizon: int,
                         beam_width: int, max_steps: int, discount: float,
                         device: str) -> dict:
    env = DetourWorld(seed=int(seed), max_episode_steps=int(max_steps))
    obs = env.reset()
    hidden = model.nucleus.initial_state(1, device)
    memory = torch.zeros(1, model.cfg.memory_dim, dtype=torch.float32, device=device)
    previous_action = torch.zeros(1, dtype=torch.long, device=device)
    total_reward = 0.0
    planner_overrides = 0
    completed = False
    steps_used = max_steps
    for step in range(1, int(max_steps) + 1):
        image = _env_image_tensor(env, model.cfg.image_size, device)
        context, context7 = _env_context(obs, device)
        semantic = model.vision(image)
        spatial_online = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial_online)
        nuc = model.nucleus(fused, context, memory, previous_action, hidden)
        hidden = nuc["hidden"]
        reactive_action = int(torch.argmax(model.motor(nuc["workspace"]), dim=-1)[0].item())
        if mode == "reactive":
            action = reactive_action
            imagined_seq = []
        elif mode == "planning":
            stable_spatial = model.target_spatial_bus(image)
            action, _, imagined_seq = _plan_action(
                model, stable_spatial, context7, horizon=int(horizon),
                beam_width=int(beam_width), discount=float(discount), device=device,
            )
            planner_overrides += int(action != reactive_action)
        else:
            raise ValueError(mode)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        previous_action = torch.tensor([action + 1], dtype=torch.long, device=device)
        if bool(info.get("reached")):
            completed = True
            steps_used = step
        if done:
            break
    return {
        "seed": int(seed), "mode": mode,
        "horizon": int(horizon if mode == "planning" else 0),
        "reward": float(total_reward), "completed": bool(completed),
        "steps": int(steps_used), "collisions": int(env.collisions),
        "final_energy": float(env.energy), "planner_overrides": int(planner_overrides),
    }


def _summary(rows, max_steps):
    return {
        "episodes": len(rows),
        "completion_rate": _mean(float(r["completed"]) for r in rows),
        "mean_reward": _mean(r["reward"] for r in rows),
        "mean_steps_penalized": _mean(r["steps"] if r["completed"] else max_steps + 1 for r in rows),
        "mean_collisions": _mean(r["collisions"] for r in rows),
        "mean_final_energy": _mean(r["final_energy"] for r in rows),
        "mean_planner_overrides": _mean(r["planner_overrides"] for r in rows),
    }


def evaluate_planning(checkpoint: str | Path, *, episodes=10, steps=120,
                      seed_start=9001, horizons=(3, 6, 12), beam_width=32,
                      discount=0.95, device="cpu", output=None) -> dict:
    model, saved = load_checkpoint(checkpoint, device)
    model.eval()
    metadata = dict(saved.get("metadata") or {})
    if not metadata.get("planning_cortex_version"):
        raise RuntimeError("Checkpoint has not completed v0.10 planning-cortex training")
    reactive = []
    planned = {int(h): [] for h in horizons}
    for i in range(int(episodes)):
        seed = int(seed_start) + i
        reactive.append(run_planning_episode(
            model, seed, "reactive", horizon=1, beam_width=beam_width,
            max_steps=steps, discount=discount, device=device,
        ))
        for h in horizons:
            planned[int(h)].append(run_planning_episode(
                model, seed, "planning", horizon=int(h), beam_width=beam_width,
                max_steps=steps, discount=discount, device=device,
            ))
        if (i + 1) % 10 == 0:
            print(f"paired_v100_planning_episodes={i+1}/{episodes}")

    r = _summary(reactive, int(steps))
    horizon_results = {}
    any_pass = False
    any_completion = False
    for h, rows in planned.items():
        m = _summary(rows, int(steps))
        completion_adv = m["completion_rate"] - r["completion_rate"]
        collision_regression = m["mean_collisions"] - r["mean_collisions"]
        passed = bool(
            m["completion_rate"] >= 0.20
            and completion_adv > 0.05
            and collision_regression <= 2.0
        )
        any_pass = any_pass or passed
        any_completion = any_completion or m["completion_rate"] > 0.0
        horizon_results[str(h)] = {
            "planning": m,
            "advantages": {
                "completion_percentage_points": 100.0 * completion_adv,
                "reward_absolute": m["mean_reward"] - r["mean_reward"],
                "collisions_reduced": r["mean_collisions"] - m["mean_collisions"],
                "steps_saved_penalized": r["mean_steps_penalized"] - m["mean_steps_penalized"],
            },
            "p4b_passed": passed,
        }
    result = {
        "format": "continuous_brain.p4b_dedicated_planning_gate.v100",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "source_training_updates": int(saved.get("updates", metadata.get("updates", 0))),
        "planning_updates": int(metadata.get("planning_updates", 0)),
        "planning_cortex_version": metadata.get("planning_cortex_version"),
        "experiment": {
            "paired_episodes": int(episodes), "steps_per_episode": int(steps),
            "seed_start": int(seed_start), "seed_end": int(seed_start) + int(episodes) - 1,
            "horizons": [int(h) for h in horizons], "beam_width": int(beam_width),
            "discount": float(discount), "parents_used": False,
            "training_performed_during_evaluation": False, "memory_written": False,
            "planner_state": "dedicated recurrent child-owned latent",
            "planner_score": "imagined discounted reward + terminal learned action-value",
        },
        "reactive": r,
        "horizons": horizon_results,
        "thresholds": {
            "absolute_completion": 0.20,
            "completion_advantage": 0.05,
            "max_collision_regression_per_episode": 2.0,
        },
        "any_planning_completion": bool(any_completion),
        "p4b_passed": bool(any_pass),
        "episodes": {"reactive": reactive, **{f"h{h}": rows for h, rows in planned.items()}},
    }
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
