"""v0.10.1 conservative Bellman action-value repair for the planning cortex.

v0.10.0 established that the dedicated recurrent planning latent learns action
semantics and multi-step physical dynamics, but its Q head was supervised with
Monte-Carlo *behavior* returns.  In an exploratory nursery that target answers
"what happened after the collector took this action?", not "what is the best
future achievable after this action?".

v0.10.1 therefore freezes the entire child except ``planning.q_head`` and runs
fitted offline Bellman Q-learning on actual lived transitions.  A slowly moving
target Q network stabilizes bootstrapping, balanced action sampling prevents the
collector's common actions from dominating, and a conservative log-sum-exp
penalty discourages unsupported counterfactual optimism.  Bellman targets are
clamped to the empirical lived-return support so the offline learner cannot
invent values outside demonstrated experience.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import statistics
from typing import Iterable

import torch
from torch.nn import functional as F

from .infant import InfantBrain, one_hot_action
from .parented_data import NurseryDataset
from .parented_train import load_checkpoint, save_checkpoint
from .planning_v100 import (
    _checkpoint_sha256,
    _complete_episode_rows,
    _extract_episode_features,
    _split_episodes,
)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def _all_action_values(planner, latent: torch.Tensor) -> torch.Tensor:
    """Return Q values with shape [batch, action_dim]."""
    batch = int(latent.shape[0])
    action_dim = int(planner.cfg.action_dim)
    z = latent[:, None, :].expand(batch, action_dim, latent.shape[-1]).reshape(batch * action_dim, -1)
    actions = torch.arange(action_dim, device=latent.device).repeat(batch)
    aoh = one_hot_action(actions, action_dim)
    return planner.action_value(z, aoh).reshape(batch, action_dim)


def _all_action_values_head(q_head, latent: torch.Tensor, action_dim: int) -> torch.Tensor:
    batch = int(latent.shape[0])
    z = latent[:, None, :].expand(batch, action_dim, latent.shape[-1]).reshape(batch * action_dim, -1)
    actions = torch.arange(action_dim, device=latent.device).repeat(batch)
    aoh = one_hot_action(actions, action_dim)
    return q_head(torch.cat((z, aoh), dim=-1)).squeeze(-1).reshape(batch, action_dim)


def _configure_q_only(model: InfantBrain) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.planning.q_head.parameters():
        p.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _flatten_transitions(features: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ep in features:
        n = len(ep["action"])
        for i in range(n):
            terminal = bool(i == n - 1 or float(ep["success"][i]) > 0.5)
            out.append({
                "episode_id": int(ep["episode_id"]),
                "index": int(i),
                "spatial": ep["spatial"][i],
                "context": ep["context"][i],
                "next_spatial": ep["next_spatial"][i],
                "next_context": ep["next_context"][i],
                "action": int(ep["action"][i]),
                "reward": float(ep["reward"][i]),
                "terminal": terminal,
                "behavior_return": float(ep["return"][i]),
            })
    return out


def _action_buckets(transitions: list[dict], action_dim: int) -> dict[int, list[int]]:
    buckets = {a: [] for a in range(int(action_dim))}
    for i, row in enumerate(transitions):
        a = int(row["action"])
        if a in buckets:
            buckets[a].append(i)
    return buckets


def _sample_balanced_indices(buckets: dict[int, list[int]], batch_size: int,
                             rng: random.Random) -> list[int]:
    available = [a for a, rows in buckets.items() if rows]
    if not available:
        raise RuntimeError("No action-labelled transitions for Bellman Q training")
    indices = []
    for _ in range(int(batch_size)):
        a = rng.choice(available)
        indices.append(rng.choice(buckets[a]))
    return indices


def _transition_batch(model: InfantBrain, transitions: list[dict], indices: list[int],
                      device: str) -> dict[str, torch.Tensor]:
    rows = [transitions[i] for i in indices]
    spatial = torch.stack([r["spatial"] for r in rows]).to(device)
    context = torch.stack([r["context"] for r in rows]).to(device)
    next_spatial = torch.stack([r["next_spatial"] for r in rows]).to(device)
    next_context = torch.stack([r["next_context"] for r in rows]).to(device)
    action = torch.tensor([r["action"] for r in rows], dtype=torch.long, device=device)
    reward = torch.tensor([r["reward"] for r in rows], dtype=torch.float32, device=device)
    terminal = torch.tensor([float(r["terminal"]) for r in rows], dtype=torch.float32, device=device)
    behavior_return = torch.tensor([r["behavior_return"] for r in rows], dtype=torch.float32, device=device)
    with torch.no_grad():
        z = model.planning.encode_state(spatial, context)
        next_z = model.planning.encode_state(next_spatial, next_context)
        predicted_next_z = model.planning.step(z, one_hot_action(action, model.cfg.action_dim))["latent"]
    return {
        "z": z, "next_z": next_z, "predicted_next_z": predicted_next_z,
        "action": action, "reward": reward, "terminal": terminal,
        "behavior_return": behavior_return,
    }


def _bellman_loss(model: InfantBrain, target_q, batch: dict[str, torch.Tensor], *,
                  gamma: float, cql_alpha: float, support_min: float,
                  support_max: float, consistency_weight: float = 0.10):
    planner = model.planning
    action_dim = int(model.cfg.action_dim)
    q_all = _all_action_values(planner, batch["z"])
    q_data = q_all.gather(1, batch["action"][:, None]).squeeze(1)
    with torch.no_grad():
        next_online = _all_action_values(planner, batch["next_z"])
        next_action = next_online.argmax(dim=1)
        next_target = _all_action_values_head(target_q, batch["next_z"], action_dim)
        next_q = next_target.gather(1, next_action[:, None]).squeeze(1)
        target = batch["reward"] + float(gamma) * (1.0 - batch["terminal"]) * next_q
        target = target.clamp(float(support_min), float(support_max))
    td = F.smooth_l1_loss(q_data, target)
    conservative = (torch.logsumexp(q_all, dim=1) - q_data).mean()

    # Make Q stable on the planner's own one-step latent, not only on real encoded
    # states.  This reduces the small real-vs-imagined latent mismatch without
    # changing the already-qualified dynamics model.
    actual_next_all = _all_action_values(planner, batch["next_z"]).detach()
    imagined_next_all = _all_action_values(planner, batch["predicted_next_z"])
    consistency = F.smooth_l1_loss(imagined_next_all, actual_next_all)
    total = td + float(cql_alpha) * conservative + float(consistency_weight) * consistency
    return total, {
        "loss": float(total.detach()),
        "td": float(td.detach()),
        "conservative": float(conservative.detach()),
        "imagined_consistency": float(consistency.detach()),
        "q_data_mean": float(q_data.detach().mean()),
        "target_mean": float(target.detach().mean()),
    }


@torch.no_grad()
def _validation_metrics(model: InfantBrain, target_q, transitions: list[dict], *,
                        gamma: float, support_min: float, support_max: float,
                        device: str) -> dict:
    if not transitions:
        return {}
    planner = model.planning
    action_dim = int(model.cfg.action_dim)
    td_abs, behavior_abs, conservative_gap, consistency = [], [], [], []
    greedy_counts = {str(a): 0 for a in range(action_dim)}
    executed_counts = {str(a): 0 for a in range(action_dim)}
    for start in range(0, len(transitions), 256):
        idx = list(range(start, min(start + 256, len(transitions))))
        b = _transition_batch(model, transitions, idx, device)
        q_all = _all_action_values(planner, b["z"])
        q_data = q_all.gather(1, b["action"][:, None]).squeeze(1)
        next_online = _all_action_values(planner, b["next_z"])
        next_action = next_online.argmax(dim=1)
        next_target = _all_action_values_head(target_q, b["next_z"], action_dim)
        next_q = next_target.gather(1, next_action[:, None]).squeeze(1)
        target = b["reward"] + float(gamma) * (1.0 - b["terminal"]) * next_q
        target = target.clamp(float(support_min), float(support_max))
        td_abs.extend((q_data - target).abs().cpu().tolist())
        behavior_abs.extend((q_data - b["behavior_return"]).abs().cpu().tolist())
        conservative_gap.extend((torch.logsumexp(q_all, dim=1) - q_data).cpu().tolist())
        pred_all = _all_action_values(planner, b["predicted_next_z"])
        true_all = _all_action_values(planner, b["next_z"])
        consistency.extend((pred_all - true_all).abs().mean(dim=1).cpu().tolist())
        for a in q_all.argmax(dim=1).cpu().tolist():
            greedy_counts[str(int(a))] += 1
        for a in b["action"].cpu().tolist():
            executed_counts[str(int(a))] += 1
    return {
        "bellman_residual_mae": _mean(td_abs),
        "behavior_return_mae": _mean(behavior_abs),
        "conservative_gap_mean": _mean(conservative_gap),
        "imagined_vs_actual_q_mae": _mean(consistency),
        "greedy_action_counts": greedy_counts,
        "executed_action_counts": executed_counts,
        "transitions": len(transitions),
    }


def train_conservative_bellman_q(dataset_path: str | Path, state_dir: str | Path, *,
                                 updates=3000, batch_size=256, gamma=0.95,
                                 learning_rate=3e-4, cql_alpha=0.10,
                                 target_tau=0.01, seed=10110, device="cpu") -> dict:
    dataset = NurseryDataset(dataset_path)
    episodes, dropped = _complete_episode_rows(dataset.records())
    if len(episodes) < 4:
        raise RuntimeError("Need at least four complete lived episodes for Bellman Q training")
    train_eps, val_eps = _split_episodes(episodes, int(seed))
    state_dir = Path(state_dir)
    checkpoint = state_dir / "infant_training.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Bellman Q training requires an existing v0.10 child checkpoint: {checkpoint}")

    before_hash = _checkpoint_sha256(checkpoint)
    model, saved = load_checkpoint(checkpoint, device)
    metadata = dict(saved.get("metadata") or {})
    if not metadata.get("planning_cortex_version"):
        raise RuntimeError("Run v0.10 planning-cortex training before v0.10.1 Bellman Q training")
    source_updates = int(saved.get("updates", metadata.get("updates", 0)))

    # Snapshot every tensor except Q so the release can prove causal isolation.
    before_non_q = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        if not k.startswith("planning.q_head.")
    }
    trainable = _configure_q_only(model)
    model.eval(); model.planning.q_head.train()

    # Reuse the same stable child-owned dorsal features as v0.10.0.  No parent
    # model and no simulator-generated counterfactual transition is used here.
    print("Extracting lived states for conservative Bellman Q learning...")
    goal_gamma = float(metadata.get("planning_goal_gamma", 0.97))
    train_features = _extract_episode_features(model, dataset, train_eps, device, gamma, goal_gamma)
    val_features = _extract_episode_features(model, dataset, val_eps, device, gamma, goal_gamma)
    train_transitions = _flatten_transitions(train_features)
    val_transitions = _flatten_transitions(val_features)
    if not train_transitions or not val_transitions:
        raise RuntimeError("Bellman Q training requires non-empty train and validation transitions")

    behavior_returns = [r["behavior_return"] for r in train_transitions]
    support_min = float(min(behavior_returns))
    support_max = float(max(behavior_returns))
    buckets = _action_buckets(train_transitions, model.cfg.action_dim)
    if any(len(v) == 0 for v in buckets.values()):
        missing = [a for a, v in buckets.items() if not v]
        raise RuntimeError(f"Bellman Q requires lived support for every action; missing={missing}")

    target_q = copy.deepcopy(model.planning.q_head).to(device).eval()
    for p in target_q.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.planning.q_head.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    rng = random.Random(int(seed) + 17)

    best_state = None
    best_target_state = None
    best_val = float("inf")
    best_update = 0
    latest = {}
    for step in range(1, int(updates) + 1):
        model.planning.q_head.train()
        indices = _sample_balanced_indices(buckets, int(batch_size), rng)
        batch = _transition_batch(model, train_transitions, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss, latest = _bellman_loss(
            model, target_q, batch, gamma=gamma, cql_alpha=cql_alpha,
            support_min=support_min, support_max=support_max,
        )
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.planning.q_head.parameters(), 1.0))
        optimizer.step()
        with torch.no_grad():
            tau = float(target_tau)
            for tp, op in zip(target_q.parameters(), model.planning.q_head.parameters()):
                tp.mul_(1.0 - tau).add_(op, alpha=tau)

        if step == 1 or step % 250 == 0 or step == int(updates):
            model.planning.q_head.eval()
            val = _validation_metrics(
                model, target_q, val_transitions, gamma=gamma,
                support_min=support_min, support_max=support_max, device=device,
            )
            selection = float(val["bellman_residual_mae"]) + 0.10 * float(val["imagined_vs_actual_q_mae"])
            if selection < best_val:
                best_val = selection
                best_update = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.planning.q_head.state_dict().items()}
                best_target_state = {k: v.detach().cpu().clone() for k, v in target_q.state_dict().items()}
            print(
                f"bellman_q_update={step}/{updates} loss={latest['loss']:.5f} "
                f"td={latest['td']:.5f} cql={latest['conservative']:.5f} "
                f"consistency={latest['imagined_consistency']:.5f} grad={grad:.3f} "
                f"val_td_mae={val['bellman_residual_mae']:.4f} "
                f"val_imag_q_mae={val['imagined_vs_actual_q_mae']:.4f}"
            )
    if best_state is not None:
        model.planning.q_head.load_state_dict(best_state)
    if best_target_state is not None:
        target_q.load_state_dict(best_target_state)
    model.eval()
    validation = _validation_metrics(
        model, target_q, val_transitions, gamma=gamma,
        support_min=support_min, support_max=support_max, device=device,
    )

    changed_non_q = [
        k for k, v in before_non_q.items()
        if not torch.equal(v, model.state_dict()[k].detach().cpu())
    ]
    if changed_non_q:
        raise RuntimeError(f"Bellman Q isolation failure; non-Q tensors changed: {changed_non_q[:20]}")

    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.target_vision.parameters():
        p.requires_grad_(False)
    for p in model.target_spatial.parameters():
        p.requires_grad_(False)

    previous_q_updates = int(metadata.get("planning_bellman_q_updates", 0))
    metadata.update({
        "planning_value_version": "conservative-bellman-q-v1",
        "planning_bellman_q_updates": previous_q_updates + int(updates),
        "planning_bellman_q_best_update": int(best_update),
        "planning_bellman_q_gamma": float(gamma),
        "planning_bellman_q_cql_alpha": float(cql_alpha),
        "planning_bellman_q_target_tau": float(target_tau),
        "planning_bellman_q_dataset_digest": dataset.digest(),
        "planning_bellman_q_return_support": [support_min, support_max],
        "planning_cortex_version": "recurrent-latent-v1+conservative-bellman-q-v1",
    })
    parent_id = str(metadata.get("parent_id", "vision.dinov3.base"))
    save_checkpoint(
        checkpoint, model, updates=source_updates, parent_id=parent_id,
        dataset_digest=str(metadata.get("dataset_digest", dataset.digest())),
        optimizer=None, metadata=metadata,
    )
    after_hash = _checkpoint_sha256(checkpoint)

    return {
        "format": "continuous_brain.planning_bellman_q_training.v101",
        "checkpoint": str(checkpoint),
        "source_training_updates": source_updates,
        "planning_updates": int(metadata.get("planning_updates", 0)),
        "planning_bellman_q_updates": previous_q_updates + int(updates),
        "planning_value_version": metadata["planning_value_version"],
        "q_only": True,
        "q_only_trainable_parameters": trainable,
        "changed_non_q_tensors": len(changed_non_q),
        "episodes_complete": len(episodes),
        "episodes_train": len(train_eps),
        "episodes_validation": len(val_eps),
        "dropped_incomplete_episode_ids": dropped,
        "train_transitions": len(train_transitions),
        "validation_transitions": len(val_transitions),
        "train_action_counts": {str(a): len(v) for a, v in buckets.items()},
        "return_support": [support_min, support_max],
        "best_update": int(best_update),
        "training": {
            "updates": int(updates), "batch": int(batch_size), "gamma": float(gamma),
            "learning_rate": float(learning_rate), "cql_alpha": float(cql_alpha),
            "target_tau": float(target_tau), "seed": int(seed),
            "balanced_action_sampling": True,
            "target": "r + gamma * max_a Q_target(s',a), terminal-masked and support-clamped",
            "conservative_regularizer": "logsumexp(Q(s,*)) - Q(s,a_lived)",
        },
        "validation": validation,
        "source_checkpoint_sha256": before_hash,
        "trained_checkpoint_sha256": after_hash,
        "legacy_optimizer_carried_forward": False,
    }
