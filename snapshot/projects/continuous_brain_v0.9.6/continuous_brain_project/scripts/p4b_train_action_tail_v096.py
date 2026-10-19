from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from continuous_brain.parented_train import load_checkpoint


class ActionTailValue(nn.Module):
    def __init__(self, input_dim: int = 12, hidden: int = 128):
        super().__init__()
        inner = max(32, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, inner), nn.SiLU(),
            nn.Linear(inner, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_rows(dataset: Path):
    rows = []
    with (dataset / "lessons.jsonl").open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def complete_episodes(rows, max_episode_step: int = 119):
    episodes = defaultdict(list)
    for row in rows:
        episodes[int(row["episode_id"])].append(row)
    complete = {}
    dropped = []
    for eid, erows in episodes.items():
        erows.sort(key=lambda r: int(r["episode_step"]))
        success = any(float(r.get("success", 0.0)) > 0.5 for r in erows)
        timeout = int(erows[-1].get("episode_step", -1)) >= int(max_episode_step)
        if success or timeout:
            complete[eid] = erows
        else:
            dropped.append(eid)
    return complete, dropped


def episode_is_success(erows) -> bool:
    return any(float(r.get("success", 0.0)) > 0.5 for r in erows)


def stratified_episode_split(episodes, seed: int, val_fraction: float = 0.20):
    rng = random.Random(seed)
    success_ids = [eid for eid, rows in episodes.items() if episode_is_success(rows)]
    fail_ids = [eid for eid, rows in episodes.items() if not episode_is_success(rows)]
    rng.shuffle(success_ids); rng.shuffle(fail_ids)

    def take_val(ids):
        if not ids:
            return []
        if len(ids) == 1:
            return []
        n = max(1, int(round(len(ids) * val_fraction)))
        return ids[:min(n, len(ids) - 1)]

    val_ids = set(take_val(success_ids) + take_val(fail_ids))
    if not val_ids:
        all_ids = list(episodes)
        rng.shuffle(all_ids)
        val_ids = {all_ids[0]}
    train_ids = set(episodes) - val_ids
    if not train_ids:
        raise RuntimeError("Need at least two complete episodes")
    return train_ids, val_ids


def build_features(episodes, gamma: float, action_dim: int = 5):
    features = []
    targets = []
    episode_ids = []
    actions_out = []
    success_flags = []

    for eid in sorted(episodes):
        erows = episodes[eid]
        # future_after[i] deliberately EXCLUDES reward[i]. The planner supplies
        # current reward from the learned world model, then adds gamma * this tail.
        future_after = [0.0] * len(erows)
        running = 0.0
        for i in range(len(erows) - 1, -1, -1):
            future_after[i] = running
            running = float(erows[i].get("reward", 0.0)) + float(gamma) * running

        flag = 1.0 if episode_is_success(erows) else 0.0
        for i, (row, target) in enumerate(zip(erows, future_after)):
            next_context = np.asarray(row["next_context"], dtype=np.float32)
            if next_context.size < 7:
                raise ValueError(f"episode {eid} row {i} next_context has only {next_context.size} values")
            action = int(row.get("executed_action", -1))
            if not (0 <= action < action_dim):
                raise ValueError(f"episode {eid} row {i} invalid executed_action={action}")
            one_hot = np.zeros(action_dim, dtype=np.float32)
            one_hot[action] = 1.0
            feature = np.concatenate([next_context[:7], one_hot], axis=0)
            features.append(torch.tensor(feature, dtype=torch.float32))
            targets.append(float(target))
            episode_ids.append(int(eid))
            actions_out.append(action)
            success_flags.append(flag)

    return (
        torch.stack(features),
        torch.tensor(targets, dtype=torch.float32),
        episode_ids,
        actions_out,
        torch.tensor(success_flags, dtype=torch.float32),
    )


def mae(a, b):
    return float(torch.mean(torch.abs(a - b)).item())


def correlation(a, b):
    a = a.float(); b = b.float()
    if len(a) < 2 or float(a.std().item()) < 1e-10 or float(b.std().item()) < 1e-10:
        return 0.0
    c = torch.corrcoef(torch.stack((a, b)))[0, 1]
    return float(torch.nan_to_num(c, nan=0.0).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=4545)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    if not (0.0 < a.gamma <= 1.0):
        raise ValueError("--gamma must be in (0,1]")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    # Load only to prove lineage/compatibility. No child parameter is trainable or saved.
    model, saved = load_checkpoint(a.checkpoint, a.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = Path(a.dataset)
    episodes, dropped = complete_episodes(load_rows(dataset))
    train_ids, val_ids = stratified_episode_split(episodes, a.seed)
    x, y, eids, actions, success_flags = build_features(episodes, a.gamma, int(model.cfg.action_dim))

    train_mask = torch.tensor([eid in train_ids for eid in eids], dtype=torch.bool)
    val_mask = torch.tensor([eid in val_ids for eid in eids], dtype=torch.bool)

    # Standardize only the continuous 7-state prefix. Keep action one-hot exact.
    state_mean = x[train_mask, :7].mean(0)
    state_std = x[train_mask, :7].std(0).clamp_min(1e-5)
    xz = x.clone()
    xz[:, :7] = (xz[:, :7] - state_mean) / state_std
    y_mean = y[train_mask].mean()
    y_std = y[train_mask].std().clamp_min(1e-5)
    yz = (y - y_mean) / y_std

    critic = ActionTailValue(int(x.shape[1]), int(a.hidden)).to(a.device)
    opt = torch.optim.AdamW(critic.parameters(), lr=a.learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(xz[train_mask], yz[train_mask]),
        batch_size=a.batch,
        shuffle=True,
    )

    best = None; best_state = None; best_epoch = None
    for epoch in range(1, a.epochs + 1):
        critic.train()
        for bx, by in loader:
            bx = bx.to(a.device); by = by.to(a.device)
            opt.zero_grad(set_to_none=True)
            pred = critic(bx)
            loss = torch.nn.functional.smooth_l1_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
        critic.eval()
        with torch.no_grad():
            vp = critic(xz[val_mask].to(a.device)).cpu() * y_std + y_mean
            vy = y[val_mask]
            vmae = mae(vp, vy)
            vcorr = correlation(vp, vy)
        score = (vmae, -vcorr)
        if best is None or score < best:
            best = score; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == a.epochs:
            print(f"action_tail_epoch={epoch} val_mae={vmae:.6f} val_corr={vcorr:.4f}")

    critic.load_state_dict(best_state); critic.eval()
    with torch.no_grad():
        tp = critic(xz[train_mask].to(a.device)).cpu() * y_std + y_mean
        vp = critic(xz[val_mask].to(a.device)).cpu() * y_std + y_mean
    baseline = torch.full_like(y[val_mask], float(y[train_mask].mean().item()))

    train_actions = Counter(a for a, eid in zip(actions, eids) if eid in train_ids)
    val_actions = Counter(a for a, eid in zip(actions, eids) if eid in val_ids)
    val_success_mask = val_mask & (success_flags > 0.5)
    val_fail_mask = val_mask & (success_flags <= 0.5)

    result = {
        "format": "continuous_brain.p4b_action_tail_value.v096.v4",
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "training_updates": int(saved.get("updates", (saved.get("metadata") or {}).get("updates", 0))),
        "dataset": str(dataset.resolve()),
        "input": "actual_next_structured_context7 + executed_action_one_hot5",
        "input_dim": int(x.shape[1]),
        "gamma": float(a.gamma),
        "target": "discounted lived reward strictly AFTER the current action/reward",
        "planner_decomposition": "world_reward(s,a) - uncertainty + gamma * tail(predicted_next_context,a)",
        "hidden": int(a.hidden),
        "best_epoch": int(best_epoch),
        "episodes_complete": len(episodes),
        "episodes_train": len(train_ids),
        "episodes_validation": len(val_ids),
        "successful_episodes": sum(episode_is_success(rows) for rows in episodes.values()),
        "failed_complete_episodes": sum(not episode_is_success(rows) for rows in episodes.values()),
        "dropped_incomplete_episode_ids": dropped,
        "samples_train": int(train_mask.sum()),
        "samples_validation": int(val_mask.sum()),
        "train_action_counts": {str(k): int(train_actions[k]) for k in range(int(model.cfg.action_dim))},
        "validation_action_counts": {str(k): int(val_actions[k]) for k in range(int(model.cfg.action_dim))},
        "min_train_action_count": int(min(train_actions[k] for k in range(int(model.cfg.action_dim)))),
        "validation_mae": mae(vp, y[val_mask]),
        "validation_baseline_mean_mae": mae(baseline, y[val_mask]),
        "validation_correlation": correlation(vp, y[val_mask]),
        "training_mae": mae(tp, y[train_mask]),
        "validation_success_samples": int(val_success_mask.sum()),
        "validation_fail_samples": int(val_fail_mask.sum()),
        "recursive_world_rollout_used": False,
        "imagined_workspace_used_by_critic": False,
        "imagined_semantic_bus_used_by_critic": False,
        "child_checkpoint_modified": False,
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": result["format"],
        "critic": best_state,
        "input_dim": int(x.shape[1]),
        "hidden": int(a.hidden),
        "gamma": float(a.gamma),
        "state_mean": state_mean,
        "state_std": state_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "action_dim": int(model.cfg.action_dim),
        "metadata": result,
    }, out)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"action_tail_critic={out}")


if __name__ == "__main__":
    main()
