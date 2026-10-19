from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from continuous_brain.parented_train import load_checkpoint


class ValueToGo(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
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


def complete_episodes(rows):
    episodes = defaultdict(list)
    for row in rows:
        episodes[int(row["episode_id"])].append(row)
    out = {}
    dropped = []
    for eid, erows in episodes.items():
        erows.sort(key=lambda r: int(r["episode_step"]))
        terminal_success = any(float(r.get("success", 0.0)) > 0.5 for r in erows)
        terminal_timeout = int(erows[-1].get("episode_step", -1)) >= 119
        if terminal_success or terminal_timeout:
            out[eid] = erows
        else:
            dropped.append(eid)
    return out, dropped


def image_tensor(path: Path, size: int, device: str):
    arr = np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return torch.tensor(np.transpose(arr, (2, 0, 1))[None], dtype=torch.float32, device=device)


@torch.no_grad()
def build_features(model, dataset: Path, episodes, gamma: float, device: str):
    features = []
    targets = []
    episode_ids = []
    count = 0
    for eid in sorted(episodes):
        erows = episodes[eid]
        returns = [0.0] * len(erows)
        running = 0.0
        for i in range(len(erows) - 1, -1, -1):
            running = float(erows[i].get("reward", 0.0)) + float(gamma) * running
            returns[i] = running

        hidden = model.nucleus.initial_state(1, device)
        memory = torch.zeros(1, model.cfg.memory_dim, dtype=torch.float32, device=device)
        for row, ret in zip(erows, returns):
            img = image_tensor(dataset / row["image"], model.cfg.image_size, device)
            context = torch.tensor([row["context"]], dtype=torch.float32, device=device)
            semantic = model.vision(img)
            online_spatial = model.spatial(img)
            target_spatial = model.target_spatial_bus(img)
            fused = model.fuse_vision(semantic, online_spatial)
            prev = int(row.get("previous_action", -1))
            prev_token = 0 if prev < 0 else prev + 1
            prev_action = torch.tensor([prev_token], dtype=torch.long, device=device)
            nuc = model.nucleus(fused, context, memory, prev_action, hidden)
            hidden = nuc["hidden"]
            feat = torch.cat((
                nuc["workspace"],
                target_spatial,
                context[:, :model.cfg.dynamics_context_dim],
            ), dim=-1)
            features.append(feat[0].cpu())
            targets.append(float(ret))
            episode_ids.append(int(eid))
            count += 1
            if count % 500 == 0:
                print(f"value_features={count}")
    return torch.stack(features), torch.tensor(targets, dtype=torch.float32), episode_ids


def mae(a, b):
    return float(torch.mean(torch.abs(a - b)).item())


def correlation(a, b):
    a = a.float(); b = b.float()
    if len(a) < 2 or float(a.std().item()) < 1e-10 or float(b.std().item()) < 1e-10:
        return 0.0
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1].item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dataset = Path(a.dataset)
    model, saved = load_checkpoint(a.checkpoint, a.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    episodes, dropped = complete_episodes(load_rows(dataset))
    ids = sorted(episodes)
    rng = random.Random(a.seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * 0.20)))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    if not train_ids:
        raise RuntimeError("Need at least two complete episodes")

    x, y, eids = build_features(model, dataset, episodes, a.gamma, a.device)
    train_mask = torch.tensor([eid in train_ids for eid in eids], dtype=torch.bool)
    val_mask = ~train_mask

    x_mean = x[train_mask].mean(0)
    x_std = x[train_mask].std(0).clamp_min(1e-5)
    y_mean = y[train_mask].mean()
    y_std = y[train_mask].std().clamp_min(1e-5)
    xz = (x - x_mean) / x_std
    yz = (y - y_mean) / y_std

    critic = ValueToGo(int(x.shape[1]), int(a.hidden)).to(a.device)
    opt = torch.optim.AdamW(critic.parameters(), lr=a.learning_rate, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(xz[train_mask], yz[train_mask]), batch_size=a.batch, shuffle=True)

    best = None
    best_state = None
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
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1 or epoch == a.epochs:
            print(f"value_epoch={epoch} val_mae={vmae:.6f} val_corr={vcorr:.4f}")

    critic.load_state_dict(best_state)
    critic.eval()
    with torch.no_grad():
        tp = critic(xz[train_mask].to(a.device)).cpu() * y_std + y_mean
        vp = critic(xz[val_mask].to(a.device)).cpu() * y_std + y_mean
    baseline = torch.full_like(y[val_mask], float(y[train_mask].mean().item()))

    result = {
        "format": "continuous_brain.p4b_value_to_go.v096",
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "training_updates": int(saved.get("updates", (saved.get("metadata") or {}).get("updates", 0))),
        "dataset": str(dataset.resolve()),
        "gamma": float(a.gamma),
        "input": "workspace + target_spatial_bus + structured_context7",
        "input_dim": int(x.shape[1]),
        "hidden": int(a.hidden),
        "episodes_complete": len(ids),
        "episodes_train": len(train_ids),
        "episodes_validation": len(val_ids),
        "dropped_incomplete_episode_ids": dropped,
        "samples_train": int(train_mask.sum()),
        "samples_validation": int(val_mask.sum()),
        "validation_mae": mae(vp, y[val_mask]),
        "validation_baseline_mean_mae": mae(baseline, y[val_mask]),
        "validation_correlation": correlation(vp, y[val_mask]),
        "training_mae": mae(tp, y[train_mask]),
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
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "metadata": result,
    }, out)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"value_critic={out}")


if __name__ == "__main__":
    main()
