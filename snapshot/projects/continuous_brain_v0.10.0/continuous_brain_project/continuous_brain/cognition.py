"""v0.6 recurrent workspace and sequence-trained skill/world learner.

This is an experimental cognitive architecture, not a consciousness detector.
No simulator metadata or free-form diagnostics are accepted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SKILL_COUNT = 6
GOAL_COUNT = 4
SELF_DIM = 24


@dataclass
class CognitiveConfig:
    width: int = 768
    layers: int = 3
    heads: int = 8
    slots: int = 4
    sensory_dim: int = 54
    memory_k: int = 4
    sequence: int = 6
    burn_in: int = 2
    horizon: int = 3
    batch: int = 2
    replay_capacity: int = 2048
    learning_rate: float = 0.0001
    gamma: float = 0.98
    train_every: int = 4
    teacher_decay: int = 5000
    imagination_after_updates: int = 200
    imagination_weight: float = 0.10

    @classmethod
    def profile(cls, name: str, sensory_dim: int = 54):
        sizes = {"tiny": (64, 1, 4), "compact": (256, 2, 8), "standard": (768, 3, 8)}
        if name not in sizes:
            raise ValueError(f"Unknown profile: {name}")
        w, layers, heads = sizes[name]
        return cls(width=w, layers=layers, heads=heads, sensory_dim=sensory_dim)

    def __post_init__(self):
        if self.width % self.heads or min(self.width, self.layers, self.slots) < 1:
            raise ValueError("Invalid workspace dimensions")
        if self.sequence <= self.burn_in or self.horizon < 1 or self.batch < 1:
            raise ValueError("Sequence must be longer than burn-in; batch/horizon positive")
        if min(self.train_every, self.teacher_decay, self.replay_capacity, self.memory_k) < 1:
            raise ValueError("Training frequencies and memory/replay capacities must be positive")

    @property
    def memory_dim(self):
        return self.sensory_dim + SKILL_COUNT + 2


class CognitiveCore(nn.Module):
    def __init__(self, cfg: CognitiveConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.sensory = nn.Sequential(nn.Linear(cfg.sensory_dim, w), nn.LayerNorm(w), nn.SiLU())
        self.self_encoder = nn.Linear(SELF_DIM, w)
        self.goal = nn.Embedding(GOAL_COUNT, w)
        self.memory = nn.Linear(cfg.memory_dim, w)
        self.slot_roles = nn.Parameter(torch.randn(cfg.slots, w) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(w, cfg.heads, 3 * w, dropout=0.0,
                                       batch_first=True, norm_first=True)
            for _ in range(cfg.layers)
        ])
        self.recurrence = nn.GRUCell(w, w)
        self.norm = nn.LayerNorm(w)
        self.skill_embedding = nn.Embedding(SKILL_COUNT, w)
        self.skill_values = nn.Linear(w, SKILL_COUNT)
        self.dynamics = nn.GRUCell(w, w)
        self.future = nn.Linear(w, 2 * cfg.sensory_dim)
        self.reward = nn.Linear(w, 1)
        self.success = nn.Linear(w, 1)

    def initial_state(self, batch=1, device=None):
        return torch.zeros(batch, self.cfg.slots, self.cfg.width,
                           device=device or self.slot_roles.device)

    def forward(self, sensory, self_state, goals, memories, workspace):
        tokens = torch.cat((self.sensory(sensory)[:, None],
                            self.self_encoder(self_state)[:, None],
                            self.goal(goals)[:, None], self.memory(memories)), dim=1)
        updated = workspace + self.slot_roles
        for block in self.blocks:
            updated = block(updated, tokens)
        shape = updated.shape
        updated = self.recurrence(updated.reshape(-1, shape[-1]),
                                  workspace.reshape(-1, shape[-1])).reshape(shape)
        pooled = self.norm(updated.mean(dim=1))
        return updated, pooled, self.skill_values(pooled)

    def imagine(self, latent, skills):
        future = self.dynamics(self.skill_embedding(skills), latent)
        mean, logvar = self.future(future).chunk(2, dim=-1)
        return (future, mean, logvar.clamp(-5.0, 2.0),
                self.reward(future).squeeze(-1), self.success(future).squeeze(-1))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def tensor_packet(packet, device):
    if set(packet) != {"sensory", "self", "goal", "memories"}:
        raise ValueError("Cognitive input contract rejects evaluator/teacher/diagnostic metadata")
    return (
        torch.as_tensor(packet["sensory"], dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(packet["self"], dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor([packet["goal"]], dtype=torch.long, device=device),
        torch.as_tensor(packet["memories"], dtype=torch.float32, device=device).unsqueeze(0),
    )


class SequenceLearner:
    """Sequential replay with burn-in, option TD, planner distillation and rollouts.

    Old stored states initialize burn-in, never pretend to be fresh encodings.
    Sampling balances goal groups and retains a protected early-experience set.
    Multi-step prediction stops at physical episode/fault boundaries.
    """
    def __init__(self, net, seed=7):
        self.net = net
        self.cfg = net.cfg
        self.target = copy.deepcopy(net).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=self.cfg.learning_rate)
        self.replay = deque(maxlen=self.cfg.replay_capacity)
        self.protected = []
        self.rng = random.Random(seed)
        self.updates = 0
        self.last = {}
        self.control_revision = None

    def _eligible(self, seq):
        return (all(b["serial"] == a["serial"] + 1 for a, b in zip(seq, seq[1:]))
                and len({row.get("control_revision", "v060") for row in seq}) == 1
                and (self.control_revision is None or all(
                    row.get("control_revision", "v060") == self.control_revision for row in seq)))

    def active_replay_count(self):
        return sum(self.control_revision is None or row.get("control_revision", "v060") == self.control_revision
                   for row in self.replay)

    def append(self, row):
        self.replay.append(row)
        if len(self.replay) >= self.cfg.sequence and sum(self._eligible(seq) for seq in self.protected) < 32:
            if len(self.replay) % self.cfg.sequence == 0:
                seq = list(self.replay)[-self.cfg.sequence:]
                if self._eligible(seq):
                    self.protected.append(copy.deepcopy(seq))

    def _samples(self):
        rows = list(self.replay)
        n = self.cfg.sequence
        if len(rows) < n:
            return []
        groups = {}
        for start in range(len(rows) - n + 1):
            # A sensor fault leaves an explicit gap: never infer through it.
            seq = rows[start:start+n]
            if self._eligible(seq):
                groups.setdefault(seq[-1]["packet"]["goal"], []).append(start)
        if not groups:
            return []
        result = []
        protected = [seq for seq in self.protected if self._eligible(seq)]
        for _ in range(self.cfg.batch):
            if protected and self.rng.random() < 0.20:
                result.append(self.rng.choice(protected))
            else:
                group = self.rng.choice(list(groups.values()))
                start = self.rng.choice(group)
                result.append(rows[start:start+n])
        return result

    def loss(self, seq):
        device = next(self.net.parameters()).device
        h = torch.as_tensor(seq[0]["workspace"], device=device).float().unsqueeze(0)
        losses, metrics = [], {"td": 0.0, "imitation": 0.0, "world": 0.0}
        for i, row in enumerate(seq):
            packet = tensor_packet(row["packet"], device)
            if i < self.cfg.burn_in:
                with torch.no_grad():
                    h, _, _ = self.net(*packet, h)
                continue
            h, latent, q = self.net(*packet, h)
            mask = torch.tensor(row["mask"], device=device, dtype=torch.bool)[None]
            q_masked = q.masked_fill(~mask, -1e9)
            with torch.no_grad():
                _, next_latent, next_q = self.target(*tensor_packet(row["next"], device), h.detach())
                next_mask = torch.tensor(row["next_mask"], device=device, dtype=torch.bool)[None]
                value = next_q.masked_fill(~next_mask, -1e9).max(dim=-1).values
                target = row["reward"] + (0.0 if row["boundary"] else
                    self.cfg.gamma ** row["duration"]) * value
            td = F.smooth_l1_loss(q[:, row["action"]], target)
            imitation = F.cross_entropy(q_masked, torch.tensor([row["teacher"]], device=device))
            world_losses = []
            imagined = latent
            for j in range(i, min(len(seq), i + self.cfg.horizon)):
                transition = seq[j]
                if j > i and transition["serial"] != seq[j-1]["serial"] + 1:
                    break
                action = torch.tensor([transition["action"]], device=device)
                imagined, mean, logvar, reward, success = self.net.imagine(imagined, action)
                truth = torch.as_tensor(transition["next"]["sensory"], device=device)[None]
                # Heteroscedastic Gaussian predictive loss; uncertainty is not consciousness.
                predict = 0.5 * ((mean - truth).square() * (-logvar).exp() + logvar).mean()
                rloss = F.smooth_l1_loss(reward, torch.tensor([transition["reward"]], device=device))
                sloss = F.binary_cross_entropy_with_logits(success,
                    torch.tensor([float(transition["success"])], device=device))
                if j == i:
                    predict = predict + 0.1 * F.smooth_l1_loss(imagined, next_latent)
                world_losses.append(predict + 0.5 * rloss + 0.2 * sloss)
                if transition["boundary"]:
                    break
            world = torch.stack(world_losses).mean()
            losses.append(td + 0.15 * imitation + world)
            for key, item in (("td", td), ("imitation", imitation), ("world", world)):
                metrics[key] += float(item.detach())
        return torch.stack(losses).mean(), {k: v / len(losses) for k, v in metrics.items()}

    def train(self):
        sequences = self._samples()
        if not sequences:
            return None
        self.optimizer.zero_grad(set_to_none=True)
        totals = {}
        for seq in sequences:
            loss, metrics = self.loss(seq)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite cognitive loss; refusing an optimizer update")
            (loss / len(sequences)).backward()
            for k, v in {"loss": float(loss.detach()), **metrics}.items():
                totals[k] = totals.get(k, 0.0) + v / len(sequences)
        norm = nn.utils.clip_grad_norm_(self.net.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for p, target in zip(self.net.parameters(), self.target.parameters()):
                target.lerp_(p, 0.01)
        self.updates += 1
        self.last = {**totals, "gradient_norm": float(norm), "updates": self.updates}
        return self.last

    def state_dict(self):
        return {"target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                "replay": list(self.replay), "protected": self.protected,
                "rng": self.rng.getstate(), "updates": self.updates, "last": self.last}

    def load_state_dict(self, state):
        self.target.load_state_dict(state["target"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.extend(state["replay"])
        self.protected = state["protected"]
        self.rng.setstate(state["rng"])
        self.updates = state["updates"]
        self.last = state["last"]
