"""Connected, independently inspectable neural components for the local lab."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .world import SIZE, SENSOR_DIM, MEMORY_DIM, STYLE_DIM


@dataclass(frozen=True)
class BrainConfig:
    width: int = 128
    hidden: int = 192


class DevelopmentBrain(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or BrainConfig()
        d, h = self.cfg.width, self.cfg.hidden
        self.perception = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(24 * SIZE * SIZE, d), nn.Tanh())
        self.interoception = nn.Sequential(nn.Linear(SENSOR_DIM + STYLE_DIM, d), nn.Tanh())
        self.memory_adapter = nn.Sequential(nn.Linear(MEMORY_DIM, d), nn.Tanh())
        self.workspace = nn.GRUCell(d * 3, h)
        self.policy = nn.Sequential(nn.Linear(h, d), nn.ReLU(), nn.Linear(d, 5))
        self.q = nn.Sequential(nn.Linear(h, d), nn.ReLU(), nn.Linear(d, 5))
        # A consequence model, conditioned on the lived workspace and action.
        self.world = nn.Sequential(nn.Linear(h + 5, d), nn.ReLU(), nn.Linear(d, 8))

    def initial(self, n, device):
        return torch.zeros(n, self.cfg.hidden, device=device)

    def forward(self, grid, sensors, memory, style, hidden):
        x = torch.cat((self.perception(grid),
                       self.interoception(torch.cat((sensors, style), -1)),
                       self.memory_adapter(memory)), -1)
        hidden = self.workspace(x, hidden)
        return {"hidden": hidden, "logits": self.policy(hidden), "q": self.q(hidden)}

    def predict(self, hidden, action_onehot):
        pred = self.world(torch.cat((hidden, action_onehot), -1))
        return {"body": torch.sigmoid(pred[..., :3]),
                "reward": pred[..., 3], "collision_logit": pred[..., 4],
                "resource_logits": pred[..., 5:8]}

    def choose(self, result):
        # Q is a small learned correction to the demonstrator-initialized policy.
        # Imagined consequences are logged until planning earns its own test.
        return result["logits"] + .10 * torch.tanh(result["q"])


class DevelopmentForecaster(nn.Module):
    """Predict measured learning-block deltas; forecasts are advisory in v0.12."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 3), nn.Tanh())

    def forward(self, summary):
        return self.net(summary)


def tensor(array, device):
    # Construct on CPU first: avoid unsupported DirectML numpy/scalar paths.
    return torch.as_tensor(np.asarray(array), dtype=torch.float32).to(device)


def observation_tensors(obs, memory, style, device):
    return (tensor(obs["grid"][None], device), tensor(obs["sensors"][None], device),
            tensor(np.asarray(memory)[None], device), tensor(np.asarray(style)[None], device))


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def fingerprint(module):
    h = hashlib.sha256()
    for name, t in sorted(cpu_state(module).items()):
        h.update(name.encode())
        h.update(str((str(t.dtype), tuple(t.shape))).encode())
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def model_payload(model):
    return {"config": asdict(model.cfg), "model": cpu_state(model), "model_sha256": fingerprint(model)}


def load_model(payload, device):
    model = DevelopmentBrain(BrainConfig(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    if fingerprint(model) != payload["model_sha256"]:
        raise ValueError("Development model fingerprint mismatch")
    return model.to(device)


def finite_step(loss, model, optimizer, clip=1.):
    if not bool(torch.isfinite(loss.detach().cpu()).all()):
        raise FloatingPointError("Nonfinite learning loss; candidate rejected")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # The small scalar reduction is deliberately on CPU; training stays on DML.
    total = 0.
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().cpu()
            if not bool(torch.isfinite(g).all()):
                raise FloatingPointError("Nonfinite gradient; candidate rejected")
            total += float(g.square().sum())
    factor = min(1., clip / (total ** .5 + 1e-8))
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(factor)
    optimizer.step()
    return float(loss.detach().cpu())
