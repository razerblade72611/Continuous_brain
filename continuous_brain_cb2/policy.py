"""The first learned skill: a policy that acts from the predictive core's state, first by imitating the routine.

Two arms (Linear plan Q3), so the dependence on the hand-built map is measured, not assumed:

- ``P-A``: core state + body sense + its memory organ's readout (the 62 belief features, without any routine-
  internal state, and a 33 x 33 window of its remembered map);
- ``P-B``: core state + body sense only.

Outputs: the 17 Crafter actions, and an auxiliary guess at which routine option is running (its intent).  The
core is frozen while the policy learns, so a probe of the core is a probe of what the world model learned alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .archive import CROP, FEATURES
from .data import BODY_DIM, N_ACTIONS

N_OPTIONS = 4
N_MATERIALS = 20            # remembered map: -1 unknown -> 0, materials 0..18 -> 1..19


@dataclass
class PolicyConfig:
    arm: str = "P-A"
    units: int = 512
    layers: int = 3
    map_channels: int = 32
    lr: float = 3e-4
    aux_scale: float = 0.2


class MapEncoder(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.emb = nn.Embedding(N_MATERIALS, 8)
        self.net = nn.Sequential(nn.Conv2d(8, ch, 3, 2, 1), nn.SiLU(), nn.Conv2d(ch, ch, 3, 2, 1), nn.SiLU(),
                                 nn.Conv2d(ch, ch, 3, 2, 1), nn.SiLU())       # 33 -> 17 -> 9 -> 5
        self.out = ch * 5 * 5

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        if crop.shape[-1] == CROP * CROP:
            crop = crop.reshape(crop.shape[:-1] + (CROP, CROP))
        x = self.emb((crop.long() + 1).clamp(0, N_MATERIALS - 1))           # [..., 33, 33, 8]
        lead = x.shape[:-3]
        x = x.reshape((-1, CROP, CROP, 8)).permute(0, 3, 1, 2)
        return self.net(x).reshape(lead + (self.out,))


class SkillPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig, feat_size: int, device: str = "cpu"):
        super().__init__()
        self.cfg, self.device, self.feat_size = cfg, device, feat_size
        inp = feat_size + BODY_DIM
        self.map = None
        if cfg.arm == "P-A":
            self.map = MapEncoder(cfg.map_channels)
            inp += FEATURES + self.map.out
        elif cfg.arm != "P-B":
            raise ValueError(cfg.arm)
        layers, d = [], inp
        for _ in range(cfg.layers):
            layers += [nn.Linear(d, cfg.units), nn.LayerNorm(cfg.units), nn.SiLU()]
            d = cfg.units
        self.trunk = nn.Sequential(*layers)
        self.action = nn.Linear(d, N_ACTIONS)
        self.option = nn.Linear(d, N_OPTIONS)
        self.to(device)
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def forward(self, feat: torch.Tensor, body, features=None, map_crop=None):
        x = [feat, torch.as_tensor(body, dtype=torch.float32, device=self.device)]
        if self.map is not None:
            x += [self._t(features), self.map(self._t(map_crop, torch.int64))]
        h = self.trunk(torch.cat(x, -1))
        return self.action(h), self.option(h)

    def losses(self, feat, batch: dict, mask=None):
        logits, opt_logits = self(feat, batch["body"], batch.get("features"), batch.get("map_crop"))
        target = self._t(batch["teacher_action"], torch.int64)
        ce = F.cross_entropy(logits.reshape(-1, N_ACTIONS), target.reshape(-1), reduction="none")
        aux = F.cross_entropy(opt_logits.reshape(-1, N_OPTIONS),
                              self._t(batch["teacher_option"], torch.int64).clamp(0, N_OPTIONS - 1).reshape(-1),
                              reduction="none")
        m = torch.ones_like(ce) if mask is None else self._t(mask).reshape(-1)
        loss = ((ce + self.cfg.aux_scale * aux) * m).sum() / m.sum().clamp(min=1)
        acc = ((logits.reshape(-1, N_ACTIONS).argmax(-1) == target.reshape(-1)).float() * m).sum() / m.sum().clamp(min=1)
        return loss, {"bc_loss": float(loss.detach()), "agreement": float(acc)}

    def train_step(self, feat, batch, mask=None) -> dict:
        self.train()
        loss, m = self.losses(feat, batch, mask)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        self.opt.step()
        return m

    @torch.no_grad()
    def act(self, feat, body, features=None, map_crop=None, greedy: bool = True, rng=None) -> np.ndarray:
        self.eval()
        logits, _ = self(feat, body, features, map_crop)
        if greedy:
            return logits.argmax(-1).cpu().numpy()
        p = torch.softmax(logits, -1).cpu().numpy()
        rng = rng or np.random.default_rng()
        return np.array([rng.choice(N_ACTIONS, p=pi / pi.sum()) for pi in p.reshape(-1, N_ACTIONS)])

    def state_dict_full(self) -> dict:
        return {"config": asdict(self.cfg), "feat_size": self.feat_size, "model": self.state_dict()}

    @classmethod
    def from_state(cls, saved: dict, device: str = "cpu") -> "SkillPolicy":
        p = cls(PolicyConfig(**saved["config"]), saved["feat_size"], device)
        p.load_state_dict(saved["model"])
        return p
