"""CB2-L (L-C): choosing among commands by their imagined consequences, learned from its own lived experience.

Machinery, not answers: nothing here says which command is right.  Three learned parts, all trained in sleep on
the individual's own archive:

- ``HandsModel``: what its own hands do -- the primitive action its executor produces, given the core's state and
  a command (learned from its lived (state, command, action) triples).  Used only inside imagination.
- ``ValueNet``: how its life tends to go from a state, in its own drives (the environment's reward, and death),
  learned by TD(lambda) on its lived returns.
- ``choose``: for each command, the core imagines H steps with the hands model acting under that command; the
  imagined drive (predicted reward, minus the death signal times the probability the life ends) plus the discounted
  value at the horizon is its score.  The selector's own preference enters as a prior with strength ``tau``:
  ``score_c = G_c + tau * log pi(c)``; with the chooser disabled, the selector acts alone (the Gate 2 ablation).

Deterministic: imagination uses the core's mode (no sampling), so a life with a chooser replays exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import N_ACTIONS

N_COMMANDS = 8
DEATH = 10.0                      # the death signal (drive), as in S2b
VCLIP = 15.0                      # value head output bound (symlog space: |v| <= ~3.3e6, far above any real return);
GRAD_CLIP = 100.0                 # without it the value head overflowed to inf/NaN in the first L1 GPU sleep


@dataclass
class ChooserConfig:
    units: int = 256
    horizon: int = 15
    gamma: float = 0.99
    lam: float = 0.95
    tau: float = 1.0              # strength of the selector's own preference
    enabled: bool = True
    lr: float = 3e-4


def _mlp(i, o, units):
    return nn.Sequential(nn.Linear(i, units), nn.SiLU(), nn.Linear(units, units), nn.SiLU(), nn.Linear(units, o))


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class Chooser(nn.Module):
    def __init__(self, cfg: ChooserConfig, feat_size: int, device: str = "cpu"):
        super().__init__()
        self.cfg, self.feat_size, self.device = cfg, feat_size, device
        self.hands = _mlp(feat_size + N_COMMANDS, N_ACTIONS, cfg.units)
        self.value = _mlp(feat_size, 1, cfg.units)
        self.to(device)
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    # ----------------------------------------------------------------------------------------- parts
    def hands_logits(self, feat, command):
        return self.hands(torch.cat([feat, F.one_hot(command, N_COMMANDS).float()], -1))

    def v(self, feat):
        return symexp(self.value(feat).squeeze(-1).clamp(-VCLIP, VCLIP))

    # ----------------------------------------------------------------------------------------- acting
    @torch.no_grad()
    def imagined_returns(self, core, post: dict) -> torch.Tensor:
        """[N_COMMANDS] imagined drive return from the core's present state, one rollout per command."""
        st = {k: v.expand(N_COMMANDS, *v.shape[1:]).contiguous() for k, v in post.items()}
        cmd = torch.arange(N_COMMANDS, device=self.device)
        g, disc = torch.zeros(N_COMMANDS, device=self.device), torch.ones(N_COMMANDS, device=self.device)
        for _ in range(self.cfg.horizon):
            f = core.feat(st)
            a = self.hands_logits(f, cmd).argmax(-1)
            st = core.rssm.img_step(st, core._onehot(a), sample=False)
            f2 = core.feat(st)
            r = core.heads["reward"](f2).mode().squeeze(-1)
            cont = core.heads["cont"](f2).mean.squeeze(-1)
            g = g + disc * (r - DEATH * (1.0 - cont))
            disc = disc * self.cfg.gamma * cont
        return g + disc * self.v(core.feat(st))

    @torch.no_grad()
    def choose(self, core, post: dict, prior_logits: torch.Tensor) -> tuple[int, dict]:
        g = self.imagined_returns(core, post)
        logp = F.log_softmax(prior_logits.reshape(-1), -1)
        score = g + self.cfg.tau * logp
        bad = not bool(torch.isfinite(score).all())
        c = int(logp.argmax()) if bad else int(score.argmax())   # never act on non-finite imagination: the prior decides
        return c, {"imagined": [round(float(x), 3) for x in g], "prior_choice": int(logp.argmax()), "chosen": c,
                   "changed": c != int(logp.argmax()), "nonfinite": bad}

    # ----------------------------------------------------------------------------------------- learning
    def losses(self, feat, command, action, reward, done, is_first):
        """feat [B,T,F] (detached core states of its lived windows), command/action/reward/done [B,T]."""
        hl = F.cross_entropy(self.hands_logits(feat, command).reshape(-1, N_ACTIONS), action.reshape(-1))
        with torch.no_grad():
            r = reward - DEATH * done.float()
            v = self.v(feat)
            T = feat.shape[1]
            ret = torch.zeros_like(v)
            nxt = v[:, -1]
            for t in range(T - 1, -1, -1):
                end = done[:, t].float()
                boot = v[:, t + 1] if t + 1 < T else v[:, t]
                # a life's first row starts a new life: the previous row must not bootstrap across it
                cut = is_first[:, t + 1].float() if t + 1 < T else torch.zeros_like(end)
                nxt = r[:, t] + self.cfg.gamma * (1 - end) * (1 - cut) * ((1 - self.cfg.lam) * boot + self.cfg.lam * nxt)
                ret[:, t] = nxt
        vl = F.mse_loss(self.value(feat).squeeze(-1)[:, : T - 1], symlog(ret)[:, : T - 1])
        acc = (self.hands_logits(feat, command).argmax(-1) == action).float().mean()
        return hl + vl, {"hands_loss": float(hl), "hands_acc": float(acc), "value_loss": float(vl)}

    def train_step(self, *args) -> dict:
        self.train()
        loss, m = self.losses(*args)
        self.opt.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):                # skip (and count) a non-finite update instead of applying it
            return {**m, "skipped": 1}
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(gn)):
            self.opt.zero_grad(set_to_none=True)
            return {**m, "skipped": 1, "grad_norm": float(gn)}
        self.opt.step()
        return {**m, "skipped": 0, "grad_norm": float(gn)}

    # ----------------------------------------------------------------------------------------- persistence
    def state_dict_full(self) -> dict:
        return {"config": asdict(self.cfg), "feat_size": self.feat_size, "model": self.state_dict()}

    @classmethod
    def from_state(cls, saved: dict, device: str = "cpu", **override) -> "Chooser":
        ch = cls(ChooserConfig(**{**saved["config"], **override}), saved["feat_size"], device)
        ch.load_state_dict(saved["model"])
        return ch
