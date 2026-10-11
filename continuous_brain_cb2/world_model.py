"""The predictive core: one recurrent world model whose state is the individual's present moment.

The dynamics are DreamerV3's RSSM (Hafner et al. 2023), from the MIT-licensed PyTorch port vendored in
``third_party/dreamerv3_torch``: a GRU deterministic state ``deter`` and a 32 x 32 categorical ``stoch``.  It sees
the frame and its body sense; it predicts the frame, its body, the reward, whether its life goes on, and the 12
risk labels of ``data.RISKS`` x ``data.RISK_H``.

The interface is the contract; the RSSM behind it is replaceable:

- ``initial(B)`` -> state
- ``step(state, prev_action, image, body, is_first, sample=False)`` -> state   (one waking moment)
- ``scan(image, body, prev_action, is_first, state=None, sample=False)`` -> states [B, T]   (training, probes)
- ``imagine(state, policy, H)`` -> imagined states, actions                       (dreams, planning)
- ``feat(state)`` -> the vector everything downstream reads

``step`` repeated and ``scan`` agree exactly (tests), so a core trained with ``scan`` lives with ``step``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .data import BODY_DIM, N_ACTIONS, N_RISK

MAP_CLASSES = 14                  # believed-map values -1 (never seen) .. 12 (vision MATERIALS), shifted by +1
VIEW_H, VIEW_W = 7, 9             # the view: 7 cells in y (rows of the image), 9 in x


def in_view_crop(d: int) -> np.ndarray:
    """[d, d] bool, True where a cell of the egocentric crop is in the current view.  The crop is indexed [x, y]
    (record.map_crop slices memory["map"][x, y]), NOT image [y, x]; built from the memory's own view geometry
    (memory.view_cells) so the two cannot drift apart.  Sol's review (S-002): the old mask was transposed and,
    at radius 8, counted 14 visible cells as out of view and dropped 14 hidden ones."""
    from continuous_brain_individual.memory import view_cells
    c = d // 2
    m = np.zeros((d, d), bool)
    for _, _, x, y in view_cells((c, c)):
        if 0 <= x < d and 0 <= y < d:
            m[x, y] = True
    return m
from .third_party.dreamerv3_torch import networks, tools


@dataclass
class CoreConfig:
    deter: int = 512
    hidden: int = 512
    stoch: int = 32
    discrete: int = 32
    cnn_depth: int = 32
    units: int = 512
    head_layers: int = 2
    body_layers: int = 2
    kl_free: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    risk_scale: float = 1.0
    map_radius: int = 0          # CB2-L (L-A): >0 adds a head predicting its own remembered map around it
    map_scale: float = 1.0
    lr: float = 1e-4
    eps: float = 1e-8
    grad_clip: float = 1000.0

    @classmethod
    def sized(cls, name: str) -> "CoreConfig":
        """Named sizes for the scaling curve (Linear plan Q4): S ~ the local core, M, L for the first GPU."""
        return {"XS": cls(deter=256, hidden=256, cnn_depth=16, units=256),
                "S": cls(),
                "M": cls(deter=1024, hidden=768, cnn_depth=48, units=768),
                "L": cls(deter=2048, hidden=1024, cnn_depth=64, units=1024)}[name]


class PredictiveCore(nn.Module):
    def __init__(self, cfg: CoreConfig, device: str = "cpu"):
        super().__init__()
        self.cfg, self.device = cfg, device
        self.image_enc = networks.ConvEncoder((64, 64, 3), cfg.cnn_depth, "SiLU", True, 4, 4)
        self.body_enc = networks.MLP(BODY_DIM, None, cfg.body_layers, cfg.units, "SiLU", True, symlog_inputs=True,
                                     device=device, name="Body")
        self.embed_size = self.image_enc.outdim + cfg.units
        self.rssm = networks.RSSM(cfg.stoch, cfg.deter, cfg.hidden, 1, cfg.discrete, "SiLU", True, "none",
                                  "sigmoid2", 0.1, 0.01, "learned", N_ACTIONS, self.embed_size, device)
        f = self.feat_size
        self.image_dec = networks.ConvDecoder(f, (3, 64, 64), cfg.cnn_depth, "SiLU", True, 4, 4, 1.0, False)
        self.heads = nn.ModuleDict({
            "body": networks.MLP(f, (BODY_DIM,), cfg.head_layers, cfg.units, "SiLU", True, dist="symlog_mse",
                                 device=device, name="BodyHead"),
            "reward": networks.MLP(f, (255,), cfg.head_layers, cfg.units, "SiLU", True, dist="symlog_disc",
                                   outscale=0.0, device=device, name="Reward"),
            "cont": networks.MLP(f, (), cfg.head_layers, cfg.units, "SiLU", True, dist="binary", device=device,
                                 name="Cont"),
            "risk": networks.MLP(f, (N_RISK,), cfg.head_layers, cfg.units, "SiLU", True, dist="binary",
                                 device=device, name="Risk")})
        if cfg.map_radius:
            # L-A: what it itself perceived at the places around it, in view or not (its believed map, written only
            # from its own vision) -- a target it can only meet out of view by remembering
            d = 2 * cfg.map_radius + 1
            self.heads["map"] = nn.Sequential(nn.Linear(f, cfg.units), nn.SiLU(), nn.Linear(cfg.units, cfg.units),
                                              nn.SiLU(), nn.Linear(cfg.units, d * d * MAP_CLASSES))
        self.to(device)
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.lr, eps=cfg.eps)

    # ------------------------------------------------------------------------------------------ geometry
    @property
    def feat_size(self) -> int:
        return self.cfg.stoch * self.cfg.discrete + self.cfg.deter

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict:
        return {"core": "dreamerv3-rssm", "config": asdict(self.cfg), "params": self.n_params(),
                "feat_size": self.feat_size}

    # ------------------------------------------------------------------------------------------ interface
    def _t(self, x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dtype)
        x = np.asarray(x)
        if not x.flags.writeable:
            x = x.copy()
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def embed(self, image, body) -> torch.Tensor:
        """image uint8 [..., 64, 64, 3], body float [..., BODY_DIM] -> embedding [..., E]."""
        img = self._t(image) / 255.0
        return torch.cat([self.image_enc(img), self.body_enc(self._t(body))], -1)

    def initial(self, B: int) -> dict:
        return self.rssm.initial(B)

    def feat(self, state: dict) -> torch.Tensor:
        return self.rssm.get_feat(state)

    def _onehot(self, a) -> torch.Tensor:
        return nn.functional.one_hot(self._t(a, torch.int64), N_ACTIONS).float()

    @torch.no_grad()
    def step(self, state, prev_action, image, body, is_first, sample: bool = False) -> dict:
        """One waking moment for a batch of B: arrays shaped [B, ...]; ``state`` None at the very first."""
        emb = self.embed(image, body)
        first = self._t(is_first)
        post, _ = self.rssm.obs_step(None if state is None else dict(state), self._onehot(prev_action), emb, first,
                                     sample=sample)
        return post

    def scan(self, image, body, prev_action, is_first, state=None, sample: bool = False, grad: bool = False):
        """Posterior and prior states for sequences [B, T, ...]."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            emb = self.embed(image, body)
            act = self._onehot(prev_action)
            first = self._t(is_first)
            posts, priors = [], []
            st = state
            for t in range(emb.shape[1]):
                post, prior = self.rssm.obs_step(None if st is None else dict(st), act[:, t].clone(), emb[:, t],
                                                 first[:, t], sample=sample)
                posts.append(post)
                priors.append(prior)
                st = post
            stack = lambda xs: {k: torch.stack([x[k] for x in xs], 1) for k in xs[0]}
            return stack(posts), stack(priors)

    @torch.no_grad()
    def imagine(self, state: dict, policy, H: int, sample: bool = True):
        """Roll the dynamics forward H steps under ``policy(feat) -> action indices``; nothing is observed."""
        feats, acts = [], []
        st = state
        for _ in range(H):
            f = self.feat(st)
            a = policy(f)
            feats.append(f)
            acts.append(a)
            st = self.rssm.img_step(st, self._onehot(a), sample=sample)
        return torch.stack(feats, 1), torch.stack(acts, 1), st

    # ------------------------------------------------------------------------------------------ learning
    def losses(self, batch: dict) -> tuple[torch.Tensor, dict, dict]:
        emb = self.embed(batch["image"], batch["body"])
        post, prior = self.rssm.observe(emb, self._onehot(batch["prev_action"]), self._t(batch["is_first"]))
        kl, kl_value, dyn, rep = self.rssm.kl_loss(post, prior, self.cfg.kl_free, self.cfg.dyn_scale,
                                                   self.cfg.rep_scale)
        feat = self.feat(post)
        img = self._t(batch["image"]) / 255.0
        losses = {"image": -tools.MSEDist(self.image_dec(feat)).log_prob(img),
                  "body": -self.heads["body"](feat).log_prob(self._t(batch["body"])),
                  "reward": -self.heads["reward"](feat).log_prob(self._t(batch["reward"])[..., None]),
                  "cont": -self.heads["cont"](feat).log_prob(self._t(batch["cont"])[..., None]),
                  "risk": -self.heads["risk"](feat).log_prob(self._t(batch["risk"])) * self.cfg.risk_scale}
        if "map" in self.heads and "map_crop" in batch:
            logits, tgt = self.map_logits(feat), self.map_target(batch["map_crop"])
            losses["map"] = nn.functional.cross_entropy(logits.reshape(-1, MAP_CLASSES), tgt.reshape(-1),
                                                        reduction="none").reshape(tgt.shape).mean(-1) * self.cfg.map_scale
        total = sum(v.mean() for v in losses.values()) + kl.mean()
        metrics = {f"{k}_loss": float(v.mean().detach()) for k, v in losses.items()}
        metrics.update(kl=float(kl_value.mean().detach()), dyn_loss=float(dyn.mean().detach()),
                       rep_loss=float(rep.mean().detach()))
        return total, metrics, post

    def map_logits(self, feat: torch.Tensor) -> torch.Tensor:
        d = 2 * self.cfg.map_radius + 1
        return self.heads["map"](feat).reshape(*feat.shape[:-1], d * d, MAP_CLASSES)

    def map_target(self, crop) -> torch.Tensor:
        """The centre (2r+1)^2 of the 33 x 33 believed-map crop, as class indices [..., d*d]."""
        r, c = self.cfg.map_radius, 16
        crop = np.asarray(crop)
        if crop.shape[-1] == 33 * 33:                       # stored flat
            crop = crop.reshape(*crop.shape[:-1], 33, 33)
        sub = crop[..., c - r:c + r + 1, c - r:c + r + 1].astype(np.int64) + 1
        return self._t(sub.reshape(*sub.shape[:-2], -1), torch.int64)

    def out_of_view_mask(self) -> np.ndarray:
        return ~in_view_crop(2 * self.cfg.map_radius + 1).reshape(-1)

    def train_step(self, batch: dict) -> dict:
        self.train()
        total, metrics, _ = self.losses(batch)
        self.opt.zero_grad(set_to_none=True)
        total.backward()
        metrics["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.grad_clip))
        self.opt.step()
        metrics["loss"] = float(total.detach())
        return metrics

    @torch.no_grad()
    def evaluate(self, batch: dict) -> dict:
        """Held-out metrics: per-pixel error, risk calibration (Brier) and the open-loop prediction error over the
        second half of each window (the core observes the first half, then only imagines under the actions
        that were taken)."""
        self.eval()
        _, metrics, post = self.losses(batch)
        feat = self.feat(post)
        risk_p = self.heads["risk"](feat).mean.cpu().numpy()
        metrics["risk_brier"] = float(((risk_p - batch["risk"]) ** 2).mean())
        metrics["risk_brier_base"] = float(((batch["risk"].mean((0, 1)) - batch["risk"]) ** 2).mean())
        T = batch["image"].shape[1]
        h = T // 2
        last = {k: v[:, h - 1] for k, v in post.items()}
        acts = self._onehot(batch["prev_action"][:, h:])
        prior = self.rssm.imagine_with_action(acts, last)
        pred = self.image_dec(self.feat(prior)).cpu().numpy()
        truth = batch["image"][:, h:].astype(np.float32) / 255.0
        metrics["openloop_mse"] = float(((pred - truth) ** 2).mean())
        metrics["copy_last_mse"] = float(((batch["image"][:, h - 1:h].astype(np.float32) / 255.0 - truth) ** 2).mean())
        if "map" in self.heads and "map_crop" in batch:
            metrics.update(self.map_metrics(feat, batch["map_crop"]))
        return metrics

    @torch.no_grad()
    def map_metrics(self, feat, crop, skip: int = 0) -> dict:
        """Out-of-view accuracy of its remembered-map head on tiles it had seen, against the best constant guess."""
        pred = self.map_logits(feat).argmax(-1)[:, skip:].cpu().numpy()
        tgt = self.map_target(crop)[:, skip:].cpu().numpy()
        oov = self.out_of_view_mask()
        p, t = pred[..., oov], tgt[..., oov]
        seen = t > 0
        if not seen.any():
            return {}
        const = np.bincount(t[seen], minlength=MAP_CLASSES).argmax()
        cls = [k for k in np.unique(t[seen])]
        bal = float(np.mean([(p[seen][t[seen] == k] == k).mean() for k in cls]))
        return {"map_oov_seen_acc": float((p[seen] == t[seen]).mean()), "map_oov_seen_balanced": bal,
                "map_oov_seen_balanced_const": 1.0 / len(cls),
                "map_oov_seen_const": float((t[seen] == const).mean()),
                "map_oov_known_acc": float(((p > 0) == seen).mean())}

    # ------------------------------------------------------------------------------------------ persistence
    def state_dict_full(self) -> dict:
        return {"config": asdict(self.cfg), "model": self.state_dict(), "opt": self.opt.state_dict()}

    @classmethod
    def from_state(cls, saved: dict, device: str = "cpu", **override) -> "PredictiveCore":
        """``override`` may add a head (e.g. map_radius=8) to a saved core: its new parameters start fresh and the
        optimiser restarts; everything the core already had is loaded exactly."""
        cfg = {**saved["config"], **override}
        core = cls(CoreConfig(**cfg), device)
        missing, unexpected = core.load_state_dict(saved["model"], strict=not override)
        if unexpected or any(not k.startswith("heads.map.") for k in missing):
            raise ValueError(f"saved core does not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
        if saved.get("opt") and not missing:
            core.opt.load_state_dict(saved["opt"])
        return core
