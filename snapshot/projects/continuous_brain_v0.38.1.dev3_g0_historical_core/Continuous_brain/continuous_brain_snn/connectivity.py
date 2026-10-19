"""Fixed-capacity sparse connectivity slots (DAV-35).

Every postsynaptic neuron owns exactly `fanin_capacity` preallocated slots:

    src[N, K]          presynaptic neuron id
    weight[N, K]       non-negative magnitude (sign comes from the source's cell type)
    candidate[N, K]    slot holds a valid anatomical candidate edge
    active[N, K]       edge is live (active implies candidate)
    eligibility[N, K]  reserved for plasticity (E1+)
    utility[N, K]      reserved for structural plasticity (E1+)

Structural change only flips `active`; tensor shapes never change, so wake-time
updates never reallocate indices. Hard compaction is a sleep-time operation
(not part of E0).

Candidates are sampled once at genesis with p(src) ~ exp(-distance / length)
within the allowed region pairs below: anatomy supplies the neighbourhood.

Spike propagation is gather-based (each post reads its own K sources), so it
has no atomic scatter-adds and is deterministic on a given device/backend.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from .anatomy import REGIONS, Anatomy
from .config import ConnectivityConfig

# target region -> allowed source regions. Sensory regions receive no recurrent input.
ALLOWED_SOURCES: dict[str, tuple[str, ...]] = {
    "retina": (),
    "proprioception": (),
    "cortex": ("retina", "proprioception", "cortex", "motor"),
    "motor": ("cortex",),
}

SLOT_FIELDS = ("src", "weight", "candidate", "active", "eligibility", "utility")


class SparseSlots:
    def __init__(self, src, weight, candidate, active, eligibility, utility, src_exc,
                 max_occupancy: float = 1.0, fanout_cap: int | None = None):
        self.max_occupancy = float(max_occupancy)
        self.fanout_cap = fanout_cap
        self.src, self.weight = src, weight
        self.candidate, self.active = candidate, active
        self.eligibility, self.utility = eligibility, utility
        self.src_exc = src_exc
        self._version = 0

    # ------------------------------------------------------------------ genesis

    @classmethod
    def build(cls, anatomy: Anatomy, cfg: ConnectivityConfig, seed: int) -> "SparseSlots":
        g = torch.Generator().manual_seed(seed + 7919)
        n, k = anatomy.n, cfg.fanin_capacity
        coords = torch.from_numpy(anatomy.coords_mm)
        region = torch.from_numpy(anatomy.region.astype(np.int64))
        src = torch.zeros(n, k, dtype=torch.long)
        candidate = torch.zeros(n, k, dtype=torch.bool)

        for target, sources in ALLOWED_SOURCES.items():
            posts = torch.nonzero(region == REGIONS.index(target)).flatten()
            if not sources or posts.numel() == 0:
                continue
            pool = torch.nonzero(torch.isin(region, torch.tensor([REGIONS.index(s) for s in sources]))).flatten()
            k_eff = min(k, int(pool.numel()) - 1)
            for chunk in torch.split(posts, 512):
                d = torch.cdist(coords[chunk], coords[pool])                     # [c, S] mm
                p = torch.exp(-d / cfg.length_constant_mm)
                p[pool[None, :] == chunk[:, None]] = 0.0                         # no autapses
                p = p + 1e-12 * (pool[None, :] != chunk[:, None])                # keep rows sampleable
                pick = torch.multinomial(p, k_eff, replacement=False, generator=g)
                src[chunk, :k_eff] = pool[pick]
                candidate[chunk, :k_eff] = True

        is_exc = torch.from_numpy(anatomy.is_exc)
        src_exc = is_exc[src]
        base = torch.where(src_exc, cfg.w_exc, cfg.w_inh)
        weight = (base * torch.exp(cfg.w_jitter_sigma * torch.randn(n, k, generator=g))).clamp(max=cfg.w_max)
        weight = weight * candidate
        active = candidate & (torch.rand(n, k, generator=g) < cfg.initial_active_fraction)
        slots = cls(src, weight, candidate, active, torch.zeros(n, k), torch.zeros(n, k), src_exc,
                    cfg.max_occupancy, cfg.fanout_cap)
        slots._enforce_fanout(cfg.fanout_cap, g)
        return slots

    def _enforce_fanout(self, cap: int, g: torch.Generator) -> None:
        """Deterministically deactivate random excess edges from over-cap sources."""
        for _ in range(8):
            out = self.fanout()
            over = torch.nonzero(out > cap).flatten()
            if over.numel() == 0:
                return
            for s in over.tolist():
                rows, cols = torch.nonzero(self.active & (self.src == s), as_tuple=True)
                drop = torch.randperm(rows.numel(), generator=g)[: rows.numel() - cap]
                self.active[rows[drop], cols[drop]] = False
        self._version += 1

    # ------------------------------------------------------------------ structure

    @property
    def n(self) -> int:
        return int(self.src.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.src.shape[1])

    @property
    def version(self) -> int:
        return self._version

    def fanin(self) -> torch.Tensor:
        return self.active.sum(dim=1)

    def fanout(self) -> torch.Tensor:
        return torch.bincount(self.src[self.active], minlength=self.n)

    def occupancy(self) -> torch.Tensor:
        return self.fanin().float() / self.capacity

    def set_active(self, post: torch.Tensor, slot: torch.Tensor, value: bool) -> None:
        """Activate or prune specific slots.

        Activation is refused (nothing changes) if a slot holds no candidate edge
        or if the result would break the occupancy or fan-out cap.
        """
        post, slot = post.long(), slot.long()
        if value:
            if not bool(self.candidate[post, slot].all()):
                raise ValueError("cannot activate a slot that holds no candidate edge")
            trial = self.active.clone()
            trial[post, slot] = True
            occ = trial.sum(dim=1).float() / self.capacity
            if float(occ.max()) > self.max_occupancy + 1e-9:
                raise ValueError(f"activation would exceed the occupancy cap ({self.max_occupancy})")
            if self.fanout_cap is not None:
                fan = torch.bincount(self.src[trial], minlength=self.n)
                if int(fan.max()) > self.fanout_cap:
                    raise ValueError(f"activation would exceed the fan-out cap ({self.fanout_cap})")
            self.active = trial
        else:
            self.active[post, slot] = False
        self._version += 1

    def check_invariants(self, cfg: ConnectivityConfig) -> list[str]:
        problems = []
        shapes = {f: tuple(getattr(self, f).shape) for f in SLOT_FIELDS}
        if len(set(shapes.values())) != 1:
            problems.append(f"slot tensors disagree in shape: {shapes}")
        if bool((self.active & ~self.candidate).any()):
            problems.append("active slot without a candidate edge")
        if bool((self.weight < 0).any()):
            problems.append("negative weight magnitude")
        if int(self.src.min()) < 0 or int(self.src.max()) >= self.n:
            problems.append("source index out of range")
        if bool((self.src[self.candidate] == torch.arange(self.n, device=self.src.device)[:, None]
                 .expand_as(self.src)[self.candidate]).any()):
            problems.append("autapse present")
        occ = self.occupancy()
        if float(occ.max()) > cfg.max_occupancy + 1e-9:
            problems.append(f"occupancy cap exceeded: {float(occ.max()):.3f} > {cfg.max_occupancy}")
        if int(self.fanout().max()) > cfg.fanout_cap:
            problems.append(f"fan-out cap exceeded: {int(self.fanout().max())} > {cfg.fanout_cap}")
        # duplicate candidate edges within a neuron's slots
        s = torch.where(self.candidate, self.src, -1 - torch.arange(self.capacity, device=self.src.device)[None, :])
        srt = s.sort(dim=1).values
        if bool((srt[:, 1:] == srt[:, :-1]).any()):
            problems.append("duplicate candidate edge")
        return problems

    # ------------------------------------------------------------------ dynamics

    def effective(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(excitatory, inhibitory) weight matrices over slots, zero where inactive."""
        live = self.active.to(self.weight.dtype)
        return (self.weight * live * self.src_exc.to(self.weight.dtype),
                self.weight * live * (~self.src_exc).to(self.weight.dtype))

    def to(self, device) -> "SparseSlots":
        moved = SparseSlots(*(getattr(self, f).to(device) for f in SLOT_FIELDS), self.src_exc.to(device),
                            self.max_occupancy, self.fanout_cap)
        moved._version = self._version
        return moved

    # ------------------------------------------------------------------ persistence

    def state_dict(self) -> dict:
        return {f: getattr(self, f).detach().cpu().clone() for f in SLOT_FIELDS} | \
               {"src_exc": self.src_exc.cpu().clone(), "max_occupancy": self.max_occupancy,
                "fanout_cap": self.fanout_cap}

    @classmethod
    def from_state_dict(cls, d: dict) -> "SparseSlots":
        return cls(*(d[f].clone() for f in SLOT_FIELDS), d["src_exc"].clone(),
                   d["max_occupancy"], d["fanout_cap"])

    def connectivity_hash(self) -> str:
        h = hashlib.sha256()
        for f in SLOT_FIELDS:
            t = getattr(self, f).detach().cpu().contiguous()
            h.update(f.encode())
            h.update(t.numpy().tobytes())
        return h.hexdigest()
