"""Procedural anatomy and population registry (DAV-34).

Anatomy supplies the neighbourhood: every neuron gets a region, a cell type
(excitatory/inhibitory), and a position in millimetres. Populations are named
groups of neurons with physical metadata only.

Regions (fixed vocabulary):
  retina           ON-centre and OFF-centre cells on a square sheet
  proprioception   per joint, population codes for angle and angular velocity
  motor            per joint, antagonistic M+ / M- and co-contraction Mco
  cortex           everything else; 80/20 E/I; partitioned into spatial columns

No semantic object labels may enter sensory populations: `validate_physical`
rejects any population whose metadata goes beyond the physical keys allowed
for its modality. The anatomy hash is a pure function of config + seed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import AnatomyConfig, E0Config

REGIONS = ("retina", "proprioception", "motor", "cortex")
SENSORY_REGIONS = ("retina", "proprioception")

# Physical metadata allowed per region. Anything else is rejected.
ALLOWED_META: dict[str, set[str]] = {
    "retina": {"polarity"},
    "proprioception": {"joint", "quantity", "units", "range"},
    "motor": {"joint", "channel"},
    "cortex": {"column", "cell_type"},
}


@dataclass
class Population:
    name: str
    region: str
    index: np.ndarray              # neuron ids (int64)
    meta: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return {"name": self.name, "region": self.region, "meta": self.meta,
                "index": [int(self.index.min()), int(self.index.max()), int(self.index.size)]}


@dataclass
class Anatomy:
    region: np.ndarray             # [N] int8, index into REGIONS
    is_exc: np.ndarray             # [N] bool
    coords_mm: np.ndarray          # [N, 3] float32
    preferred: np.ndarray          # [N] float32 tuning centre (physical units) or NaN
    populations: list[Population]
    config: AnatomyConfig

    @property
    def n(self) -> int:
        return int(self.region.shape[0])

    def pop(self, name: str) -> Population:
        for p in self.populations:
            if p.name == name:
                return p
        raise KeyError(name)

    def region_index(self, name: str) -> np.ndarray:
        return np.nonzero(self.region == REGIONS.index(name))[0]

    def anatomy_hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.config.__dict__, sort_keys=True, default=list).encode())
        for arr in (self.region.astype(np.int8), self.is_exc.astype(np.uint8),
                    self.coords_mm.astype(np.float32), np.nan_to_num(self.preferred, nan=-1e9).astype(np.float32)):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(json.dumps([p.as_json() for p in self.populations], sort_keys=True).encode())
        return h.hexdigest()

    def population_mean_matrix(self) -> torch.Tensor:
        """Sparse [P, N] matrix averaging each population's members (populations may overlap)."""
        rows, cols, vals = [], [], []
        for i, p in enumerate(self.populations):
            rows.append(np.full(p.index.size, i))
            cols.append(p.index)
            vals.append(np.full(p.index.size, 1.0 / p.index.size, dtype=np.float32))
        idx = torch.from_numpy(np.stack([np.concatenate(rows), np.concatenate(cols)])).long()
        return torch.sparse_coo_tensor(idx, torch.from_numpy(np.concatenate(vals)),
                                       (len(self.populations), self.n), check_invariants=True).coalesce()


def validate_physical(anatomy: Anatomy) -> list[str]:
    """Return violations of the semantic firewall in population metadata."""
    problems = []
    for p in anatomy.populations:
        if p.region not in REGIONS:
            problems.append(f"{p.name}: unknown region {p.region!r}")
            continue
        extra = set(p.meta) - ALLOWED_META[p.region]
        if extra:
            problems.append(f"{p.name}: non-physical metadata {sorted(extra)}")
        if p.index.size and not np.all(anatomy.region[p.index] == REGIONS.index(p.region)):
            problems.append(f"{p.name}: members outside region {p.region}")
    return problems


def build_anatomy(cfg: E0Config) -> Anatomy:
    a = cfg.anatomy
    rng = np.random.default_rng(cfg.seed)
    n_ret = 2 * a.retina_side ** 2
    n_prop = a.n_joints * (a.proprio_angle_cells + a.proprio_velocity_cells)
    n_mot = a.n_joints * 3 * a.motor_cells_per_channel
    n_ctx = a.n_neurons_total - n_ret - n_prop - n_mot
    if n_ctx < a.cortex_columns_per_side ** 2 * 2:
        raise ValueError("n_neurons_total too small for the sensory/motor anatomy")

    region = np.empty(a.n_neurons_total, dtype=np.int8)
    is_exc = np.ones(a.n_neurons_total, dtype=bool)
    coords = np.zeros((a.n_neurons_total, 3), dtype=np.float32)
    preferred = np.full(a.n_neurons_total, np.nan, dtype=np.float32)
    pops: list[Population] = []
    cursor = 0
    ex, ey, ez = a.cortex_extent_mm

    # ---- retina: two co-registered sheets under the cortex
    side = a.retina_side
    span = side * a.retina_spacing_mm
    gx, gy = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    xy = np.stack([(gx.ravel() + 0.5) * a.retina_spacing_mm - span / 2,
                   (gy.ravel() + 0.5) * a.retina_spacing_mm - span / 2], axis=1) * (ex / span)
    for polarity, z in (("on", -0.50), ("off", -0.52)):
        idx = np.arange(cursor, cursor + side * side)
        region[idx] = REGIONS.index("retina")
        coords[idx, :2], coords[idx, 2] = xy, z
        pops.append(Population(f"retina_{polarity}", "retina", idx, {"polarity": polarity}))
        cursor += idx.size

    # ---- proprioception: tuned population codes, left of cortex
    for j in range(a.n_joints):
        for quantity, n_cells, rng_range, units in (
                ("angle", a.proprio_angle_cells, a.angle_range_rad, "rad"),
                ("velocity", a.proprio_velocity_cells, a.velocity_range_rad_s, "rad/s")):
            idx = np.arange(cursor, cursor + n_cells)
            region[idx] = REGIONS.index("proprioception")
            coords[idx, 0] = -ex / 2 - 0.3
            coords[idx, 1] = np.linspace(-ey / 2, ey / 2, n_cells) * 0.4 + (j - (a.n_joints - 1) / 2) * 0.5
            coords[idx, 2] = ez / 2 + (0.05 if quantity == "velocity" else 0.0)
            preferred[idx] = np.linspace(rng_range[0], rng_range[1], n_cells)
            pops.append(Population(f"proprio_{quantity}_j{j}", "proprioception", idx,
                                   {"joint": j, "quantity": quantity, "units": units,
                                    "range": list(rng_range)}))
            cursor += n_cells

    # ---- motor: M+ / M- / Mco per joint, right of cortex
    for j in range(a.n_joints):
        for c, channel in enumerate(("plus", "minus", "co")):
            idx = np.arange(cursor, cursor + a.motor_cells_per_channel)
            region[idx] = REGIONS.index("motor")
            coords[idx, 0] = ex / 2 + 0.3
            coords[idx, 1] = np.linspace(-0.1, 0.1, idx.size) + (j - (a.n_joints - 1) / 2) * 0.5
            coords[idx, 2] = ez / 2 + 0.1 * (c - 1)
            pops.append(Population(f"motor_{channel}_j{j}", "motor", idx, {"joint": j, "channel": channel}))
            cursor += idx.size

    # ---- cortex: random positions in a slab, 80/20 E/I, spatial columns
    idx = np.arange(cursor, a.n_neurons_total)
    region[idx] = REGIONS.index("cortex")
    coords[idx] = (rng.random((idx.size, 3)) * np.array([ex, ey, ez]) - np.array([ex / 2, ey / 2, 0.0])).astype(np.float32)
    exc_draw = rng.random(idx.size) < a.cortex_frac_excitatory
    is_exc[idx] = exc_draw
    cps = a.cortex_columns_per_side
    col_x = np.clip(((coords[idx, 0] + ex / 2) / ex * cps).astype(int), 0, cps - 1)
    col_y = np.clip(((coords[idx, 1] + ey / 2) / ey * cps).astype(int), 0, cps - 1)
    for cx in range(cps):
        for cy in range(cps):
            in_col = (col_x == cx) & (col_y == cy)
            for cell_type, sel in (("exc", exc_draw), ("inh", ~exc_draw)):
                members = idx[in_col & sel]
                if members.size:
                    pops.append(Population(f"cortex_c{cx}_{cy}_{cell_type}", "cortex", members,
                                           {"column": [cx, cy], "cell_type": cell_type}))

    anatomy = Anatomy(region, is_exc, coords, preferred, pops, a)
    problems = validate_physical(anatomy)
    if problems:
        raise AssertionError("anatomy violates the semantic firewall: " + "; ".join(problems))
    return anatomy
