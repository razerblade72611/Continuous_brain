"""E0 configuration. Every field feeds the config hash recorded with each run.

Defaults give the minimal ~10k-neuron organism described in DAV-5 / the
Architecture & Scientific Charter. Nothing here is semantic: sizes, physical
tuning ranges, time constants, and connection geometry only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "cbsnn/e0/1"


@dataclass
class AnatomyConfig:
    # Retinal sheet: a square grid of ON-centre and OFF-centre cells (DAV-34).
    retina_side: int = 32
    retina_spacing_mm: float = 0.05
    # Two-joint body for E2; proprioception = population codes for angle and velocity.
    n_joints: int = 2
    proprio_angle_cells: int = 32
    proprio_velocity_cells: int = 32
    angle_range_rad: tuple[float, float] = (-1.5708, 1.5708)
    velocity_range_rad_s: tuple[float, float] = (-6.0, 6.0)
    # Motor: antagonistic M+ / M- and co-contraction (stiffness) populations per joint.
    motor_cells_per_channel: int = 32
    # Everything else is cortex, 80% excitatory / 20% inhibitory.
    n_neurons_total: int = 10_000
    cortex_frac_excitatory: float = 0.8
    cortex_extent_mm: tuple[float, float, float] = (1.6, 1.6, 0.8)
    # Cortex is partitioned into spatial columns: the populations whose
    # summaries E1 reads (topography metrics need spatially local groups).
    cortex_columns_per_side: int = 8


@dataclass
class ConnectivityConfig:
    # Fixed capacity per postsynaptic neuron (DAV-35): all slots preallocated.
    fanin_capacity: int = 96
    # Fraction of candidate slots active at genesis; the rest are dormant.
    initial_active_fraction: float = 0.5
    # Structural caps (validated invariants).
    max_occupancy: float = 0.9          # active / capacity per neuron
    fanout_cap: int = 400               # active outgoing slots per neuron
    # Candidate sampling: p(src) ~ exp(-distance / length) within allowed region pairs.
    length_constant_mm: float = 0.25
    # Chosen from a genesis sweep (w_exc 0.05/0.4/0.8): 0.4/1.2 gives an
    # inhibition-dominated asynchronous regime, cortex E ~2 Hz / I ~7 Hz.
    w_exc: float = 0.4
    w_inh: float = 1.2
    w_jitter_sigma: float = 0.3
    w_max: float = 3.6


@dataclass
class NeuronConfig:
    dt_s: float = 0.001                 # nominal 1 ms Euler step
    tau_mem_exc_s: float = 0.020
    tau_mem_inh_s: float = 0.010
    tau_syn_exc_s: float = 0.005
    tau_syn_inh_s: float = 0.010
    v_rest: float = 0.0
    v_reset: float = 0.0
    threshold_base: float = 1.0
    refractory_s: float = 0.002
    background_mean: float = 0.75
    noise_std: float = 0.9
    # Homeostatic threshold: slow drift toward a target rate (DAV-5 state).
    homeostasis: bool = True
    target_rate_hz: float = 5.0
    tau_homeostasis_s: float = 10.0
    homeostasis_gain: float = 0.02
    threshold_bounds: tuple[float, float] = (0.5, 3.0)
    # Noise is drawn on CPU so every backend sees identical noise (DirectML
    # has no device-side torch.Generator).
    noise_on_cpu: bool = True


@dataclass
class TraceConfig:
    taus_s: tuple[float, ...] = (0.005, 0.025, 0.1, 0.5, 2.0)


@dataclass
class E0Config:
    seed: int = 0
    anatomy: AnatomyConfig = field(default_factory=AnatomyConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    traces: TraceConfig = field(default_factory=TraceConfig)
    stage: str = "E0"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON form of the whole configuration."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> "E0Config":
        def tup(x):
            return tuple(x) if isinstance(x, list) else x

        return cls(seed=d["seed"],
                   anatomy=AnatomyConfig(**{k: tup(v) for k, v in d["anatomy"].items()}),
                   connectivity=ConnectivityConfig(**d["connectivity"]),
                   neuron=NeuronConfig(**{k: tup(v) for k, v in d["neuron"].items()}),
                   traces=TraceConfig(taus_s=tuple(d["traces"]["taus_s"])),
                   stage=d["stage"], schema_version=d["schema_version"])


def small_config(seed: int = 0, n_neurons: int = 1200) -> E0Config:
    """A reduced organism for tests and CPU smoke runs (same structure, fewer cells)."""
    cfg = E0Config(seed=seed)
    cfg.anatomy.retina_side = 8
    cfg.anatomy.proprio_angle_cells = 8
    cfg.anatomy.proprio_velocity_cells = 8
    cfg.anatomy.motor_cells_per_channel = 8
    cfg.anatomy.n_neurons_total = n_neurons
    cfg.anatomy.cortex_columns_per_side = 4
    cfg.connectivity.fanin_capacity = 48
    cfg.connectivity.fanout_cap = 200
    return cfg
