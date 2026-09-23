"""Configuration dataclasses for every component.

Defaults follow the architecture spec (Continuous Brain - Hybrid Architecture
Spec v0.1). Anything marked "placeholder" must be re-checked against the real
InfantBrain v0.11.3 once it is ported (neuron counts, integration step).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransducerConfig:
    # Synaptic-trace time constants, seconds (fast -> slow).
    taus_s: tuple[float, ...] = (0.005, 0.025, 0.1, 0.5, 2.0)
    # Integration bin; should match the core's integration step (InfantBrain: 1 ms).
    bin_s: float = 0.001
    # Time constant for spike recency r = exp(-(t - t_last) / tau).
    recency_tau_s: float = 0.05
    # Number of projection populations (placeholder until InfantBrain's real
    # region/stream structure is mapped onto populations).
    n_populations: int = 64
    # Learned latent tokens handed to the bridge.
    n_latent_tokens: int = 32
    latent_dim: int = 256
    n_heads: int = 8
    dropout: float = 0.0


@dataclass
class BridgeConfig:
    # Hidden width of the MLP that maps latent tokens into the backbone width.
    prefix_mlp_hidden: int = 1024
    # Write-back (LLM -> core) decoder.
    writeback_hidden: int = 512
    max_current: float = 2.0
    # Exploration noise on write-back currents (needed to train the decoder
    # by policy gradient, since spikes are not differentiable).
    writeback_noise_std: float = 0.25
    # Pending write-back current decays with this time constant after injection.
    writeback_decay_s: float = 0.2


@dataclass
class GateConfig:
    # Populations whose activity requests a backbone consultation.
    trigger_populations: tuple[int, ...] = (0,)
    threshold: float = 0.5
    refractory_s: float = 1.0


@dataclass
class MemoryConfig:
    # Salience = w_outcome*|outcome| + w_novelty*novelty + w_explicit*explicit
    w_outcome: float = 1.0
    w_novelty: float = 0.5
    w_explicit: float = 2.0
    # Episodes recalled into the prompt at each consultation.
    recall_k: int = 3
    recall_min_similarity: float = 0.2


@dataclass
class SleepConfig:
    max_episodes: int = 64
    bridge_steps: int = 50
    bridge_lr: float = 1e-4
    bridge_batch: int = 8
    writeback_lr: float = 1e-4
    writeback_steps: int = 20
    core_modulator_scale: float = 1.0
    # Regression gate: a guarded metric may drop by at most this much
    # (absolute, in the metric's own units) versus the pre-sleep baseline.
    regression_tolerance: float = 0.02
    seed: int = 0


@dataclass
class LoopConfig:
    # One cognitive cycle: the core runs this long between transducer reads.
    cycle_s: float = 0.02
    max_new_tokens: int = 64
    # Cycles of core input kept with each episode for sleep replay: this many
    # before the consultation, plus the injected write-back's aftermath.
    history_cycles: int = 25
    aftermath_cycles: int = 25
    # Prompt used when the core itself triggers a consultation (no message).
    internal_prompt: str = "Continue your current line of thought."
    device: str = "cpu"


@dataclass
class BrainConfig:
    transducer: TransducerConfig = field(default_factory=TransducerConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
