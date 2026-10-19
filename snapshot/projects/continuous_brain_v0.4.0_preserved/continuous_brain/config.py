from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BrainConfig:
    obs_dim: int = 6
    action_dim: int = 5
    embed_dim: int = 64
    hidden_dim: int = 128
    replay_capacity: int = 50_000
    batch_size: int = 64
    gamma: float = 0.98
    learning_rate: float = 3e-4
    brain_hz: float = 10.0
    learn_every: int = 4
    consolidate_every: int = 50
    memory_surprise_threshold: float = 0.04
    epsilon_start: float = 0.30
    epsilon_end: float = 0.03
    epsilon_decay_steps: int = 30_000
    checkpoint_every: int = 2_000
    seed: int = 7

    # v0.3 lifetime-learning signals. Kept intentionally small so migrated v0.2
    # behavior is not destroyed when these signals are first enabled.
    curiosity_weight: float = 0.02
    curiosity_cap: float = 1.0
    homeostasis_weight: float = 0.01
    visual_memory_threshold: float = 0.08
