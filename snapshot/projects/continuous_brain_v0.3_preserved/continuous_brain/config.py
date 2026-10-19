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
