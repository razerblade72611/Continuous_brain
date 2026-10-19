from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable
import torch

class SubstrateError(RuntimeError): pass
class UnsupportedOperation(SubstrateError): pass
class GradientFirewallError(SubstrateError): pass

@dataclass(frozen=True)
class BudgetReport:
    learned_scalars: int
    active_connections: int | None = None
    notes: str = ""

@dataclass(frozen=True)
class ComputeReport:
    inference_steps: int = 0
    local_plasticity_steps: int = 0
    gradient_steps: int = 0
    examples_seen: int = 0

@dataclass
class SubstrateState:
    vector: torch.Tensor
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingBatch:
    sequence: torch.Tensor  # [T, D] or [T, B, D]
    dt: float = 0.01

@runtime_checkable
class Substrate(Protocol):
    @property
    def name(self) -> str: ...
    def reset(self, seed: int | None = None) -> None: ...
    def step(self, observation: torch.Tensor, dt: float) -> SubstrateState: ...
    def expose_state(self) -> SubstrateState: ...
    def plasticity_step(self, modulators: Mapping[str, float] | None = None) -> None: ...
    def train_step(self, batch: TrainingBatch) -> Mapping[str, float]: ...
    def freeze(self) -> None: ...
    def save(self, path: str | Path) -> None: ...
    def restore(self, path: str | Path) -> None: ...
    def budget_report(self) -> BudgetReport: ...
    def compute_report(self) -> ComputeReport: ...
    def lesion(self, unit_indices: list[int]) -> None: ...
