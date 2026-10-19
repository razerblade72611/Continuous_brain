from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
import torch
from torch import nn
from .contracts import BudgetReport, ComputeReport, SubstrateState, TrainingBatch, UnsupportedOperation

@dataclass(frozen=True)
class DenseRNNConfig:
    input_dim: int = 8
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    seed: int = 1
    device: str = "cpu"

class DenseRNNSubstrate(nn.Module):
    """GRU comparator for engineering and later preregistered E2 comparison.

    Its default training objective is causal t -> t+1 input prediction. This code
    does not by itself approve Amendment A1 or an E2 decision rule.
    """
    def __init__(self, config: DenseRNNConfig):
        super().__init__()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.gru = nn.GRUCell(config.input_dim, config.hidden_dim)
            self.decoder = nn.Linear(config.hidden_dim, config.input_dim)
        self.to(torch.device(config.device))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        self._hidden = torch.zeros(config.hidden_dim, device=config.device)
        self._mask = torch.ones(config.hidden_dim, device=config.device)
        self._timestamp = 0.0
        self._frozen = False
        self._inference_steps = self._gradient_steps = self._examples_seen = 0

    @property
    def name(self) -> str: return "DENSE-RNN/GRU"

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self._hidden = torch.zeros(self.config.hidden_dim, device=self._hidden.device)
        self._timestamp = 0.0

    def _coerce(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32, device=self._hidden.device)
        if x.ndim != 1 or x.shape[0] != self.config.input_dim:
            raise ValueError(f"expected observation [{self.config.input_dim}], got {tuple(x.shape)}")
        return x

    @torch.no_grad()
    def step(self, observation: torch.Tensor, dt: float) -> SubstrateState:
        x = self._coerce(observation)
        self._hidden = self.gru(x, self._hidden) * self._mask
        self._timestamp += float(dt)
        self._inference_steps += 1
        return self.expose_state()

    def expose_state(self) -> SubstrateState:
        return SubstrateState(self._hidden.detach().cpu().clone(), self._timestamp,
                              {"feature_kind":"dense_hidden", "name":self.name})

    def plasticity_step(self, modulators: Mapping[str, float] | None = None) -> None:
        raise UnsupportedOperation("DENSE-RNN uses gradient training; local plasticity is not implemented")

    def train_step(self, batch: TrainingBatch) -> Mapping[str, float]:
        if self._frozen: raise UnsupportedOperation("substrate is frozen")
        seq = torch.as_tensor(batch.sequence, dtype=torch.float32, device=self._hidden.device)
        if seq.ndim == 2: seq = seq[:, None, :]
        if seq.ndim != 3 or seq.shape[-1] != self.config.input_dim or seq.shape[0] < 2:
            raise ValueError("sequence must be [T,D] or [T,B,D], T>=2")
        self.optimizer.zero_grad(set_to_none=True)
        h = torch.zeros(seq.shape[1], self.config.hidden_dim, device=seq.device)
        losses=[]
        for t in range(seq.shape[0]-1):
            h = self.gru(seq[t], h)
            pred = self.decoder(h)
            losses.append(torch.mean((pred - seq[t+1])**2))
        loss = torch.stack(losses).mean()
        loss.backward()
        self.optimizer.step()
        self._hidden = h[-1].detach() * self._mask
        self._gradient_steps += 1
        self._examples_seen += int((seq.shape[0]-1)*seq.shape[1])
        return {"prediction_mse": float(loss.detach().cpu())}

    def freeze(self) -> None:
        self._frozen = True
        for p in self.parameters(): p.requires_grad_(False)

    def save(self, path: str | Path) -> None:
        path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config":asdict(self.config), "model":self.state_dict(),
                    "optimizer":self.optimizer.state_dict() if not self._frozen else None,
                    "hidden":self._hidden.detach().cpu(), "mask":self._mask.detach().cpu(),
                    "timestamp":self._timestamp, "frozen":self._frozen,
                    "compute":self.compute_report().__dict__}, path)

    def restore(self, path: str | Path) -> None:
        d=torch.load(path, map_location=self._hidden.device, weights_only=False)
        self.load_state_dict(d["model"])
        if d.get("optimizer") is not None and not self._frozen: self.optimizer.load_state_dict(d["optimizer"])
        self._hidden=d["hidden"].to(self._hidden.device); self._mask=d["mask"].to(self._hidden.device)
        self._timestamp=float(d["timestamp"]); self._frozen=bool(d["frozen"])
        c=d.get("compute",{}); self._inference_steps=int(c.get("inference_steps",0)); self._gradient_steps=int(c.get("gradient_steps",0)); self._examples_seen=int(c.get("examples_seen",0))
        if self._frozen:
            for p in self.parameters(): p.requires_grad_(False)

    def budget_report(self) -> BudgetReport:
        n=sum(p.numel() for p in self.parameters())
        return BudgetReport(learned_scalars=n, active_connections=n,
                            notes="learned scalar parameter count; not equated to SNN active synapses")

    def compute_report(self) -> ComputeReport:
        return ComputeReport(self._inference_steps,0,self._gradient_steps,self._examples_seen)

    def lesion(self, unit_indices: list[int]) -> None:
        for i in unit_indices:
            if not 0 <= i < self.config.hidden_dim: raise IndexError(i)
            self._mask[i]=0.0
        self._hidden *= self._mask
