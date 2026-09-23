"""When does the core consult the backbone?

    LLM request  iff  a_trigger(t) > theta  and  time since last request >= refractory

a_trigger is the mean pooled fast-trace feature over designated trigger
populations. External events (an incoming message) can force a consultation.
The backbone is otherwise idle: the core runs continuously, the backbone is a
periodically consulted organ.
"""

from __future__ import annotations

import math

import torch


class InvocationGate:
    def __init__(self, trigger_populations: tuple[int, ...], threshold: float,
                 refractory_s: float, fast_channel: int = 0):
        if not trigger_populations:
            raise ValueError("at least one trigger population is required")
        self.trigger = torch.tensor(trigger_populations, dtype=torch.long)
        self.threshold = threshold
        self.refractory_s = refractory_s
        self.fast_channel = fast_channel
        self.last_fire_s = -math.inf

    def activation(self, pooled: torch.Tensor) -> torch.Tensor:
        """[B, P, C] pooled features -> [B] trigger activation."""
        return pooled[:, self.trigger.to(pooled.device), self.fast_channel].mean(dim=-1)

    def __call__(self, pooled: torch.Tensor, t_s: float, force: bool = False) -> bool:
        ready = (t_s - self.last_fire_s) >= self.refractory_s
        fire = force or (ready and bool((self.activation(pooled) > self.threshold).any()))
        if fire:
            self.last_fire_s = t_s
        return fire

    def state_dict(self) -> dict:
        return {"last_fire_s": self.last_fire_s}

    def load_state_dict(self, state: dict) -> None:
        self.last_fire_s = float(state["last_fire_s"])
