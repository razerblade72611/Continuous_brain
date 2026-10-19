#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    HISTORY_LEN,
    AutobiographicalTrajectoryMemory,
    load_memory,
    save_memory,
)

STRATEGY_DIRECT = 0
STRATEGY_SCAFFOLD = 1


@dataclass
class DevelopmentalState:
    """
    Persistent non-neural developmental state.

    capability is an evaluator/environment-side latent skill variable. It is
    intentionally NOT visible to the autobiographical self-model.
    """
    capability_a: float
    capability_b: float
    cycle: int = 0

    def capability_for(self, primary: int) -> float:
        if int(primary) == 1:
            return float(self.capability_a)
        if int(primary) == 2:
            return float(self.capability_b)
        raise ValueError("primary must be A(1) or B(2)")

    def set_capability_for(self, primary: int, value: float):
        value = float(max(0.0, min(1.0, value)))
        if int(primary) == 1:
            self.capability_a = value
        elif int(primary) == 2:
            self.capability_b = value
        else:
            raise ValueError("primary must be A(1) or B(2)")


class DevelopmentalController:
    """
    Fixed controller translating autobiographical self-estimate into strategy.

    No learned weights.

    If the frozen autobiographical model judges the targeted commitment
    unreliable, gather information/scaffold before working. Once the historical
    self-model becomes sufficiently reliable, act directly.
    """
    def __init__(self, direct_threshold: float = 0.50):
        self.direct_threshold = float(direct_threshold)

    def choose(self, reliability_for_primary: float) -> int:
        if float(reliability_for_primary) < self.direct_threshold:
            return STRATEGY_SCAFFOLD
        return STRATEGY_DIRECT


def strategy_name(strategy: int) -> str:
    return (
        "SCAFFOLD"
        if int(strategy) == STRATEGY_SCAFFOLD
        else "DIRECT"
    )


def strategy_failure_probability(
    capability: float,
    strategy: int,
) -> float:
    """
    Surprise probability for a work step.

    Scaffolded behavior uses information gathering and sharply reduces
    avoidable failure when capability is still developing.
    """
    capability = float(max(0.0, min(1.0, capability)))
    lack = 1.0 - capability
    if int(strategy) == STRATEGY_SCAFFOLD:
        return min(0.95, 0.15 * lack)
    return min(0.95, 0.85 * lack)


def capability_update(
    capability: float,
    *,
    success: bool,
    strategy: int,
) -> float:
    """
    Practice changes future capability.

    Successful scaffolded work provides a larger early learning increment;
    direct successful work still learns. Failed attempts produce a smaller
    learning increment.
    """
    capability = float(capability)
    if success:
        gain = (
            0.13
            if int(strategy) == STRATEGY_SCAFFOLD
            else 0.085
        )
    else:
        gain = 0.035

    # Diminishing returns near mastery.
    capability = capability + gain * (1.0 - capability)
    return float(max(0.0, min(1.0, capability)))


def reliability_from_memory(
    autobiographical_model,
    memory: AutobiographicalTrajectoryMemory,
):
    x, mask = memory.feature_tensor(HISTORY_LEN)
    with torch.no_grad():
        out = autobiographical_model(
            x.unsqueeze(0),
            mask.unsqueeze(0),
        )
    return out["reliability"][0].detach().cpu()


def choose_strategy_from_memory(
    autobiographical_model,
    memory: AutobiographicalTrajectoryMemory,
    primary: int,
    controller: DevelopmentalController,
):
    reliability = reliability_from_memory(
        autobiographical_model,
        memory,
    )
    idx = 0 if int(primary) == 1 else 1
    value = float(reliability[idx].item())
    return controller.choose(value), reliability


def save_developmental_state(
    path,
    state: DevelopmentalState,
    memory: AutobiographicalTrajectoryMemory,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    memory_path = path.with_suffix(".memory.pt")
    save_memory(memory_path, memory)

    torch.save(
        {
            "format":
                "continuous_brain.self_development_loop_state_v1",
            "capability_a":
                float(state.capability_a),
            "capability_b":
                float(state.capability_b),
            "cycle":
                int(state.cycle),
            "memory_path":
                memory_path.name,
            "metadata":
                dict(metadata or {}),
        },
        path,
    )


def load_developmental_state(path):
    path = Path(path)
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    memory_path = (
        path.parent
        / payload["memory_path"]
    )
    memory, _ = load_memory(memory_path)
    state = DevelopmentalState(
        capability_a=float(
            payload["capability_a"]
        ),
        capability_b=float(
            payload["capability_b"]
        ),
        cycle=int(payload["cycle"]),
    )
    return state, memory, payload
