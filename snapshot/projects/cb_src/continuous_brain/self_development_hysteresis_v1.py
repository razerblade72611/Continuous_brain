#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

STRATEGY_DIRECT = 0
STRATEGY_SCAFFOLD = 1

NEUTRAL_RELIABILITY = 0.50
MEANINGFUL_SELF_ESTIMATE_GAIN = 0.12


@dataclass
class CommitmentDevelopmentPhase:
    baseline_reliability: float
    strategy: int
    graduated: bool = False


class HistoryRelativeDevelopmentalController:
    """
    One-way developmental hysteresis driven only by autobiographical
    reliability estimates.

    Initialization:
      baseline >= 0.50 -> DIRECT
      baseline <  0.50 -> SCAFFOLD

    Graduation:
      a scaffolded commitment becomes DIRECT after its autobiographical
      self-estimate has improved by >= 0.12 from its own bootstrap baseline.

    The 0.12 criterion is NOT newly tuned: it is the preregistered
    `MIN_RELIABILITY_GAIN` from v0.33.0.

    Once graduated, the commitment remains DIRECT for the duration of this
    developmental assay. This prevents the fixed-calibration oscillation that
    v0.33.0 exposed and cleanly tests whether autobiographical improvement can
    drive a developmental phase transition.
    """
    def __init__(
        self,
        *,
        neutral_reliability: float = NEUTRAL_RELIABILITY,
        meaningful_gain: float = MEANINGFUL_SELF_ESTIMATE_GAIN,
    ):
        self.neutral_reliability = float(neutral_reliability)
        self.meaningful_gain = float(meaningful_gain)

    def initialize(self, baseline_reliability: float):
        baseline_reliability = float(baseline_reliability)
        direct = (
            baseline_reliability
            >= self.neutral_reliability
        )
        return CommitmentDevelopmentPhase(
            baseline_reliability=baseline_reliability,
            strategy=(
                STRATEGY_DIRECT
                if direct
                else STRATEGY_SCAFFOLD
            ),
            graduated=bool(direct),
        )

    def update(
        self,
        phase: CommitmentDevelopmentPhase,
        current_reliability: float,
    ):
        current_reliability = float(current_reliability)

        if (
            phase.strategy == STRATEGY_SCAFFOLD
            and current_reliability
                >= phase.baseline_reliability
                + self.meaningful_gain
        ):
            phase.strategy = STRATEGY_DIRECT
            phase.graduated = True

        return phase

    def strategy(
        self,
        phase: CommitmentDevelopmentPhase,
    ) -> int:
        return int(phase.strategy)
