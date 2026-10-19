#!/usr/bin/env python3
from continuous_brain.self_development_hysteresis_v1 import (
    STRATEGY_DIRECT,
    STRATEGY_SCAFFOLD,
    HistoryRelativeDevelopmentalController,
)


def main():
    c = HistoryRelativeDevelopmentalController()

    fragile = c.initialize(.10)
    assert c.strategy(fragile) == STRATEGY_SCAFFOLD

    c.update(fragile, .21)
    assert c.strategy(fragile) == STRATEGY_SCAFFOLD

    c.update(fragile, .22)
    assert c.strategy(fragile) == STRATEGY_DIRECT
    assert fragile.graduated is True

    # One-way hysteresis prevents calibration noise from undoing development.
    c.update(fragile, .15)
    assert c.strategy(fragile) == STRATEGY_DIRECT

    stable = c.initialize(.80)
    assert c.strategy(stable) == STRATEGY_DIRECT
    c.update(stable, .40)
    assert c.strategy(stable) == STRATEGY_DIRECT

    print("SELF_DEVELOPMENT_HYSTERESIS_SMOKE: PASS")


if __name__ == "__main__":
    main()
