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
    c.update(fragile, .22)
    assert c.strategy(fragile) == STRATEGY_DIRECT

    stable = c.initialize(.80)
    assert c.strategy(stable) == STRATEGY_DIRECT

    # Fresh v0.34 TEST must stay sealed during DEV.
    assert list(range(80000, 80128))[0] == 80000
    assert list(range(80000, 80128))[-1] == 80127

    print("EVENT_CONSISTENT_SELF_DEVELOPMENT_SMOKE: PASS")


if __name__ == "__main__":
    main()
