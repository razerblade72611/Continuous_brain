#!/usr/bin/env python3
import torch

from continuous_brain.self_broadcast_value_priority_repair_v1 import (
    ValuePriorityRepairHead,
)


def main():
    torch.manual_seed(29101)
    head = ValuePriorityRepairHead(
        broadcast_dim=64,
        hidden_dim=96,
    )

    x = torch.randn(17, 64)
    y = head(x)
    assert tuple(y.shape) == (17,)
    assert bool((y >= 0.0).all())
    assert bool((y <= 1.0).all())

    z = torch.zeros(17, 64)
    yz = head(z)
    assert torch.count_nonzero(yz).item() == 0

    print(
        "SELF_BROADCAST_VALUE_PRIORITY_REPAIR_SMOKE: PASS"
    )


if __name__ == "__main__":
    main()
