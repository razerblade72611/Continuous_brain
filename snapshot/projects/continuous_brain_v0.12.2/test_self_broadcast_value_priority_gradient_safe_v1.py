#!/usr/bin/env python3
import torch

from continuous_brain.self_broadcast_value_priority_gradient_safe_v1 import (
    GradientSafeValuePriorityHead,
)


def main():
    torch.manual_seed(29201)
    head = GradientSafeValuePriorityHead(
        broadcast_dim=64,
        hidden_dim=128,
    )

    zero = torch.zeros(
        17, 64, requires_grad=True
    )
    y0 = head(zero)
    assert torch.count_nonzero(y0).item() == 0

    y0.sum().backward()
    assert zero.grad is not None
    assert torch.isfinite(zero.grad).all()
    assert float(zero.grad.abs().max().item()) > 0.0

    x = torch.randn(17, 64)
    y = head(x)
    assert tuple(y.shape) == (17,)
    assert bool((y > -1.0).all())
    assert bool((y < 1.0).all())

    print(
        "SELF_BROADCAST_VALUE_PRIORITY_GRADIENT_SAFE_SMOKE: PASS"
    )


if __name__ == "__main__":
    main()
