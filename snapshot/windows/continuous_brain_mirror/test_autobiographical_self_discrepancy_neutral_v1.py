#!/usr/bin/env python3
import torch

from continuous_brain.autobiographical_self_discrepancy_v1 import (
    HISTORY_LEN,
    RECORD_FEATURE_DIM,
)
from continuous_brain.autobiographical_self_discrepancy_neutral_v1 import (
    NeutralAutobiographicalSelfModel,
)


def main():
    torch.manual_seed(32101)
    model = NeutralAutobiographicalSelfModel(
        record_dim=32,
        hidden_dim=48,
    )

    x = torch.zeros(
        3,
        HISTORY_LEN,
        RECORD_FEATURE_DIM,
        requires_grad=True,
    )
    mask = torch.ones(
        3,
        HISTORY_LEN,
        dtype=torch.bool,
    )

    with torch.no_grad():
        for b in range(3):
            for t in range(HISTORY_LEN):
                x[b, t, 0 if t % 2 == 0 else 1] = 1.0

    out = model(x, mask)
    expected = torch.full((3, 2), .5)
    assert torch.equal(
        out["reliability"],
        expected,
    )

    diff = (
        out["reliability"][:, 0]
        - out["reliability"][:, 1]
    ).sum()
    diff.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(
        x.grad[..., 2:].abs().max().item()
    ) > 0.0

    # Nonzero discrepancy may produce non-neutral output.
    y = x.detach().clone()
    y[0, -1, 2] = .4
    yo = model(y, mask)
    assert tuple(yo["reliability"].shape) == (3, 2)

    print(
        "AUTOBIOGRAPHICAL_SELF_DISCREPANCY_NEUTRAL_SMOKE: PASS"
    )


if __name__ == "__main__":
    main()
