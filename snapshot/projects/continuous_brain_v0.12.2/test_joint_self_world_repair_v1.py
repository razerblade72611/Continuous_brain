#!/usr/bin/env python3
import torch

from continuous_brain.joint_self_world_repair_v1 import (
    FutureGoalRepairHead,
)


def main():
    torch.manual_seed(30101)
    head = FutureGoalRepairHead(
        latent_dim=96,
        hidden_dim=128,
    )

    fake = {
        "latent":
            torch.randn(7, 96),
        "next_self_continuous":
            torch.rand(7, 6),
        "next_goal_logits":
            torch.randn(7, 4),
    }
    out = head(fake)
    assert tuple(out.shape) == (7, 4)

    print(
        "JOINT_SELF_WORLD_REPAIR_SMOKE: PASS"
    )


if __name__ == "__main__":
    main()
