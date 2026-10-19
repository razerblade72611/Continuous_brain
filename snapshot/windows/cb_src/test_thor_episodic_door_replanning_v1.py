#!/usr/bin/env python3
import torch

from continuous_brain.thor_episodic_door_replanning_v1 import (
    DoorEpisode,
    bfs_path,
    crossed_far_side,
    select_bound_episode,
)
from continuous_brain.multi_item_episodic_memory_v1 import (
    QUERY_FIRST,
    QUERY_SECOND,
)


class DummyReader:
    def __call__(self, slots, valid, q):
        b = slots.shape[0]
        weights = torch.zeros(b, 2)
        for i in range(b):
            weights[i, 0 if int(q[i]) == QUERY_FIRST else 1] = 1.0
        return {
            "attention": weights,
            "logits": torch.zeros(b, 2),
        }


def main():
    pts = []
    for ix in range(4):
        for iz in range(3):
            pts.append({
                "x": ix * .25,
                "y": 0.0,
                "z": iz * .25,
            })

    path = bfs_path(
        pts,
        (0.0, 0.0, 0.0),
        (.75, 0.0, .50),
    )
    assert path[0] == (0.0, 0.0, 0.0)
    assert path[-1] == (.75, 0.0, .50)
    assert len(path) == 6

    ep0 = DoorEpisode(
        0, "north_door", "north",
        (0.0, 0.0, 1.0),
        "z", 1,
        (0.0, 0.0, .75),
        (0.0, 0.0, 1.5),
    )
    ep1 = DoorEpisode(
        1, "east_door", "east",
        (1.0, 0.0, 0.0),
        "x", 1,
        (.75, 0.0, 0.0),
        (1.5, 0.0, 0.0),
    )

    assert crossed_far_side(
        ep0, {"x": 0.0, "z": 1.5}
    )
    assert not crossed_far_side(
        ep0, {"x": 0.0, "z": .75}
    )

    reader = DummyReader()
    k0 = torch.randn(1, 128)
    k1 = torch.randn(1, 128)

    first = select_bound_episode(
        reader, k0, k1, [ep0, ep1], QUERY_FIRST
    )
    second = select_bound_episode(
        reader, k0, k1, [ep0, ep1], QUERY_SECOND
    )
    assert first["payload"].object_id == "north"
    assert second["payload"].object_id == "east"

    print("THOR_EPISODIC_DOOR_REPLANNING_SMOKE: PASS")


if __name__ == "__main__":
    main()
