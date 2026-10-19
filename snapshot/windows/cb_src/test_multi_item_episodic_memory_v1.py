#!/usr/bin/env python3
import torch

from continuous_brain.multi_item_episodic_memory_v1 import (
    EpisodicLateQueryReader,
    TwoSlotMemory,
    QUERY_FIRST,
)


def main():
    m = TwoSlotMemory.empty(2, 128)
    a = torch.randn(2, 128)
    b = torch.randn(2, 128)

    m = m.append(a)
    assert m.valid[:, 0].all()
    assert not m.valid[:, 1].any()
    assert torch.equal(m.slots[:, 0], a)

    m = m.append(b)
    assert m.valid.all()
    assert torch.equal(m.slots[:, 1], b)

    s = m.swap()
    assert torch.equal(s.slots[:, 0], b)
    assert torch.equal(s.slots[:, 1], a)

    z = m.zero_slot(0)
    assert not z.valid[:, 0].any()
    assert z.valid[:, 1].all()
    assert torch.count_nonzero(z.slots[:, 0]).item() == 0

    d = torch.randn(2, 128)
    r = m.replace_slot(1, d)
    assert torch.equal(r.slots[:, 1], d)

    reader = EpisodicLateQueryReader()
    q = torch.full((2,), QUERY_FIRST, dtype=torch.long)
    out = reader(m.slots, m.valid, q)
    assert tuple(out["logits"].shape) == (2, 2)
    assert tuple(out["attention"].shape) == (2, 2)
    assert torch.allclose(
        out["attention"].sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )

    print("MULTI_ITEM_EPISODIC_MEMORY_SMOKE: PASS")


if __name__ == "__main__":
    main()
