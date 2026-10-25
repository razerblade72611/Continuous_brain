import torch

from being.v3.slots import SlotMemory


def test_born_silent_and_deterministic():
    a, b = SlotMemory(64, n_keys=16, k=4, d_key=32, birth_seed=7), SlotMemory(64, n_keys=16, k=4, d_key=32, birth_seed=7)
    x = torch.randn(5, 64)
    assert torch.all(a(x) == 0)
    assert a.weight_hash() == b.weight_hash()


def test_only_top_t_rows_change_and_background_slots_protected():
    torch.manual_seed(0)
    m = SlotMemory(64, n_keys=16, k=4, d_key=32, birth_seed=1)
    bg = torch.randn(500, 64)
    m.observe_background(bg)
    x = torch.randn(6, 64, requires_grad=False)
    before = m.V.detach().clone()
    out = m(x)
    loss = ((out - torch.ones_like(out)) ** 2).sum()
    loss.backward()
    rows = m.sparse_step(top_t=3, lr=0.1)
    changed = (m.V.detach() != before).any(-1).nonzero().flatten()
    assert set(changed.tolist()) <= set(rows.tolist()) and len(rows) == 3
    # the chosen rows have no higher background usage than the median touched slot (idf protects common slots)
    touched = torch.unique(m.last_slots)
    assert m.bg_count[rows].mean() <= m.bg_count[touched].float().median() + 1e-6


def test_update_moves_output_toward_target():
    torch.manual_seed(0)
    m = SlotMemory(64, n_keys=16, k=4, d_key=32, birth_seed=2)
    m.observe_background(torch.randn(200, 64))
    x = torch.randn(3, 64)
    for _ in range(20):
        out = m(x)
        ((out - 1) ** 2).sum().backward()
        m.sparse_step(top_t=12, lr=0.2)
    assert ((m(x) - 1) ** 2).mean() < 0.5


def test_fit_keys_spreads_background_usage():
    torch.manual_seed(0)
    x = torch.randn(2000, 64) @ torch.randn(64, 64) * 0.2 + torch.randn(64)   # anisotropic, like real hidden states
    a = SlotMemory(64, n_keys=16, k=4, d_key=32, birth_seed=3)
    b = SlotMemory(64, n_keys=16, k=4, d_key=32, birth_seed=3)
    b.fit_keys(x)
    a.observe_background(x); b.observe_background(x)
    assert int((b.bg_count > 0).sum()) >= int((a.bg_count > 0).sum())
    assert torch.all(b.V == 0)
