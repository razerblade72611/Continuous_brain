"""Repairs from Sol's L1 audit: terminal rewards reach value learning, short lives are not dropped from replay, and
episode ids are unique per world whatever the worker split."""
import json

import numpy as np

from continuous_brain_cb2.archive import FRAME_BYTES, Archive, StreamWriter
from continuous_brain_cb2.data import SequenceSampler


def _life(w, life, n, terminal_reward):
    for t in range(n):
        last = t == n - 1
        w.append({"life": life, "step": t, "seed": life, "frame": bytes(FRAME_BYTES), "inventory": [9] * 16,
                  "pos": [0, 0], "facing": [0, 1], "sleeping": False, "messages": json.dumps([]), "action": t % 17,
                  "utterance": "", "who": 0, "reward": terminal_reward if last else 0.1, "reward_caretaker": 0.0,
                  "done": last, "dying": False, "store_hash": ""})


def _archive(tmp_path, lives):
    for k, (n, r) in enumerate(lives):
        w = StreamWriter(tmp_path, f"s{k:02d}", "PRACTICE", "ind")
        _life(w, 100 + k, n, r)
        w.close()
    a = Archive(tmp_path)
    return a, a.shards(("PRACTICE",))


def test_terminal_reward_is_kept_for_value_learning(tmp_path):
    a, refs = _archive(tmp_path, [(5, 2.0), (6, -1.0)])                 # two short lives, chained into one index
    s = SequenceSampler(a, refs, T=8, seed=0, chain_short=True)
    idx = s.streams[0]
    w = s.window(idx, 0)
    first = np.flatnonzero(w["is_first"])
    assert list(first) == [0, 5]                                        # the second life starts fresh
    assert w["reward_out"][4] == 2.0                                    # the first life's terminal reward survives
    assert w["reward"][5] == 0.0                                        # ...while the core's input is still masked there
    assert w["reward_out"][:4].tolist() == [np.float32(0.1)] * 4


def test_short_lives_are_chained_not_dropped(tmp_path):
    a, refs = _archive(tmp_path, [(5, 1.0), (6, 1.0), (300, 1.0)])
    old = SequenceSampler(a, refs, T=8, seed=0)
    assert old.rows() == 300 and old.excluded_rows == 11                # the old behaviour, measured
    new = SequenceSampler(a, refs, T=8, seed=0, chain_short=True)
    assert new.rows() == 311 and new.excluded_rows == 0
    chain = [s for s in new.streams if s.components][0]
    assert chain.components == (("s00", 5), ("s01", 6))


def test_life_ids_are_unique_per_world_whatever_the_worker_split():
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("l1", Path(__file__).parents[1] / "scripts/explore/cb2_l1.py")
    sys.argv = ["x"]
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class Pool:                                                         # records the jobs instead of running them
        def __init__(self):
            self.jobs = []

        def submit(self, fn, job):
            self.jobs.append(job)

            class F:
                def result(self_inner):
                    return [{"seed": s, "life": l} for s, l in zip(job["seeds"], job["lives"])]
            return F()
    ids = {}
    for workers in (1, 3, 15, 40):
        for c in range(11):
            p = Pool()
            seeds = list(range(89_000_000 + c * 10_000, 89_000_000 + c * 10_000 + 20))
            res = m.submit_lives(p, "home", seeds, "life", 30_000_000 + c * 1000, workers)()
            for r in res:
                assert ids.setdefault((c, r["seed"]), r["life"]) == r["life"]   # same id for a world, any split
    lives = list(ids.values())
    assert len(set(lives)) == len(lives) == 11 * 20                       # no collisions across cycles


def test_rows_after_a_true_life_start_are_targets_even_inside_burn_in(tmp_path):
    from continuous_brain_cb2.data import train_mask
    a, refs = _archive(tmp_path, [(5, 2.0), (6, -1.0)])
    s = SequenceSampler(a, refs, T=8, seed=0, chain_short=True)
    idx = s.streams[0]
    head = s.window(idx, 0)                                     # starts at a real life start
    m = train_mask(head["life_start"][None], burn=6)[0]
    assert m.all()                                              # the short first life's terminal row (4) is a target
    mid = s.window(idx, 2)                                      # starts mid-life: rows before the next life need burn-in
    m2 = train_mask(mid["life_start"][None], burn=6)[0]
    assert mid["life_start"].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert m2.tolist() == [False, False, False, True, True, True, True, True]


def test_chooser_value_loss_uses_the_masked_rows_and_a_terminal_last_row():
    import torch
    from continuous_brain_cb2.chooser import Chooser, ChooserConfig, N_COMMANDS
    from continuous_brain_cb2.world_model import N_ACTIONS
    torch.manual_seed(0)
    ch = Chooser(ChooserConfig(), 16)
    B, T = 2, 6
    f = torch.randn(B, T, 16)
    c, a = torch.randint(0, N_COMMANDS, (B, T)), torch.randint(0, N_ACTIONS, (B, T))
    r, d, s = torch.zeros(B, T), torch.zeros(B, T, dtype=torch.bool), torch.zeros(B, T)
    r[:, -1], d[:, -1] = 5.0, True                              # a life ends on the window's last row
    none = torch.zeros(B, T, dtype=torch.bool)
    _, m0 = ch.losses(f, c, a, r, d, s, none)
    assert m0["value_loss"] == 0.0 and m0["hands_loss"] == 0.0  # nothing masked in -> nothing learned
    last = none.clone(); last[:, -1] = True
    _, m1 = ch.losses(f, c, a, r, d, s, last)
    assert m1["value_loss"] > 0.0                               # the terminal last row is a value target


def _l1():
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("l1", Path(__file__).parents[1] / "scripts/explore/cb2_l1.py")
    sys.argv = ["x"]
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_short_life_terminal_reaches_chooser_targets_at_real_burn(tmp_path):
    """Sol's case: lives of 50 and 206 rows, T=256, burn=64 -- the only window starts at row 0 (a true life start), so
    the first life's terminal row 49 must be a chooser target."""
    from continuous_brain_cb2.data import train_mask
    a, refs = _archive(tmp_path, [(50, 3.0), (206, 1.0)])
    s = SequenceSampler(a, refs, T=256, seed=0, chain_short=True)
    assert s.rows() == 256 and s.excluded_rows == 0
    w = s.window(s.streams[0], 0)
    assert w["reward_out"][49] == 3.0 and w["cont"][49] == 0.0
    assert train_mask(w["life_start"][None], burn=64)[0, 49]


def test_life_id_plan_rejects_collisions_before_any_work():
    import argparse

    import pytest
    m = _l1()
    ok = argparse.Namespace(cycles=10, probe_worlds=40, lives=20, holdout_lives=2, situation_worlds=24, run=10)
    m._check_worlds(ok)
    for bad in (dict(lives=m.LIFE_STRIDE), dict(cycles=1_000_000 // m.LIFE_STRIDE)):
        with pytest.raises(AssertionError):
            m._check_worlds(argparse.Namespace(**{**vars(ok), **bad}))
    # every family x cycle block of ids is disjoint for the accepted maximum
    bases = (20_000_000, 21_000_000, 22_000_000, 30_000_000, 31_000_000)
    blocks = [(b + c * m.LIFE_STRIDE, b + c * m.LIFE_STRIDE + m.LIFE_STRIDE - 1) for b in bases for c in range(11)]
    blocks.sort()
    assert all(x[1] < y[0] for x, y in zip(blocks, blocks[1:]))


def test_a_stream_head_mid_life_is_not_a_life_start(tmp_path):
    from continuous_brain_cb2.data import train_mask
    w = StreamWriter(tmp_path, "mid", "PRACTICE", "ind")
    for t in range(40, 60):                                     # a stream that begins at step 40 of a life
        w.append({"life": 7, "step": t, "seed": 7, "frame": bytes(FRAME_BYTES), "inventory": [9] * 16, "pos": [0, 0],
                  "facing": [0, 1], "sleeping": False, "messages": json.dumps([]), "action": 0, "utterance": "",
                  "who": 0, "reward": 0.0, "reward_caretaker": 0.0, "done": t == 59, "dying": False, "store_hash": ""})
    w.close()
    a = Archive(tmp_path)
    s = SequenceSampler(a, a.shards(("PRACTICE",)), T=8, seed=0)
    win = s.window(s.streams[0], 0)
    assert win["is_first"][0] == 1 and win["life_start"][0] == 0   # the window resets state, but it is no life start
    assert not train_mask(win["life_start"][None], burn=4)[0, :4].any()


def test_terminal_return_does_not_see_the_next_life():
    import torch
    from continuous_brain_cb2.chooser import DEATH, Chooser, ChooserConfig, N_COMMANDS, symlog
    from continuous_brain_cb2.world_model import N_ACTIONS
    torch.manual_seed(0)
    ch = Chooser(ChooserConfig(), 16)
    B, T = 1, 6
    f = torch.randn(B, T, 16)
    c, a = torch.zeros(B, T, dtype=torch.long), torch.zeros(B, T, dtype=torch.long)
    d = torch.zeros(B, T, dtype=torch.bool); d[0, 2] = True        # life ends at row 2, next life starts at row 3
    first = torch.zeros(B, T); first[0, 3] = 1
    r1 = torch.tensor([[0.0, 0.0, 2.0, 100.0, 100.0, 100.0]])       # the next life is very rewarding
    r2 = torch.tensor([[0.0, 0.0, 2.0, -50.0, -50.0, -50.0]])
    only_terminal = torch.zeros(B, T, dtype=torch.bool); only_terminal[0, 2] = True
    l1, _ = ch.losses(f, c, a, r1, d, first, only_terminal)
    l2, _ = ch.losses(f, c, a, r2, d, first, only_terminal)
    assert torch.allclose(l1, l2)                                   # the terminal target is r - DEATH, nothing more
    raw = ch.value(f).squeeze(-1)[0, 2]
    hands = l1 - (raw - symlog(torch.tensor(2.0 - DEATH))) ** 2
    assert torch.isfinite(hands)
