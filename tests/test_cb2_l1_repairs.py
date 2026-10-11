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
        w = StreamWriter(tmp_path, f"s{k:02d}", "PRACTICE", "ind", shard_rows=4096, row_group=64)
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
