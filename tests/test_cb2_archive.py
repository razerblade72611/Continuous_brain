"""CB2-S1 archive (exploration lane): recording never changes a life; shards, manifests and the lived chain are
exact; provenance is enforced, not promised."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from continuous_brain_cb2.archive import Archive, ArchiveError, StreamWriter, fixed, frames
from continuous_brain_cb2.record import record_life
from continuous_brain_individual.individual import Individual, IndividualConfig, first_info
from continuous_brain_organs.language import LanguageOrgan
from continuous_brain_world.env import W1Config, W1Env

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "runs/explore/vision/tile_decoder_v1.pt"
CARD = REPO / "continuous_brain_organs/cards/tile_decoder_v1.json"
SEED = 75101


def _vision():
    if WEIGHTS.exists():
        from continuous_brain_organs.vision import VisionOrgan
        return VisionOrgan(WEIGHTS, CARD)
    return None


VISION = _vision()
PERCEPTION = "organ" if VISION is not None else "oracle"


def _ind():
    return Individual(IndividualConfig(perception=PERCEPTION, lifelong=True), VISION,
                      LanguageOrgan.from_config({"backend": "dev"}))


def _unrecorded(seed, steps):
    ind = _ind()
    env = W1Env(W1Config())
    obs = env.reset(seed)
    info = first_info(env)
    acts = []
    for _ in range(steps):
        a, u = ind.decide(obs, info)
        acts.append(a)
        obs, _, done, info = env.step(a, u)
        if done:
            break
    return acts, ind.state_hash()


def test_recording_does_not_change_the_life_and_reads_back_exactly(tmp_path):
    acts, h = _unrecorded(SEED, 300)
    ind = _ind()
    w = StreamWriter(tmp_path, "lived-a", "LIVED", "ind-a", {"code": "test"}, shard_rows=128, row_group=64)
    out = record_life(ind, W1Env(W1Config()), SEED, w, steps=300)
    w.close()
    assert out["steps"] == len(acts) and ind.state_hash() == h          # record/no-record twins
    arch = Archive(tmp_path)
    assert arch.verify()["rows"] == len(acts)
    refs = arch.shards()
    assert [r.manifest["rows"] for r in refs][:2] == [128, 128]
    t = arch.read(refs[0], ("student", "teacher", "organ"))
    assert t["student"]["action"].to_pylist() == acts[:128]
    assert t["teacher"]["teacher_action"].to_pylist() == acts[:128]     # the routine is its own skill here
    assert t["student"]["store_hash"].to_pylist()[-1] != ""
    env = W1Env(W1Config())
    first = env.reset(SEED)["image"]
    assert np.array_equal(frames(t["student"])[0], first)
    assert fixed(t["organ"], "map_crop", np.int8).shape == (128, 33 * 33)
    assert all(r.manifest["seeds"] == [SEED] for r in refs) and SEED in arch.seeds_used()


def test_one_chronology_and_the_lived_chain(tmp_path):
    def row(i, who=0):
        return {"life": 0, "step": i, "seed": 1, "frame": bytes(12288), "inventory": [9] * 16, "pos": [32, 32],
                "facing": [0, 1], "sleeping": False, "messages": "[]", "action": 0, "utterance": "", "who": who,
                "reward": 0.0, "reward_caretaker": 0.0, "done": False, "dying": False, "store_hash": "x"}
    with StreamWriter(tmp_path, "lived", "LIVED", "ind", shard_rows=8, row_group=4) as w:
        for i in range(20):
            w.append(row(i, who=i % 2))
    with pytest.raises(ArchiveError):
        StreamWriter(tmp_path, "lived-2", "LIVED", "ind")               # a second lived stream: refused
    with StreamWriter(tmp_path, "lived", "LIVED", "ind", shard_rows=8, row_group=4) as w:   # it continues
        w.append(row(20))
    arch = Archive(tmp_path)
    ms = arch.manifests("lived")
    assert [m["tick_range"] for m in ms] == [[0, 7], [8, 15], [16, 19], [20, 20]]
    assert arch.verify()["shards"] == 4
    assert sum(len(t) for t in arch.lived_rows("ind")) == 11              # only LIVED x SELF happened to it
    dream = StreamWriter(tmp_path, "dream", "DREAMED", "ind")
    with pytest.raises(ArchiveError):
        dream.append(row(0))                                            # a dream carries no store hash
    assert sum(len(t) for t in arch.lived_rows("ind")) == 11
    m = json.loads((tmp_path / "lived/shard-000001.json").read_text())
    m["rows"] = 9
    (tmp_path / "lived/shard-000001.json").write_text(json.dumps(m))
    with pytest.raises(ArchiveError, match="manifest altered"):
        arch.verify()


def test_a_tampered_shard_is_caught(tmp_path):
    ind = _ind()
    with StreamWriter(tmp_path, "lived", "LIVED", "ind", shard_rows=64, row_group=32) as w:
        record_life(ind, W1Env(W1Config()), SEED, w, steps=64)
    p = tmp_path / "lived/shard-000000.student.parquet"
    t = pq.read_table(p)
    pq.write_table(t.slice(1), p)
    with pytest.raises(ArchiveError, match="student file altered"):
        Archive(tmp_path).verify()
