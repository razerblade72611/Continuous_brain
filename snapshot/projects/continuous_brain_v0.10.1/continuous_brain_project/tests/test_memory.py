import numpy as np
from continuous_brain.memory import EpisodicMemory


def test_memory_roundtrip(tmp_path):
    mem = EpisodicMemory(tmp_path / "m.sqlite3")
    mem.store(step=1, reward=1.0, surprise=0.2, action=3, latent=np.array([1, 0, 0], dtype=np.float32))
    mem.store(step=2, reward=0.0, surprise=0.1, action=4, latent=np.array([0, 1, 0], dtype=np.float32))
    got = mem.retrieve(np.array([0.9, 0.1, 0], dtype=np.float32), k=1)
    assert got[0]["step"] == 1
    mem.close()


def test_visual_memory_roundtrip(tmp_path):
    mem = EpisodicMemory(tmp_path / "mv.sqlite3")
    mem.store_visual(
        step=3, reward=0.5, novelty=0.4, action=2, backbone="test",
        latent=np.array([1, 0, 0], dtype=np.float32), metadata={"kind": "frame"}
    )
    mem.store_visual(
        step=4, reward=0.0, novelty=0.1, action=1, backbone="test",
        latent=np.array([0, 1, 0], dtype=np.float32)
    )
    got = mem.retrieve_visual(np.array([0.95, 0.05, 0], dtype=np.float32), k=1)
    assert got[0]["step"] == 3
    assert got[0]["backbone"] == "test"
    assert mem.visual_count() == 2
    mem.close()
