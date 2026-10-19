"""Store authority and rebuild tests."""
from continuous_individual.store import CanonicalStore
from continuous_individual.agent import Individual


def test_erase_and_rebuild():
    store = CanonicalStore()
    ind = Individual(store=store)
    original_hash = store.state_hash()
    store.erase("controller.weights")
    assert not store.has("controller.weights")
    # New individual from same store starts with fresh weights
    ind2 = Individual(store=store)
    assert store.has("controller.weights")
    assert store.state_hash() != original_hash


def test_save_load_roundtrip(tmp_path):
    store = CanonicalStore()
    store.write("test", {"a": 1, "b": [2, 3]})
    p = tmp_path / "s.pkl"
    store.save(p)
    loaded = CanonicalStore.load(p)
    assert loaded.read("test") == {"a": 1, "b": [2, 3]}
    assert loaded.state_hash() == store.state_hash()
