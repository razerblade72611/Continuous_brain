import numpy as np

from continuous_brain.replay import ReplayBuffer, Transition


def transition(level):
    z = np.zeros(2, dtype=np.float32)
    return Transition(z, z, 0, 0, 0.0, z, False, skill_level=level, valid_actions=(0, 2, 3) if level == 0 else (0, 1, 2, 3))


def test_stratified_replay_keeps_prior_skill_rehearsal():
    rb = ReplayBuffer(100, seed=3)
    for _ in range(30): rb.append(transition(0))
    for _ in range(30): rb.append(transition(1))
    batch = rb.sample(20, current_level=1, current_fraction=0.75)
    levels = [t.skill_level for t in batch]
    assert levels.count(1) == 15
    assert levels.count(0) == 5


def test_replay_state_round_trip_preserves_skill_tags():
    rb = ReplayBuffer(100, seed=4)
    for level in (0, 1, 1, 0, 1):
        rb.append(transition(level))
    clone = ReplayBuffer(100, seed=5)
    loaded = clone.load_state_dict(rb.state_dict(limit=4))
    assert loaded == 4
    assert len(clone) == 4
    assert clone.level_counts()[1] == 3
    assert clone.level_counts()[0] == 1


def test_persistent_replay_reserves_prior_skill_rows():
    rb = ReplayBuffer(200, seed=6)
    for _ in range(20): rb.append(transition(0))
    for _ in range(100): rb.append(transition(1))
    state = rb.state_dict(limit=20)
    levels = [row["skill_level"] for row in state["items"]]
    assert 0 in levels
    assert levels.count(0) >= 5
    assert len(levels) == 20
