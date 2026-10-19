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
