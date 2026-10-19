import torch
from continuous_brain.model import ContinuousBrainNet


def test_shapes():
    net = ContinuousBrainNet(6, 5, 16, 32)
    obs = torch.zeros(2, 6)
    h = net.initial_state(2, torch.device("cpu"))
    a = torch.zeros(2, 5)
    h2 = net.encode_step(obs, a, h)
    assert h2.shape == (2, 32)
    q = net.q_values(h2)
    assert q.shape == (2, 5)
    p, r = net.predict(h2, a)
    assert p.shape == (2, 6)
    assert r.shape == (2,)
