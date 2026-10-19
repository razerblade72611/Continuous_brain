import numpy as np
import torch

from continuous_brain.environment import ForagingWorld
from continuous_brain.model import ContinuousBrainNet
from continuous_brain.vision import PixelStatsBackbone


def test_render_and_pixel_vision():
    env = ForagingWorld(seed=3)
    frame = env.render_rgb(64)
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8
    v = PixelStatsBackbone().encode(frame)
    assert v.shape == (48,)
    assert np.isfinite(v).all()
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_visual_adapter_preserves_core_shapes():
    net = ContinuousBrainNet(6, 5, 16, 32, visual_dim=48)
    obs = torch.zeros(2, 6)
    visual = torch.zeros(2, 48)
    h = net.initial_state(2, torch.device("cpu"))
    a = torch.zeros(2, 5)
    h2 = net.encode_step(obs, a, h, visual=visual)
    assert h2.shape == (2, 32)
    assert 0.0 < net.visual_gate() < 0.1
