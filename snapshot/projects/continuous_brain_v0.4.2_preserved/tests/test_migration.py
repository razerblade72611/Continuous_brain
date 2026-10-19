import torch

from continuous_brain.agent import ContinuousAgent
from continuous_brain.config import BrainConfig
from continuous_brain.environment import ForagingWorld
from continuous_brain.model import ContinuousBrainNet
from continuous_brain.vision import PixelStatsBackbone


def test_v02_style_checkpoint_migrates_to_visual_agent(tmp_path):
    cfg = BrainConfig(embed_dim=16, hidden_dim=32)
    old = ContinuousBrainNet(6, 5, 16, 32)
    old_target = ContinuousBrainNet(6, 5, 16, 32)
    opt = torch.optim.AdamW(old.parameters(), lr=cfg.learning_rate)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    torch.save({
        "model": old.state_dict(),
        "target": old_target.state_dict(),
        "optimizer": opt.state_dict(),
        "steps": 123,
        "reward": 4.5,
        "goals": 7,
    }, state_dir / "brain.pt")

    agent = ContinuousAgent(
        ForagingWorld(seed=1), cfg, state_dir,
        device="cpu", vision_backbone=PixelStatsBackbone(), vision_every=2,
    )
    try:
        assert agent.total_steps == 123
        assert agent.goals == 7
        assert agent.net.visual_adapter is not None
    finally:
        agent.episodic.close()


def test_visual_backbone_dimension_can_change(tmp_path):
    cfg = BrainConfig(embed_dim=16, hidden_dim=32)
    state_dir = tmp_path / "switch"
    state_dir.mkdir()
    old_visual = ContinuousBrainNet(6, 5, 16, 32, visual_dim=48)
    torch.save({
        "model": old_visual.state_dict(),
        "target": old_visual.state_dict(),
        "steps": 9,
        "reward": 1.0,
        "goals": 1,
    }, state_dir / "brain.pt")

    class FakeVision:
        name = "fake-384"
        output_dim = 384
        pretrained = True
        def encode(self, rgb):
            import numpy as np
            return np.zeros(384, dtype=np.float32)
        def describe(self):
            return {"name": self.name, "output_dim": self.output_dim, "pretrained": True}

    agent = ContinuousAgent(
        ForagingWorld(seed=2), cfg, state_dir,
        device="cpu", vision_backbone=FakeVision(),
    )
    try:
        assert agent.total_steps == 9
        assert agent.net.visual_adapter[0].in_features == 384
    finally:
        agent.episodic.close()


def test_bootstrap_checkpoint_imports_weights_without_counters(tmp_path):
    cfg = BrainConfig(embed_dim=16, hidden_dim=32)
    old = ContinuousBrainNet(6, 5, 16, 32)
    with torch.no_grad():
        old.perception[0].weight.fill_(0.123)
    source = tmp_path / "source.pt"
    torch.save({"model": old.state_dict(), "steps": 999, "reward": 55.0, "goals": 44}, source)

    agent = ContinuousAgent(
        ForagingWorld(seed=3), cfg, tmp_path / "fresh",
        device="cpu", bootstrap_checkpoint=source,
    )
    try:
        assert agent.total_steps == 0
        assert agent.goals == 0
        assert torch.allclose(agent.net.perception[0].weight, old.perception[0].weight)
    finally:
        agent.episodic.close()
