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


def test_agent_persists_bounded_replay_across_restart(tmp_path):
    import numpy as np
    from continuous_brain.replay import Transition
    cfg = BrainConfig(embed_dim=16, hidden_dim=32, replay_checkpoint_items=16)
    state_dir = tmp_path / "persist"
    agent = ContinuousAgent(ForagingWorld(seed=4), cfg, state_dir, device="cpu")
    try:
        zobs = np.zeros(6, dtype=np.float32)
        zhid = np.zeros(32, dtype=np.float32)
        for level in (0, 1, 1):
            agent.replay.append(Transition(
                zobs.copy(), zhid.copy(), 0, 0, 0.0, zobs.copy(), False,
                skill_level=level, valid_actions=(0, 2, 3) if level == 0 else (0, 1, 2, 3),
            ))
        agent.total_steps = 12
        agent.save()
    finally:
        agent.episodic.close()
    clone = ContinuousAgent(ForagingWorld(seed=5), cfg, state_dir, device="cpu")
    try:
        assert len(clone.replay) == 3
        assert clone.replay.level_counts()[1] == 2
    finally:
        clone.episodic.close()


def test_mastered_skill_snapshot_and_atomic_previous_checkpoint(tmp_path):
    cfg = BrainConfig(embed_dim=16, hidden_dim=32, replay_checkpoint_items=0)
    state_dir = tmp_path / "snapshot"
    agent = ContinuousAgent(ForagingWorld(seed=6), cfg, state_dir, device="cpu")
    try:
        agent.total_steps = 50
        agent.save()
        agent.total_steps = 51
        agent.save()
        assert (state_dir / "brain.previous.pt").exists()
        stats = {"recent_attempts": 100, "recent_success_rate": 0.99, "median_success_steps": 4.0}
        agent._save_skill_snapshot(0, stats)
        assert (state_dir / "skills" / "L0_mastered.pt").exists()
        assert 0 in agent.mastered_levels
    finally:
        agent.episodic.close()


def test_corrupt_primary_checkpoint_falls_back_to_previous(tmp_path):
    cfg = BrainConfig(embed_dim=16, hidden_dim=32, replay_checkpoint_items=0)
    state_dir = tmp_path / "recovery"
    agent = ContinuousAgent(ForagingWorld(seed=7), cfg, state_dir, device="cpu")
    try:
        agent.total_steps = 40
        agent.save()
        agent.total_steps = 41
        agent.save()  # previous now contains step 40
    finally:
        agent.episodic.close()
    (state_dir / "brain.pt").write_bytes(b"not a torch checkpoint")
    clone = ContinuousAgent(ForagingWorld(seed=8), cfg, state_dir, device="cpu")
    try:
        assert clone.total_steps == 40
    finally:
        clone.episodic.close()


def test_shape_migration_resets_incompatible_adam_moments(tmp_path):
    old_cfg = BrainConfig(embed_dim=16, hidden_dim=32)
    old = ContinuousBrainNet(6, 5, 16, 32)
    opt = torch.optim.AdamW(old.parameters(), lr=old_cfg.learning_rate)
    # Populate Adam exp_avg / exp_avg_sq so a shape mismatch would otherwise crash
    # only on the first optimizer.step().
    loss = sum((p.float() ** 2).mean() for p in old.parameters())
    loss.backward(); opt.step()
    state_dir = tmp_path / "shape_opt"; state_dir.mkdir()
    torch.save({
        "model": old.state_dict(), "target": old.state_dict(),
        "optimizer": opt.state_dict(), "steps": 5, "reward": 0.0, "goals": 0,
    }, state_dir / "brain.pt")

    new_cfg = BrainConfig(embed_dim=16, hidden_dim=64)
    agent = ContinuousAgent(ForagingWorld(seed=9), new_cfg, state_dir, device="cpu")
    try:
        assert agent.total_steps == 5
        assert len(agent.optimizer.state) == 0
    finally:
        agent.episodic.close()
