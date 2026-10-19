from pathlib import Path

import numpy as np
from PIL import Image
import torch

from continuous_brain.infant import InfantBrain, InfantConfig, one_hot_action
from continuous_brain.parented_data import NurseryDataset
from continuous_brain.parented_train import load_checkpoint, save_checkpoint
from continuous_brain.planning_v101 import (
    _all_action_values,
    _bellman_loss,
    _configure_q_only,
    train_conservative_bellman_q,
)


def _tiny_q_nursery(path: Path, episodes=6, steps=5):
    ds = NurseryDataset(path)
    for eid in range(episodes):
        x = -0.7 + 0.02 * eid
        y = -0.3
        for step in range(steps):
            rid = f"q-{eid:02d}-{step:02d}"
            # Cover every action across the tiny set.
            action = (eid + step) % 5
            context = np.zeros(48, dtype=np.float32)
            context[0] = x; context[1] = y
            context[2] = 0.65 - x; context[3] = 0.10 - y
            context[4] = 1.0 - 0.002 * step
            context[5] = -0.004
            context[6] = context[4]
            context[30] = 1.0
            nx, ny = x, y
            if action == 1: ny += 0.08
            elif action == 2: ny -= 0.08
            elif action == 3: nx -= 0.08
            elif action == 4: nx += 0.08
            next_context = context.copy()
            next_context[0] = nx; next_context[1] = ny
            next_context[2] = 0.65 - nx; next_context[3] = 0.10 - ny
            next_context[4] = max(0.0, context[4] - 0.002)
            next_context[6] = next_context[4]
            success = float(step == steps - 1)
            reward = 1.0 if success else (-0.02 if action == 2 else -0.004)
            next_context[5] = reward
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            img[:, :, 1] = 25 + eid * 20
            img[8 + step:13 + step, 7:12, 2] = 220
            nxt = np.roll(img, 1 if action == 4 else -1, axis=0)
            ip = ds.images / f"{rid}.png"; nip = ds.images / f"{rid}-next.png"
            Image.fromarray(img).save(ip); Image.fromarray(nxt).save(nip)
            ds.append({
                "episode_id": eid, "episode_step": step, "record_id": rid,
                "image": str(ip.relative_to(path)), "next_image": str(nip.relative_to(path)),
                "context": context.tolist(), "next_context": next_context.tolist(),
                "executed_action": action, "action": 0, "motor_valid": False,
                "reward": reward, "success": success, "collision": False,
                "world_valid": True,
            })
            x, y = nx, ny
    ds.write_manifest({"format": "continuous_brain.nursery.v101.test", "records": episodes * steps})
    return ds


def test_v101_q_only_freezes_everything_else():
    model = InfantBrain(InfantConfig.for_profile("proof"))
    trainable = _configure_q_only(model)
    assert trainable
    assert all(name.startswith("planning.q_head.") for name in trainable)
    assert all(p.requires_grad for p in model.planning.q_head.parameters())
    assert all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("planning.q_head."))


def test_v101_bellman_loss_is_finite_and_backpropagates_q_only():
    model = InfantBrain(InfantConfig.for_profile("proof"))
    _configure_q_only(model)
    target_q = type(model.planning.q_head)()
    # Sequential cannot be constructed empty with copied layers this way; use deepcopy.
    import copy
    target_q = copy.deepcopy(model.planning.q_head)
    for p in target_q.parameters(): p.requires_grad_(False)
    b = 8
    z = torch.randn(b, model.cfg.planning_dim)
    next_z = torch.randn(b, model.cfg.planning_dim)
    batch = {
        "z": z, "next_z": next_z, "predicted_next_z": next_z + 0.01,
        "action": torch.arange(b) % model.cfg.action_dim,
        "reward": torch.linspace(-0.1, 0.2, b),
        "terminal": torch.zeros(b),
        "behavior_return": torch.zeros(b),
    }
    loss, metrics = _bellman_loss(
        model, target_q, batch, gamma=0.95, cql_alpha=0.10,
        support_min=-2.0, support_max=2.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert metrics["td"] >= 0.0
    assert all(p.grad is not None for p in model.planning.q_head.parameters())
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("planning.q_head."))


def test_v101_training_changes_only_q_head(tmp_path):
    ds = _tiny_q_nursery(tmp_path / "nursery")
    state = tmp_path / "child"; state.mkdir()
    model = InfantBrain(InfantConfig.for_profile("proof"), vision_parent_dims={"vision.test": 8})
    save_checkpoint(
        state / "infant_training.pt", model, updates=11000, parent_id="vision.test",
        dataset_digest="old", metadata={
            "planning_cortex_version": "recurrent-latent-v1",
            "planning_updates": 4000,
            "planning_goal_gamma": 0.97,
        },
    )
    before_model, _ = load_checkpoint(state / "infant_training.pt", "cpu")
    before = {k: v.detach().clone() for k, v in before_model.state_dict().items()}
    result = train_conservative_bellman_q(
        ds.directory, state, updates=3, batch_size=10, gamma=0.95,
        learning_rate=1e-3, cql_alpha=0.10, target_tau=0.05, seed=19, device="cpu",
    )
    assert result["q_only"] is True
    assert result["changed_non_q_tensors"] == 0
    after_model, saved = load_checkpoint(state / "infant_training.pt", "cpu")
    changed_q = False
    for key, value in before.items():
        if key.startswith("planning.q_head."):
            changed_q = changed_q or (not torch.equal(value, after_model.state_dict()[key]))
        else:
            assert torch.equal(value, after_model.state_dict()[key])
    assert changed_q
    assert saved["metadata"]["planning_value_version"] == "conservative-bellman-q-v1"
