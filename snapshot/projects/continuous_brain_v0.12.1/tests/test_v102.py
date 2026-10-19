from pathlib import Path

import numpy as np
from PIL import Image
import torch

from continuous_brain.infant import InfantBrain, InfantConfig, one_hot_action
from continuous_brain.parented_data import NurseryDataset
from continuous_brain.parented_train import load_checkpoint, save_checkpoint
from continuous_brain.planning_v102 import _configure_options_only, train_temporal_options


def _tiny_option_nursery(path: Path, episodes=8, steps=12):
    ds = NurseryDataset(path)
    for eid in range(episodes):
        x = -0.72 + 0.01 * eid
        y = -0.22 + 0.01 * eid
        target = np.array([0.64, -0.05], dtype=np.float32)
        for step in range(steps):
            rid = f"o-{eid:02d}-{step:02d}"
            # Three coherent successful phases, with small episode variation.
            if step < 4:
                action = 1
            elif step < 8:
                action = 4
            else:
                action = 2 if step < 10 else 4
            context = np.zeros(48, dtype=np.float32)
            context[0] = x; context[1] = y
            context[2] = target[0] - x; context[3] = target[1] - y
            context[4] = 1.0 - 0.002 * step
            context[5] = -0.004; context[6] = context[4]; context[30] = 1.0
            nx, ny = x, y
            if action == 1: ny += 0.08
            elif action == 2: ny -= 0.08
            elif action == 3: nx -= 0.08
            elif action == 4: nx += 0.08
            next_context = context.copy()
            next_context[0] = nx; next_context[1] = ny
            next_context[2] = target[0] - nx; next_context[3] = target[1] - ny
            next_context[4] = max(0.0, context[4] - 0.002); next_context[6] = next_context[4]
            success = float(step == steps - 1)
            reward = 2.0 if success else -0.004
            next_context[5] = reward
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            img[:, :, 1] = 25 + eid * 15
            img[8 + min(step, 10):13 + min(step, 10), 7:12, 2] = 220
            nxt = np.roll(img, 1 if action in (2,4) else -1, axis=0)
            ip = ds.images / f"{rid}.png"; nip = ds.images / f"{rid}-next.png"
            Image.fromarray(img).save(ip); Image.fromarray(nxt).save(nip)
            ds.append({
                "episode_id": eid, "episode_step": step, "record_id": rid,
                "image": str(ip.relative_to(path)), "next_image": str(nip.relative_to(path)),
                "context": context.tolist(), "next_context": next_context.tolist(),
                "executed_action": action, "action": 0, "motor_valid": False,
                "reward": reward, "success": success, "collision": False, "world_valid": True,
            })
            x, y = nx, ny
    ds.write_manifest({"format": "continuous_brain.nursery.v102.test", "records": episodes * steps})
    return ds


def test_v101_checkpoint_migrates_by_adding_only_option_cortex(tmp_path):
    model = InfantBrain(InfantConfig.for_profile("proof"), vision_parent_dims={"vision.test": 8})
    payload = model.checkpoint_payload(metadata={"planning_value_version": "conservative-bellman-q-v1"})
    old_model = {k: v.clone() for k, v in payload["model"].items() if not k.startswith("options.")}
    payload["model"] = old_model
    path = tmp_path / "v101.pt"
    torch.save(payload, path)
    loaded, state = load_checkpoint(path, "cpu")
    assert state["_v102_missing_keys"]
    assert all(k.startswith("options.") for k in state["_v102_missing_keys"])
    for key, value in old_model.items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_option_cortex_has_selector_termination_policy_and_action_value():
    model = InfantBrain(InfantConfig.for_profile("proof"))
    z = torch.randn(4, model.cfg.planning_dim)
    option = torch.tensor([0, 1, 2, 3])
    action = one_hot_action(torch.tensor([1, 4, 2, 0]), model.cfg.action_dim)
    assert model.options.selector_logits(z).shape == (4, model.cfg.option_count)
    assert model.options.termination_logit(z, option).shape == (4,)
    assert model.options.action_logits(z, option).shape == (4, model.cfg.action_dim)
    assert model.options.action_value(z, option, action).shape == (4,)


def test_options_only_freezes_every_older_tensor():
    model = InfantBrain(InfantConfig.for_profile("proof"))
    trainable = _configure_options_only(model)
    assert trainable
    assert all(name.startswith("options.") for name in trainable)
    assert all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("options."))


def test_option_training_changes_only_option_cortex(tmp_path):
    ds = _tiny_option_nursery(tmp_path / "nursery")
    state = tmp_path / "child"; state.mkdir()
    model = InfantBrain(InfantConfig.for_profile("proof"), vision_parent_dims={"vision.test": 8})
    payload = model.checkpoint_payload(metadata={
        "planning_cortex_version": "recurrent-latent-v1+conservative-bellman-q-v1",
        "planning_updates": 4000,
        "planning_bellman_q_updates": 3000,
        "planning_value_version": "conservative-bellman-q-v1",
        "planning_goal_gamma": 0.97,
    })
    payload["updates"] = 11000
    payload["model"] = {k: v for k, v in payload["model"].items() if not k.startswith("options.")}
    torch.save(payload, state / "infant_training.pt")
    before_model, _ = load_checkpoint(state / "infant_training.pt", "cpu")
    before = {k: v.detach().clone() for k, v in before_model.state_dict().items() if not k.startswith("options.")}
    result = train_temporal_options(
        ds.directory, state, updates=2, batch_size=16, lookahead=3,
        gamma=0.95, goal_gamma=0.97, learning_rate=1e-3,
        q_margin=0.05, seed=23, device="cpu",
    )
    assert result["option_only"] is True
    assert result["changed_non_option_tensors"] == 0
    assert result["option_migrated"] is True
    after, saved = load_checkpoint(state / "infant_training.pt", "cpu")
    for key, value in before.items():
        assert torch.equal(value, after.state_dict()[key])
    assert saved["metadata"]["option_cortex_version"] == "latent-temporal-options-v1"
