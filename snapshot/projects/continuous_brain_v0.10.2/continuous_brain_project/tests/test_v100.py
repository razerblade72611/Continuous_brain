from pathlib import Path

import numpy as np
from PIL import Image
import torch

from continuous_brain.infant import InfantBrain, InfantConfig, one_hot_action
from continuous_brain.parented_data import NurseryDataset
from continuous_brain.parented_train import load_checkpoint, save_checkpoint
from continuous_brain.planning_v100 import _configure_planning_only, train_planning_cortex


def _tiny_planning_nursery(path: Path, episodes=5, steps=4):
    ds = NurseryDataset(path)
    for eid in range(episodes):
        x = -0.7 + 0.03 * eid
        y = -0.2
        for step in range(steps):
            rid = f"p-{eid:02d}-{step:02d}"
            action = 1 if step < 2 else 4
            context = np.zeros(48, dtype=np.float32)
            context[0] = x; context[1] = y
            context[2] = 0.65 - x; context[3] = 0.10 - y
            context[4] = 1.0 - 0.002 * step
            context[5] = -0.004
            context[6] = context[4]
            context[30] = 1.0
            nx, ny = x, y
            if action == 1: ny += 0.08
            if action == 4: nx += 0.08
            next_context = context.copy()
            next_context[0] = nx; next_context[1] = ny
            next_context[2] = 0.65 - nx; next_context[3] = 0.10 - ny
            next_context[4] = max(0.0, context[4] - 0.002)
            next_context[6] = next_context[4]
            success = float(step == steps - 1)
            reward = 2.0 if success else -0.004
            next_context[5] = reward
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            img[:, :, 1] = 30 + eid * 20
            img[8 + step:13 + step, 7:12, 2] = 220
            nxt = np.roll(img, 1 if action == 4 else -1, axis=0)
            ip = ds.images / f"{rid}.png"
            nip = ds.images / f"{rid}-next.png"
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
    ds.write_manifest({"format": "continuous_brain.nursery.v100.test", "records": episodes * steps})
    return ds


def test_v096_checkpoint_migrates_by_adding_only_planning_cortex(tmp_path):
    cfg = InfantConfig.for_profile("proof")
    model = InfantBrain(cfg, vision_parent_dims={"vision.test": 8})
    payload = model.checkpoint_payload(metadata={"updates": 11000})
    payload["updates"] = 11000
    old_model = {k: v.clone() for k, v in payload["model"].items() if not k.startswith("planning.")}
    payload["model"] = old_model
    path = tmp_path / "v096.pt"
    torch.save(payload, path)

    loaded, state = load_checkpoint(path, "cpu")
    assert state["_v100_missing_keys"]
    assert all(k.startswith("planning.") for k in state["_v100_missing_keys"])
    loaded_state = loaded.state_dict()
    for key, value in old_model.items():
        assert torch.equal(value, loaded_state[key])


def test_planning_cortex_has_separate_recurrent_state_and_action_dependence():
    model = InfantBrain(InfantConfig.for_profile("proof"))
    p = model.planning
    spatial = torch.randn(2, model.cfg.bus_dim)
    context = torch.randn(2, model.cfg.dynamics_context_dim)
    z = p.encode_state(spatial, context)
    assert z.shape == (2, model.cfg.planning_dim)
    a1 = one_hot_action(torch.tensor([1, 1]), model.cfg.action_dim)
    a4 = one_hot_action(torch.tensor([4, 4]), model.cfg.action_dim)
    s1 = p.step(z, a1)
    s4 = p.step(z, a4)
    assert s1["latent"].shape == z.shape
    assert s1["context"].shape == (2, model.cfg.dynamics_context_dim)
    assert s1["spatial"].shape == (2, model.cfg.bus_dim)
    assert not torch.allclose(s1["latent"], s4["latent"])


def test_planning_only_freezes_every_pre_v100_parameter():
    model = InfantBrain(InfantConfig.for_profile("proof"))
    trainable = _configure_planning_only(model)
    assert trainable
    assert all(name.startswith("planning.") for name in trainable)
    assert all(p.requires_grad for p in model.planning.parameters())
    assert all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("planning."))


def test_planning_training_does_not_change_existing_child(tmp_path):
    ds = _tiny_planning_nursery(tmp_path / "nursery")
    state = tmp_path / "child"
    state.mkdir()
    model = InfantBrain(InfantConfig.for_profile("proof"), vision_parent_dims={"vision.test": 8})
    save_checkpoint(state / "infant_training.pt", model, updates=11000,
                    parent_id="vision.test", dataset_digest="old")
    before_model, _ = load_checkpoint(state / "infant_training.pt", "cpu")
    before = {k: v.detach().clone() for k, v in before_model.state_dict().items() if not k.startswith("planning.")}

    result = train_planning_cortex(
        ds.directory, state, updates=2, batch_size=2, unroll=2,
        gamma=0.95, goal_gamma=0.97, learning_rate=1e-3, seed=12, device="cpu",
    )
    assert result["planning_only"] is True
    assert result["source_training_updates"] == 11000
    after_model, saved = load_checkpoint(state / "infant_training.pt", "cpu")
    for key, value in before.items():
        assert torch.equal(value, after_model.state_dict()[key])
    assert saved["metadata"]["planning_cortex_version"] == "recurrent-latent-v1"
