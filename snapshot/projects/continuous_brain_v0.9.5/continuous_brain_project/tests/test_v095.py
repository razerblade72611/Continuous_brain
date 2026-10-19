from pathlib import Path

import numpy as np
from PIL import Image
import torch

from continuous_brain.infant import InfantBrain, InfantConfig, one_hot_action
from continuous_brain.parented_data import NurseryDataset
from continuous_brain.parented_train import load_checkpoint, train


def _tiny_nursery(path: Path, n=24):
    ds = NurseryDataset(path)
    rng = np.random.default_rng(5)
    for i in range(n):
        rid = f"w-{i:04d}"
        action = 1 + (i % 4)
        cur = np.zeros(48, dtype=np.float32)
        cur[0] = -0.5 + (i % 8) * 0.12
        cur[1] = -0.4 + ((i // 8) % 3) * 0.25
        cur[2] = 0.4 - cur[0]
        cur[3] = 0.3 - cur[1]
        cur[4] = 0.9
        cur[6] = cur[4]
        nxt = cur.copy()
        if action == 1: nxt[1] += 0.08; nxt[3] -= 0.08
        elif action == 2: nxt[1] -= 0.08; nxt[3] += 0.08
        elif action == 3: nxt[0] -= 0.08; nxt[2] += 0.08
        elif action == 4: nxt[0] += 0.08; nxt[2] -= 0.08
        nxt[4] -= 0.002
        nxt[6] = nxt[4]
        reward = 0.02 if action in (1, 4) else -0.01
        nxt[5] = reward
        cur[30] = nxt[30] = 1.0

        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[8:16, 8:16, 0] = 180
        img[:, :, 1] = (i * 7) % 255
        next_img = np.roll(img, shift={1:1,2:-1,3:-2,4:2}[action], axis=1)
        ip = ds.images / f"{rid}.png"
        nip = ds.images / f"{rid}-next.png"
        Image.fromarray(img).save(ip)
        Image.fromarray(next_img).save(nip)
        vp = ds.parent_vision / f"{rid}.npy"
        np.save(vp, rng.normal(size=8).astype(np.float32))
        ds.append({
            "record_id": rid,
            "image": str(ip.relative_to(path)),
            "next_image": str(nip.relative_to(path)),
            "context": cur.tolist(),
            "next_context": nxt.tolist(),
            "qas": [{"question": "What are you doing?", "answer": "I am learning."}],
            "action": action,
            "motor_valid": True,
            "executed_action": action,
            "previous_action": -1,
            "reward": reward,
            "success": 0.0,
            "care": [0.0, 0.0, 0.0, 1.0, 0.05, 0.5],
            "parent_vision": str(vp.relative_to(path)),
            "world_valid": True,
            "source": "v095-test",
        })
    ds.write_manifest({
        "format": "continuous_brain.nursery.v095",
        "vision_parent_id": "vision.test",
        "vision_parent_dim": 8,
        "records": n,
        "source": "test",
        "digest": ds.digest(),
    })
    return ds


def test_v094_checkpoint_adds_zero_state_conditioners_without_changing_predictions(tmp_path):
    cfg = InfantConfig.for_profile("proof")
    model = InfantBrain(cfg, vision_parent_dims={"vision.test": 8})
    model.eval()
    workspace = torch.randn(3, cfg.bus_dim)
    action = one_hot_action(torch.tensor([1, 2, 4]), cfg.action_dim)
    base_bus = torch.randn(3, cfg.bus_dim)
    base_context = torch.randn(3, cfg.context_dim)
    with torch.no_grad():
        before = model.world(workspace, action, base_bus=base_bus, base_context=base_context)

    payload = model.checkpoint_payload(metadata={"world_dynamics_version": "dual-stream-spatial-v1", "updates": 6000})
    payload["updates"] = 6000
    payload["model"] = {
        k: v for k, v in payload["model"].items()
        if ".state_bus." not in k and ".state_context." not in k
    }
    path = tmp_path / "v094.pt"
    torch.save(payload, path)

    loaded, state = load_checkpoint(path, "cpu")
    assert state["_v095_missing_keys"]
    assert all(
        (".state_bus." in k or ".state_context." in k)
        for k in state["_v095_missing_keys"]
    )
    loaded.eval()
    with torch.no_grad():
        after = loaded.world(workspace, action, base_bus=base_bus, base_context=base_context)
    assert torch.allclose(before["next_bus"], after["next_bus"], atol=1e-7)
    assert torch.allclose(before["next_context"], after["next_context"], atol=1e-7)


def test_world_focus_resume_reports_state_conditioning_migration(tmp_path):
    ds = _tiny_nursery(tmp_path / "nursery")
    state = tmp_path / "child"
    first = train(ds.directory, state, profile="proof", updates=1, batch_size=8, device="cpu", seed=3)
    assert first["updates"] == 1

    # Simulate a v0.9.4 checkpoint by removing the new adapter tensors and changing metadata.
    ckpt = state / "infant_training.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload["model"] = {
        k: v for k, v in payload["model"].items()
        if ".state_bus." not in k and ".state_context." not in k
    }
    payload["metadata"]["world_dynamics_version"] = "dual-stream-spatial-v1"
    payload.pop("optimizer", None)
    torch.save(payload, ckpt)

    second = train(ds.directory, state, profile="proof", updates=1, batch_size=8, device="cpu",
                   seed=4, world_fraction=0.75, world_focus=True)
    assert second["updates"] == 2
    assert second["world_state_conditioning_migrated"] is True
    assert second["optimizer_reinitialized"] is True
    assert second["world_focus"] is True
