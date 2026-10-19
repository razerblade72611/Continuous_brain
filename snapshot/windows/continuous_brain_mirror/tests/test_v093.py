from pathlib import Path
import random

import numpy as np
import torch

from continuous_brain.infant import InfantConfig, InfantBrain, one_hot_action
from continuous_brain.parented_data import NurseryDataset
from continuous_brain.parented_train import train


def test_residual_world_starts_at_no_change_baseline():
    cfg = InfantConfig.for_profile("proof")
    model = InfantBrain(cfg, vision_parent_dims={"vision.test": 8})
    b = 4
    base = torch.randn(b, cfg.bus_dim)
    workspace = torch.randn(b, cfg.bus_dim)
    actions = one_hot_action(torch.tensor([0, 1, 2, 3]), cfg.action_dim)
    out = model.world(workspace, actions, base_bus=base)
    assert torch.allclose(out["next_bus"], base, atol=1e-7)
    assert torch.allclose(out["delta_bus"], torch.zeros_like(base), atol=1e-7)


def _write_minimal_mixed(path: Path, n_static=20, n_world=20):
    from PIL import Image
    ds = NurseryDataset(path)
    rng = np.random.default_rng(2)
    total = n_static + n_world
    for i in range(total):
        rid = f"r-{i:04d}"
        img = np.full((32, 32, 3), i % 255, dtype=np.uint8)
        ip = ds.images / f"{rid}.png"
        Image.fromarray(img).save(ip)
        vp = ds.parent_vision / f"{rid}.npy"
        np.save(vp, rng.normal(size=8).astype(np.float32))
        row = {
            "record_id": rid,
            "image": str(ip.relative_to(path)),
            "context": [0.0] * 48,
            "qas": [{"question": "What are you doing?", "answer": "I am learning."}],
            "action": i % 5,
            "executed_action": i % 5,
            "previous_action": -1,
            "reward": 0.0,
            "success": 0.0,
            "care": [0.0, 0.0, 0.0, 1.0, 0.05, 0.5],
            "parent_vision": str(vp.relative_to(path)),
            "world_valid": i >= n_static,
        }
        if i >= n_static:
            nxt = img.copy(); nxt[:, :, 1] = (i * 7) % 255
            nip = ds.images / f"{rid}-next.png"
            Image.fromarray(nxt).save(nip)
            row["next_image"] = str(nip.relative_to(path))
        ds.append(row)
    ds.write_manifest({
        "format": "continuous_brain.nursery.v093",
        "vision_parent_id": "vision.test",
        "vision_parent_dim": 8,
        "records": total,
        "source": "test",
        "digest": ds.digest(),
    })
    return ds


def test_mixed_training_reports_stratified_world_sampling(tmp_path):
    ds = _write_minimal_mixed(tmp_path / "nursery")
    state = tmp_path / "child"
    result = train(ds.directory, state, profile="proof", updates=1, batch_size=8,
                   device="cpu", seed=3, world_fraction=0.5)
    assert result["sampling"]["world_fraction"] == 0.5
    assert result["sampling"]["world_records"] > 0
    assert result["sampling"]["replay_records"] > 0


def test_v092_checkpoint_migrates_only_world_residual_heads(tmp_path):
    ds = _write_minimal_mixed(tmp_path / "nursery")
    state = tmp_path / "child"
    first = train(ds.directory, state, profile="proof", updates=1, batch_size=4, device="cpu", seed=4)
    ckpt = state / "infant_training.pt"
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Simulate a pre-v0.9.3 checkpoint by removing the migration marker and making
    # the old absolute-future head visibly nonzero.
    saved["metadata"].pop("world_dynamics_version", None)
    for k, v in saved["model"].items():
        if ".next_bus." in k:
            v.fill_(0.25)
    torch.save(saved, ckpt)
    resumed = train(ds.directory, state, profile="proof", updates=1, batch_size=4,
                    device="cpu", seed=4)
    assert resumed["world_dynamics_migrated"] is True
    assert resumed["optimizer_reinitialized"] is True
