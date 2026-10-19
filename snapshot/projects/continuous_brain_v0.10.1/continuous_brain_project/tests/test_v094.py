from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

from continuous_brain.infant import InfantBrain, InfantConfig
from continuous_brain.parented_data import NurseryDataset
from continuous_brain.parented_train import _batch, load_checkpoint, save_checkpoint, train, evaluate, training_step


def _structured_nursery(path: Path, n=80):
    ds = NurseryDataset(path)
    rng = np.random.default_rng(9)
    for i in range(n):
        rid = f"s-{i:04d}"
        action = 0 if i == 0 else 1 + (i % 4)
        cur = np.zeros(48, dtype=np.float32)
        cur[0] = (i % 10) / 10.0
        cur[1] = ((i // 10) % 8) / 8.0
        cur[2] = 0.5 - cur[0]
        cur[3] = 0.5 - cur[1]
        cur[4] = 1.0
        nxt = cur.copy()
        if action == 1: nxt[1] += 0.08; nxt[3] -= 0.08
        elif action == 2: nxt[1] -= 0.08; nxt[3] += 0.08
        elif action == 3: nxt[0] -= 0.08; nxt[2] += 0.08
        elif action == 4: nxt[0] += 0.08; nxt[2] -= 0.08
        nxt[4] -= 0.002
        reward = 0.1 if action in (1, 4) else -0.01
        nxt[5] = reward
        cur[30] = nxt[30] = 1.0

        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 0] = (i * 5) % 255
        img[8:16, 8:16, 1] = 180
        next_img = img.copy()
        shift = {0: 0, 1: 1, 2: -1, 3: -2, 4: 2}[action]
        next_img = np.roll(next_img, shift=shift, axis=1)
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
            "action": 1 + (i % 4),
            "motor_valid": True,
            "executed_action": action,
            "previous_action": -1 if i == 0 else 1 + ((i - 1) % 4),
            "reward": reward,
            "success": float(i % 17 == 0),
            "care": [0.0, 0.0, 0.0, 1.0, 0.05, 0.5],
            "parent_vision": str(vp.relative_to(path)),
            "world_valid": True,
            "source": "v094-test",
        })
    ds.write_manifest({
        "format": "continuous_brain.nursery.v094",
        "vision_parent_id": "vision.test",
        "vision_parent_dim": 8,
        "records": n,
        "source": "test",
        "digest": ds.digest(),
    })
    return ds


def test_ema_target_is_frozen_and_tracks_online():
    model = InfantBrain(InfantConfig.for_profile("proof"), vision_parent_dims={"vision.test": 8})
    assert all(not p.requires_grad for p in model.target_vision.parameters())
    with torch.no_grad():
        first_online = next(model.vision.parameters())
        first_target = next(model.target_vision.parameters())
        first_online.add_(1.0)
        before = first_target.clone()
        model.sync_target_vision(tau=0.5)
        assert not torch.allclose(before, first_target)
        expected = before * 0.5 + first_online * 0.5
        assert torch.allclose(first_target, expected)


def test_static_rows_do_not_supervise_motor(tmp_path):
    ds = _structured_nursery(tmp_path / "nursery", n=8)
    rows = ds.records()
    for row in rows:
        row["world_valid"] = False
        row["motor_valid"] = False
        row.pop("next_image", None)
        row.pop("next_context", None)
    ds.rewrite(rows)
    cfg = InfantConfig.for_profile("proof")
    model = InfantBrain(cfg, vision_parent_dims={"vision.test": 8})
    batch = _batch(ds, rows[:4], cfg, random.Random(1))
    _, metrics = training_step(model, batch, "vision.test", "cpu")
    assert metrics["motor"] == 0.0
    assert metrics["world"] == 0.0
    assert metrics["world_context"] == 0.0


def test_old_checkpoint_migrates_without_resetting_existing_bus_head(tmp_path):
    cfg = InfantConfig.for_profile("proof")
    model = InfantBrain(cfg, vision_parent_dims={"vision.test": 8})
    with torch.no_grad():
        for member in model.world.members:
            member.next_bus.weight.fill_(0.123)
            member.next_bus.bias.fill_(0.045)
    path = tmp_path / "old.pt"
    payload = model.checkpoint_payload(metadata={"world_dynamics_version": "residual-v1"})
    payload["model"] = {
        k: v for k, v in payload["model"].items()
        if not (
            k.startswith("target_vision.")
            or k.startswith("spatial.")
            or k.startswith("target_spatial.")
            or k.startswith("spatial_state_head.")
            or k.startswith("vision_fusion.")
            or k.startswith("dynamics_temporal.")
            or ".next_context." in k
        )
    }
    torch.save(payload, path)
    loaded, state = load_checkpoint(path, "cpu")
    assert any(k.startswith("target_vision.") for k in state["_v094_missing_keys"])
    assert any(".next_context." in k for k in state["_v094_missing_keys"])
    assert any(k.startswith("spatial.") for k in state["_v094_missing_keys"])
    assert any(k.startswith("target_spatial.") for k in state["_v094_missing_keys"])
    for member in loaded.world.members:
        assert torch.allclose(member.next_bus.weight, torch.full_like(member.next_bus.weight, 0.123))
        assert torch.allclose(member.next_context.weight, torch.zeros_like(member.next_context.weight))
    for online, target in zip(loaded.vision.parameters(), loaded.target_vision.parameters()):
        assert torch.allclose(online, target)


def test_structured_world_evidence_allows_sparse_idle(tmp_path):
    ds = _structured_nursery(tmp_path / "nursery", n=80)
    state = tmp_path / "child"
    result = train(ds.directory, state, profile="proof", updates=2, batch_size=8,
                   device="cpu", seed=5, world_fraction=0.75)
    assert result["updates"] == 2
    assert "world_context" in result["latest"]
    ev = evaluate(ds.directory, state, device="cpu", seed=7)
    assert ev["evidence"]["world_context_valid_records"] >= 8
    assert "world_next_context_error" in ev["metrics"]
    # With an 80-record nursery, the 10% split is intentionally too small for the
    # production world gate, but the supported-action diagnostic must ignore the
    # single sparse idle class rather than making min(counts) the criterion.
    assert 0 not in ev["evidence"]["supported_executed_action_counts"]
    assert all(c >= 3 for c in ev["evidence"]["supported_executed_action_counts"].values())
