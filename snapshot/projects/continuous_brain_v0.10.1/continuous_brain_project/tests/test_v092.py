import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from continuous_brain.infant import InfantConfig, InfantBrain
from continuous_brain.parented_data import NurseryDataset, merge_nurseries
from continuous_brain.parented_train import _batch, train, evaluate


def _make_nursery(path: Path, prefix: str, n=60):
    ds = NurseryDataset(path)
    rng = np.random.default_rng(7)
    for i in range(n):
        rid = f"{prefix}-{i:04d}"
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 0] = (i * 3) % 255
        nxt = img.copy(); nxt[:, :, 1] = (i * 5 + 20) % 255
        ip = ds.images / f"{rid}.png"; nip = ds.images / f"{rid}-next.png"
        Image.fromarray(img).save(ip); Image.fromarray(nxt).save(nip)
        vec = rng.normal(size=8).astype(np.float32)
        vp = ds.parent_vision / f"{rid}.npy"; np.save(vp, vec)
        action = 1 + (i % 4)
        executed = (action + (1 if i % 7 == 0 else 0)) % 5
        previous = -1 if i % 10 == 0 else ((i - 1) % 5)
        ds.append({
            "record_id": rid,
            "image": str(ip.relative_to(path)),
            "next_image": str(nip.relative_to(path)),
            "context": [0.0] * 48,
            "qas": [{"kind": "state", "question": "What are you doing?", "answer": "I am learning."}],
            "action": action,
            "executed_action": executed,
            "previous_action": previous,
            "reward": float(i % 3) / 10.0,
            "success": float(i % 11 == 0),
            "care": [0.0, 0.0, 0.0, 1.0, 0.05, 0.5],
            "parent_vision": str(vp.relative_to(path)),
            "world_valid": True,
            "source": "test",
        })
    ds.write_manifest({
        "format": "continuous_brain.nursery.v092",
        "vision_parent_id": "vision.test",
        "vision_parent_dim": 8,
        "records": n,
        "source": "test",
        "digest": ds.digest(),
    })
    return ds


def test_batch_separates_motor_executed_and_previous(tmp_path):
    ds = _make_nursery(tmp_path / "nursery", "a", n=8)
    rows = ds.records()[:8]
    cfg = InfantConfig.for_profile("proof")
    import random
    batch = _batch(ds, rows, cfg, random.Random(1))
    assert batch["actions"].tolist() == [r["action"] for r in rows]
    assert batch["executed_actions"].tolist() == [r["executed_action"] for r in rows]
    expected = [(r["previous_action"] + 1) if r["previous_action"] >= 0 else 0 for r in rows]
    assert batch["previous_action_tokens"].tolist() == expected
    assert any(a != e for a, e in zip(batch["actions"].tolist(), batch["executed_actions"].tolist()))


def test_merge_nurseries_rewrites_files(tmp_path):
    a = _make_nursery(tmp_path / "a", "a", n=3)
    b = _make_nursery(tmp_path / "b", "b", n=4)
    out = tmp_path / "merged"
    manifest = merge_nurseries([a.directory, b.directory], out)
    ds = NurseryDataset(out)
    rows = ds.records()
    assert manifest["records"] == 7
    assert len(rows) == 7
    for r in rows:
        assert (out / r["image"]).exists()
        assert (out / r["next_image"]).exists()
        assert (out / r["parent_vision"]).exists()


def test_embodied_training_and_world_eval_smoke(tmp_path):
    ds = _make_nursery(tmp_path / "nursery", "a", n=60)
    state = tmp_path / "child"
    result = train(ds.directory, state, profile="proof", updates=2, batch_size=4, device="cpu", seed=3)
    assert result["updates"] == 2
    assert "temporal" in result["latest"]
    ev = evaluate(ds.directory, state, device="cpu", seed=5)
    assert ev["evidence"]["world_valid_records"] > 0
    assert "world_next_bus_error" in ev["metrics"]
    assert "world_wrong_action_error" in ev["metrics"]
    assert ev["world_model_passed"] in (True, False, None)
