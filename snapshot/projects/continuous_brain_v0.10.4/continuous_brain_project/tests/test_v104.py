from pathlib import Path

import numpy as np
from PIL import Image
import torch

from continuous_brain import __version__
from continuous_brain.infant import InfantBrain, InfantConfig
from continuous_brain.parented_data import NurseryDataset
from continuous_brain.parented_train import load_checkpoint, save_checkpoint
from continuous_brain.world_consolidation_v104 import (
    collect_world_holdout,
    evaluate_world_consolidation,
    seal_world_holdouts,
    train_world_consolidation,
    transition_digest,
)


def _tiny_world_dataset(path: Path, *, shift: float, rows: int = 30, holdout: bool = False):
    ds = NurseryDataset(path)
    for i in range(rows):
        rid = f"w-{i:04d}"
        action = i % 5
        ctx = np.zeros(48, dtype=np.float32)
        ctx[0] = -0.5 + 0.02 * i + shift
        ctx[1] = -0.2 + shift
        ctx[2] = 0.7 - ctx[0]
        ctx[3] = 0.2 - ctx[1]
        ctx[4] = 1.0 - 0.002 * i
        ctx[5] = -0.004
        ctx[6] = ctx[4]
        ctx[30] = 1.0
        nxt = ctx.copy()
        if action == 1: nxt[1] += 0.08
        elif action == 2: nxt[1] -= 0.08
        elif action == 3: nxt[0] -= 0.08
        elif action == 4: nxt[0] += 0.08
        nxt[2] = 0.7 - nxt[0]
        nxt[3] = 0.2 - nxt[1]
        nxt[4] = max(0.0, ctx[4] - 0.002)
        nxt[6] = nxt[4]
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 0] = (30 + i * 3) % 255
        img[6:10, 4 + (i % 10):8 + (i % 10), 1] = 220
        nimg = np.roll(img, 1 if action in (2, 4) else -1, axis=0)
        ip = ds.images / f"{rid}.png"
        npth = ds.images / f"{rid}-next.png"
        Image.fromarray(img).save(ip)
        Image.fromarray(nimg).save(npth)
        ds.append({
            "record_id": rid,
            "episode_id": i // 10,
            "episode_step": i % 10,
            "previous_action": (i - 1) % 5 if i % 10 else -1,
            "image": str(ip.relative_to(path)),
            "next_image": str(npth.relative_to(path)),
            "context": ctx.tolist(),
            "next_context": nxt.tolist(),
            "executed_action": action,
            "reward": float(0.1 if i % 11 == 0 else -0.004),
            "success": float(i % 11 == 0),
            "world_valid": True,
            "motor_valid": False,
            "training_prohibited": bool(holdout),
        })
    if holdout:
        manifest = {
            "format": "continuous_brain.world_retention_holdout.v104",
            "domain": "foraging" if shift < 0.25 else "detour",
            "records": rows,
            "seed": 1,
            "exploration": .45,
            "parent_models_used": False,
            "child_model_used": False,
            "training_prohibited": True,
        }
        manifest["transition_digest"] = transition_digest(path)
        ds.write_manifest(manifest)
    else:
        ds.write_manifest({"format": "test.world.v104", "records": rows})
    return ds


def test_version_is_v104():
    assert __version__ == "0.10.4"


def test_v104_raw_holdout_collection_uses_no_parent(tmp_path):
    result = collect_world_holdout(tmp_path / "holdout", domain="foraging", samples=8, seed=123, exploration=.5)
    ds = NurseryDataset(tmp_path / "holdout")
    assert result["parent_models_used"] is False
    assert result["training_prohibited"] is True
    assert len(ds.records()) == 8
    assert all("parent_vision" not in row for row in ds.records())
    assert result["transition_digest"] == transition_digest(ds.directory)


def test_consolidation_changes_world_only_and_verifies_seal(tmp_path):
    train_a = _tiny_world_dataset(tmp_path / "train_a", shift=0.0)
    train_b = _tiny_world_dataset(tmp_path / "train_b", shift=0.5)
    hold_f = _tiny_world_dataset(tmp_path / "hold_f", shift=0.0, holdout=True)
    hold_d = _tiny_world_dataset(tmp_path / "hold_d", shift=0.5, holdout=True)
    # Correct the synthetic holdout domain labels explicitly.
    for ds, domain in ((hold_f, "foraging"), (hold_d, "detour")):
        m = ds.manifest(); m["domain"] = domain; ds.write_manifest(m)
    seal = tmp_path / "seal.json"
    seal_world_holdouts(
        foraging_dataset=hold_f.directory, detour_dataset=hold_d.directory,
        training_datasets=[train_a.directory, train_b.directory], output=seal,
    )

    state = tmp_path / "child"; state.mkdir()
    model = InfantBrain(InfantConfig.for_profile("proof"), vision_parent_dims={"vision.test": 8})
    save_checkpoint(
        state / "infant_training.pt", model, updates=11000, parent_id="vision.test",
        dataset_digest="source", metadata={
            "planning_cortex_version": "recurrent-latent-v1+conservative-bellman-q-v1+temporal-options-v1",
            "planning_updates": 4000,
            "planning_bellman_q_updates": 3000,
            "option_cortex_version": "latent-temporal-options-v1",
            "option_updates": 3000,
        },
    )
    before_model, _ = load_checkpoint(state / "infant_training.pt", "cpu")
    before = {k: v.detach().clone() for k, v in before_model.state_dict().items() if not k.startswith("world.")}
    result = train_world_consolidation(
        [train_a.directory, train_b.directory], state, holdout_seal=seal,
        updates=2, batch_per_domain=5, learning_rate=1e-3, seed=77, device="cpu",
    )
    assert result["world_only"] is True
    assert result["changed_non_world_tensors"] == 0
    assert result["balanced_domain_weighting"] is True
    assert result["action_balanced_within_domain"] is True
    after, saved = load_checkpoint(state / "infant_training.pt", "cpu")
    for key, value in before.items():
        assert torch.equal(value, after.state_dict()[key])
    assert saved["metadata"]["world_consolidation_version"] == "balanced-multidomain-action-rehearsal-v1"
    assert saved["metadata"]["option_cortex_version"] == "latent-temporal-options-v1"


def test_world_consolidation_evaluator_uses_sealed_parent_free_holdouts(tmp_path):
    train_a = _tiny_world_dataset(tmp_path / "train_a", shift=0.0)
    train_b = _tiny_world_dataset(tmp_path / "train_b", shift=0.5)
    hold_f = _tiny_world_dataset(tmp_path / "hold_f", shift=0.0, rows=15, holdout=True)
    hold_d = _tiny_world_dataset(tmp_path / "hold_d", shift=0.5, rows=15, holdout=True)
    for ds, domain in ((hold_f, "foraging"), (hold_d, "detour")):
        m = ds.manifest(); m["domain"] = domain; ds.write_manifest(m)
    seal = tmp_path / "seal.json"
    seal_world_holdouts(
        foraging_dataset=hold_f.directory, detour_dataset=hold_d.directory,
        training_datasets=[train_a.directory, train_b.directory], output=seal,
    )
    model = InfantBrain(InfantConfig.for_profile("proof"), vision_parent_dims={"vision.test": 8})
    checkpoint = tmp_path / "brain.pt"
    save_checkpoint(checkpoint, model, updates=11000, parent_id="vision.test", dataset_digest="x")
    result = evaluate_world_consolidation(
        checkpoint, foraging_holdout=hold_f.directory, detour_holdout=hold_d.directory,
        holdout_seal=seal, device="cpu",
    )
    assert result["parents_used"] is False
    assert set(result["domains"]) == {"foraging", "detour"}
    assert result["domains"]["foraging"]["records"] == 15
    assert result["domains"]["detour"]["records"] == 15
