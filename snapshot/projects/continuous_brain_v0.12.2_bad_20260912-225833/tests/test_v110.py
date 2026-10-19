from pathlib import Path

import torch

from continuous_brain import __version__
from continuous_brain.curriculum_v110 import RandomizedDetourWorld, TwoBarrierWorld
from continuous_brain.growth_v110 import is_growth_parameter
from continuous_brain.infant import InfantBrain, InfantConfig
from continuous_brain.parented_train import save_checkpoint, load_checkpoint
from continuous_brain.whole_brain_v110 import expand_capacity, _set_route_trainable


def test_version_is_v110_or_compatible_patch():
    assert __version__.startswith(("0.11.", "0.12."))


def test_balanced_growth_capacity_and_zero_identity():
    torch.manual_seed(1)
    base = InfantBrain(InfantConfig())
    grown = InfantBrain(InfantConfig(growth_enabled=True, growth_profile="balanced-v110"))
    # Copy every legacy tensor; only growth keys remain new.
    missing, unexpected = grown.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(is_growth_parameter(k) for k in missing)
    base.eval(); grown.eval()
    x = torch.rand(1,3,96,96)
    assert torch.equal(base.vision(x), grown.vision(x))
    assert 50_000_000 <= grown.parameter_count <= 56_000_000
    assert sum(p.numel() for n,p in grown.named_parameters() if is_growth_parameter(n)) > 40_000_000


def test_expand_capacity_preserves_all_audited_functions_and_legacy_tensors(tmp_path):
    src = tmp_path / "source"; src.mkdir()
    model = InfantBrain(InfantConfig(), vision_parent_dims={"vision.test": 8})
    save_checkpoint(src/"infant_training.pt", model, updates=14000,
                    parent_id="vision.test", dataset_digest="x",
                    metadata={"planning_updates":4000,"option_updates":3000})
    result = expand_capacity(src, tmp_path/"grown", device="cpu")
    assert result["legacy_tensor_mismatches"] == 0
    assert result["equivalence"]["exact_function_preservation"] is True
    assert result["equivalence"]["max_abs_difference"] == 0.0
    assert result["expanded_trainable_parameters"] > 50_000_000
    grown, saved = load_checkpoint(tmp_path/"grown"/"infant_training.pt", "cpu")
    assert grown.cfg.growth_enabled is True
    assert saved["metadata"]["growth_version"] == "zero-residual-whole-brain-v1"


def test_route_masks_select_growth_only():
    model = InfantBrain(InfantConfig(growth_enabled=True, growth_profile="balanced-v110"))
    for route in ("perception","language","embodiment","world","planning","options"):
        names = _set_route_trainable(model, route)
        assert names
        assert all(is_growth_parameter(n) for n in names)
        assert all((not p.requires_grad) or is_growth_parameter(n) for n,p in model.named_parameters())


def test_generalized_worlds_change_layout_across_resets():
    env = RandomizedDetourWorld(seed=123)
    layouts=[]
    for _ in range(4):
        env.reset(); layouts.append((round(env.wall_x,3),round(env.gap_center_y,3)))
    assert len(set(layouts)) > 1
    two = TwoBarrierWorld(seed=456)
    two.reset()
    assert two.wall_x < two.wall_x2
    assert two.gap_center_y > 0 and two.gap_center_y2 < 0
