import torch

from continuous_brain import __version__
from continuous_brain.dml_transformer_v113 import directml_causal_mask, directml_transformer_safe_path
from continuous_brain.infant import InfantBrain, InfantConfig


def test_version_is_v113():
    assert __version__ == "0.11.3"


def test_directml_causal_mask_is_float_finite_and_correct():
    ref = torch.randn(2, 4, 8, dtype=torch.float32)
    mask = directml_causal_mask(4, ref)
    assert mask.dtype == torch.float32
    assert mask.shape == (4, 4)
    assert torch.isfinite(mask).all()
    assert torch.equal(torch.diag(mask), torch.zeros(4))
    assert float(mask[0, 1]) < -1000.0
    assert float(mask[3, 0]) == 0.0


def test_safe_path_is_noop_on_cpu_and_restores_fastpath():
    backend = getattr(torch.backends, "mha", None)
    if backend is None or not hasattr(backend, "get_fastpath_enabled"):
        return
    before = backend.get_fastpath_enabled()
    with directml_transformer_safe_path(torch.zeros(1)):
        assert backend.get_fastpath_enabled() == before
    assert backend.get_fastpath_enabled() == before


def test_language_cpu_path_and_state_dict_schema_remain_stable():
    torch.manual_seed(113)
    cfg = InfantConfig.for_profile("proof")
    model = InfantBrain(cfg).eval()
    keys_before = tuple(model.language.state_dict())
    workspace = torch.randn(2, cfg.bus_dim)
    q = torch.randint(3, 20, (2, 5), dtype=torch.long)
    a = torch.randint(3, 20, (2, 4), dtype=torch.long)
    with torch.no_grad():
        out = model.language(workspace, q, a)
    assert out.shape == (2, 4, 259)
    assert tuple(model.language.state_dict()) == keys_before
    assert not any("dml_transformer" in k for k in keys_before)


def test_mask_helper_introduces_no_parameters():
    cfg = InfantConfig.for_profile("proof")
    model = InfantBrain(cfg)
    assert not any("causal_mask" in k or "dml_transformer" in k for k in model.state_dict())
