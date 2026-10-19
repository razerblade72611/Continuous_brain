import torch

from continuous_brain import __version__
from continuous_brain.competitive_infant_brain import CompetitiveInfantBrain
from continuous_brain.infant import InfantBrain, InfantConfig
from continuous_brain.dml_ops_v121 import dml_identity_clone


def test_version_v122():
    assert __version__ == "0.12.2"


def test_identity_clone_cpu_is_noop_reference():
    x = torch.randn(2, 3, requires_grad=True)
    y = dml_identity_clone(x)
    assert y is x


def test_minimal_competitive_path_preserves_train_outputs_exactly():
    torch.manual_seed(12201)
    cfg = InfantConfig()
    base = InfantBrain(cfg).eval()
    model = CompetitiveInfantBrain.grow_from(base, use_shared=True).eval()
    x = dict(
        images=torch.randn(1, 3, cfg.image_size, cfg.image_size),
        context=torch.randn(1, cfg.context_dim),
        memory=torch.randn(1, cfg.memory_dim),
        prev_action=torch.randint(0, cfg.action_dim + 1, (1,)),
        hidden=torch.randn(1, cfg.nucleus_dim),
    )
    with torch.no_grad():
        full = model.step(**x, use_competition=True, compute_aux=True)
        mini = model.step(**x, use_competition=True, compute_aux=False)
    for key in ("hidden", "logits", "workspace_for_world"):
        assert torch.equal(full[key], mini[key]), key
    assert mini["value"] is None
    assert mini["router"] is None
