import sys
import types

import torch

from continuous_brain import __version__
from continuous_brain.device_v111 import resolve_device, checkpoint_map_location, is_directml_spec


def test_version_is_v111():
    assert __version__ == "0.11.1"


def test_cpu_resolution_is_unchanged():
    d = resolve_device("cpu")
    assert d == torch.device("cpu")
    assert checkpoint_map_location(d) == torch.device("cpu")


def test_directml_aliases_are_recognized():
    assert is_directml_spec("dml")
    assert is_directml_spec("dml:0")
    assert is_directml_spec("directml")
    assert is_directml_spec("privateuseone:0")
    assert not is_directml_spec("cuda:0")


def test_directml_resolver_uses_torch_directml_device(monkeypatch):
    fake = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        device=lambda idx=0: torch.device(f"privateuseone:{idx}"),
        device_name=lambda idx=0: f"Fake DML {idx}",
    )
    monkeypatch.setitem(sys.modules, "torch_directml", fake)
    assert resolve_device("dml") == torch.device("privateuseone:0")
    assert resolve_device("dml:1") == torch.device("privateuseone:1")
    assert checkpoint_map_location(resolve_device("dml:1")) == torch.device("cpu")


def test_directml_out_of_range_fails(monkeypatch):
    fake = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        device=lambda idx=0: torch.device(f"privateuseone:{idx}"),
    )
    monkeypatch.setitem(sys.modules, "torch_directml", fake)
    try:
        resolve_device("dml:1")
    except RuntimeError as exc:
        assert "only 1 device" in str(exc)
    else:
        raise AssertionError("expected DirectML index failure")
