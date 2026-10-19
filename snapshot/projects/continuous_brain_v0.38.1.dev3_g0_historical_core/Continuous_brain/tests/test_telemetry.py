"""DAV-37: local/offline-safe logger; a smoke run yields a complete dashboard-ready record."""

import json
import math

import pytest

from continuous_brain_snn.inputs import SyntheticDrive
from continuous_brain_snn.telemetry import REQUIRED_KEYS, JsonlLogger, WindowStats, collect


def test_smoke_run_record_is_complete(engine, tmp_path):
    w = WindowStats(engine)
    engine.run(500, input_fn=SyntheticDrive(engine), recorder=w)
    rec = collect(engine, w)
    assert all(k in rec for k in REQUIRED_KEYS)
    assert rec["health/nonfinite"] == 0
    numeric = [v for k, v in rec.items() if k in REQUIRED_KEYS and isinstance(v, float)]
    assert all(math.isfinite(v) for v in numeric)
    log = JsonlLogger(tmp_path / "t.jsonl")
    log.log(rec)
    assert json.loads((tmp_path / "t.jsonl").read_text())["step"] == 500


def test_rates_are_split_by_region_and_ei(engine):
    w = WindowStats(engine)
    engine.run(300, recorder=w)
    rec = collect(engine, w)
    assert rec["rate_hz/inh"] > 0 and rec["rate_hz/exc"] > 0
    assert rec["drive/g_inh_mean"] > 0 and rec["drive/g_exc_mean"] > 0


def test_empty_window_is_rejected(engine):
    with pytest.raises(ValueError):
        collect(engine, WindowStats(engine))


def test_wandb_is_optional():
    import continuous_brain_snn.telemetry as t  # imports without wandb installed
    assert hasattr(t, "WandbLogger")
