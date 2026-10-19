import math

import numpy as np
import pytest

from continuous_brain_protocols import (ChannelKind, ChannelSpec, InterventionKind, InterventionSpec, LesionSpec,
                                        ResultStatus, SeedFamily, SeedPurpose, StreamSpec, canonical_json,
                                        read_only, scientific_claim_allowed, sha256_json, state_digest)


def test_semantic_names_and_units_are_rejected():
    with pytest.raises(ValueError, match="semantic"):
        ChannelSpec("object_label", (1,), "1", ChannelKind.SENSOR)
    with pytest.raises(ValueError, match="not physical"):
        ChannelSpec("thing", (1,), "category", ChannelKind.SENSOR)
    with pytest.raises(ValueError, match="snake_case"):
        ChannelSpec("Joint Angle", (1,), "rad", ChannelKind.SENSOR)
    ChannelSpec("joint_angle", (2,), "rad", ChannelKind.SENSOR, -1.0, 1.0)


def test_stream_spec_validation():
    spec = StreamSpec((ChannelSpec("joint_angle", (2,), "rad", ChannelKind.SENSOR, -1.0, 1.0),), 0.01)
    assert spec.validate({"joint_angle": np.zeros(2)}) == []
    assert any("shape" in p for p in spec.validate({"joint_angle": np.zeros(3)}))
    assert any("above" in p for p in spec.validate({"joint_angle": np.array([0.0, 2.0])}))
    assert any("non-finite" in p for p in spec.validate({"joint_angle": np.array([np.nan, 0.0])}))
    assert any("undeclared" in p for p in spec.validate({"joint_angle": np.zeros(2), "extra": np.zeros(1)}))
    with pytest.raises(ValueError):
        StreamSpec((ChannelSpec("a", (1,), "rad", ChannelKind.SENSOR),) * 2, 0.01)


def test_canonical_json_is_stable_and_refuses_nan():
    assert canonical_json({"b": 1, "a": [1.0, 2]}) == '{"a":[1.0,2],"b":1}'
    assert sha256_json({"x": 1, "y": 2}) == sha256_json({"y": 2, "x": 1})
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


def test_state_digest_detects_single_bit_changes():
    import torch
    a = {"w": torch.zeros(3), "n": np.arange(4), "k": [1, "x"]}
    b = {"w": torch.zeros(3), "n": np.arange(4), "k": [1, "x"]}
    assert state_digest(a) == state_digest(b)
    b["w"][1] = 1e-30
    assert state_digest(a) != state_digest(b)


def test_seed_family_overlap():
    a = SeedFamily("a", 0, 10, SeedPurpose.DEV)
    assert a.overlaps(SeedFamily("b", 10, 20, SeedPurpose.DEV))
    assert not a.overlaps(SeedFamily("c", 11, 20, SeedPurpose.DEV))
    with pytest.raises(ValueError):
        SeedFamily("bad", 5, 4, SeedPurpose.DEV)


def test_intervention_and_lesion_specs_validate():
    with pytest.raises(ValueError):
        InterventionSpec(InterventionKind.DONOR, "x.y", 0)
    with pytest.raises(ValueError):
        InterventionSpec(InterventionKind.ERASE, "x.y", 5, 5)
    s = InterventionSpec(InterventionKind.ERASE, "x.y", 2, 4)
    assert not s.active(1) and s.active(2) and s.active(3) and not s.active(4)
    with pytest.raises(ValueError):
        LesionSpec("synapses:cortex", 0.0, 0)


def test_scientific_claims_need_prereg_and_artifact():
    assert scientific_claim_allowed(ResultStatus.ENGINEERING_ONLY, None, None)
    assert not scientific_claim_allowed(ResultStatus.PASS, None, {"a": "b"})
    assert not scientific_claim_allowed(ResultStatus.FAIL, "X", {})
    assert scientific_claim_allowed(ResultStatus.PASS, "X", {"a": "b"})


def test_read_only_arrays():
    r = read_only(np.zeros(2))
    with pytest.raises(ValueError):
        r[0] = 1.0
