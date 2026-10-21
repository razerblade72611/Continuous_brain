"""Resume logic of gate_v2.run (no model needed): recorded rows kept verbatim, only remaining seeds run."""
import pytest

import being.gate_v2 as g


@pytest.fixture
def stub(monkeypatch):
    calls = []
    monkeypatch.setattr(g, "pool_ids", lambda tok: list(range(80)))
    monkeypatch.setattr(g, "load_sentences", lambda name: ["x"])
    monkeypatch.setattr(g, "mean_loss", lambda m, t, s: 1.0)
    monkeypatch.setattr(g, "backbone_hash", lambda m: "H")
    monkeypatch.setattr(g, "summarize", lambda rows, cfg: {"checks": {"ok": True}, "backbone_unattested_seeds": []})

    def fake_seed(model, tok, cfg, s, *a):
        calls.append(s)
        return {"seed": s, "backbone_sha256": "H", "backbone_sha256_end": "H"}
    monkeypatch.setattr(g, "run_seed", fake_seed)
    return calls


def test_resume_runs_only_remaining(stub, tmp_path):
    prior = [{"seed": 1000, "backbone_sha256": "H", "backbone_sha256_end": "H", "marker": "kept"}]
    res = g.run(None, None, g.V2Config(), [1000, 1001, 1002], tmp_path, None, prior_rows=prior)
    assert stub == [1001, 1002]
    assert res["per_seed"][0]["marker"] == "kept" and res["complete"]
    assert res["summary"]["checks"]["backbone_unchanged"]


def test_max_new_stops_without_summary(stub, tmp_path):
    res = g.run(None, None, g.V2Config(), [1000, 1001, 1002], tmp_path, None, max_new=2)
    assert stub == [1000, 1001] and not res["complete"] and "summary" not in res


def test_prior_must_be_prefix(stub, tmp_path):
    with pytest.raises(ValueError):
        g.run(None, None, g.V2Config(), [1000, 1001], tmp_path, None, prior_rows=[{"seed": 1001}])


def test_backbone_mismatch_across_segments_fails(stub, tmp_path):
    prior = [{"seed": 1000, "backbone_sha256": "OTHER", "backbone_sha256_end": "OTHER"}]
    res = g.run(None, None, g.V2Config(), [1000, 1001], tmp_path, None, prior_rows=prior)
    assert not res["summary"]["checks"]["backbone_unchanged"]
    res = g.run(None, None, g.V2Config(), [1000], tmp_path, None,
                prior_rows=[{"seed": 1000, "backbone_sha256": "H", "backbone_sha256_end": "H"}], prior_segments_ok=False)
    assert not res["summary"]["checks"]["backbone_unchanged"]
