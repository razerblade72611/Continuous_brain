import json

import numpy as np
import pytest

from continuous_brain_protocols import ResultStatus, SeedFamily, SeedPurpose
from continuous_brain_eval.decision_rules import RULES, rule
from continuous_brain_eval.features import RawHistoryFeatures, select_features
from continuous_brain_eval.indicators import CANONICAL_IDS, IndicatorError, load_indicators
from continuous_brain_eval.results import ResultRecord, StatusViolation
from continuous_brain_eval.ridge import blocked_folds, blocked_split, classify, fit_ridge, probe, r2_score
from continuous_brain_eval.runs import RunExistsError, RunStore
from continuous_brain_eval.seeds import SeedOverlapError, SeedRegistry
from continuous_brain_eval.stats import (bootstrap_mean_ci, linear_trend, moving_block_bootstrap,
                                         wilcoxon_signed_rank_one_sided)
from continuous_brain_eval.streams import StreamConfig, SyntheticBodyStream
from continuous_brain_substrates import EncoderConfig


def test_ridge_recovers_a_linear_map_and_scores_noise_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 20))
    y = x[:, :3] @ np.array([[1.0], [-2.0], [0.5]]) + 0.1 * rng.normal(size=(600, 1))
    assert probe(x, y, gap=5).mean_r2 > 0.95
    noise = probe(x, rng.normal(size=(600, 1)), gap=5).mean_r2
    assert noise < 0.05


def test_near_constant_training_features_cannot_explode_test_predictions():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 5))
    x[:240, 4] = 1e-10 * rng.normal(size=240)       # almost silent while training, active at test time
    x[240:, 4] = rng.normal(size=60)
    y = x[:, :1]
    fit = fit_ridge(x[:240], y[:240])
    assert np.all(np.isfinite(fit.predict(x[240:]))) and np.max(np.abs(fit.predict(x[240:]))) < 100


def test_blocked_folds_and_split_leave_gaps():
    for tr, va in blocked_folds(100, 5, gap=4):
        assert not np.intersect1d(tr, va).size
        assert np.min(np.abs(tr[:, None] - va[None, :])) > 4
    tr, te = blocked_split(100, 0.2, gap=7)
    assert te[0] - tr[-1] == 8 and te[-1] == 99


def test_classify_and_r2():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 4, 400)
    x = np.eye(4)[labels] + 0.1 * rng.normal(size=(400, 4))
    assert classify(x, labels, 4)["accuracy"] > 0.95
    assert r2_score(np.arange(5.0), np.arange(5.0))[0] == pytest.approx(1.0)


def test_stats():
    ci = bootstrap_mean_ci(np.ones(10) * 2.0)
    assert ci.low == ci.high == 2.0
    assert bootstrap_mean_ci(np.random.default_rng(0).normal(1.0, 0.1, 50)).excludes_zero()
    assert moving_block_bootstrap(np.random.default_rng(0).normal(size=500), 20).within(0.5)
    # exact Wilcoxon: n=5 all positive -> p = 1/32
    assert wilcoxon_signed_rank_one_sided([1, 2, 3, 4, 5]) == pytest.approx(1 / 32)
    assert wilcoxon_signed_rank_one_sided([-1, -2, -3]) == pytest.approx(1.0)
    slope, se = linear_trend([0.0, 1.0, 2.0, 3.0])
    assert slope == pytest.approx(1.0) and se == pytest.approx(0.0, abs=1e-12)


def test_feature_subset_rule():
    assert np.array_equal(select_features(50, 2000), np.arange(50))
    a, b = select_features(10_000, 2000, 3), select_features(10_000, 2000, 3)
    assert a.size == 2000 and np.array_equal(a, b) and np.all(np.diff(a) > 0)


def test_raw_history_features_lag_order():
    from continuous_brain_protocols import ChannelKind, ChannelSpec, StreamSpec
    spec = StreamSpec((ChannelSpec("joint_angle", (1,), "rad", ChannelKind.SENSOR),), 0.01)
    r = RawHistoryFeatures(spec, lags=3)
    for v in (1.0, 2.0, 3.0, 4.0):
        r.push({"joint_angle": np.array([v])})
    assert r.read_features().tolist() == [4.0, 3.0, 2.0]


def test_stream_is_deterministic_physical_and_restorable():
    cfg = StreamConfig(encoder=EncoderConfig(retina_side=8, angle_cells=8, velocity_cells=8)).with_vision(
        cue_positions=4, cue_min_interval_s=0.2, cue_max_interval_s=0.3, rhythm_period_s=0.25)
    a, b = SyntheticBodyStream(cfg), SyntheticBodyStream(cfg)
    fa, fb = a.reset(7), b.reset(7)
    for _ in range(80):
        fa, fb = a.next(), b.next()
        assert a.spec.validate(fa.values) == []
    np.testing.assert_array_equal(fa["luminance"], fb["luminance"])
    assert set(a.last_targets) >= {"q", "qd", "cue_onset", "time_to_flash_s"}
    st = a.state_dict()
    x = [a.next()["joint_angle"] for _ in range(5)]
    c = SyntheticBodyStream(cfg)
    c.reset(0)
    c.load_state_dict(st)
    np.testing.assert_array_equal(x, [c.next()["joint_angle"] for _ in range(5)])
    assert np.any(np.abs(fa["joint_velocity"]) > 0)         # babbling actually moves the body


def test_seed_registry_refuses_spent_ranges_and_reports_the_known_overlap():
    reg = SeedRegistry.load()
    for name in ("v0.34.1-TEST", "anything"):                    # same name or not: compared by range
        with pytest.raises(SeedOverlapError):
            reg.check_fresh(SeedFamily(name, 80000, 80127, SeedPurpose.TEST))
    with pytest.raises(SeedOverlapError):
        reg.check_usable(SeedFamily("v0.34.1-TEST", 80000, 80127, SeedPurpose.HISTORICAL))   # spent
    with pytest.raises(SeedOverlapError, match="legacy-line"):
        reg.check_usable(SeedFamily("v0.37.x-TEST", 140000, 140127, SeedPurpose.HISTORICAL))
    with pytest.raises(SeedOverlapError):
        reg.check_seeds([80005])
    with pytest.raises(SeedOverlapError, match="not inside"):
        reg.check_seeds([12345678])
    # Completed engineering preflights are now frozen as spent; use still-unallocated reserve seeds.
    reg.check_seeds([905000, 909999])
    with pytest.raises(SeedOverlapError, match="v0.34.1"):
        reg.check_fresh(SeedFamily("new", 80100, 80200, SeedPurpose.TEST))
    reg.check_fresh(SeedFamily("fresh", 200000, 200127, SeedPurpose.TEST))
    anomalies = reg.anomalies()
    assert any("v0.31.1" in a and "v0.33.5" in a and a.startswith("known") for a in anomalies)
    assert not any(a.startswith("UNEXPECTED") for a in anomalies)


def test_engineering_reserve_allows_children_but_not_overlapping_siblings(tmp_path):
    import shutil
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "registry"
    shutil.copy(src / "seed_ranges.json", tmp_path / "seed_ranges.json")
    reg = SeedRegistry.load(tmp_path)
    a = SeedFamily("eng-a", 905000, 905099, SeedPurpose.ENGINEERING)
    b = SeedFamily("eng-b", 905050, 905149, SeedPurpose.ENGINEERING)
    c = SeedFamily("eng-c", 905100, 905199, SeedPurpose.ENGINEERING)
    reg.register(a, source="test")
    with pytest.raises(SeedOverlapError, match="eng-a"):
        reg.register(b, source="test")
    reg.register(c, source="test")
    assert not any("eng-a" in x and "eng-c" in x for x in reg.anomalies())


def _approved_registry(tmp_path, items=("G2",)):
    from continuous_brain_unified.prereg import PreregistrationRegistry
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "preregistrations.json").write_text(json.dumps({"preregistrations": [
        {"id": "OK", "title": "t", "status": "APPROVED", "approved_by": "David", "allowed_items": list(items)}]}))
    return PreregistrationRegistry.load(tmp_path)


def test_result_records_cannot_claim_pass_without_prereg_artifact_and_scientific_context(tmp_path):
    reg = _approved_registry(tmp_path)
    with pytest.raises(StatusViolation):
        ResultRecord("G2", ResultStatus.PASS).validate(reg)
    only_spec = ResultRecord("G2", ResultStatus.PASS, prereg_id="OK", provenance={"artifacts": {"spec.json": "x"}},
                             scientific_context=True)
    with pytest.raises(StatusViolation, match="primary artifact"):
        only_spec.validate(reg)
    draft = ResultRecord("G2", ResultStatus.PASS, prereg_id="E2-PREREG-v1", provenance={"artifacts": {"a": "b"}},
                         scientific_context=True)
    with pytest.raises(StatusViolation, match="DRAFT"):
        draft.validate()                                          # the registry, not a flag, decides
    other_item = ResultRecord("G1", ResultStatus.PASS, prereg_id="OK", provenance={"artifacts": {"a": "b"}},
                              scientific_context=True)
    with pytest.raises(StatusViolation, match="does not cover"):
        other_item.validate(reg)
    ok = ResultRecord("G2", ResultStatus.PASS, prereg_id="OK", provenance={"artifacts": {"a": "b"}})
    with pytest.raises(StatusViolation, match="non-scientific"):
        ok.validate(reg)
    ok.scientific_context = True
    ok.validate(reg)
    with pytest.raises(StatusViolation):
        ResultRecord("G2", ResultStatus.FAIL).to_json(reg)
    ResultRecord("B4", ResultStatus.ENGINEERING_ONLY).validate()


def test_gate_verdicts_become_status_only_at_finalization(tmp_path):
    reg = _approved_registry(tmp_path / "r")
    store = RunStore(tmp_path / "runs")
    rec = ResultRecord("G2", ResultStatus.UNVERIFIED, prereg_id="OK", metrics={"verdict": "PASS"},
                       scientific_context=True)
    run = store.create({"item": "G2", "x": 1})
    with pytest.raises(StatusViolation, match="primary artifact"):
        store.finalize(run, rec, reg)                              # only spec.json present
    run2 = store.create({"item": "G2", "x": 2})
    run2.artifact("cells.json").write_text("{}")
    rec2 = ResultRecord("G2", ResultStatus.UNVERIFIED, prereg_id="OK", metrics={"verdict": "FAIL"},
                        scientific_context=True)
    store.finalize(run2, rec2, reg)
    assert rec2.status == ResultStatus.FAIL and store.entries()[-1]["status"] == "FAIL"
    with pytest.raises(RunExistsError, match="finalized"):
        store.finalize(run2, rec2, reg)


def test_run_store_is_content_addressed_and_manifested(tmp_path):
    store = RunStore(tmp_path)
    run = store.create({"item": "B5", "seed": 900001})
    run.artifact("features.npy").write_bytes(b"123")
    with pytest.raises(RunExistsError):
        store.create({"seed": 900001, "item": "B5"})
    store.create({"item": "B5", "seed": 900001, "replicate": 2})
    manifest = store.finalize(run, ResultRecord("B5", ResultStatus.ENGINEERING_ONLY))
    assert store.verify(run.run_id)
    (run.path / "extra.txt").write_text("added later")
    assert not store.verify(run.run_id)
    (run.path / "extra.txt").unlink()
    assert "features.npy" in json.loads(manifest.read_text())["files"]
    assert store.entries()[0]["status"] == "ENGINEERING_ONLY"
    import stat
    assert not (run.path / "result.json").stat().st_mode & stat.S_IWUSR      # read-only (root may still write)
    (run.path / "result.json").chmod(0o644)
    (run.path / "result.json").write_text("tampered")
    assert not store.verify(run.run_id)


def test_decision_rules_are_versioned_and_hashed():
    assert RULES["DAV-32-v1"].status == "CURRENT" and RULES["A1-proposed"].status == "PROPOSED"
    rec = rule("A1-proposed").record()
    assert len(rec["implementation_sha256"]) == 64
    out = rule("E2-primary-v1").fn([0.1, 0.08, 0.12, 0.09, 0.11, 0.1, 0.07, 0.13, 0.1, 0.1])
    assert out["criterion_met"] and out["wilcoxon_one_sided_p"] < 0.01
    assert not rule("E2-primary-v1").fn([0.01] * 10)["criterion_met"]
    with pytest.raises(KeyError):
        rule("nope")


def test_indicator_registry_has_14_rows_none_admitted(tmp_path):
    rows, chain = load_indicators()
    assert tuple(r.id for r in rows) == CANONICAL_IDS
    assert not any(r.admitted for r in rows)
    assert len(chain) == 6
    import shutil
    from pathlib import Path
    reg = Path(__file__).resolve().parents[1] / "registry"
    for f in ("indicators.json", "preregistrations.json"):
        shutil.copy(reg / f, tmp_path / f)
    data = json.loads((tmp_path / "indicators.json").read_text())
    data["indicators"][10]["admitted"] = True                    # try to turn AST-1 green by editing JSON
    (tmp_path / "indicators.json").write_text(json.dumps(data))
    with pytest.raises(IndicatorError, match="AST-1"):
        load_indicators(tmp_path)


def test_a_test_family_is_usable_exactly_once(tmp_path):
    import shutil
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "registry"
    shutil.copy(src / "seed_ranges.json", tmp_path / "seed_ranges.json")
    shutil.copy(src / "preregistrations.json", tmp_path / "preregistrations.json")
    reg = SeedRegistry.load(tmp_path)
    fam = SeedFamily("G1-fresh-TEST", 500000, 500127, SeedPurpose.TEST, sealed=True)
    reg.register(fam, source="test")
    reg.check_usable(fam)
    reg.spend(fam, source="test")
    with pytest.raises(SeedOverlapError, match="spent"):
        SeedRegistry.load(tmp_path).check_usable(fam)
