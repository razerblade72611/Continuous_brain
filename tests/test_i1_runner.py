"""I1 runner guards (v0.38.1.dev7): science is locked until approval; plans target the right ticks."""

import json

import pytest

from continuous_brain_unified.integration import runner
from continuous_brain_unified.prereg import PreregistrationError


def test_science_is_locked_until_david_approves(tmp_path):
    with pytest.raises((PreregistrationError, runner.I1Refused)):
        runner.run("science", legacy_root=str(tmp_path), historical_core_archive=str(tmp_path / "none.zip"),
                   label="never", confirm_open=runner.PREREG_ID)
    with pytest.raises(runner.I1Refused):
        runner.run("science", legacy_root=str(tmp_path), historical_core_archive=str(tmp_path / "none.zip"),
                   label="never", confirm_open="wrong")
    assert not (runner.REPO / "runs" / "i1" / "never").exists()


def test_dev_mode_refuses_scientific_families(tmp_path):
    with pytest.raises((runner.I1Refused, KeyError)):
        runner.run("dev", legacy_root=str(tmp_path), historical_core_archive="x", label="never",
                   family="g0-v0361-dev", start=307000, n=2)


def test_dev_slices_are_confined_to_the_named_family():
    from continuous_brain_eval.seeds import SeedRegistry
    reg = SeedRegistry.load()
    prereg = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    fam, sids = runner._dev_sids(reg, prereg, "cloud-dev-claude-eng", 906680, 4)
    assert sids == [906680, 906681, 906682, 906683] and fam.name == "cloud-dev-claude-eng"
    with pytest.raises(runner.I1Refused, match="not inside"):
        runner._dev_sids(reg, prereg, "cloud-dev-claude-eng", 906998, 4)        # runs past the family
    with pytest.raises(runner.I1Refused, match="touch registered family"):
        runner._dev_sids(reg, prereg, "v0.38-ENGINEERING", 900000, 2)           # a reserve touching spent G0 seeds
    shadow = json.loads(json.dumps(prereg))
    shadow["families"]["engineering_preflight_proposed"].update(start=906700, stop=906701)
    with pytest.raises(runner.I1Refused, match="preregistered"):
        runner._dev_sids(reg, shadow, "cloud-dev-claude-eng", 906700, 2)
    shadow["families"]["engineering_preflight_proposed"].update(start=906700 - 20000, stop=906701 - 20000)
    with pytest.raises(runner.I1Refused, match="preregistered"):                # derived-id cross collision
        runner._dev_sids(reg, shadow, "cloud-dev-claude-eng", 906700, 2)


def test_design_digest_binds_the_approval():
    p = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    d = runner.design_digest(p)
    assert d == runner.design_digest(json.loads(json.dumps(p)))
    p2 = json.loads(json.dumps(p))
    p2["status"], p2["approved_by"] = "APPROVED", "David"                     # approval fields are outside
    assert runner.design_digest(p2) == d
    p2["decision_rule"]["primary"]["C2_first_goal_completes"]["threshold"] = "0.5"
    assert runner.design_digest(p2) != d


def test_preregistration_is_proposed_and_cites_frozen_thresholds():
    p = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    assert p["status"] == "PROPOSED" and p["approved_by"] is None and "preflight_manifest_sha256" not in p
    assert "approved_design_sha256" not in p
    assert len(p["decision_rule"]["primary"]) == 11          # C1-C10 + gate
    assert "AUTHORITY_SKILLS / AUTHORITY_GOAL / AUTHORITY_CURRICULUM" in p["design"]["arms"]
    assert p["families"]["engineering_preflight_proposed"]["start"] == 907100
    from continuous_brain_eval.seeds import SeedRegistry
    from continuous_brain_protocols import SeedFamily, SeedPurpose
    reg = SeedRegistry.load()
    for key in ("scientific_dev_proposed", "engineering_preflight_proposed"):
        f = p["families"][key]
        reg.check_fresh(SeedFamily(f["name"], f["start"], f["stop"], SeedPurpose(f["purpose"])))


def test_manifest_is_deterministic_and_covers_all_code():
    a, b = runner.manifest(2), runner.manifest(2)
    assert a["manifest_sha256"] == b["manifest_sha256"] and a["legacy_pins_count"] == 28
    code = a["code"]
    for rel in ("continuous_brain_unified/integration/route.py", "continuous_brain_unified/composition/route.py",
                "continuous_brain_unified/integration/evaluate.py", "continuous_brain_unified/runtime.py",
                "continuous_brain_eval/seeds.py", "scripts/i1_run.py", "legacy/historical_core_v0122.json"):
        assert rel in code                                    # keyed by repo path: no name collisions
    assert runner.manifest(3)["manifest_sha256"] != a["manifest_sha256"]      # thread count is part of it


def test_plans_target_the_recorded_ticks():
    normal = {
        10: {"record": {"marks": {"imprint": 40, "before_second": 120}, "birth_state_b64": "QUE=",
                        "first_completed": True, "second_goal": {"__type__": "G", "primary": 2}}},
        11: {"record": {"marks": {"imprint": 41}, "birth_state_b64": "QkI=", "first_completed": False}},
    }
    plans = runner._plans(normal)
    assert plans["ERASED"]["10"][0] == {"kind": "erase", "target": "autobiography.state", "start_tick": 40,
                                        "end_tick": 41, "why": "erase autobiography after birth imprint"}
    assert plans["DONOR"]["10"][0]["payload_b64"] == "QkI=" and plans["DONOR"]["11"][0]["payload_b64"] == "QUE="
    rb = plans["ROLLBACK"]["10"]
    assert rb["interventions"][0]["start_tick"] == 120 and rb["interventions"][0]["payload_b64"] == "QUE="
    assert rb["compare"]["normal_second_goal"] == {"primary": 2}
    assert "11" not in plans["ROLLBACK"]                     # only first-goal completers are rolled back
    for arm, target in runner.AUTHORITY_TARGETS.items():
        assert plans[arm] == {"10": [{"kind": "erase", "target": target, "start_tick": 120, "end_tick": 121,
                                      "why": f"authority control: erase {target} before the second-goal decision"}]}
