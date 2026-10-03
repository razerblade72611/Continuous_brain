"""I1 runner guards (v0.38.1.dev7): science is locked until approval; plans target the right ticks.

Lifecycle: the repository's I1 preregistration moves PROPOSED -> (preflight spent) -> APPROVED ->
(scientific family spent). Nothing in this file asserts one particular state. The repository's current
state is checked against the lifecycle invariants (i1_lifecycle_problems), every legal state is
simulated on copies and must pass them, and illegal states must fail them, so the suite is green in
every legal state, before and after approval."""

import json
import re

import pytest

from continuous_brain_unified.integration import runner
from continuous_brain_unified.prereg import PreregistrationError, PreregistrationRegistry

APPROVAL_FIELDS = ("approved_utc", "preflight_manifest_sha256", "approved_design_sha256", "preflight_run")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _exact(reg, f):
    from continuous_brain_eval.seeds import _key
    from continuous_brain_protocols import SeedFamily, SeedPurpose
    fam = SeedFamily(f["name"], int(f["start"]), int(f["stop"]), SeedPurpose(f["purpose"]))
    return fam, [r for r in reg.ranges if _key(r.family) == _key(fam)]


def i1_lifecycle_problems(prereg: dict, central: dict, reg) -> list[str]:
    """Every invariant the I1 preregistration must satisfy in whatever legal state it is in."""
    out = []
    st = prereg.get("status")
    entries = [e for e in central["preregistrations"] if e.get("id") == runner.PREREG_ID]
    if len(entries) != 1:
        return [f"central registry has {len(entries)} {runner.PREREG_ID} entries"]
    c = entries[0]
    if st not in ("PROPOSED", "APPROVED", "FROZEN"):
        out.append(f"illegal status {st!r}")
    if c.get("status") != st:
        out.append(f"central status {c.get('status')!r} != preregistration status {st!r}")
    pf, pf_rows = _exact(reg, prereg["families"]["engineering_preflight_proposed"])
    sci, sci_rows = _exact(reg, prereg["families"]["scientific_dev_proposed"])
    for name, fam, rows in (("preflight", pf, pf_rows), ("scientific", sci, sci_rows)):
        if not rows:                                    # unregistered: must still be fresh
            try:
                reg.check_fresh(fam)
            except Exception as e:
                out.append(f"{name} family is neither registered exactly nor fresh: {e}")
        elif not all(r.family.spent for r in rows):      # the runner registers and spends in one step
            out.append(f"{name} family is registered but not spent")
    if st == "PROPOSED":
        if prereg.get("approved_by") is not None or any(k in prereg for k in APPROVAL_FIELDS):
            out.append("PROPOSED preregistration carries approval fields")
        if c.get("approved_by"):
            out.append("PROPOSED central entry carries approved_by")
        if sci_rows:
            out.append("the scientific family was opened before approval")
    else:
        if not (prereg.get("approved_by") or "").strip() or not prereg.get("approved_utc"):
            out.append("approval without approver or time")
        if prereg.get("approved_design_sha256") != runner.design_digest(prereg):
            out.append("the approved design digest does not match the design as it stands")
        if not HEX64.match(str(prereg.get("preflight_manifest_sha256", ""))):
            out.append("approval without a preflight manifest SHA-256")
        if not c.get("approved_by"):
            out.append("APPROVED central entry without approved_by")
        if not pf_rows:
            out.append("approved without a spent preflight family in this registry")
    return out


def _load_central(root=runner.REPO):
    return json.loads((root / "registry" / "preregistrations.json").read_text())


def _as_proposed(prereg: dict, central: dict) -> tuple[dict, dict]:
    """The PROPOSED form of the repository's preregistration (approval fields removed), for simulations."""
    p = json.loads(json.dumps(prereg))
    p["status"], p["approved_by"] = "PROPOSED", None
    for k in APPROVAL_FIELDS:
        p.pop(k, None)
    c = json.loads(json.dumps(central))
    for e in c["preregistrations"]:
        if e["id"] == runner.PREREG_ID:
            e["status"] = "PROPOSED"
            e.pop("approved_by", None)
            e.pop("approved_utc", None)
    return p, c


def test_science_is_locked_until_david_approves(tmp_path):
    proposed = json.loads((runner.REPO / runner.PREREG_FILE).read_text())["status"] == "PROPOSED"
    with pytest.raises(PreregistrationError if proposed else runner.I1Refused):
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


def _registry_with_cloud_dev():
    """The live registry, plus cloud-dev-claude-eng in memory if this checkout's local ledger lacks it
    (the ledger is per machine and not in git). Nothing is written."""
    from continuous_brain_eval.seeds import RegisteredRange, SeedRegistry
    from continuous_brain_protocols import SeedFamily, SeedPurpose
    reg = SeedRegistry.load()
    if not any(r.family.name == "cloud-dev-claude-eng" for r in reg.ranges):
        reg.ranges.append(RegisteredRange(SeedFamily("cloud-dev-claude-eng", 906000, 906999, SeedPurpose.ENGINEERING),
                                          "test", "in-memory only"))
    return reg


def test_dev_slices_are_confined_to_the_named_family():
    reg = _registry_with_cloud_dev()
    prereg = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    fam, sids = runner._dev_sids(reg, prereg, "cloud-dev-claude-eng", 906680, 4)
    assert sids == [906680, 906681, 906682, 906683] and fam.name == "cloud-dev-claude-eng"
    with pytest.raises(runner.I1Refused, match="not inside"):
        runner._dev_sids(reg, prereg, "cloud-dev-claude-eng", 906998, 4)        # runs past the family
    with pytest.raises(runner.I1Refused, match="touch ids of registered family"):
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


def test_preregistration_cites_frozen_thresholds_and_fixed_families():
    p = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    assert len(p["decision_rule"]["primary"]) == 11          # C1-C10 + gate
    assert "AUTHORITY_SKILLS / AUTHORITY_GOAL / AUTHORITY_CURRICULUM" in p["design"]["arms"]
    fam = p["families"]
    assert (fam["engineering_preflight_proposed"]["start"], fam["engineering_preflight_proposed"]["stop"]) == (907100, 907163)
    assert (fam["scientific_dev_proposed"]["start"], fam["scientific_dev_proposed"]["stop"]) == (328000, 328127)
    for name, rule in p["decision_rule"]["primary"].items():
        if name != "gate":
            assert "threshold" in rule and "denominator" in rule


def test_repository_i1_state_satisfies_the_lifecycle_invariants():
    """Holds in every legal state: PROPOSED, preflight spent, APPROVED, scientific family spent."""
    from continuous_brain_eval.seeds import SeedRegistry
    p = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    assert i1_lifecycle_problems(p, _load_central(), SeedRegistry.load()) == []


def test_every_legal_lifecycle_state_passes_and_illegal_ones_fail(tmp_path):
    from continuous_brain_eval.seeds import RegisteredRange, SeedRegistry
    from continuous_brain_protocols import SeedFamily, SeedPurpose
    p0, c0 = _as_proposed(json.loads((runner.REPO / runner.PREREG_FILE).read_text()), _load_central())

    def registry(*spent_keys):
        reg = SeedRegistry.load()
        keep = {(p0["families"][k]["name"]) for k in ("engineering_preflight_proposed", "scientific_dev_proposed")}
        reg.ranges = [r for r in reg.ranges if r.family.name not in keep]      # start from "never opened"
        for k in spent_keys:
            f = p0["families"][k]
            reg.ranges.append(RegisteredRange(SeedFamily(f["name"], f["start"], f["stop"], SeedPurpose(f["purpose"]),
                                                         False, True), "test", "spent"))
        return reg

    # legal: PROPOSED, never opened / PROPOSED, preflight spent
    assert i1_lifecycle_problems(p0, c0, registry()) == []
    assert i1_lifecycle_problems(p0, c0, registry("engineering_preflight_proposed")) == []
    # legal: APPROVED through the real approval function / APPROVED with the scientific family spent
    root = tmp_path / "root"
    (root / "registry").mkdir(parents=True)
    (root / runner.PREREG_FILE).write_text(json.dumps(p0, indent=2))
    (root / "registry" / "preregistrations.json").write_text(json.dumps(c0, indent=2))
    runner.approve(preflight_dir=str(_fake_preflight(tmp_path, p0)), approver="David", confirm=runner.PREREG_ID, root=root)
    pa, ca = json.loads((root / runner.PREREG_FILE).read_text()), _load_central(root)
    PreregistrationRegistry.load(root / "registry").require(runner.PREREG_ID, item="I1")
    assert i1_lifecycle_problems(pa, ca, registry("engineering_preflight_proposed")) == []
    assert i1_lifecycle_problems(pa, ca, registry("engineering_preflight_proposed", "scientific_dev_proposed")) == []
    # illegal states are caught
    bad = json.loads(json.dumps(pa)); bad["decision_rule"]["primary"]["C2_first_goal_completes"]["threshold"] = "0.5"
    assert any("design digest" in x for x in i1_lifecycle_problems(bad, ca, registry("engineering_preflight_proposed")))
    assert any("central status" in x for x in i1_lifecycle_problems(pa, c0, registry("engineering_preflight_proposed")))
    assert any("without a spent preflight" in x for x in i1_lifecycle_problems(pa, ca, registry()))
    assert any("before approval" in x for x in i1_lifecycle_problems(p0, c0, registry("scientific_dev_proposed")))
    leaky = json.loads(json.dumps(p0)); leaky["preflight_manifest_sha256"] = "0" * 64
    assert any("approval fields" in x for x in i1_lifecycle_problems(leaky, c0, registry()))


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
    assert plans["DONOR"]["10"][0]["donor_sid"] == 11 and plans["DONOR"]["11"][0]["donor_sid"] == 10   # fixed by ids
    rb = plans["ROLLBACK"]["10"]
    assert rb["interventions"][0]["start_tick"] == 120 and rb["interventions"][0]["payload_b64"] == "QUE="
    assert rb["compare"]["normal_second_goal"] == {"primary": 2}
    assert "11" not in plans["ROLLBACK"]                     # only first-goal completers are rolled back
    for arm, target in runner.AUTHORITY_TARGETS.items():
        assert plans[arm] == {"10": [{"kind": "erase", "target": target, "start_tick": 120, "end_tick": 121,
                                      "why": f"authority control: erase {target} before the second-goal decision"}]}


def test_i1_families_collide_with_nothing_at_any_offset_pair():
    reg = _registry_with_cloud_dev()
    p = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    offs = (0, *runner.DERIVED_OFFSETS)
    for key in ("scientific_dev_proposed", "engineering_preflight_proposed"):
        f = p["families"][key]
        for r in reg.ranges:
            g = r.family
            if (g.name, g.start, g.stop) == (f["name"], f["start"], f["stop"]):
                continue                         # the family's own registration (once it is opened)
            container = r.reserve and g.start <= f["start"] and f["stop"] <= g.stop
            for oi in offs:
                for og in offs:
                    if container and oi == og:
                        continue
                    assert not (f["start"] + oi <= g.stop + og and g.start + og <= f["stop"] + oi), (key, oi, g.name, og)


def _fake_preflight(tmp_path, prereg, *, valid=True, mode="preflight"):
    d = tmp_path / "pf"
    d.mkdir()
    pf = prereg["families"]["engineering_preflight_proposed"]
    (d / "seed_opened.json").write_text(json.dumps({"mode": mode, "family": {**pf, "spent": True}}))
    (d / "i1_validity.json").write_text(json.dumps({"valid": valid, "validity": {k: valid for k in VALIDITY_KEYS}}))
    (d / "manifest.json").write_text(json.dumps(runner.manifest(2)))
    return d


VALIDITY_KEYS = ("fidelity_to_frozen_v036_driver", "replay_bit_exact_before_every_intervention",
                 "interventions_applied_exactly_as_planned", "manifest_consistent_across_arms",
                 "store_authority_controls_change_the_individual", "host_state_read_from_store",
                 "driver_record_matches_store_lifecycle", "no_driver_side_model_forwards",
                 "no_driver_side_mechanism_calls", "no_errors", "legacy_tree_unchanged", "pins_verified")


def _registry_copy(tmp_path):
    """A PROPOSED copy of the repository's two registry files (whatever state the repository is in)."""
    root = tmp_path / "root"
    (root / "registry").mkdir(parents=True)
    p, c = _as_proposed(json.loads((runner.REPO / runner.PREREG_FILE).read_text()), _load_central())
    (root / runner.PREREG_FILE).write_text(json.dumps(p, indent=2))
    (root / "registry" / "preregistrations.json").write_text(json.dumps(c, indent=2))
    return root


def test_approval_is_one_checked_act_over_both_registry_files(tmp_path):
    repo_bytes = [(runner.REPO / "registry" / f).read_bytes() for f in ("i1_integration_v1.json", "preregistrations.json")]
    root = _registry_copy(tmp_path)
    prereg = json.loads((root / runner.PREREG_FILE).read_text())
    d = _fake_preflight(tmp_path, prereg)
    with pytest.raises(runner.I1Refused, match="--i-approve"):
        runner.approve(preflight_dir=str(d), approver="David", confirm="yes", root=root)
    out = runner.approve(preflight_dir=str(d), approver="David", confirm=runner.PREREG_ID, root=root)
    p = json.loads((root / runner.PREREG_FILE).read_text())
    assert p["status"] == "APPROVED" and p["approved_by"] == "David"
    assert p["preflight_manifest_sha256"] == runner.manifest(2)["manifest_sha256"] == out["preflight_manifest_sha256"]
    assert p["approved_design_sha256"] == runner.design_digest(p) == runner.design_digest(prereg)
    central = json.loads((root / "registry" / "preregistrations.json").read_text())
    assert [e["status"] for e in central["preregistrations"] if e["id"] == runner.PREREG_ID] == ["APPROVED"]
    from continuous_brain_unified.prereg import PreregistrationRegistry
    PreregistrationRegistry.load(root / "registry").require(runner.PREREG_ID, item="I1")      # the science gate opens
    with pytest.raises(runner.I1Refused, match="not PROPOSED"):          # approval happens once
        runner.approve(preflight_dir=str(d), approver="David", confirm=runner.PREREG_ID, root=root)
    assert repo_bytes == [(runner.REPO / "registry" / f).read_bytes()                       # repo untouched
                          for f in ("i1_integration_v1.json", "preregistrations.json")]


def test_approval_refuses_an_invalid_or_foreign_preflight(tmp_path):
    root = _registry_copy(tmp_path)
    prereg = json.loads((root / runner.PREREG_FILE).read_text())
    bad = _fake_preflight(tmp_path, prereg, valid=False)
    with pytest.raises(runner.I1Refused, match="not valid"):
        runner.approve(preflight_dir=str(bad), approver="David", confirm=runner.PREREG_ID, root=root)
    m = json.loads((bad / "manifest.json").read_text())
    (bad / "i1_validity.json").write_text(json.dumps({"valid": True, "validity": {k: True for k in VALIDITY_KEYS}}))
    m["environment"]["cpu"] = "another machine"
    import hashlib
    m["manifest_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"},
                                                     sort_keys=True).encode()).hexdigest()
    (bad / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(runner.I1Refused, match="differs from the preflight"):
        runner.approve(preflight_dir=str(bad), approver="David", confirm=runner.PREREG_ID, root=root)
    o = json.loads((bad / "seed_opened.json").read_text())
    o["mode"] = "dev"
    (bad / "seed_opened.json").write_text(json.dumps(o))
    with pytest.raises(runner.I1Refused, match="spent preflight"):
        runner.approve(preflight_dir=str(bad), approver="David", confirm=runner.PREREG_ID, root=root)
    assert json.loads((root / runner.PREREG_FILE).read_text())["status"] == "PROPOSED"


def test_later_arms_include_the_hidden_state_control():
    from continuous_brain_unified.integration import evaluate, life
    assert "RNG_PERTURBED" in runner.LATER_ARMS and "RNG_PERTURBED" in life.ARMS and "RNG_PERTURBED" in evaluate.ARMS
    assert evaluate.S4_K == (1, 2, 4)


def test_registry_binding_covers_both_registry_files_and_the_design():
    p = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    b = runner._registry_binding(p)
    from continuous_brain_protocols import sha256_file
    assert b["design_sha256"] == runner.design_digest(p)
    assert b["prereg_file_sha256"] == sha256_file(runner.REPO / runner.PREREG_FILE)
    assert b["central_registry_sha256"] == sha256_file(runner.REPO / "registry" / "preregistrations.json")
    assert b["prereg_status"] == p["status"] and runner._registry_binding(p) == b      # deterministic
