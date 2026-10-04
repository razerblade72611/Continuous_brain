"""I1 runner guards (v0.38.1.dev7): science is locked until approval; plans target the right ticks.

Lifecycle: the repository's I1 preregistration moves PROPOSED -> (preflight spent) -> APPROVED ->
(scientific family spent). Nothing in this file asserts one particular state. The repository's current
state is checked against the lifecycle invariants (runner.lifecycle_problems), every legal state is
simulated on copies and must pass them, and illegal states must fail them, so the suite is green in
every legal state, before and after approval."""

import json
import re
from pathlib import Path

import pytest

from continuous_brain_unified.integration import runner
from continuous_brain_unified.prereg import PreregistrationError, PreregistrationRegistry

from i1_fixtures import as_proposed as _as_proposed, proposed_root, synthetic_preflight

APPROVAL_FIELDS = runner.APPROVAL_FIELDS
i1_lifecycle_problems = runner.lifecycle_problems          # the production validator (approve runs it too)


def _load_central(root=runner.REPO):
    return json.loads((root / "registry" / "preregistrations.json").read_text())


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
    assert i1_lifecycle_problems(p, _load_central(), SeedRegistry.load(runner.REPO / "registry"), runner.REPO) == []


def test_every_legal_lifecycle_state_passes_and_illegal_ones_fail(tmp_path):
    from continuous_brain_eval.seeds import RegisteredRange, SeedRegistry
    from continuous_brain_protocols import SeedFamily, SeedPurpose
    root = proposed_root(tmp_path)
    p0, c0 = json.loads((root / runner.PREREG_FILE).read_text()), _load_central(root)
    fresh = SeedRegistry.load(root / "registry")            # neither I1 family opened

    def with_rows(reg, *rows):
        reg = SeedRegistry(list(reg.ranges), None)
        for key, source in rows:
            f = p0["families"][key]
            reg.ranges.append(RegisteredRange(SeedFamily(f["name"], f["start"], f["stop"], SeedPurpose(f["purpose"]),
                                                         False, True), source, "spent"))
        return reg

    sci = ("scientific_dev_proposed", runner.spend_source("science", "sci"))
    # legal: PROPOSED, never opened / PROPOSED, preflight spent by a real opening
    assert i1_lifecycle_problems(p0, c0, fresh, root) == []
    d = synthetic_preflight(root)
    spent = SeedRegistry.load(root / "registry")
    assert i1_lifecycle_problems(p0, c0, spent, root) == []
    # legal: APPROVED through the real approval function / APPROVED with the scientific family spent
    runner.approve(preflight_dir=str(d), approver="David", confirm=runner.PREREG_ID, root=root)
    pa, ca = json.loads((root / runner.PREREG_FILE).read_text()), _load_central(root)
    PreregistrationRegistry.load(root / "registry").require(runner.PREREG_ID, item="I1")
    assert i1_lifecycle_problems(pa, ca, spent, root) == []
    assert i1_lifecycle_problems(pa, ca, with_rows(spent, sci), root) == []
    # illegal states are caught
    bad = json.loads(json.dumps(pa)); bad["decision_rule"]["primary"]["C2_first_goal_completes"]["threshold"] = "0.5"
    assert any("design digest" in x for x in i1_lifecycle_problems(bad, ca, spent))
    assert any("central status" in x for x in i1_lifecycle_problems(pa, c0, spent))
    assert any("without a spent preflight" in x for x in i1_lifecycle_problems(pa, ca, fresh))
    assert any("before approval" in x for x in i1_lifecycle_problems(p0, c0, with_rows(fresh, sci)))
    leaky = json.loads(json.dumps(p0)); leaky["preflight_manifest_sha256"] = "0" * 64
    assert any("approval fields" in x for x in i1_lifecycle_problems(leaky, c0, fresh))
    simulated = with_rows(fresh, ("engineering_preflight_proposed", "lifecycle-suite:simulated"))
    assert any("not by a preflight run" in x for x in i1_lifecycle_problems(p0, c0, simulated))
    other = with_rows(fresh, ("engineering_preflight_proposed", runner.spend_source("preflight", "another_run")))
    assert any("another run" in x for x in i1_lifecycle_problems(pa, ca, other))
    nobasis = json.loads(json.dumps(pa)); nobasis.pop("approval_basis")
    assert any("approval basis" in x for x in i1_lifecycle_problems(nobasis, ca, spent))
    wrongbasis = json.loads(json.dumps(pa)); wrongbasis["approval_basis"]["manifest_sha256"] = "1" * 64
    assert any("disagrees" in x for x in i1_lifecycle_problems(wrongbasis, ca, spent))
    (root / runner.JOURNAL).write_text("{}")
    assert any("journal" in x for x in i1_lifecycle_problems(pa, ca, spent, root))


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


from continuous_brain_unified.integration.evaluate import VALIDITY_KEYS  # the one authoritative schema
from continuous_brain_unified.integration.evaluate import ARMS as I1_ARMS  # every arm must be bound


def _approve(root, d):
    return runner.approve(preflight_dir=str(d), approver="David", confirm=runner.PREREG_ID, root=root)


def _status(root):
    return json.loads((root / runner.PREREG_FILE).read_text())["status"]


def test_approval_is_one_checked_act_over_both_registry_files(tmp_path):
    repo_bytes = [(runner.REPO / "registry" / f).read_bytes() for f in ("i1_integration_v1.json", "preregistrations.json")]
    root = proposed_root(tmp_path)
    prereg = json.loads((root / runner.PREREG_FILE).read_text())
    d = synthetic_preflight(root)
    with pytest.raises(runner.I1Refused, match="--i-approve"):
        runner.approve(preflight_dir=str(d), approver="David", confirm="yes", root=root)
    out = _approve(root, d)
    p = json.loads((root / runner.PREREG_FILE).read_text())
    assert p["status"] == "APPROVED" and p["approved_by"] == "David" and p["preflight_run"] == d.name
    assert p["preflight_manifest_sha256"] == runner.manifest(2)["manifest_sha256"] == out["preflight_manifest_sha256"]
    assert p["approved_design_sha256"] == runner.design_digest(p) == runner.design_digest(prereg)
    assert set(p["approval_basis"]) == set(runner.APPROVAL_BASIS_KEYS) and p["approval_basis"] == out["approval_basis"]
    central = json.loads((root / "registry" / "preregistrations.json").read_text())
    assert [e["status"] for e in central["preregistrations"] if e["id"] == runner.PREREG_ID] == ["APPROVED"]
    PreregistrationRegistry.load(root / "registry").require(runner.PREREG_ID, item="I1")      # the science gate opens
    assert not (root / runner.JOURNAL).exists()
    # science re-verifies the same chain and reproduces the recorded basis
    assert runner.verify_preflight_chain(d, root=root, stage="science") == p["approval_basis"]
    with pytest.raises(runner.I1Refused, match="not PROPOSED"):          # approval happens once
        _approve(root, d)
    assert repo_bytes == [(runner.REPO / "registry" / f).read_bytes()                       # repo untouched
                          for f in ("i1_integration_v1.json", "preregistrations.json")]


def test_approval_refuses_an_invalid_or_foreign_preflight(tmp_path):
    import hashlib
    root = proposed_root(tmp_path / "a")
    bad = synthetic_preflight(root, valid=False)
    with pytest.raises(runner.I1Refused, match="not valid"):
        _approve(root, bad)
    root = proposed_root(tmp_path / "b")
    d = synthetic_preflight(root)
    m = json.loads((d / "manifest.json").read_text())
    m["environment"]["cpu"] = "another machine"
    m["manifest_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"},
                                                     sort_keys=True).encode()).hexdigest()
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(runner.I1Refused, match="another manifest"):    # not the manifest it was opened under
        _approve(root, d)
    o = json.loads((d / "seed_opened.json").read_text())
    o["manifest_sha256"] = m["manifest_sha256"]
    (d / "seed_opened.json").write_text(json.dumps(o))
    with pytest.raises(runner.I1Refused, match="differs from the preflight"):    # not this machine's manifest
        _approve(root, d)
    o["mode"] = "dev"
    (d / "seed_opened.json").write_text(json.dumps(o))
    with pytest.raises(runner.I1Refused, match="opening record"):
        _approve(root, d)
    assert _status(root) == "PROPOSED"


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
    assert b["seed_ledger_sha256"] == runner.seed_ledger_digest(runner.REPO)           # the spend ledger too
    assert set(b) == set(runner.BINDING_KEYS)
    assert b["prereg_status"] == p["status"] and runner._registry_binding(p) == b      # deterministic


def test_validity_schema_is_the_fourteen_preregistered_checks():
    p = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    registered = {k for k in p["decision_rule"]["validity_blocking"]
                  if k not in ("on_failure", "legacy_tree_unchanged_and_pins_verified")}
    registered |= {"legacy_tree_unchanged", "pins_verified"}
    assert len(VALIDITY_KEYS) == 14 and len(set(VALIDITY_KEYS)) == 14
    assert set(VALIDITY_KEYS) == registered            # code schema == preregistered checks
    assert {"no_hidden_mutable_state", "donor_pairing_fixed_by_ids"} <= set(VALIDITY_KEYS)


def _approve_with(tmp_path, mutate):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    v = json.loads((d / "i1_validity.json").read_text())
    mutate(v)
    (d / "i1_validity.json").write_text(json.dumps(v))
    try:
        _approve(root, d)
    finally:
        assert _status(root) in ("PROPOSED", "APPROVED")
    return _status(root)


@pytest.mark.parametrize("key", VALIDITY_KEYS)
def test_approval_refuses_a_preflight_missing_any_one_check(tmp_path, key):
    with pytest.raises(runner.I1Refused, match="exactly the I1 schema"):
        _approve_with(tmp_path, lambda v: v["validity"].pop(key))


def test_approval_refuses_the_old_twelve_check_schema(tmp_path):
    old = {"no_hidden_mutable_state", "donor_pairing_fixed_by_ids"}
    with pytest.raises(runner.I1Refused, match="missing"):
        _approve_with(tmp_path, lambda v: [v["validity"].pop(k) for k in old])


def test_approval_refuses_unknown_checks_false_values_and_unbound_manifests(tmp_path):
    cases = {
        "unexpected": lambda v: v["validity"].__setitem__("some_new_check", True),
        "false": lambda v: v["validity"].__setitem__("no_hidden_mutable_state", False),
        "valid flag": lambda v: v.__setitem__("valid", False),
        "no diagnostics": lambda v: v.pop("validity_diagnostics"),
        "no expected manifest": lambda v: v["validity_diagnostics"].pop("expected_manifest_sha256"),
        "other manifest": lambda v: v["validity_diagnostics"].__setitem__("expected_manifest_sha256", "0" * 64),
        "one arm elsewhere": lambda v: v["validity_diagnostics"]["arm_manifests"].__setitem__("DONOR", "0" * 64),
        "no arms": lambda v: v["validity_diagnostics"].__setitem__("arm_manifests", {}),
        "only NORMAL bound": lambda v: v["validity_diagnostics"].__setitem__(
            "arm_manifests", {"NORMAL": v["validity_diagnostics"]["expected_manifest_sha256"]}),
        "unexpected arm": lambda v: v["validity_diagnostics"]["arm_manifests"].__setitem__(
            "SOME_OTHER_ARM", v["validity_diagnostics"]["expected_manifest_sha256"]),
    }
    for i, (name, mutate) in enumerate(cases.items()):
        with pytest.raises(runner.I1Refused):
            _approve_with(tmp_path / str(i), mutate)



@pytest.mark.parametrize("arm", I1_ARMS)
def test_approval_refuses_a_preflight_with_any_one_arm_unbound(tmp_path, arm):
    with pytest.raises(runner.I1Refused, match="exactly the I1 arms"):
        _approve_with(tmp_path, lambda v: v["validity_diagnostics"]["arm_manifests"].pop(arm))


# ---- chain of custody (ChatGPT's 64g reproductions and the hardening around them) --------------------------

def _ledger(root):
    return root / "registry" / "seed_ranges.local.jsonl"


def _is_pf_spend(root, row):
    return bool(row.get("spent")) and row.get("name") == _pf_name(root)


def _rewrite_ledger(root, fn):
    rows = [json.loads(x) for x in _ledger(root).read_text().splitlines() if x.strip()]
    _ledger(root).write_text("".join(json.dumps(r) + "\n" for r in fn(rows)))


def _pf_name(root):
    return json.loads((root / runner.PREREG_FILE).read_text())["families"]["engineering_preflight_proposed"]["name"]


def test_chatgpt_repro_design_changed_after_the_preflight_was_opened(tmp_path):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    p = json.loads((root / runner.PREREG_FILE).read_text())
    p["decision_rule"]["primary"]["C2_first_goal_completes"]["threshold"] = "0.5"      # a frozen threshold
    (root / runner.PREREG_FILE).write_text(json.dumps(p, indent=2) + "\n")
    with pytest.raises(runner.I1Refused, match="design changed after the preflight was opened"):
        _approve(root, d)
    assert _status(root) == "PROPOSED"


def test_chatgpt_repro_fabricated_preflight_without_a_ledger_spend_or_sealed_report(tmp_path):
    root = proposed_root(tmp_path)
    prereg = json.loads((root / runner.PREREG_FILE).read_text())
    d = root / "runs" / "i1" / "fabricated"
    d.mkdir(parents=True)
    m = runner.manifest(2)
    (d / "manifest.json").write_text(json.dumps(m))
    (d / "seed_opened.json").write_text(json.dumps({"mode": "preflight", "family": {
        **prereg["families"]["engineering_preflight_proposed"], "spent": True}}))
    (d / "i1_validity.json").write_text(json.dumps({
        "valid": True, "validity": {k: True for k in VALIDITY_KEYS},
        "validity_diagnostics": {"expected_manifest_sha256": m["manifest_sha256"],
                                 "arm_manifests": {a: m["manifest_sha256"] for a in I1_ARMS}}}))
    with pytest.raises(runner.I1Refused, match="no spend"):
        _approve(root, d)
    assert _status(root) == "PROPOSED"


def _chain_case(tmp_path, attack):
    """A genuine synthetic preflight, attacked by `attack(root, d)` (which may return another directory)."""
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    moved = attack(root, d)
    d = moved if isinstance(moved, Path) else d
    try:
        _approve(root, d)
    finally:
        assert _status(root) in ("PROPOSED", "APPROVED")


def _edit_json(path, fn):
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _reseal(d, fn):
    """Alter the sealed report and recommit its SHA-256 in the visible view (a consistent forgery)."""
    import hashlib
    rep = json.loads((d / "sealed" / "i1_report.json").read_text())
    fn(rep)
    data = json.dumps(rep, indent=2, sort_keys=True) + "\n"
    (d / "sealed" / "i1_report.json").write_text(data)
    _edit_json(d / "i1_validity.json", lambda v: v.__setitem__("sealed_report_sha256",
                                                                hashlib.sha256(data.encode()).hexdigest()))


def _copy_dir(src, dst):
    import shutil
    shutil.copytree(src, dst)
    return dst


CHAIN_ATTACKS = {
    "prereg file edited after opening (design unchanged)": (
        lambda r, d: _edit_json(r / runner.PREREG_FILE, lambda p: p.setdefault("deviation_log", []).append("late")),
        "changed after the preflight was opened"),
    "central registry edited after opening": (
        lambda r, d: _edit_json(r / "registry" / "preregistrations.json", lambda c: c.__setitem__("touched", 1)),
        "changed after the preflight was opened"),
    "ledger spend removed": (
        lambda r, d: _rewrite_ledger(r, lambda rows: [x for x in rows if not _is_pf_spend(r, x)]), "no spend"),
    "ledger spend by another source": (
        lambda r, d: _rewrite_ledger(r, lambda rows: [{**x, "source": "lifecycle-suite:simulated"} if _is_pf_spend(r, x)
                                                      else x for x in rows]), "was made by"),
    "ledger spend recorded twice": (
        lambda r, d: _rewrite_ledger(r, lambda rows: rows + [x for x in rows if _is_pf_spend(r, x)]), "2 spends"),
    "ledger row after the spend": (
        lambda r, d: _rewrite_ledger(r, lambda rows: rows + [{**[x for x in rows if _is_pf_spend(r, x)][0],
                                                               "spent": False, "note": "unspent again"}]),
        "supersedes"),
    "family marked spent in the shared ranges file": (
        lambda r, d: _edit_json(r / "registry" / "seed_ranges.json", lambda j: j["ranges"].append(
            {**json.loads((d / "seed_opened.json").read_text())["family"], "source": "hand", "note": ""})),
        "shared ranges file"),
    "sealed report missing": (lambda r, d: (d / "sealed" / "i1_report.json").unlink(), "sealed report is missing"),
    "sealed report altered": (
        lambda r, d: (d / "sealed" / "i1_report.json").write_text(
            (d / "sealed" / "i1_report.json").read_text().replace('"n": 64', '"n": 63')), "does not match the SHA-256"),
    "sealed report recommitted with other validity": (
        lambda r, d: _reseal(d, lambda rep: rep["validity"].__setitem__("no_errors", False)), "disagrees"),
    "sealed report recommitted with other arm bindings": (
        lambda r, d: _reseal(d, lambda rep: rep["validity_diagnostics"]["arm_manifests"].__setitem__("DONOR", "0" * 64)),
        "disagrees"),
    "sealed report of another run": (lambda r, d: _reseal(d, lambda rep: rep.__setitem__("run", "other")), "disagrees"),
    "opened under an APPROVED status": (
        lambda r, d: _edit_json(d / "seed_opened.json", lambda o: o.__setitem__("prereg_status", "APPROVED")),
        "not opened under the PROPOSED"),
    "approval fields at opening": (
        lambda r, d: _edit_json(d / "seed_opened.json", lambda o: o.__setitem__("approved_design_sha256", "0" * 64)),
        "not opened under the PROPOSED"),
    "opening record without the binding": (
        lambda r, d: _edit_json(d / "seed_opened.json", lambda o: o.pop("seed_ledger_sha256")), "lacks the registry"),
    "opening record of another run": (
        lambda r, d: _edit_json(d / "seed_opened.json", lambda o: o.__setitem__("run", "other")), "opening record"),
    "directory renamed": (lambda r, d: _copy_dir(d, d.parent / "renamed"), "was made by"),
    "directory outside runs/i1": (lambda r, d: _copy_dir(d, r / "elsewhere" / d.name), "runs/i1/<label>"),
    "interrupted approval journal present": (
        lambda r, d: (r / runner.JOURNAL).write_text("{}"), "journal"),
    "validity view of another run": (
        lambda r, d: _edit_json(d / "i1_validity.json", lambda v: v.__setitem__("run", "other")), "validity view"),
}


@pytest.mark.parametrize("name", sorted(CHAIN_ATTACKS))
def test_approval_refuses_every_break_in_the_chain_of_custody(tmp_path, name):
    attack, msg = CHAIN_ATTACKS[name]
    with pytest.raises(runner.I1Refused, match=re.escape(msg)):
        _chain_case(tmp_path, attack)


def test_preflight_approval_and_science_refuse_a_foreign_registry(tmp_path, monkeypatch):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    monkeypatch.setenv("CONTINUOUS_BRAIN_REGISTRY", str(tmp_path / "elsewhere"))
    with pytest.raises(runner.I1Refused, match="CONTINUOUS_BRAIN_REGISTRY"):
        _approve(root, d)
    with pytest.raises(runner.I1Refused, match="CONTINUOUS_BRAIN_REGISTRY"):
        runner.run("preflight", legacy_root=str(tmp_path), historical_core_archive="none", label="never")
    with pytest.raises(runner.I1Refused, match="CONTINUOUS_BRAIN_REGISTRY"):
        runner.run("science", legacy_root=str(tmp_path), historical_core_archive="none", label="never",
                   confirm_open=runner.PREREG_ID)
    monkeypatch.setenv("CONTINUOUS_BRAIN_REGISTRY", str(root / "registry"))       # the same directory is fine
    _approve(root, d)
    assert _status(root) == "APPROVED"
    assert not (runner.REPO / "runs" / "i1" / "never").exists()


def test_science_reverification_catches_changes_after_approval(tmp_path):
    def approved():
        root = proposed_root(tmp_path / str(len(list(tmp_path.iterdir()))))
        d = synthetic_preflight(root)
        _approve(root, d)
        return root, d
    root, d = approved()
    basis = runner.verify_preflight_chain(d, root=root, stage="science")
    from continuous_brain_eval.seeds import SeedRegistry
    from continuous_brain_protocols import SeedFamily, SeedPurpose
    f = json.loads((root / runner.PREREG_FILE).read_text())["families"]["scientific_dev_proposed"]
    reg = SeedRegistry.load(root / "registry")      # opening the scientific family does not disturb the chain
    reg.register(SeedFamily(f["name"], f["start"], f["stop"], SeedPurpose(f["purpose"])), source="t")
    reg.spend(SeedFamily(f["name"], f["start"], f["stop"], SeedPurpose(f["purpose"])),
              source=runner.spend_source("science", "sci"))
    assert runner.verify_preflight_chain(d, root=root, stage="science") == basis
    attacks = {
        "validity view edited": (lambda r, d: _edit_json(d / "i1_validity.json", lambda v: v.__setitem__("note", "x")),
                                 "no longer reproduces"),
        "opening record edited": (lambda r, d: _edit_json(d / "seed_opened.json", lambda o: o.__setitem__("utc", "x")),
                                  "no longer reproduces"),
        "design edited": (lambda r, d: _edit_json(r / runner.PREREG_FILE, lambda p: p["decision_rule"]["primary"][
            "C2_first_goal_completes"].__setitem__("threshold", "0.5")), "design changed"),
        "basis edited": (lambda r, d: _edit_json(r / runner.PREREG_FILE, lambda p: p["approval_basis"].__setitem__(
            "sealed_report_sha256", "0" * 64)), "no longer reproduces"),
        "central disagrees": (lambda r, d: _edit_json(r / "registry" / "preregistrations.json", lambda c: [
            e.__setitem__("approved_by", "Someone") for e in c["preregistrations"] if e["id"] == runner.PREREG_ID]),
            "does not agree"),
        "preflight spend removed": (lambda r, d: _rewrite_ledger(r, lambda rows: [
            x for x in rows if not (x.get("spent") and x["name"] == _pf_name(r))]), "no spend"),
        "sealed report removed": (lambda r, d: (d / "sealed" / "i1_report.json").unlink(), "sealed report is missing"),
    }
    for name, (attack, msg) in attacks.items():
        root, d = approved()
        attack(root, d)
        with pytest.raises(runner.I1Refused, match=re.escape(msg)):
            runner.verify_preflight_chain(d, root=root, stage="science")


def test_science_run_reverifies_the_chain_before_opening(monkeypatch):
    import inspect
    src = inspect.getsource(runner.run)
    assert src.index('stage="science"') < src.index("open_family(")             # verified before the family opens
    assert "_registry_binding(json.loads((REPO / PREREG_FILE).read_text()), REPO)" in src   # incl. the ledger at end


def test_an_interrupted_approval_is_refused_until_recovered(tmp_path, monkeypatch):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    real, calls = runner._atomic_write, []

    def crash_after_first_registry_file(path, data):
        real(path, data)
        calls.append(path.name)
        if path.name == "i1_integration_v1.json":
            raise OSError("power cut")
    monkeypatch.setattr(runner, "_atomic_write", crash_after_first_registry_file)
    with pytest.raises(OSError, match="power cut"):
        _approve(root, d)
    monkeypatch.setattr(runner, "_atomic_write", real)
    assert (root / runner.JOURNAL).exists() and calls[0] == (root / runner.JOURNAL).name
    assert _status(root) == "APPROVED" and [e["status"] for e in _load_central(root)["preregistrations"]
                                            if e["id"] == runner.PREREG_ID] == ["PROPOSED"]      # half-written
    from continuous_brain_eval.seeds import SeedRegistry
    assert any("journal" in x for x in runner.lifecycle_problems(
        json.loads((root / runner.PREREG_FILE).read_text()), _load_central(root), SeedRegistry.load(root / "registry"),
        root))
    with pytest.raises(runner.I1Refused, match="journal"):
        runner.verify_preflight_chain(d, root=root, stage="science")
    assert runner.recover_approval(root) == "approval completed from the journal"
    assert not (root / runner.JOURNAL).exists()
    assert [e["status"] for e in _load_central(root)["preregistrations"] if e["id"] == runner.PREREG_ID] == ["APPROVED"]
    assert runner.verify_preflight_chain(d, root=root, stage="science") == \
        json.loads((root / runner.PREREG_FILE).read_text())["approval_basis"]
    assert runner.recover_approval(root).startswith("no approval journal")


def test_recovery_refuses_a_file_in_neither_state(tmp_path, monkeypatch):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    real = runner._atomic_write

    def crash(path, data):
        real(path, data)
        if path.name == "i1_integration_v1.json":
            raise OSError("power cut")
    monkeypatch.setattr(runner, "_atomic_write", crash)
    with pytest.raises(OSError):
        _approve(root, d)
    monkeypatch.setattr(runner, "_atomic_write", real)
    _edit_json(root / "registry" / "preregistrations.json", lambda c: c.__setitem__("hand_edit", 1))
    with pytest.raises(runner.I1Refused, match="neither"):
        runner.recover_approval(root)
    assert (root / runner.JOURNAL).exists()


def test_preflight_output_shows_no_timings_or_error_counts():
    import inspect
    src = inspect.getsource(runner._run_arm)
    assert '{"arm": arm, "finished": True}' in src
    assert 'quiet = mode == "preflight"' in inspect.getsource(runner.run)


def test_the_opening_record_binds_spend_run_manifest_and_registry_state(tmp_path):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root, label="lbl")
    o = json.loads((d / "seed_opened.json").read_text())
    assert o["run"] == "lbl" and o["spend_source"] == runner.spend_source("preflight", "lbl") == \
        "I1-INTEGRATED-INDIVIDUAL-v1:preflight:lbl:opened"
    assert o["manifest_sha256"] == json.loads((d / "manifest.json").read_text())["manifest_sha256"]
    assert o["seed_ledger_sha256"] == runner.seed_ledger_digest(root)          # the ledger after the spend
    assert o["prereg_status"] == "PROPOSED" and o["approved_design_sha256"] is None
    with pytest.raises(runner.I1Refused, match="runs/i1"):
        runner.open_family("preflight", "x", prereg=json.loads((root / runner.PREREG_FILE).read_text()),
                           man=runner.manifest(2), run_dir=root / "elsewhere", root=root)


# ---- ChatGPT's fourth gate audit (64h): approval authority, ledger freeze, derived ids at every spend ----

def test_only_the_frozen_approval_authority_can_approve(tmp_path):
    root = proposed_root(tmp_path)
    p = json.loads((root / runner.PREREG_FILE).read_text())
    assert p["approval_authority"] == "David"
    assert runner.design_digest({**p, "approval_authority": "Claude"}) != runner.design_digest(p)      # digested
    d = synthetic_preflight(root)
    for who in ("Claude", "ChatGPT", "david", " David", "David ", "DAVID"):
        with pytest.raises(runner.I1Refused, match="approval authority"):
            runner.approve(preflight_dir=str(d), approver=who, confirm=runner.PREREG_ID, root=root)
    assert _status(root) == "PROPOSED"
    _approve(root, d)
    assert json.loads((root / runner.PREREG_FILE).read_text())["approved_by"] == "David"


def test_chatgpt_repro_approval_by_claude_is_refused(tmp_path):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    with pytest.raises(runner.I1Refused, match="approval authority"):
        runner.approve(preflight_dir=str(d), approver="Claude", confirm=runner.PREREG_ID, root=root)
    assert _status(root) == "PROPOSED"


def test_a_preregistration_without_an_approval_authority_cannot_be_approved(tmp_path):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    _edit_json(root / runner.PREREG_FILE, lambda p: p.pop("approval_authority"))
    with pytest.raises(runner.I1Refused, match="names no approval_authority"):
        _approve(root, d)


def test_science_and_lifecycle_refuse_an_approval_by_anyone_else(tmp_path):
    from continuous_brain_eval.seeds import SeedRegistry
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    _approve(root, d)
    _edit_json(root / runner.PREREG_FILE, lambda p: p.__setitem__("approved_by", "Claude"))
    _edit_json(root / "registry" / "preregistrations.json", lambda c: [
        e.__setitem__("approved_by", "Claude") for e in c["preregistrations"] if e["id"] == runner.PREREG_ID])
    with pytest.raises(runner.I1Refused, match="approval authority"):
        runner.verify_preflight_chain(d, root=root, stage="science")
    problems = runner.lifecycle_problems(json.loads((root / runner.PREREG_FILE).read_text()), _load_central(root),
                                         SeedRegistry.load(root / "registry"), root)
    assert any("approval authority" in x for x in problems)


def test_chatgpt_repro_an_unrelated_ledger_row_after_opening_blocks_approval(tmp_path):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    with open(_ledger(root), "a") as f:
        f.write(json.dumps({"name": "unrelated-later-family", "start": 990000, "stop": 990001,
                            "purpose": "ENGINEERING", "source": "someone", "note": ""}) + "\n")
    with pytest.raises(runner.I1Refused, match="seed ledger changed after the preflight was opened"):
        _approve(root, d)
    assert _status(root) == "PROPOSED"


def _register(root, name, start, stop, purpose="ENGINEERING"):
    from continuous_brain_eval.seeds import SeedRegistry
    from continuous_brain_protocols import SeedFamily, SeedPurpose
    SeedRegistry.load(root / "registry").register(SeedFamily(name, start, stop, SeedPurpose(purpose)), source="test")


def _sci_rows(root):
    from continuous_brain_eval.seeds import SeedRegistry
    name = json.loads((root / runner.PREREG_FILE).read_text())["families"]["scientific_dev_proposed"]["name"]
    return [r for r in SeedRegistry.load(root / "registry").ranges if r.family.name == name]


def test_chatgpt_repro_derived_id_collision_blocks_the_science_spend(tmp_path):
    root = proposed_root(tmp_path)
    d = synthetic_preflight(root)
    _approve(root, d)
    sci = json.loads((root / runner.PREREG_FILE).read_text())["families"]["scientific_dev_proposed"]
    _register(root, "later-family", sci["start"] + 70000, sci["stop"] + 70000)        # 398000-398127
    run_dir = root / "runs" / "i1" / "sci"
    run_dir.mkdir(parents=True)
    with pytest.raises(runner.I1Refused, match=r"cannot be opened.*offset 70000"):
        runner.open_family("science", "sci", prereg=json.loads((root / runner.PREREG_FILE).read_text()),
                           man=runner.manifest(2), run_dir=run_dir, root=root)
    assert _sci_rows(root) == [] and not (run_dir / "seed_opened.json").exists()        # nothing registered or spent


@pytest.mark.parametrize("offset", (0, 70000, 90000))
def test_every_preflight_spend_is_preceded_by_the_derived_id_check(tmp_path, offset):
    root = proposed_root(tmp_path)
    pf = json.loads((root / runner.PREREG_FILE).read_text())["families"]["engineering_preflight_proposed"]
    lo = pf["start"] + offset + (10 if offset == 0 else 0)
    if offset == 0:                                    # inside the v0.38 reserve: a sibling, not a parent
        lo = pf["stop"] + 1 - 4
    _register(root, "colliding-family", lo, lo + 7)
    with pytest.raises(runner.I1Refused, match="cannot be opened"):
        synthetic_preflight(root)
    from continuous_brain_eval.seeds import SeedRegistry
    assert not [r for r in SeedRegistry.load(root / "registry").ranges if r.family.name == pf["name"]]


def test_derived_id_check_is_one_function_for_dev_and_for_every_spend():
    import inspect
    assert "offset_collisions(" in inspect.getsource(runner._dev_sids)
    src = inspect.getsource(runner.open_family)
    assert src.index("offset_collisions(") < src.index("_family(reg") < src.index("reg.spend(")
    reg = _registry_with_cloud_dev()
    p = json.loads((runner.REPO / runner.PREREG_FILE).read_text())
    from continuous_brain_protocols import SeedFamily, SeedPurpose
    for key in ("scientific_dev_proposed", "engineering_preflight_proposed"):
        f = p["families"][key]
        home = SeedFamily(f["name"], f["start"], f["stop"], SeedPurpose(f["purpose"]))
        assert runner.offset_collisions(reg, home.start, home.stop, p, home) == []     # the real families are clean

