import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from continuous_brain_unified import ChannelDecl, Component, UnifiedRuntime
from continuous_brain_unified.legacy import (ENV_ROOT, LegacyHashMismatch, LegacyImportConflict, LegacyManifest,
                                             ManifestError, StepContract, build_legacy_components,
                                             import_verified_module, parse_entry, register_step_contract,
                                             resolve_legacy_root, verify_entry)
from continuous_brain_unified.legacy import adapters as adapters_mod
from continuous_brain_unified.legacy import inventory as inv
from continuous_brain_unified.legacy.adapters import legacy_provenance

REPO = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def legacy_root(tmp_path):
    root = tmp_path / "legacy_tree"
    pkg = root / "continuous_brain"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "toy_memory_v1.py").write_text("class ToyMemory:\n    def __init__(self):\n        self.x = 1.5\n")
    (root / "evaluate_toy_v1.py").write_text("print('eval')\n")
    (root / "state").mkdir()
    (root / "state" / "result.json").write_text(json.dumps({"format": "toy", "test_admission_passed": True}))
    return root


def entry_dict(root: Path, **over):
    d = {"mechanism_id": "toy_memory_v1", "slot": "episodic_memory", "lineage_version": "v0.0", "evidence_status":
         "ADMITTED/CLOSED PASS", "status": "VERIFIED",
         "files": [{"relpath": "continuous_brain/toy_memory_v1.py", "sha256": sha(root / "continuous_brain/toy_memory_v1.py"),
                    "role": "module"},
                   {"relpath": "continuous_brain/__init__.py", "sha256": sha(root / "continuous_brain/__init__.py"),
                    "role": "support"},
                   {"relpath": "evaluate_toy_v1.py", "sha256": sha(root / "evaluate_toy_v1.py"), "role": "evaluator"},
                   {"relpath": "state/result.json", "sha256": sha(root / "state/result.json"), "role": "result"}],
         "entry_points": {"class": "ToyMemory"}, "audit_ref": "DAV-74#test", "verified_by": "test",
         "verified_utc": "2026-09-27T00:00:00Z"}
    d.update(over)
    return d


def test_repository_manifest_is_valid_and_has_no_verified_entries_yet():
    m = LegacyManifest.load(REPO / "legacy" / "manifest.json")
    assert m.entries == ()
    schema = json.loads((REPO / "legacy" / "manifest.schema.json").read_text())
    assert schema["properties"]["format"]["const"] == "cb-legacy-manifest/1"


def test_manifest_rejects_unverified_absolute_and_malformed_entries(legacy_root):
    with pytest.raises(ManifestError, match="only VERIFIED"):
        parse_entry(entry_dict(legacy_root, status="UNVERIFIED"))
    bad = entry_dict(legacy_root)
    bad["files"][0]["relpath"] = "/home/davei/projects/x.py"
    with pytest.raises(ManifestError, match="relative"):
        parse_entry(bad)
    bad = entry_dict(legacy_root)
    bad["files"][0]["relpath"] = "C:\\Users\\davei\\x.py"
    with pytest.raises(ManifestError):
        parse_entry(bad)
    bad = entry_dict(legacy_root)
    bad["files"] = bad["files"][:1]
    with pytest.raises(ManifestError, match="required file roles"):
        parse_entry(bad)
    with pytest.raises(ManifestError, match="mechanism slot"):
        parse_entry(entry_dict(legacy_root, slot="persistence"))
    with pytest.raises(ManifestError, match="audit"):
        parse_entry(entry_dict(legacy_root, audit_ref=" "))


def test_verification_refuses_mismatch_unless_logged_override(legacy_root, tmp_path):
    e = parse_entry(entry_dict(legacy_root))
    assert verify_entry(e, legacy_root).ok
    (legacy_root / "evaluate_toy_v1.py").write_text("print('changed')\n")
    with pytest.raises(LegacyHashMismatch):
        verify_entry(e, legacy_root)
    log = tmp_path / "overrides.jsonl"
    with pytest.raises(ValueError, match="log"):
        verify_entry(e, legacy_root, dev_override=True)
    res = verify_entry(e, legacy_root, dev_override=True, override_log=log)
    assert res.overridden and not res.ok
    assert json.loads(log.read_text().splitlines()[0])["mechanism_id"] == "toy_memory_v1"


def test_manifest_entries_can_resolve_named_immutable_roots(legacy_root, tmp_path):
    e = parse_entry(entry_dict(legacy_root, root_name="legacy-main"))
    from continuous_brain_unified.legacy import root_for_entry
    assert root_for_entry(e, {"legacy-main": legacy_root}) == legacy_root.resolve()
    overlay = parse_entry(entry_dict(legacy_root, mechanism_id="toy_overlay", root_name="overlay-v036"))
    with pytest.raises(ManifestError, match="not configured"):
        root_for_entry(overlay, {"legacy-main": legacy_root})
    assert root_for_entry(overlay, {"overlay-v036": legacy_root}) == legacy_root.resolve()
    with pytest.raises(ManifestError, match="root mapping"):
        root_for_entry(overlay, legacy_root)


def test_root_resolution(monkeypatch, legacy_root):
    monkeypatch.delenv(ENV_ROOT, raising=False)
    with pytest.raises(ManifestError, match="no legacy root"):
        resolve_legacy_root()
    monkeypatch.setenv(ENV_ROOT, str(legacy_root))
    assert resolve_legacy_root() == legacy_root.resolve()


def test_loader_imports_only_from_the_verified_root(legacy_root, monkeypatch):
    e = parse_entry(entry_dict(legacy_root))
    for k in [k for k in sys.modules if k == "continuous_brain" or k.startswith("continuous_brain.")]:
        monkeypatch.delitem(sys.modules, k)
    mod = import_verified_module(e, legacy_root.resolve())
    assert mod.ToyMemory().x == 1.5
    assert not list(legacy_root.rglob("__pycache__"))            # nothing written under the root
    other = legacy_root.parent / "other"
    (other / "continuous_brain").mkdir(parents=True)
    (other / "continuous_brain" / "toy_memory_v1.py").write_bytes(
        (legacy_root / "continuous_brain" / "toy_memory_v1.py").read_bytes())
    with pytest.raises(LegacyImportConflict):          # same bytes, but a different tree is already imported
        import_verified_module(e, other.resolve())
    (other / "continuous_brain" / "toy_memory_v1.py").write_text("x = 1\n")
    with pytest.raises(LegacyHashMismatch):
        import_verified_module(e, other.resolve())
    for k in [k for k in sys.modules if k == "continuous_brain" or k.startswith("continuous_brain.")]:
        monkeypatch.delitem(sys.modules, k)


def test_verified_entries_without_an_audited_step_contract_stay_unbound(legacy_root):
    manifest = LegacyManifest((parse_entry(entry_dict(legacy_root)),))
    outcomes = build_legacy_components(manifest, legacy_root.resolve())
    assert len(outcomes) == 1 and not outcomes[0].bound and "no audited step contract" in outcomes[0].reason


def test_a_registered_step_contract_binds_under_the_verified_only_policy(legacy_root, monkeypatch):
    e = parse_entry(entry_dict(legacy_root))

    class ToyEpisodic(Component):
        name, slot = "toy", "episodic_memory"
        writes = (ChannelDecl("episodic_memory.slots", "episodic_memory", (1,)),)

        def __init__(self, entry, module, result, contract):
            self.obj = module.ToyMemory()
            self._prov = legacy_provenance(entry, result, contract)

        def provenance(self):
            return self._prov

        def step(self, ctx):
            return {"episodic_memory.slots": np.array([self.obj.x])}

    contract = StepContract("toy_memory_v1", e.file("module").sha256,
                            lambda entry, module, root, result: ToyEpisodic(entry, module, result, contract), "DAV-74#t")
    monkeypatch.setattr(adapters_mod, "_CONTRACTS", {})
    register_step_contract(contract)
    for k in [k for k in sys.modules if k == "continuous_brain" or k.startswith("continuous_brain.")]:
        monkeypatch.delitem(sys.modules, k)
    outcomes = build_legacy_components(LegacyManifest((e,)), legacy_root.resolve())
    assert outcomes[0].bound
    rt = UnifiedRuntime()
    rt.bind(outcomes[0].component)
    assert rt.scientific
    rt.run(2)
    assert rt.view().read("episodic_memory.slots")[0] == 1.5
    report = {r.slot: r for r in rt.binding_report()}
    assert report["episodic_memory"].status == "bound"
    assert report["episodic_memory"].provenance["detail"]["mechanism_id"] == "toy_memory_v1"
    for k in [k for k in sys.modules if k == "continuous_brain" or k.startswith("continuous_brain.")]:
        monkeypatch.delitem(sys.modules, k)


def test_inventory_is_read_only_and_marks_everything_unverified(legacy_root, tmp_path):
    spec = {"format": inv.SPEC_FORMAT,
            "protected_checkpoint": {"relpath": "state/base.pt", "sha256": "0" * 64},
            "candidates": [{"mechanism_id": "toy", "slot_options": ["episodic_memory"], "lineage_version": "v0",
                            "dossier_status": "x", "admissible": True,
                            "module": ["continuous_brain/toy_memory_v1.py"], "evaluator": ["evaluate_*.py"],
                            "result_dirs": ["state"]}]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    before = {p: sha(p) for p in legacy_root.rglob("*") if p.is_file()}
    assert inv.main(["--root", str(legacy_root), "--spec", str(spec_path), "--root-label", "t",
                     "--out", str(legacy_root / "out.json")]) == 2
    out = tmp_path / "cand.json"
    assert inv.main(["--root", str(legacy_root), "--spec", str(spec_path), "--root-label", "t", "--out", str(out)]) == 0
    after = {p: sha(p) for p in legacy_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert {k: v for k, v in before.items() if "__pycache__" not in k.parts} == after
    data = json.loads(out.read_text())
    c = data["candidates"][0]
    assert c["status"] == "UNVERIFIED"
    assert c["module"][0]["api"]["classes"][0]["name"] == "ToyMemory"
    assert c["evaluator"][0]["relpath"] == "evaluate_toy_v1.py"
    assert c["result_dirs"][0]["results"][0]["passed_flags"] == {"test_admission_passed": True}
    assert data["protected_checkpoint"]["exists"] is False
    assert str(legacy_root) not in out.read_text()          # the root path itself is never recorded


def test_committed_mirror_inventory_is_consistent():
    data = json.loads((REPO / "legacy" / "candidates.json").read_text())
    assert data["format"] == "cb-legacy-candidates/1" and data["root_label"] == "windows-source-mirror"
    assert all(c["status"] == "UNVERIFIED" for c in data["candidates"])
    assert data["protected_checkpoint"]["exists"] is False
    ids = {c["mechanism_id"] for c in data["candidates"]}
    assert "event_consistent_self_development_v0341" in ids
    text = (REPO / "legacy" / "candidates.json").read_text()
    assert "davei" not in text and "/home/" not in text and "C:\\" not in text   # no personal absolute paths


def test_loader_refuses_unhashed_in_package_dependencies(legacy_root, monkeypatch):
    (legacy_root / "continuous_brain" / "helper_v1.py").write_text("VALUE = 3\n")
    (legacy_root / "continuous_brain" / "toy_memory_v1.py").write_text(
        "from .helper_v1 import VALUE\nclass ToyMemory:\n    x = VALUE\n")
    e = parse_entry(entry_dict(legacy_root))
    for k in [k for k in sys.modules if k == "continuous_brain" or k.startswith("continuous_brain.")]:
        monkeypatch.delitem(sys.modules, k)
    with pytest.raises(LegacyImportConflict, match="helper_v1.py"):
        import_verified_module(e, legacy_root.resolve())
    d = entry_dict(legacy_root)
    d["files"].append({"relpath": "continuous_brain/helper_v1.py",
                       "sha256": sha(legacy_root / "continuous_brain/helper_v1.py"), "role": "support"})
    assert import_verified_module(parse_entry(d), legacy_root.resolve()).ToyMemory.x == 3
    for k in [k for k in sys.modules if k == "continuous_brain" or k.startswith("continuous_brain.")]:
        monkeypatch.delitem(sys.modules, k)


def test_hand_made_legacy_provenance_is_refused():
    from continuous_brain_unified import BindingError, ComponentProvenance, ProvenanceKind

    class Fake(Component):
        name, slot = "fake", "episodic_memory"
        writes = (ChannelDecl("episodic_memory.slots", "episodic_memory", (1,)),)

        def provenance(self):
            return ComponentProvenance(ProvenanceKind.LEGACY_VERIFIED, {"mechanism_id": "x", "entry_digest": "x",
                                                                        "module_sha256": "x", "audit_ref": "x"})

        def step(self, ctx):
            return {"episodic_memory.slots": np.zeros(1)}

    class FakePrereg(Fake):
        def provenance(self):
            return ComponentProvenance(ProvenanceKind.EXPERIMENTAL_PREREG, {})

    for cls in (Fake, FakePrereg):
        with pytest.raises(BindingError):
            UnifiedRuntime().bind(cls())
