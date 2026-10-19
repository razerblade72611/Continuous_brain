import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import continuous_brain_unified
from continuous_brain_unified import ChannelDecl, UnifiedRuntime
from continuous_brain_unified.prereg import PreregistrationError, PreregistrationRegistry
from continuous_brain_unified.workspace import (AttentionSchemaMechanism, WorkspaceMechanism,
                                                experimental_provenance, register_experimental)


class CandidateWorkspace(WorkspaceMechanism):
    """A test-only concrete subclass; the package itself ships none (C4)."""
    name = "candidate_workspace"
    writes = (ChannelDecl("workspace.state", "workspace", (2,)),)

    @property
    def capacity(self):
        return 1

    @property
    def candidate_channels(self):
        return ()

    def provenance(self):
        return experimental_provenance(self)

    def step(self, ctx):
        return {"workspace.state": np.zeros(2)}


def test_repository_only_g0_preregistration_unlocks():
    reg = PreregistrationRegistry.load()
    assert {"E2-PREREG-v1", "A1-PROPOSED", "H1H2-BATTERY-v0", "UI-GATE-v0"} <= set(reg.entries)
    # G0 is approved; I1 unlocks only once David's approval has been recorded (lifecycle-dependent)
    i1 = json.loads((Path(__file__).resolve().parents[1] / "registry/i1_integration_v1.json").read_text())
    expected = {"G0-LEGACY-REQUAL-v1"} | ({"I1-INTEGRATED-INDIVIDUAL-v1"} if i1["status"] in ("APPROVED", "FROZEN") else set())
    assert {p.id for p in reg.entries.values() if p.unlocks} == expected
    with pytest.raises(PreregistrationError, match="DRAFT"):
        reg.require("UI-GATE-v0")
    with pytest.raises(PreregistrationError):
        register_experimental(CandidateWorkspace, "UI-GATE-v0", reg)


def test_an_approved_preregistration_admits_a_mechanism_under_the_default_policy(tmp_path):
    (tmp_path / "preregistrations.json").write_text(json.dumps({"preregistrations": [
        {"id": "WS-1", "title": "workspace", "status": "APPROVED", "approved_by": "David", "allowed_slots": ["workspace"]},
        {"id": "WS-2", "title": "other slot", "status": "APPROVED", "approved_by": "David", "allowed_slots": ["planner"]},
        {"id": "WS-3", "title": "no approver", "status": "DRAFT"}]}))
    reg = PreregistrationRegistry.load(tmp_path)
    with pytest.raises(PreregistrationError, match="does not cover"):
        register_experimental(CandidateWorkspace, "WS-2", reg)
    register_experimental(CandidateWorkspace, "WS-1", reg)
    rt = UnifiedRuntime()
    rt.bind(CandidateWorkspace())
    assert rt.scientific
    assert rt.binding_report()[1].provenance["detail"]["prereg_id"] == "WS-1"


def test_approved_without_approver_is_rejected(tmp_path):
    (tmp_path / "preregistrations.json").write_text(json.dumps({"preregistrations": [
        {"id": "X", "title": "x", "status": "APPROVED"}]}))
    with pytest.raises(PreregistrationError, match="approved_by"):
        PreregistrationRegistry.load(tmp_path)


def test_package_ships_no_workspace_or_attention_schema_implementation():
    """C4: interfaces and a registration point only."""
    import pkgutil, importlib
    found = []
    for m in pkgutil.walk_packages(continuous_brain_unified.__path__, "continuous_brain_unified."):
        mod = importlib.import_module(m.name)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__ and obj not in (WorkspaceMechanism, AttentionSchemaMechanism) and \
                    issubclass(obj, (WorkspaceMechanism, AttentionSchemaMechanism)):
                found.append(obj)
    assert found == []
