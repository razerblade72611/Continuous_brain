"""The E0 gate script itself: runs, passes on CPU, and registers its evidence."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e0_gate  # noqa: E402

from continuous_brain_snn.provenance import RunRegistry  # noqa: E402


def test_gate_passes_and_registers_artifacts(tmp_path):
    code = e0_gate.main(["--small", "--stability-s", "2", "--determinism-steps", "100",
                         "--parity-s", "2", "--out", str(tmp_path)])
    assert code == 0
    runs = RunRegistry(tmp_path).all()
    assert len(runs) == 1 and runs[0]["status"] == "PASS"
    report = next(p for p in runs[0]["artifacts"] if p.endswith("e0_gate_report.json"))
    body = json.loads(Path(report).read_text())
    assert body["result"] == "INFRASTRUCTURE PASS"
    assert all(v["passed"] for v in body["checks"].values() if v["passed"] is not None)


def test_registry_refuses_unevidenced_claims(tmp_path):
    import pytest
    from continuous_brain_snn.provenance import RunRecord
    rec = RunRecord(run_id="x", kind="e0_gate", linear_issue="DAV-5", seed=0, config_hash="c",
                    anatomy_hash="a", device="cpu", status="PASS")
    with pytest.raises(ValueError):
        RunRegistry(tmp_path).register(rec)
