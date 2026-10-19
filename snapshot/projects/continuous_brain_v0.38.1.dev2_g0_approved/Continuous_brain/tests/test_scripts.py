import importlib.util
import json
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runpod_plan_prints_commands_and_opens_no_connection(monkeypatch, capsys):
    def refuse(*a, **k):
        raise AssertionError("runpod_plan must not open network connections")
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    assert load("scripts/cloud/runpod_plan.py").main(["--job", "battery", "--replicates", "2", "--gpu-hourly", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "nothing has been launched" in out and "battery_smoke.py" in out


def test_unified_boot_reports_every_slot_and_checkpoints(tmp_path):
    assert load("scripts/unified_boot.py").main(["--ticks", "5", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "binding_report.json").read_text())
    assert len(report["binding"]) == 15 and report["scientific"] is True
    assert Path(report["checkpoint"]).is_dir()


def test_scripts_contain_no_personal_absolute_paths_in_code():
    for p in list((REPO / "scripts").rglob("*.py")) + list((REPO / "continuous_brain_unified").rglob("*.py")):
        text = p.read_text()
        assert "/home/davei" not in text and "C:\\Users" not in text, p
