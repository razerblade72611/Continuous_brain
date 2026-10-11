"""The final phase must work on a fresh CPU pod that received only cb2_l1b_loop.tgz (Sol, S-013): every log a later
phase needs travels inside the run directory.  Runs the real packing functions of scripts/explore/cb2_l1b_pack.sh."""
import subprocess
import tarfile
from pathlib import Path

LIB = Path(__file__).parents[1] / "scripts/explore/cb2_l1b_pack.sh"
OUT = "runs/explore/cb2/l1b"


def sh(script: str):
    return subprocess.run(["bash", "-c", f"set -euo pipefail; . {LIB}; {script}"], capture_output=True, text=True)


def test_fresh_cpu_pod_restores_the_loop_and_packs_the_final_results(tmp_path):
    w1, c1 = tmp_path / "gpu-ws", tmp_path / "gpu-code"           # the loop's pod
    (c1 / OUT).mkdir(parents=True)
    (c1 / OUT / "run.json").write_text("{}")
    w1.mkdir()
    (w1 / "l1b-run.log").write_text("loop log\n")
    assert sh(f"l1b_pack_loop {w1} {c1} {OUT}").returncode == 0
    w2, c2 = tmp_path / "cpu-ws", tmp_path / "cpu-code"           # a fresh CPU pod: only the loop tarball arrives
    w2.mkdir(); c2.mkdir()
    (w2 / "cb2_l1b_loop.tgz").write_bytes((w1 / "cb2_l1b_loop.tgz").read_bytes())
    r = sh(f"l1b_restore_loop {w2} {c2} {OUT}")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (c2 / OUT / "logs/run.log").read_text() == "loop log\n"
    (c2 / "evidence/explore/cb2").mkdir(parents=True)
    (c2 / "evidence/explore/cb2/l1b-report.json").write_text("{}")
    (w2 / "l1b-final.log").write_text("final log\n")
    r = sh(f"l1b_pack_final {w2} {c2} {OUT}")
    assert r.returncode == 0, r.stdout + r.stderr
    names = tarfile.open(w2 / "cb2_l1b_results.tgz").getnames()
    for want in (f"{OUT}/run.json", f"{OUT}/logs/run.log", f"{OUT}/logs/final.log", "evidence/explore/cb2/l1b-report.json"):
        assert want in names


def test_restore_without_loop_results_stops_loudly(tmp_path):
    r = sh(f"l1b_restore_loop {tmp_path} {tmp_path} {OUT}")
    assert r.returncode != 0 and "STOP" in r.stdout
