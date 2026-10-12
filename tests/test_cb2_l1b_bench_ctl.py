"""Billing safety of the L1b benchmark controller (Sol's review S-014), against a fake runpodctl/ssh/scp: nothing here
touches the network or the cloud."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
CTL = REPO / "scripts/explore/cb2_l1b_bench_ctl.py"
FAKE = REPO / "tests/fakes/fake_cloud.py"


@pytest.fixture
def cloud(tmp_path):
    bins = tmp_path / "bin"
    bins.mkdir()
    for n in ("runpodctl", "ssh", "scp"):
        (bins / n).symlink_to(FAKE)
    ex = tmp_path / "ex"
    (ex / "l1b").mkdir(parents=True)
    for f in ("cb2_l1b_code.bundle", "cb2_l1b_code.rev", "cb2_l1b_pod.sh"):
        (ex / "l1b" / f).write_text(f)
    (ex / "l1b" / "cb2_l1b_SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((ex / 'l1b' / f).read_bytes()).hexdigest()}  {f}\n"
        for f in ("cb2_l1b_code.bundle", "cb2_l1b_code.rev", "cb2_l1b_pod.sh")))
    remote = tmp_path / "remote"
    remote.mkdir()
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"pods": {"gjfpfnsfymn8yf": {"id": "gjfpfnsfymn8yf", "name": "cb2-l1", "desiredStatus": "EXITED"}}}))
    env = {**os.environ, "FAKE_STATE": str(state), "FAKE_REMOTE": str(remote), "RP_BIN": str(bins / "runpodctl"),
           "SSH_BIN": str(bins / "ssh"), "SCP_BIN": str(bins / "scp")}

    class C:
        def __init__(self):
            self.ex, self.remote, self.state, self.env = ex, remote, state, env

        def st(self):
            return json.loads(state.read_text())

        def set(self, **kw):
            s = self.st()
            s.update(kw)
            state.write_text(json.dumps(s))

        def run(self, *args, timeout=60):
            fast = ["--poll", "0.05", "--grace", "0", "--verify-polls", "2", "--verify-wait", "0.01", "--ssh-polls", "2",
                    "--ssh-wait", "0.01", "--retries", "2", "--skip-input-sums", "--margin", "0"]
            r = subprocess.run([sys.executable, str(CTL), *args, "--exchange", str(ex), "--key", "k",
                                "--data-dir", str(ex.parent / "wsl-data"), *fast],
                               env=env, capture_output=True, text=True, timeout=timeout)
            return r.returncode, "".join(f.read_text() for f in sorted((ex / "l1b").glob("*_status.txt")))

        def claim(self, pid, **kw):
            import time
            (ex / "l1b" / "bench_claim.json").write_text(json.dumps({"id": pid, "name": "cb2-l1b-bench-claude",
                                                                     "created": time.time(), "costPerHr": 0.34, **kw}))

        def add_pod(self, pid, status="RUNNING"):
            s = self.st()
            s["pods"][pid] = {"id": pid, "name": "cb2-l1b-bench-claude", "desiredStatus": status,
                              "runtimeStatus": "running" if status == "RUNNING" else "stopped", "costPerHr": 0.34}
            self.state.write_text(json.dumps(s))
    return C()


def test_upload_failure_after_create_deletes_the_new_pod_and_verifies_it(cloud):
    cloud.set(upload_fail=True)
    code, log = cloud.run("launch")
    assert code == 1 and "deleted (verified: not_found)" in log
    assert set(cloud.st()["pods"]) == {"gjfpfnsfymn8yf"}                  # only Sol's retained pod is left, untouched
    assert (cloud.ex / "l1b" / "bench_claim_newpod00001.closed.json").exists()


def test_unparseable_create_output_is_reconciled_by_name_then_cleaned_up(cloud):
    cloud.set(create_garbage=True, upload_fail=True)
    code, log = cloud.run("launch")
    assert code == 1 and "reconciled by name: newpod00001" in log and "newpod00001" not in cloud.st()["pods"]


def test_a_delete_that_cannot_be_verified_is_urgent_and_nonzero(cloud):
    cloud.set(upload_fail=True, delete_fail=99)
    code, log = cloud.run("launch")
    assert code == 2 and "URGENT" in log and "newpod00001" in cloud.st()["pods"]


def test_an_unrelated_existing_pod_is_refused_and_never_touched(cloud):
    code, log = cloud.run("launch", "--pod-id", "gjfpfnsfymn8yf")
    assert code == 1 and "REFUSED" in log
    calls = cloud.st()["calls"]
    assert not any(c[:2] in (["pod", "delete"], ["pod", "stop"]) for c in calls)


def test_a_duplicate_claim_is_refused(cloud):
    cloud.add_pod("other0000001")
    code, log = cloud.run("launch")
    assert code == 1 and "already exists" in log and "newpod00001" not in cloud.st()["pods"]


def test_billing_guard_stops_before_any_transfer_and_keeps_the_volume(cloud):
    cloud.add_pod("p1")
    cloud.claim("p1")
    code, log = cloud.run("watch", "--pod-id", "p1", "--maxh", "0")
    assert code == 3 and cloud.st()["pods"]["p1"]["desiredStatus"] == "EXITED"
    assert not any(c[:2] == ["pod", "delete"] for c in cloud.st()["calls"])


def test_finished_bench_is_downloaded_verified_then_deleted(cloud):
    cloud.add_pod("p1")
    cloud.claim("p1")
    (cloud.remote / "l1b-bench.log").write_text("... l1b BENCH finished ...\n")
    (cloud.remote / "cb2_l1b_bench.tgz").write_bytes(b"evidence")
    code, log = cloud.run("watch", "--pod-id", "p1")
    assert code == 0 and "sha256 MATCH" in log and "p1" not in cloud.st()["pods"]
    assert (cloud.ex.parent / "wsl-data" / "cb2_l1b_bench.tgz").read_bytes() == b"evidence"   # bulk stays WSL-side
    assert not (cloud.ex / "l1b" / "cb2_l1b_bench.tgz").exists()


@pytest.mark.parametrize("fault", ["download_fail", "download_hang"])
def test_evidence_transfer_failure_stops_the_pod_and_keeps_storage(cloud, fault):
    cloud.add_pod("p1")
    cloud.claim("p1")
    cloud.set(**{fault: True})
    (cloud.remote / "l1b-bench.log").write_text("l1b BENCH finished\n")
    (cloud.remote / "cb2_l1b_bench.tgz").write_bytes(b"evidence")
    code, log = cloud.run("watch", "--pod-id", "p1", "--download-timeout", "1")
    assert code == 4 and cloud.st()["pods"]["p1"]["desiredStatus"] == "EXITED"


def test_failed_bench_saves_logs_then_deletes(cloud):
    cloud.add_pod("p1")
    cloud.claim("p1")
    cloud.set(running=False)
    (cloud.remote / "l1b-bench.log").write_text("STOP: something\n")
    code, log = cloud.run("watch", "--pod-id", "p1")
    assert code == 5 and "p1" not in cloud.st()["pods"]
    assert (cloud.ex / "l1b" / "bench_pod.log").read_text() == "STOP: something\n"


def test_watcher_refuses_a_pod_that_is_not_our_claim(cloud):
    cloud.add_pod("p1")
    cloud.claim("someone-else")
    code, log = cloud.run("watch", "--pod-id", "p1")
    assert code == 2 and "not our claimed" in log and "p1" in cloud.st()["pods"]


def test_a_pod_that_would_breach_the_cost_ceiling_is_deleted_at_once(cloud):
    cloud.set(rate=1.78)                                       # e.g. a 2-GPU price: 2 h x 1.78 = 3.56 > $3
    code, log = cloud.run("launch", "--maxh", "2", "--max-cost", "3")
    assert code == 1 and "cost check failed" in log and "newpod00001" not in cloud.st()["pods"]


def test_the_cost_ceiling_stops_a_running_bench(cloud):
    import time
    cloud.add_pod("p1")
    cloud.claim("p1", created=time.time() - 3600, costPerHr=3.5)  # one hour at 3.5/h is past a $3 ceiling
    code, log = cloud.run("watch", "--pod-id", "p1", "--maxh", "2", "--max-cost", "3")
    assert code == 3 and cloud.st()["pods"]["p1"]["desiredStatus"] == "EXITED"


def test_launch_hands_off_to_the_watcher_after_upload_and_start(cloud, tmp_path):
    code, log = cloud.run("launch", "--maxh", "2", "--max-cost", "3")
    assert code == 0 and "cost check: 0.34/h" in log and "watcher started" in log
    claim = json.loads((cloud.ex / "l1b" / "bench_claim.json").read_text())
    assert claim["id"] == "newpod00001" and claim["maxh"] == 2.0 and claim["max_cost"] == 3.0
    assert cloud.st().get("bench_started")
    import re
    import signal
    wpid = int(re.search(r"watcher started \(pid (\d+)\)", log).group(1))   # the real watcher it spawned: end it
    os.kill(wpid, signal.SIGTERM)


def test_the_budget_counts_from_creation_not_from_when_the_watcher_starts(cloud):
    """Sol's fixture (S-019): a claim already 3 h old was treated as fresh and downloaded/deleted."""
    import time
    cloud.add_pod("p1")
    cloud.claim("p1", created=time.time() - 3 * 3600)
    (cloud.remote / "l1b-bench.log").write_text("l1b BENCH finished\n")
    (cloud.remote / "cb2_l1b_bench.tgz").write_bytes(b"evidence")
    code, log = cloud.run("watch", "--pod-id", "p1", "--maxh", "2", "--max-cost", "3")
    assert code == 3 and "budget spent" in log and cloud.st()["pods"]["p1"]["desiredStatus"] == "EXITED"
    assert not (cloud.ex.parent / "wsl-data" / "cb2_l1b_bench.tgz").exists()   # no transfer after the deadline


def test_the_deadline_cuts_a_blocked_transfer_and_stops_the_pod(cloud):
    """Expiry DURING a transfer: the download hangs; its timeout is cut to the budget left, then the pod is stopped."""
    import time
    cloud.add_pod("p1")
    cloud.claim("p1", created=time.time() - (2 * 3600 - 3))  # 3 s of a 2 h budget left
    cloud.set(download_hang=True)
    (cloud.remote / "l1b-bench.log").write_text("l1b BENCH finished\n")
    (cloud.remote / "cb2_l1b_bench.tgz").write_bytes(b"evidence")
    t = time.time()
    code, log = cloud.run("watch", "--pod-id", "p1", "--maxh", "2", "--max-cost", "3", "--download-timeout", "3600")
    assert code == 3 and "budget spent" in log and time.time() - t < 20          # not the 30 s hang, not 3600 s
    assert cloud.st()["pods"]["p1"]["desiredStatus"] == "EXITED" and "p1" in cloud.st()["pods"]   # volume kept


def test_a_stop_is_verified_by_the_runtime_not_by_the_intent(cloud):
    """Sol (S-019): desiredStatus EXITED with runtimeStatus still running is not stopped."""
    cloud.add_pod("p1")
    cloud.claim("p1")
    cloud.set(stop_runtime_lags=True)
    code, log = cloud.run("watch", "--pod-id", "p1", "--maxh", "0")
    assert code == 2 and "URGENT" in log and "could NOT be verified stopped" in log


def _run_claim(cloud, pid, **kw):
    import time
    (cloud.ex / "l1b" / "loop_claim.json").write_text(json.dumps({"id": pid, "name": "cb2-l1b-loop-claude", "created": time.time(),
                                                                  "costPerHr": 0.89, **kw}))
    s = cloud.st()
    s["pods"][pid] = {"id": pid, "name": "cb2-l1b-loop-claude", "desiredStatus": "RUNNING", "runtimeStatus": "running",
                      "costPerHr": 0.89}
    cloud.state.write_text(json.dumps(s))


def test_the_loop_phase_has_its_own_pod_name_and_does_not_collide_with_a_bench_claim(cloud):
    cloud.add_pod("benchpod0001")                              # an existing BENCH pod does not block the LOOP launch
    cloud.claim("benchpod0001")
    cloud.set(upload_fail=True)
    code, log = cloud.run("launch", "--phase", "run", "--maxh", "30", "--max-cost", "30")
    created = [c for c in cloud.st()["calls"] if c[:2] == ["pod", "create"]]
    assert created and created[0][created[0].index("--name") + 1] == "cb2-l1b-loop-claude"
    assert "benchpod0001" in cloud.st()["pods"]                 # untouched
    assert (cloud.ex / "l1b" / "loop_status.txt").exists()


def test_a_finished_loop_is_downloaded_to_wsl_with_its_progress_record_then_deleted(cloud):
    _run_claim(cloud, "lp1")
    (cloud.remote / "l1b-run-phase.log").write_text("== l1b LOOP finished -- STOP\n")
    (cloud.remote / "cb2_l1b_loop.tgz").write_bytes(b"loop evidence")
    (cloud.remote / "cycles.jsonl").write_text('{"cycle": 0}\n')
    code, log = cloud.run("watch", "--phase", "run", "--pod-id", "lp1", "--maxh", "30", "--max-cost", "30")
    assert code == 0 and "lp1" not in cloud.st()["pods"]
    assert (cloud.ex.parent / "wsl-data" / "cb2_l1b_loop.tgz").read_bytes() == b"loop evidence"
    assert (cloud.ex / "l1b" / "loop_cycles.jsonl").exists()


def test_a_failed_loop_keeps_its_volume_it_is_stopped_not_deleted(cloud):
    _run_claim(cloud, "lp1")
    cloud.set(running=False)
    (cloud.remote / "l1b-run-phase.log").write_text("TRIPWIRE: learning sleep 4\n")
    code, log = cloud.run("watch", "--phase", "run", "--pod-id", "lp1", "--maxh", "30", "--max-cost", "30")
    assert code == 5 and cloud.st()["pods"]["lp1"]["desiredStatus"] == "EXITED" and "volume kept" in log
