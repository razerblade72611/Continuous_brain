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
                    "--ssh-wait", "0.01", "--retries", "2", "--skip-input-sums"]
            r = subprocess.run([sys.executable, str(CTL), *args, "--exchange", str(ex), "--key", "k", *fast],
                               env=env, capture_output=True, text=True, timeout=timeout)
            return r.returncode, (ex / "l1b" / "bench_status.txt").read_text() if (ex / "l1b" / "bench_status.txt").exists() else ""

        def claim(self, pid):
            (ex / "l1b" / "bench_claim.json").write_text(json.dumps({"id": pid, "name": "cb2-l1b-bench-claude"}))

        def add_pod(self, pid, status="RUNNING"):
            s = self.st()
            s["pods"][pid] = {"id": pid, "name": "cb2-l1b-bench-claude", "desiredStatus": status,
                              "ssh": {"ip": "10.0.0.1", "port": 2222}}
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
    assert (cloud.ex / "l1b" / "cb2_l1b_bench.tgz").read_bytes() == b"evidence"


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
