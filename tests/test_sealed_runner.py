"""Regression tests for the sealed runner's open/resume/lock logic (Sol, DAV-104).

Each test copies the working tree (code, configs, data, tests, PREREG; no results, no model) into a temporary git
repo and drives the real scripts/run_v2.py through tests/fake_runner.py. Nothing is trained and no data file is read.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
SEEDS = list(range(1000, 1020))


@pytest.fixture
def repo(tmp_path):
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=SRC, capture_output=True,
                           text=True, check=True).stdout.split()
    for f in files:
        if f.startswith("results/") or not (SRC / f).is_file():
            continue
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / f, tmp_path / f)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-qm", "base"]):
        subprocess.run(cmd, cwd=tmp_path, check=True, env=env)
    return tmp_path


def go(repo, *args, wait=True, **env):
    cmd = [sys.executable, "-I", str(repo / "tests/fake_runner.py"), str(repo), "sealed", *args]
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    if not wait:
        return subprocess.Popen(cmd, cwd=repo, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, cwd=repo, env=e, capture_output=True, text=True)


def results(repo):
    return json.loads((repo / "results/v2_sealed.json").read_text())


def assert_complete(repo):
    d = results(repo)
    assert d["status"] == "complete" and [r["seed"] for r in d["per_seed"]] == SEEDS
    assert d["summary"]["checks"]["backbone_unchanged"]
    return d


def test_crash_during_first_seed_is_resumable(repo):
    assert go(repo, CRASH_AT=1000).returncode == 137
    assert results(repo)["per_seed"] == []          # empty resumable state was written before the model loaded
    assert go(repo).returncode != 0                 # a fresh reopen is refused
    r = go(repo, "--resume")
    assert r.returncode == 0, r.stderr
    assert_complete(repo)


def test_crash_before_initial_write_is_resumable(repo):
    assert go(repo, CRASH_FIRST_WRITE=1).returncode == 137
    assert (repo / "results/v2_sealed.OPENED").exists() and not (repo / "results/v2_sealed.json").exists()
    r = go(repo, "--resume")
    assert r.returncode == 0, r.stderr
    assert_complete(repo)


def test_completed_rows_preserved_verbatim(repo):
    assert go(repo, CRASH_AT=1005).returncode == 137
    before = results(repo)["per_seed"]
    assert [r["seed"] for r in before] == SEEDS[:5]
    assert go(repo, "--resume", "--max-seeds", "3").returncode == 0
    assert go(repo, "--resume").returncode == 0
    d = assert_complete(repo)
    assert d["per_seed"][:5] == before              # nonces unchanged: no recorded seed was rerun
    assert len(d["segments"]) == 3


def test_kill_between_row_write_and_marker_log(repo):
    assert go(repo, CRASH_BEFORE_LOG=1002).returncode == 137
    before = results(repo)["per_seed"]
    assert [r["seed"] for r in before] == SEEDS[:3]
    assert go(repo, "--resume").returncode == 0
    d = assert_complete(repo)
    assert d["per_seed"][:3] == before
    assert "logged_on_resume" in (repo / "results/v2_sealed.OPENED").read_text()


def test_competing_resume_refused(repo):
    assert go(repo, CRASH_AT=1001).returncode == 137
    first = go(repo, "--resume", wait=False, SLEEP=0.3)
    marker = repo / "results/v2_sealed.OPENED"
    for _ in range(100):
        if marker.read_text().count('"resume": true') == 1:
            break
        time.sleep(0.05)
    second = go(repo, "--resume")
    assert second.returncode != 0 and "another sealed process" in second.stderr
    out, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    assert_complete(repo)
    assert marker.read_text().count('"resume": true') == 1   # the refused process wrote nothing


def test_tampered_or_deleted_results_refused(repo):
    assert go(repo, CRASH_AT=1004).returncode == 137
    p = repo / "results/v2_sealed.json"
    d = results(repo)
    d["per_seed"][1]["nonce"] = "edited"
    p.write_text(json.dumps(d))
    r = go(repo, "--resume")
    assert r.returncode != 0 and "does not match the marker" in r.stderr
    p.unlink()
    r = go(repo, "--resume")
    assert r.returncode != 0 and "missing but the marker logs" in r.stderr


def test_resume_refused_after_code_change(repo):
    assert go(repo, CRASH_AT=1002).returncode == 137
    with open(repo / "being/memory.py", "a") as f:
        f.write("# change\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "commit", "-qam", "chg"], cwd=repo, check=True, env=env)
    r = go(repo, "--resume")
    assert r.returncode != 0 and "differ from the opened commit" in r.stderr
