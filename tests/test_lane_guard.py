"""Lane guard: on an exploration branch, the files the protected lane froze must equal its base commit.

See LANES.md. The protected lane (preregistered I1 work) is changed only through reviewed patches on its own
branch; an exploration branch builds on it but may not edit it. Skipped outside a git checkout (for example in
the lifecycle suite's copies), and on a branch whose LANES.json says it is not an exploration lane."""
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LANES = REPO / "LANES.json"


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True)


def _cfg():
    if not LANES.exists():
        pytest.skip("no LANES.json: not an exploration branch")
    cfg = json.loads(LANES.read_text())
    if cfg.get("lane") != "explore":
        pytest.skip("not an exploration lane")
    if not (REPO / ".git").exists() or _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout")
    return cfg


def test_the_base_commit_is_on_the_protected_lane():
    cfg = _cfg()
    assert _git("cat-file", "-e", cfg["base_commit"] + "^{commit}").returncode == 0, "base_commit is not in this repository"
    assert _git("merge-base", "--is-ancestor", cfg["base_commit"], "HEAD").returncode == 0, \
        "this branch does not contain the protected base commit"


def test_exploration_leaves_the_protected_files_alone():
    cfg = _cfg()
    paths = cfg["protected_paths"]
    changed = _git("diff", "--name-only", cfg["base_commit"], "--", *paths)
    assert changed.returncode == 0, changed.stderr
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *paths)
    touched = sorted(set(changed.stdout.split()) | set(untracked.stdout.split()))
    assert touched == [], f"exploration changed frozen protected-lane files: {touched}"
