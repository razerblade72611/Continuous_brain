#!/usr/bin/env python3
"""Run the full test suite with the I1 preregistration in every legal lifecycle state.

usage: python scripts/dev/i1_lifecycle_suite.py [OUT_JSON]

For each state a temporary copy of this checkout is made (the checkout itself is never modified) and
put into that state with the project's own code paths:
  PROPOSED                  the checkout as it is (expected to be PROPOSED here)
  PREFLIGHT_SPENT           + a preflight opened by runner.open_family() in the copy (a real spend of
                              i1-v1-preflight-eng-b in the copy's ledger, source
                              I1-INTEGRATED-INDIVIDUAL-v1:preflight:lifecycle_preflight:opened, and
                              runs/i1/lifecycle_preflight/seed_opened.json with the registry binding), with
                              a synthetic sealed report written by evaluate.seal_report() (tests/i1_fixtures.py)
  APPROVED                  + scripts/i1_approve.py run on that preflight directory (the full chain check)
  SCIENCE_SPENT             + i1-v1-dev registered and spent in the copy's ledger with the science source
Then `python -m pytest -q -p no:cacheprovider tests/` runs in the copy. Exit code 0 only if every state
is green. No arm is run: this checks the repository's tests and lifecycle invariants, not an experiment."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATES = ("PROPOSED", "PREFLIGHT_SPENT", "APPROVED", "SCIENCE_SPENT")


def _env(copy: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=f"{copy}{os.pathsep}{copy / 'tests'}", PYTHONDONTWRITEBYTECODE="1")
    env.pop("CONTINUOUS_BRAIN_REGISTRY", None)
    return env


def _py(copy: Path, code: str) -> str:
    r = subprocess.run([sys.executable, "-c", code], cwd=copy, env=_env(copy), text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return r.stdout


def _open_preflight(copy: Path) -> None:
    _py(copy, f"""
from i1_fixtures import synthetic_preflight
synthetic_preflight({str(copy)!r}, "lifecycle_preflight")
""")


def _spend_science(copy: Path) -> None:
    _py(copy, """
import json
from continuous_brain_eval.seeds import SeedRegistry
from continuous_brain_protocols import SeedFamily, SeedPurpose
from continuous_brain_unified.integration import runner
f = json.load(open(runner.PREREG_FILE))['families']['scientific_dev_proposed']
fam = SeedFamily(f['name'], f['start'], f['stop'], SeedPurpose(f['purpose']))
reg = SeedRegistry.load(runner.registry_root(runner.REPO))
reg.check_usable(fam)
try:
    reg.get(fam.name)
except KeyError:
    reg.register(fam, source=f"{runner.PREREG_ID}:{fam.name}", note='simulated in a temporary copy')
reg.spend(fam, source=runner.spend_source('science', 'lifecycle_science'))
""")


def _approve(copy: Path) -> None:
    r = subprocess.run([sys.executable, "scripts/i1_approve.py", "--preflight-dir", "runs/i1/lifecycle_preflight",
                        "--approver", "David", "--i-approve", "I1-INTEGRATED-INDIVIDUAL-v1"], cwd=copy, env=_env(copy),
                       text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for i, state in enumerate(STATES):
            copy = tmp / f"s{i}_{state}" / "Continuous_brain"
            shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns("runs", "__pycache__", ".pytest_cache", ".git"))
            if state in ("PREFLIGHT_SPENT", "APPROVED", "SCIENCE_SPENT"):
                _open_preflight(copy)
            if state in ("APPROVED", "SCIENCE_SPENT"):
                _approve(copy)
            if state == "SCIENCE_SPENT":
                _spend_science(copy)
            status = json.loads((copy / "registry/i1_integration_v1.json").read_text())["status"]
            env = dict(os.environ, PYTHONPATH=str(copy), PYTHONDONTWRITEBYTECODE="1")
            env.pop("CONTINUOUS_BRAIN_REGISTRY", None)
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/"], cwd=copy,
                               env=env, text=True, capture_output=True)
            tail = (r.stdout.strip().splitlines() or [""])[-1]
            results.append({"state": state, "prereg_status": status, "returncode": r.returncode, "summary": tail})
            print(f"{state:16s} status={status:9s} rc={r.returncode}  {tail}", flush=True)
            if r.returncode != 0:
                print(r.stdout[-4000:])
    ok = all(x["returncode"] == 0 for x in results)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps({"format": "cb-i1-lifecycle-suite/1", "all_green": ok,
                                                 "results": results}, indent=2) + "\n")
    print("ALL STATES GREEN" if ok else "SOME STATE NOT GREEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
