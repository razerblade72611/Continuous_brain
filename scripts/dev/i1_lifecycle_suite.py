#!/usr/bin/env python3
"""Run the full test suite with the I1 preregistration in every legal lifecycle state.

usage: python scripts/dev/i1_lifecycle_suite.py [OUT_JSON]

For each state a temporary copy of this checkout is made (the checkout itself is never modified) and
put into that state with the project's own code paths:
  PROPOSED                  the checkout as it is (expected to be PROPOSED here)
  PREFLIGHT_SPENT           + i1-v1-preflight-eng-b registered and spent in the copy's local ledger
  APPROVED                  + scripts/i1_approve.py run on a preflight directory that carries this
                              machine's manifest and an all-true validity block
  SCIENCE_SPENT             + i1-v1-dev registered and spent in the copy's local ledger
Then `python -m pytest -q -p no:cacheprovider tests/` runs in the copy. Exit code 0 only if every state
is green. The preflight directory here is synthetic (no arm is run): this checks the repository's
tests and lifecycle invariants, not an experiment."""
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


def _py(copy: Path, code: str) -> str:
    env = dict(os.environ, PYTHONPATH=str(copy), PYTHONDONTWRITEBYTECODE="1")
    env.pop("CONTINUOUS_BRAIN_REGISTRY", None)
    r = subprocess.run([sys.executable, "-c", code], cwd=copy, env=env, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return r.stdout


def _spend(copy: Path, key: str) -> None:
    _py(copy, f"""
import json
from continuous_brain_eval.seeds import SeedRegistry
from continuous_brain_protocols import SeedFamily, SeedPurpose
f = json.load(open('registry/i1_integration_v1.json'))['families']['{key}']
fam = SeedFamily(f['name'], f['start'], f['stop'], SeedPurpose(f['purpose']))
reg = SeedRegistry.load()
reg.check_usable(fam)
try:
    reg.get(fam.name)
except KeyError:
    reg.register(fam, source='lifecycle-suite', note='simulated in a temporary copy')
reg.spend(fam, source='lifecycle-suite:simulated')
""")


def _approve(copy: Path, tmp: Path) -> None:
    pf = json.loads((copy / "registry/i1_integration_v1.json").read_text())["families"]["engineering_preflight_proposed"]
    d = tmp / "synthetic_preflight"
    d.mkdir()
    (d / "seed_opened.json").write_text(json.dumps({"mode": "preflight", "family": {**pf, "spent": True}}))
    # the validity schema and the manifest come from the copy's own code (no hard-coded list here)
    _py(copy, f"""
import json
from continuous_brain_unified.integration.evaluate import VALIDITY_KEYS
from continuous_brain_unified.integration.runner import manifest
m = manifest(2)
open({str(d / 'manifest.json')!r}, 'w').write(json.dumps(m))
open({str(d / 'i1_validity.json')!r}, 'w').write(json.dumps({{
    "valid": True, "validity": {{k: True for k in VALIDITY_KEYS}},
    "validity_diagnostics": {{"expected_manifest_sha256": m["manifest_sha256"],
                              "arm_manifests": {{"NORMAL": m["manifest_sha256"], "DONOR": m["manifest_sha256"]}}}}}}))
""")
    env = dict(os.environ, PYTHONPATH=str(copy))
    env.pop("CONTINUOUS_BRAIN_REGISTRY", None)
    r = subprocess.run([sys.executable, "scripts/i1_approve.py", "--preflight-dir", str(d), "--approver", "David",
                        "--i-approve", "I1-INTEGRATED-INDIVIDUAL-v1"], cwd=copy, env=env, text=True, capture_output=True)
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
                _spend(copy, "engineering_preflight_proposed")
            if state in ("APPROVED", "SCIENCE_SPENT"):
                _approve(copy, copy.parent)
            if state == "SCIENCE_SPENT":
                _spend(copy, "scientific_dev_proposed")
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
