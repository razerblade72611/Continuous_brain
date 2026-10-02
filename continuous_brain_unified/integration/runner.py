"""I1 orchestration: dev, preflight and science modes.

  dev        a slice of one registered ENGINEERING family (e.g. cloud-dev-claude-eng); every
             individual id must lie inside that family, and the slice (and its derived ids) may
             not touch any other registered family except an engineering reserve containing it,
             nor the I1 preregistered families; full report; ENGINEERING_ONLY
  preflight  the preregistered single-use engineering family (engineering_preflight_proposed);
             registered and spent before launch; primary/secondary values withheld from display
             and committed by SHA-256 (not encrypted); writes the manifest that the
             preregistration must cite before approval
  science    the preregistered DEV family i1-v1-dev; refuses unless I1-INTEGRATED-INDIVIDUAL-v1 is
             APPROVED, the approval names the SHA-256 of the design it approves (design_digest)
             and that design is unchanged, the operator types --confirm-open, and the manifest
             (code, legacy pins, historical core, environment) equals the one frozen in the
             preregistration; the family is spent before launch; the verdict is PASS / FAIL / BLOCKED

Before any family is registered or spent, every mode verifies the legacy pins and the
historical-core archive. Once a family is spent, any failure still writes a BLOCKED report.
The legacy tree's metadata is compared before and after. Arm order: LEGACY_REF, NORMAL, then
(ticks and birth autobiographies from NORMAL) FIXED_SCAFFOLD, ERASED, DONOR, ROLLBACK,
PERSIST_SAVE, PERSIST_RESUME, AUTHORITY_SKILLS, AUTHORITY_GOAL, AUTHORITY_CURRICULUM.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from continuous_brain_protocols import SeedFamily, SeedPurpose, sha256_file
from continuous_brain_eval.seeds import SeedRegistry
from continuous_brain_unified.prereg import PreregistrationRegistry

from ..composition import boundary as _b, codec as _c, route as _r, spec as _s
from ..legacy.g0_runner import (_extract_historical_core, _historical_core_digest, _tree_metadata,
                                load_dependency_closure, load_historical_core)
from . import evaluate as _e, hosts as _h, life as _l, route as _ir

PREREG_ID = "I1-INTEGRATED-INDIVIDUAL-v1"
PREREG_FILE = "registry/i1_integration_v1.json"
SELF_LINEAGE = ["recurrent_self_trajectory_v031", "autobiographical_discrepancy_v0322",
                "event_consistent_self_development_v0341", "endogenous_goal_genesis_v0353",
                "recursive_autobiographical_curriculum_v0361"]
PINNED_CHECKPOINTS = {
    "state/recurrent_self_trajectory_v1/recurrent_self_trajectory.pt":
        "1f6997c8d36c96557772b42af964894afce7e0eba04661ac4b7dd68ccfa99457",
    "state/autobiographical_self_discrepancy_neutral_v1/neutral_autobiographical_self_model.pt":
        "df850057cb672e3a5dcb82afe477b24a8df76c67f13383fc449e59072e8b39f1",
}
BOOT = '''import importlib.abc, importlib.util, pathlib, sys
sys.dont_write_bytecode=True
import torch
torch.set_num_threads({threads})
OVERRIDES={overrides!r}
class _HistoricalCoreFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        p=OVERRIDES.get(fullname)
        if p is None: return None
        return importlib.util.spec_from_file_location(fullname,p)
sys.meta_path.insert(0,_HistoricalCoreFinder())
sys.path.insert(0,{root!r})
sys.path.insert(1,{repo!r})
from continuous_brain_unified.integration.route import run_arm
raise SystemExit(run_arm(arm={arm!r}, sids={sids!r}, out_dir={out!r}, legacy_root={root!r}, plan_path={plan!r}))
'''
REPO = Path(__file__).resolve().parents[2]
CORE_MANIFEST = REPO / "legacy" / "historical_core_v0122.json"
DERIVED_OFFSETS = (70000, 90000)        # bootstrap records and episodes are seeded at sid+70000 / sid+90000
AUTHORITY_TARGETS = {"AUTHORITY_SKILLS": "skills.state", "AUTHORITY_GOAL": "goal_genesis.state",
                     "AUTHORITY_CURRICULUM": "curriculum.state"}
AUTHORITY_N = 4                          # first-goal completers per authority control (lowest ids)
LATER_ARMS = ("FIXED_SCAFFOLD", "ERASED", "DONOR", "ROLLBACK", "PERSIST_SAVE", "PERSIST_RESUME",
              *AUTHORITY_TARGETS)


class I1Refused(RuntimeError):
    pass


def legacy_pins() -> dict[str, str]:
    closures = load_dependency_closure(str(REPO / "legacy/g0_dependency_closure.json"))
    pins = {}
    for m in SELF_LINEAGE:
        for f in closures[m]:
            pins[f["relpath"]] = f["sha256"]
    pins.update(PINNED_CHECKPOINTS)
    return dict(sorted(pins.items()))


def verify_pins(legacy: Path) -> list[str]:
    problems = []
    for rel, sha in legacy_pins().items():
        p = legacy / rel
        if not p.is_file():
            problems.append(f"missing {rel}")
        elif sha256_file(p) != sha:
            problems.append(f"hash mismatch {rel}")
    return problems


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


CODE_PACKAGES = ("continuous_brain_protocols", "continuous_brain_snn", "continuous_brain_substrates",
                 "continuous_brain_unified", "continuous_brain_eval")
CODE_FILES = ("scripts/i1_preflight.py", "scripts/i1_run.py", "scripts/dev/i1_dev.py", "pyproject.toml",
              "legacy/g0_dependency_closure.json", "legacy/historical_core_v0122.json", "legacy/g0_recipes.json")


def code_files() -> dict[str, str]:
    """Every Python file of every package plus the I1 entry points, keyed by repo-relative path."""
    out = {}
    for pkg in CODE_PACKAGES:
        for f in sorted((REPO / pkg).rglob("*.py")):
            if "__pycache__" not in f.parts:
                out[str(f.relative_to(REPO))] = sha256_file(f)
    for rel in CODE_FILES:
        out[rel] = sha256_file(REPO / rel)
    return dict(sorted(out.items()))


def design_digest(prereg: dict) -> str:
    """SHA-256 of the parts of the preregistration an approval commits to."""
    keys = ("id", "question", "scope", "design", "decision_rule", "families", "procedure")
    return hashlib.sha256(json.dumps({k: prereg.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()


def manifest(threads: int) -> dict:
    """Everything that must be identical between the preflight and the scientific run."""
    files = code_files()
    pins = legacy_pins()
    m = {"format": "cb-i1-manifest/2", "prereg_id": PREREG_ID,
         "code_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
         "code_files": len(files), "code": files, "spec_digest": _s.spec_digest(), "codec_version": _c.CODEC_VERSION,
         "legacy_pins_sha256": hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest(),
         "legacy_pins_count": len(pins),
         "historical_core_archive_sha256": load_historical_core(str(CORE_MANIFEST))["archive_sha256"],
         "historical_core_manifest_digest": _historical_core_digest(str(CORE_MANIFEST)),
         "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                         "cpu": cpu_model(), "threads": int(threads), "machine": platform.machine()}}
    m["manifest_sha256"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    return m


def _run_arm(arm, sids, run_dir, legacy, core, threads, plan):
    out = run_dir / arm
    boot = run_dir / f"bootstrap_{arm}.py"
    boot.write_text(BOOT.format(threads=threads, overrides=core, root=str(legacy), repo=str(REPO), arm=arm,
                                sids=sids, out=str(out), plan=None if plan is None else str(plan)))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    t = time.time()
    proc = subprocess.run([sys.executable, "-B", str(boot)], cwd=legacy, env=env, text=True, capture_output=True)
    (run_dir / f"{arm}.stdout.txt").write_text(proc.stdout)
    (run_dir / f"{arm}.stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise I1Refused(f"arm {arm} failed (rc {proc.returncode}): {proc.stderr[-2000:]}")
    lives = [json.loads(x) for x in (out / "lives.jsonl").read_text().splitlines() if x.strip()]
    print(json.dumps({"arm": arm, "lives": len(lives), "errors": sum(1 for r in lives if "error" in r),
                      "seconds": round(time.time() - t, 1)}), flush=True)
    return lives


def _plans(normal: dict[int, dict]) -> dict[str, dict]:
    plans: dict[str, dict] = {"ERASED": {}, "DONOR": {}, "ROLLBACK": {}, **{a: {} for a in AUTHORITY_TARGETS}}
    for sid, r in normal.items():
        rec = r.get("record")
        if rec is None:
            continue
        t_imp = rec["marks"]["imprint"]
        plans["ERASED"][str(sid)] = [{"kind": "erase", "target": "autobiography.state", "start_tick": t_imp,
                                      "end_tick": t_imp + 1, "why": "erase autobiography after birth imprint"}]
        partner = normal.get(sid ^ 1, {}).get("record")
        if partner is not None:
            plans["DONOR"][str(sid)] = [{"kind": "replace", "target": "autobiography.state", "start_tick": t_imp,
                                         "end_tick": t_imp + 1, "payload_b64": partner["birth_state_b64"],
                                         "why": f"donor birth autobiography of individual {sid ^ 1}"}]
        if rec["first_completed"] and "before_second" in rec["marks"]:
            t_b = rec["marks"]["before_second"]
            ng = rec.get("second_goal")
            plans["ROLLBACK"][str(sid)] = {
                "interventions": [{"kind": "replace", "target": "autobiography.state", "start_tick": t_b,
                                   "end_tick": t_b + 1, "payload_b64": rec["birth_state_b64"],
                                   "why": "roll autobiography back to birth before the second-goal decision"}],
                "compare": {"normal_second_goal": None if ng is None else
                            {k: v for k, v in ng.items() if k != "__type__"}}}
    completers = sorted(s for s, r in normal.items() if r.get("record") and r["record"]["first_completed"]
                        and "before_second" in r["record"]["marks"])
    for arm, target in AUTHORITY_TARGETS.items():
        for sid in completers[:AUTHORITY_N]:
            t_b = normal[sid]["record"]["marks"]["before_second"]
            plans[arm][str(sid)] = [{"kind": "erase", "target": target, "start_tick": t_b, "end_tick": t_b + 1,
                                     "why": f"authority control: erase {target} before the second-goal decision"}]
    return plans


def _family(reg: SeedRegistry, spec: dict) -> SeedFamily:
    fam = SeedFamily(spec["name"], int(spec["start"]), int(spec["stop"]), SeedPurpose(spec["purpose"]))
    try:
        existing = reg.get(fam.name)
    except KeyError:
        reg.register(fam, source=f"{PREREG_ID}:{fam.name}", note="registered by the I1 runner")
        return fam
    if (existing.start, existing.stop, existing.purpose) != (fam.start, fam.stop, fam.purpose):
        raise I1Refused(f"registered family {fam.name} does not match the preregistration")
    return existing


def _dev_sids(reg: SeedRegistry, prereg: dict, family: str | None, start: int | None, n: int | None) -> tuple:
    """Dev individuals: a slice of exactly one registered ENGINEERING family, touching nothing else."""
    if family is None:
        raise I1Refused("dev mode needs --family")
    fam = reg.get(family)
    if fam.purpose != SeedPurpose.ENGINEERING or fam.sealed or fam.spent:
        raise I1Refused(f"dev mode runs on an unspent, unsealed ENGINEERING family ({family} is not)")
    if n is None or start is None or n < 2 or n % 2 or start % 2:
        raise I1Refused("--start and --n must be even (donor arms pair even/odd individuals)")
    lo, hi = int(start), int(start) + int(n) - 1
    if lo < fam.start or hi > fam.stop:
        raise I1Refused(f"individuals {lo}-{hi} are not inside {family} [{fam.start}-{fam.stop}]")
    reserved = [(f["name"], int(f["start"]), int(f["stop"])) for k, f in prereg["families"].items()
                if isinstance(f, dict) and "start" in f]
    offsets = (0, *DERIVED_OFFSETS)
    for off in offsets:
        a, b = lo + off, hi + off
        for name, fs, ft in reserved:            # base and derived ids of the I1 families, all offset pairs
            for po in offsets:
                if a <= ft + po and fs + po <= b:
                    raise I1Refused(f"ids {a}-{b} (offset {off}) overlap ids of the I1 preregistered family "
                                    f"{name} (offset {po})")
        for r in reg.ranges:
            f = r.family
            if not (a <= f.stop and f.start <= b):
                continue
            if off == 0 and f.name == fam.name:
                continue
            parent = (r.reserve and f.purpose == SeedPurpose.ENGINEERING and f.start <= fam.start
                      and fam.stop <= f.stop and not f.sealed)
            if parent and off == 0:
                continue
            raise I1Refused(f"ids {a}-{b} (offset {off}) touch registered family {f.name} [{f.start}-{f.stop}]")
    return fam, list(range(lo, hi + 1))


def _blocked(run_dir: Path, mode: str, reason: str, man: dict) -> None:
    rep = {"format": "cb-i1-report/1", "status": "SCIENTIFIC" if mode == "science" else "ENGINEERING_ONLY",
           "run": run_dir.name, "mode": mode, "verdict": "BLOCKED", "reason": reason,
           "manifest_sha256": man["manifest_sha256"],
           "note": "the family stays spent; a retry needs a new preregistration version"}
    (run_dir / "i1_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")


def run(mode: str, *, legacy_root: str, historical_core_archive: str, label: str, threads: int = 2,
        family: str | None = None, start: int | None = None, n: int | None = None,
        confirm_open: str | None = None) -> Path:
    legacy = Path(legacy_root).resolve()
    reg = SeedRegistry.load()
    prereg = json.loads((REPO / PREREG_FILE).read_text())
    man = manifest(threads)
    if mode == "dev":
        fam, sids = _dev_sids(reg, prereg, family, start, n)
    elif mode == "preflight":
        fam = None
    elif mode == "science":
        if confirm_open != PREREG_ID:
            raise I1Refused(f"scientific I1 requires --confirm-open {PREREG_ID}")
        PreregistrationRegistry.load().require(PREREG_ID, item="I1")
        if prereg.get("status") not in ("APPROVED", "FROZEN") or not prereg.get("approved_by"):
            raise I1Refused(f"{PREREG_FILE} is not APPROVED")
        if prereg.get("approved_design_sha256") != design_digest(prereg):
            raise I1Refused("the approval does not name this design (approved_design_sha256 != design_digest)")
        frozen = prereg.get("preflight_manifest_sha256")
        if frozen != man["manifest_sha256"]:
            raise I1Refused(f"manifest {man['manifest_sha256']} differs from the frozen preflight manifest {frozen}")
        fam = None
    else:
        raise I1Refused(f"unknown mode {mode}")
    # everything that can be checked without seeds is checked before a family is registered or spent
    problems = verify_pins(legacy)
    if problems:
        raise I1Refused("legacy pins do not verify: " + "; ".join(problems[:5]))
    run_dir = (REPO / "runs" / ("dev" if mode == "dev" else "i1") / label).resolve()
    if run_dir.exists():
        raise I1Refused(f"output exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    try:
        core = _extract_historical_core(historical_core_archive, run_dir / "historical_core", str(CORE_MANIFEST))
    except Exception as e:
        raise I1Refused(f"historical core does not verify: {e}") from e
    if mode in ("preflight", "science"):
        key = "engineering_preflight_proposed" if mode == "preflight" else "scientific_dev_proposed"
        fam = _family(reg, prereg["families"][key])
        reg.check_usable(fam)
        sids = list(fam.seeds())
        reg.spend(fam, source=f"{PREREG_ID}:{mode}:{label}:opened")
        (run_dir / "seed_opened.json").write_text(json.dumps({"family": {**fam.as_dict(), "spent": True},
                                                              "spent_before_launch": True, "mode": mode,
                                                              "utc": datetime.now(timezone.utc).isoformat()},
                                                             indent=2) + "\n")
    else:
        usage = REPO / "runs/dev/seed_usage.jsonl"
        usage.parent.mkdir(parents=True, exist_ok=True)
        with open(usage, "a") as f:
            f.write(json.dumps({"family": fam.name, "start": sids[0], "stop": sids[-1], "mechanism": "I1-integration",
                                "label": label, "purpose": "ENGINEERING_ONLY",
                                "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    try:
        before = _tree_metadata(legacy)
        _run_arm("LEGACY_REF", sids, run_dir, legacy, core, threads, None)
        normal = {r["sid"]: r for r in _run_arm("NORMAL", sids, run_dir, legacy, core, threads, None)}
        plans = _plans(normal)
        for arm, p in plans.items():
            (run_dir / f"plan_{arm}.json").write_text(json.dumps(p, sort_keys=True))
        for arm in LATER_ARMS:
            _run_arm(arm, sids, run_dir, legacy, core, threads, run_dir / f"plan_{arm}.json" if arm in plans else None)
        after = _tree_metadata(legacy)
        (run_dir / "safety.json").write_text(json.dumps({"legacy_tree_unchanged": before == after, "pins_verified": True,
                                                         "manifest_sha256": man["manifest_sha256"]}) + "\n")
        rep = _e.evaluate(run_dir, sealed=(mode == "preflight"), scientific=(mode == "science"),
                          expected_manifest=man["manifest_sha256"])
    except Exception as e:
        _blocked(run_dir, mode, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}", man)
        raise
    if mode == "science":
        rep["prereg_id"], rep["manifest_sha256"] = PREREG_ID, man["manifest_sha256"]
        (run_dir / "i1_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    _e.print_report(rep)
    return run_dir
