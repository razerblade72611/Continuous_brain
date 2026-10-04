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
             preregistration, and verify_preflight_chain() re-verifies the whole frozen chain from the
             preflight's ledger spend to the approval basis; the family is spent before launch; the
             verdict is PASS / FAIL / BLOCKED

Preflight, approval and science use only this checkout's registry/ (registry_root refuses if
CONTINUOUS_BRAIN_REGISTRY points elsewhere). Preflight and science open their family through
open_family(), which spends it with source spend_source(mode, label) and writes seed_opened.json binding
the spend to the run, the manifest and the registry state (design, both registry files, the seed ledger).

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
from continuous_brain_unified.prereg import ENV_REGISTRY, PreregistrationRegistry

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
LATER_ARMS = ("FIXED_SCAFFOLD", "ERASED", "DONOR", "ROLLBACK", "PERSIST_SAVE", "PERSIST_RESUME", "RNG_PERTURBED",
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
CODE_FILES = ("scripts/i1_preflight.py", "scripts/i1_run.py", "scripts/i1_approve.py", "scripts/dev/i1_dev.py",
              "scripts/dev/i1_tamper_check.py", "scripts/dev/i1_approval_gate_check.py", "pyproject.toml",
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
    keys = ("id", "question", "scope", "design", "decision_rule", "families", "procedure", "approval_authority")
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


def _run_arm(arm, sids, run_dir, legacy, core, threads, plan, quiet=False):
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
    if quiet:      # preflight: no timings or error counts on screen (both carry weak hints about outcomes)
        print(json.dumps({"arm": arm, "finished": True}), flush=True)
    else:
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
            # the donor of individual sid is sid XOR 1 (the other member of its even/odd pair): fixed by the
            # ids alone before any arm runs, so donor selection cannot depend on any outcome
            plans["DONOR"][str(sid)] = [{"kind": "replace", "target": "autobiography.state", "start_tick": t_imp,
                                         "end_tick": t_imp + 1, "payload_b64": partner["birth_state_b64"],
                                         "donor_sid": sid ^ 1,
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


def offset_collisions(reg: SeedRegistry, lo: int, hi: int, prereg: dict, home: SeedFamily) -> list[str]:
    """Every collision of ids lo..hi, and of their derived ids (sid+70000, sid+90000), with the base or derived
    ids of any I1 preregistered family or any registered family, at every offset pair (conservative).

    `home` is the family the ids belong to. Tolerated, at the same offset only: home itself (by name), and an
    unsealed ENGINEERING reserve containing an ENGINEERING home. Used for dev slices and immediately before
    every preflight or science spend (open_family)."""
    offsets = (0, *DERIVED_OFFSETS)
    out = []
    reserved = [(f["name"], int(f["start"]), int(f["stop"])) for f in prereg["families"].values()
                if isinstance(f, dict) and "start" in f]
    for off in offsets:
        a, b = lo + off, hi + off
        for name, fs, ft in reserved:            # base and derived ids of the I1 families, all offset pairs
            for po in offsets:
                if a <= ft + po and fs + po <= b and not (name == home.name and po == off):
                    out.append(f"ids {a}-{b} (offset {off}) overlap ids of the I1 preregistered family "
                               f"{name} (offset {po})")
        for r in reg.ranges:                     # every registered family at every offset pair
            f = r.family
            parent = (r.reserve and f.purpose == SeedPurpose.ENGINEERING and home.purpose == SeedPurpose.ENGINEERING
                      and f.start <= home.start and home.stop <= f.stop and not f.sealed)
            for fo in offsets:
                if not (a <= f.stop + fo and f.start + fo <= b):
                    continue
                if fo == off and (f.name == home.name or parent):
                    continue
                out.append(f"ids {a}-{b} (offset {off}) touch ids of registered family {f.name} "
                           f"[{f.start}-{f.stop}] (offset {fo})")
    return out


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
    hits = offset_collisions(reg, lo, hi, prereg, fam)
    if hits:
        raise I1Refused(hits[0])
    return fam, list(range(lo, hi + 1))


# ---------------------------------------------------------------------------------------------------------
# Chain of custody (dev7, ChatGPT's 64g findings). The scientific decision may rest only on this chain:
#   the canonical preflight family was really spent in this checkout's ledger, once, by this run
#   -> seed_opened.json ties that spend to this run directory, its manifest and the registry state at opening
#   -> the preflight was opened under the PROPOSED preregistration, whose design equals the current design,
#      and both registry files and the seed ledger are unchanged since opening
#   -> the exact machine/code/environment manifest matches -> exactly the 14 validity checks, all true
#   -> exactly the 12 arms bound -> the sealed report exists, matches its commitment and agrees with the view
#   -> central and detailed registries agree -> the frozen approval_authority (David), and nobody else,
#      approves (approve) -> science re-verifies all of it, and every spend is preceded by the derived-id check.
# ---------------------------------------------------------------------------------------------------------
APPROVAL_FIELDS = ("approved_utc", "preflight_manifest_sha256", "approved_design_sha256", "preflight_run",
                   "approval_basis")
APPROVAL_BASIS_KEYS = ("preflight_run", "spend_source", "seed_opened_sha256", "validity_sha256", "manifest_sha256",
                       "sealed_report_sha256", "opened_design_sha256", "opened_prereg_file_sha256",
                       "opened_central_registry_sha256", "opened_seed_ledger_sha256")
BINDING_KEYS = ("design_sha256", "prereg_file_sha256", "central_registry_sha256", "seed_ledger_sha256",
                "prereg_status", "approved_design_sha256", "preflight_manifest_sha256")
JOURNAL = "registry/i1_approval.journal.json"
AUTHORITY_KEY = "approval_authority"      # frozen in the preregistration (and in its design digest)
_HEX64 = frozenset("0123456789abcdef")


def _is_hex64(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and set(x) <= _HEX64


def registry_root(root: Path = REPO) -> Path:
    """This checkout's registry/, the only ledger preflight, approval and science may use.

    CONTINUOUS_BRAIN_REGISTRY is not honored here: if it is set to any other directory (for example left
    over from tests) the run refuses, rather than spend or validate against another ledger."""
    reg = (Path(root) / "registry").resolve()
    env = os.environ.get(ENV_REGISTRY)
    if env and Path(env).resolve() != reg:
        raise I1Refused(f"{ENV_REGISTRY} is set to {env}, not to this checkout's registry {reg}; unset it")
    return reg


def spend_source(mode: str, label: str) -> str:
    """The ledger source string of the spend that opens a preflight or science run labelled `label`."""
    return f"{PREREG_ID}:{mode}:{label}:opened"


def seed_ledger_digest(root: Path = REPO) -> str:
    """SHA-256 over both seed-ledger files (shared ranges and the local append-only ledger)."""
    d = Path(root) / "registry"
    files = {n: (sha256_file(d / n) if (d / n).exists() else None)
             for n in ("seed_ranges.json", "seed_ranges.local.jsonl")}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _registry_binding(prereg: dict, root: Path = REPO) -> dict:
    """The mutable approval/spend records are outside the code manifest; bind them here instead."""
    r = Path(root)
    return {"design_sha256": design_digest(prereg), "prereg_file_sha256": sha256_file(r / PREREG_FILE),
            "central_registry_sha256": sha256_file(r / "registry" / "preregistrations.json"),
            "seed_ledger_sha256": seed_ledger_digest(r), "prereg_status": prereg.get("status"),
            "approved_design_sha256": prereg.get("approved_design_sha256"),
            "preflight_manifest_sha256": prereg.get("preflight_manifest_sha256")}


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def open_family(mode: str, label: str, *, prereg: dict, man: dict, run_dir: Path, root: Path = REPO) -> SeedFamily:
    """The one path by which a preflight or science run opens its family.

    Registers the family if it is new, checks it is usable, spends it in this checkout's ledger with source
    spend_source(mode, label), and writes run_dir/seed_opened.json, which binds the spend to the run, the
    manifest and the registry state (design, both registry files, the seed ledger after the spend)."""
    if mode not in ("preflight", "science"):
        raise I1Refused(f"only preflight and science open a preregistered family, not {mode}")
    root = Path(root).resolve()
    if Path(run_dir).resolve() != (root / "runs" / "i1" / label).resolve():
        raise I1Refused(f"a {mode} run lives in runs/i1/{label} of this checkout")
    reg = SeedRegistry.load(registry_root(root))
    key = "engineering_preflight_proposed" if mode == "preflight" else "scientific_dev_proposed"
    spec = prereg["families"][key]
    want = SeedFamily(spec["name"], int(spec["start"]), int(spec["stop"]), SeedPurpose(spec["purpose"]))
    hits = offset_collisions(reg, want.start, want.stop, prereg, want)   # base and derived ids, every offset pair
    if hits:
        raise I1Refused(f"{want.name} cannot be opened: " + "; ".join(hits[:3]))
    fam = _family(reg, spec)
    reg.check_usable(fam)
    reg.spend(fam, source=spend_source(mode, label))
    opened = {"format": "cb-i1-opened/2", "family": {**fam.as_dict(), "spent": True}, "spent_before_launch": True,
              "mode": mode, "run": label, "spend_source": spend_source(mode, label),
              "manifest_sha256": man["manifest_sha256"], "utc": datetime.now(timezone.utc).isoformat(),
              **_registry_binding(prereg, root)}
    (Path(run_dir) / "seed_opened.json").write_text(json.dumps(opened, indent=2) + "\n")
    return fam


def _read_json(path: Path, what: str):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise I1Refused(f"{what} is missing ({path})") from None
    except (OSError, ValueError) as e:
        raise I1Refused(f"{what} is unreadable ({path}): {e}") from None


def _fam_key(f: dict) -> tuple:
    return (f.get("name"), int(f.get("start", -1)), int(f.get("stop", -1)), f.get("purpose"))


def _ledger_rows(regdir: Path, name: str) -> tuple[list[dict], list[dict]]:
    """Every row naming `name` in the shared ranges file and in the local ledger (in ledger order)."""
    shared = [r for r in json.loads((regdir / "seed_ranges.json").read_text())["ranges"] if r.get("name") == name]
    local = []
    p = regdir / "seed_ranges.local.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("name") == name:
                    local.append(r)
    return shared, local


def _check_ledger_spend(regdir: Path, fam_spec: dict, mode: str, label: str) -> None:
    """The family was spent exactly once, in the local ledger, by the run `label`, and nothing superseded it."""
    want, key = spend_source(mode, label), _fam_key(fam_spec)
    shared, local = _ledger_rows(regdir, fam_spec["name"])
    if any(_fam_key(r) != key for r in shared + local):
        raise I1Refused(f"the ledger has a row named {fam_spec['name']} with another range or purpose")
    if any(r.get("spent") for r in shared):
        raise I1Refused(f"{fam_spec['name']} is marked spent in the shared ranges file, not by a run")
    spends = [r for r in local if r.get("spent")]
    if not spends:
        raise I1Refused(f"the ledger has no spend of {fam_spec['name']}: no real {mode} was opened")
    if len(spends) != 1:
        raise I1Refused(f"the ledger records {len(spends)} spends of {fam_spec['name']}")
    if spends[0].get("source") != want:
        raise I1Refused(f"the ledger's spend of {fam_spec['name']} was made by {spends[0].get('source')!r}, "
                        f"not by this run ({want!r})")
    at = next(i for i, r in enumerate(local) if r is spends[0])
    if local[at + 1:]:
        raise I1Refused(f"a later ledger row supersedes the spend of {fam_spec['name']}")
    final = [r for r in SeedRegistry.load(regdir).ranges if (r.family.name, r.family.start, r.family.stop,
                                                             r.family.purpose.value) == key]
    if len(final) != 1 or not final[0].family.spent or final[0].source != want:
        raise I1Refused(f"the loaded ledger does not show {fam_spec['name']} spent by this run")


def _i1_entries(central: dict) -> list[dict]:
    return [e for e in central.get("preregistrations", []) if e.get("id") == PREREG_ID]


def verify_preflight_chain(preflight_dir, *, root: Path = REPO, threads: int = 2, stage: str = "approve") -> dict:
    """Verify every link from the preflight spend to the decision; return the approval basis.

    stage "approve": the preregistration must still be PROPOSED and both registry files byte-identical to
    their state when the preflight was opened. stage "science": the preregistration must be APPROVED on this
    preflight, and the approval basis recorded at approval must be reproduced exactly from the files."""
    if stage not in ("approve", "science"):
        raise I1Refused(f"unknown stage {stage}")
    root = Path(root).resolve()
    regdir = registry_root(root)
    if (root / JOURNAL).exists():
        raise I1Refused(f"an interrupted approval journal exists ({JOURNAL}); run scripts/i1_approve.py --recover")
    d = Path(preflight_dir).resolve()
    if d.parent != (root / "runs" / "i1").resolve():
        raise I1Refused(f"the preflight directory must be runs/i1/<label> of this checkout, not {d}")
    label = d.name
    prereg = _read_json(root / PREREG_FILE, "the preregistration")
    central = _read_json(regdir / "preregistrations.json", "the central registry")
    pf = prereg["families"]["engineering_preflight_proposed"]
    # 1. the canonical preflight family was really spent, once, by this run
    _check_ledger_spend(regdir, pf, "preflight", label)
    # 2. the opening record ties that spend to this directory, its manifest and the registry state at opening
    opened = _read_json(d / "seed_opened.json", "the preflight's opening record")
    fam = opened.get("family") if isinstance(opened.get("family"), dict) else {}
    try:
        fam_ok = _fam_key(fam) == _fam_key(pf)
    except (TypeError, ValueError):
        fam_ok = False
    if (opened.get("format") != "cb-i1-opened/2" or opened.get("mode") != "preflight" or opened.get("run") != label
            or opened.get("spend_source") != spend_source("preflight", label) or not fam_ok
            or fam.get("spent") is not True or opened.get("spent_before_launch") is not True):
        raise I1Refused("seed_opened.json is not the opening record of a preflight of this preregistration's "
                        "preflight family in this directory")
    missing = [k for k in BINDING_KEYS if k not in opened]
    if missing:
        raise I1Refused(f"seed_opened.json lacks the registry binding {missing}")
    if (opened["prereg_status"] != "PROPOSED" or opened["approved_design_sha256"] is not None
            or opened["preflight_manifest_sha256"] is not None):
        raise I1Refused("the preflight was not opened under the PROPOSED preregistration (approval fields at opening)")
    design = design_digest(prereg)
    if opened["design_sha256"] != design:
        raise I1Refused(f"the design changed after the preflight was opened (opened under {opened['design_sha256']}, "
                        f"now {design})")
    entries = _i1_entries(central)
    if stage == "approve":
        if prereg.get("status") != "PROPOSED" or any(k in prereg for k in APPROVAL_FIELDS) or prereg.get("approved_by"):
            raise I1Refused(f"{PREREG_FILE} is {prereg.get('status')}, not PROPOSED without approval fields")
        if opened["prereg_file_sha256"] != sha256_file(root / PREREG_FILE):
            raise I1Refused(f"{PREREG_FILE} changed after the preflight was opened")
        if opened["central_registry_sha256"] != sha256_file(regdir / "preregistrations.json"):
            raise I1Refused("registry/preregistrations.json changed after the preflight was opened")
        if opened["seed_ledger_sha256"] != seed_ledger_digest(root):
            raise I1Refused("the seed ledger changed after the preflight was opened")
        if len(entries) != 1 or entries[0].get("status") != "PROPOSED" or entries[0].get("approved_by"):
            raise I1Refused(f"registry/preregistrations.json has no single PROPOSED {PREREG_ID} entry")
    else:
        if prereg.get("status") not in ("APPROVED", "FROZEN") or not (prereg.get("approved_by") or "").strip():
            raise I1Refused(f"{PREREG_FILE} is not APPROVED")
        if not prereg.get(AUTHORITY_KEY) or prereg.get("approved_by") != prereg.get(AUTHORITY_KEY):
            raise I1Refused(f"approved by {prereg.get('approved_by')!r}, not by the preregistered approval "
                            f"authority {prereg.get(AUTHORITY_KEY)!r}")
        if prereg.get("preflight_run") != label:
            raise I1Refused(f"the approval names preflight {prereg.get('preflight_run')!r}, not {label!r}")
        if prereg.get("approved_design_sha256") != design:
            raise I1Refused("the approval does not name this design (approved_design_sha256 != design_digest)")
        if (len(entries) != 1 or entries[0].get("status") != prereg.get("status")
                or entries[0].get("approved_by") != prereg.get("approved_by")):
            raise I1Refused("the central registry's I1 entry does not agree with the approved preregistration")
    # 3. the exact machine/code/environment manifest
    rec = _read_json(d / "manifest.json", "the preflight manifest")
    body = {k: v for k, v in rec.items() if k != "manifest_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != rec.get("manifest_sha256"):
        raise I1Refused("the preflight manifest.json is not self-consistent")
    if opened.get("manifest_sha256") != rec["manifest_sha256"]:
        raise I1Refused("seed_opened.json was written under another manifest than the directory's manifest.json")
    now = manifest(threads)
    if now["manifest_sha256"] != rec["manifest_sha256"]:
        diff = sorted(k for k in set(now) | set(rec) if k != "manifest_sha256" and now.get(k) != rec.get(k))
        raise I1Refused(f"this machine's manifest {now['manifest_sha256']} differs from the preflight's "
                        f"{rec['manifest_sha256']} (differing: {diff})")
    if stage == "science" and prereg.get("preflight_manifest_sha256") != rec["manifest_sha256"]:
        raise I1Refused("the approval froze another preflight manifest")
    # 4. exactly the 14 validity checks, all true
    validity = _read_json(d / "i1_validity.json", "the preflight's validity view")
    checks = validity.get("validity")
    if not isinstance(checks, dict) or set(checks) != set(_e.VALIDITY_KEYS):
        got = set(checks) if isinstance(checks, dict) else set()
        raise I1Refused("the preflight's validity block does not have exactly the I1 schema "
                        f"(missing {sorted(set(_e.VALIDITY_KEYS) - got)}, unexpected {sorted(got - set(_e.VALIDITY_KEYS))})")
    if validity.get("valid") is not True or not all(checks[k] is True for k in _e.VALIDITY_KEYS):
        raise I1Refused("the preflight is not valid: " + json.dumps(checks, sort_keys=True))
    if validity.get("run") != label or validity.get("format") != "cb-i1-validity/2" or \
            validity.get("status") != "ENGINEERING_ONLY":
        raise I1Refused("i1_validity.json is not the sealed validity view of this preflight directory")
    # 5. exactly the 12 arms, each bound to the manifest
    diag = validity.get("validity_diagnostics") if isinstance(validity.get("validity_diagnostics"), dict) else {}
    arms = diag.get("arm_manifests")
    if not isinstance(arms, dict) or set(arms) != set(_e.ARMS):
        got = set(arms) if isinstance(arms, dict) else set()
        raise I1Refused("the preflight's arm manifests do not cover exactly the I1 arms "
                        f"(missing {sorted(set(_e.ARMS) - got)}, unexpected {sorted(got - set(_e.ARMS))})")
    if diag.get("expected_manifest_sha256") != rec["manifest_sha256"] or \
            any(arms[a] != rec["manifest_sha256"] for a in _e.ARMS):
        raise I1Refused("the preflight's validity diagnostics do not bind every arm to its manifest.json")
    # 6. the sealed report exists, matches its commitment and agrees with the visible view (values not shown)
    sealed = d / "sealed" / "i1_report.json"
    if not sealed.is_file():
        raise I1Refused("the preflight's sealed report is missing (sealed/i1_report.json)")
    if not _is_hex64(validity.get("sealed_report_sha256")) or sha256_file(sealed) != validity["sealed_report_sha256"]:
        raise I1Refused("the sealed report does not match the SHA-256 committed in i1_validity.json")
    rep = _read_json(sealed, "the sealed report")
    rdiag = rep.get("validity_diagnostics") if isinstance(rep.get("validity_diagnostics"), dict) else {}
    if (rep.get("run") != label or rep.get("status") != "ENGINEERING_ONLY" or rep.get("validity") != checks
            or set(diag) != set(_e.SEALED_VISIBLE_DIAGNOSTICS)
            or any(rdiag.get(k) != diag.get(k) for k in _e.SEALED_VISIBLE_DIAGNOSTICS)
            or rep.get("n") != validity.get("n")):
        raise I1Refused("the sealed report disagrees with the visible validity view (values not shown)")
    basis = {"preflight_run": label, "spend_source": spend_source("preflight", label),
             "seed_opened_sha256": sha256_file(d / "seed_opened.json"),
             "validity_sha256": sha256_file(d / "i1_validity.json"), "manifest_sha256": rec["manifest_sha256"],
             "sealed_report_sha256": validity["sealed_report_sha256"], "opened_design_sha256": opened["design_sha256"],
             "opened_prereg_file_sha256": opened["prereg_file_sha256"],
             "opened_central_registry_sha256": opened["central_registry_sha256"],
             "opened_seed_ledger_sha256": opened["seed_ledger_sha256"]}
    if stage == "science" and prereg.get("approval_basis") != basis:
        raise I1Refused("the preflight directory no longer reproduces the basis recorded at approval")
    return basis


def lifecycle_problems(prereg: dict, central: dict, reg: SeedRegistry, root: Path | None = None) -> list[str]:
    """Every invariant the I1 preregistration must satisfy in whatever legal state it is in.

    Registry-level only (no run directory or machine needed): legal in PROPOSED (preflight unopened or
    spent by a preflight run), APPROVED and FROZEN (scientific family unopened or spent by a science run).
    approve() runs it before and after writing; the test suite runs it on the repository."""
    out = []
    if root is not None and (Path(root) / JOURNAL).exists():
        out.append(f"an interrupted approval journal exists ({JOURNAL})")
    st = prereg.get("status")
    if not isinstance(prereg.get(AUTHORITY_KEY), str) or not prereg[AUTHORITY_KEY].strip():
        out.append(f"the preregistration names no {AUTHORITY_KEY}")
    entries = _i1_entries(central)
    if len(entries) != 1:
        return out + [f"central registry has {len(entries)} {PREREG_ID} entries"]
    c = entries[0]
    if st not in ("PROPOSED", "APPROVED", "FROZEN"):
        out.append(f"illegal status {st!r}")
    if c.get("status") != st:
        out.append(f"central status {c.get('status')!r} != preregistration status {st!r}")

    def rows(key):
        f = prereg["families"][key]
        fam = SeedFamily(f["name"], int(f["start"]), int(f["stop"]), SeedPurpose(f["purpose"]))
        return fam, [r for r in reg.ranges if (r.family.name, r.family.start, r.family.stop, r.family.purpose) ==
                     (fam.name, fam.start, fam.stop, fam.purpose)]

    pf, pf_rows = rows("engineering_preflight_proposed")
    sci, sci_rows = rows("scientific_dev_proposed")
    for name, mode, fam, rr in (("preflight", "preflight", pf, pf_rows), ("scientific", "science", sci, sci_rows)):
        if not rr:                                       # unregistered: must still be fresh
            try:
                reg.check_fresh(fam)
            except Exception as e:
                out.append(f"{name} family is neither registered exactly nor fresh: {e}")
        elif not all(r.family.spent for r in rr):         # the runner registers and spends in one step
            out.append(f"{name} family is registered but not spent")
        elif not all(r.source.startswith(f"{PREREG_ID}:{mode}:") and r.source.endswith(":opened") for r in rr):
            out.append(f"{name} family was spent by {[r.source for r in rr]}, not by a {mode} run of {PREREG_ID}")
    if st == "PROPOSED":
        if prereg.get("approved_by") is not None or any(k in prereg for k in APPROVAL_FIELDS):
            out.append("PROPOSED preregistration carries approval fields")
        if c.get("approved_by"):
            out.append("PROPOSED central entry carries approved_by")
        if sci_rows:
            out.append("the scientific family was opened before approval")
    else:
        if not (prereg.get("approved_by") or "").strip() or not prereg.get("approved_utc"):
            out.append("approval without approver or time")
        if not prereg.get(AUTHORITY_KEY) or prereg.get("approved_by") != prereg.get(AUTHORITY_KEY):
            out.append("approved by someone other than the preregistered approval authority")
        if prereg.get("approved_design_sha256") != design_digest(prereg):
            out.append("the approved design digest does not match the design as it stands")
        if not _is_hex64(prereg.get("preflight_manifest_sha256")):
            out.append("approval without a preflight manifest SHA-256")
        if c.get("approved_by") != prereg.get("approved_by") or not c.get("approved_by"):
            out.append("APPROVED central entry without the same approved_by")
        if not pf_rows:
            out.append("approved without a spent preflight family in this registry")
        basis = prereg.get("approval_basis")
        if not isinstance(basis, dict) or set(basis) != set(APPROVAL_BASIS_KEYS):
            out.append("approval without the approval basis")
        else:
            run_name = prereg.get("preflight_run")
            if basis["preflight_run"] != run_name or basis["spend_source"] != spend_source("preflight", str(run_name)):
                out.append("the approval basis names another preflight run")
            if any(not _is_hex64(basis[k]) for k in APPROVAL_BASIS_KEYS if k.endswith("sha256")):
                out.append("the approval basis has a malformed SHA-256")
            if basis["manifest_sha256"] != prereg.get("preflight_manifest_sha256") or \
                    basis["opened_design_sha256"] != prereg.get("approved_design_sha256"):
                out.append("the approval basis disagrees with the approval fields")
            if pf_rows and any(r.source != basis["spend_source"] for r in pf_rows):
                out.append("the preflight family was spent by another run than the approved preflight")
    return out


def _journaled_write(root: Path, objs: dict[Path, dict]) -> None:
    """Write several registry files as one act: journal first (old and new hashes, new bytes), then each file
    atomically, then remove the journal. An interruption leaves the journal; recover_approval() completes it."""
    j = root / JOURNAL
    if j.exists():
        raise I1Refused(f"an interrupted approval journal exists ({JOURNAL}); run scripts/i1_approve.py --recover")
    files = []
    for path, obj in objs.items():
        data = json.dumps(obj, indent=2) + "\n"
        files.append({"path": str(Path(path).resolve().relative_to(root)), "old_sha256": sha256_file(path),
                      "new_sha256": hashlib.sha256(data.encode()).hexdigest(), "data": data})
    _atomic_write(j, json.dumps({"format": "cb-i1-approval-journal/1", "files": files}, indent=2) + "\n")
    for f in files:
        _atomic_write(root / f["path"], f["data"])
    os.remove(j)
    _fsync_dir(j.parent)


def recover_approval(root: Path = REPO) -> str:
    """Complete an interrupted approval from its journal (roll forward). Refuses if a file is in neither its
    old nor its new state. The journal is written only after every check passed and David confirmed."""
    root = Path(root).resolve()
    registry_root(root)
    j = root / JOURNAL
    if not j.exists():
        return "no approval journal: nothing to recover"
    journal = _read_json(j, "the approval journal")
    for f in journal["files"]:
        cur = sha256_file(root / f["path"])
        if cur not in (f["old_sha256"], f["new_sha256"]):
            raise I1Refused(f"{f['path']} is in neither its pre- nor its post-approval state; resolve it by hand")
        if hashlib.sha256(f["data"].encode()).hexdigest() != f["new_sha256"]:
            raise I1Refused("the approval journal is corrupt")
    for f in journal["files"]:
        if sha256_file(root / f["path"]) != f["new_sha256"]:
            _atomic_write(root / f["path"], f["data"])
    os.remove(j)
    _fsync_dir(j.parent)
    prereg = json.loads((root / PREREG_FILE).read_text())
    central = json.loads((root / "registry" / "preregistrations.json").read_text())
    problems = lifecycle_problems(prereg, central, SeedRegistry.load(root / "registry"), root)
    if problems:
        raise I1Refused("after recovery the registry violates the lifecycle invariants: " + "; ".join(problems))
    return "approval completed from the journal"


def approve(*, preflight_dir: str, approver: str, confirm: str, threads: int = 2, root: Path = REPO) -> dict:
    """David's approval, as one journaled act over both registry files.

    Refuses unless the operator typed the preregistration id and verify_preflight_chain() accepts the
    preflight (see the chain above), and unless the registry satisfies lifecycle_problems() before the write
    and the new state satisfies it too. Then writes, in registry/i1_integration_v1.json, status APPROVED,
    approved_by, approved_utc, preflight_manifest_sha256, approved_design_sha256, preflight_run and
    approval_basis (the hashes the science run will re-verify), and sets the I1 entry of
    registry/preregistrations.json to APPROVED."""
    if confirm != PREREG_ID:
        raise I1Refused(f"approval requires --i-approve {PREREG_ID}")
    if not approver.strip():
        raise I1Refused("approval requires --approver")
    root = Path(root).resolve()
    regdir = registry_root(root)
    authority = _read_json(root / PREREG_FILE, "the preregistration").get(AUTHORITY_KEY)
    if not isinstance(authority, str) or not authority.strip():
        raise I1Refused(f"the preregistration names no {AUTHORITY_KEY}; nobody can approve it")
    if approver != authority:
        raise I1Refused(f"only the preregistered approval authority {authority!r} can approve, not {approver!r}")
    basis = verify_preflight_chain(preflight_dir, root=root, threads=threads, stage="approve")
    prereg_path, central_path = root / PREREG_FILE, regdir / "preregistrations.json"
    prereg, central = json.loads(prereg_path.read_text()), json.loads(central_path.read_text())
    reg = SeedRegistry.load(regdir)
    problems = lifecycle_problems(prereg, central, reg, root)
    if problems:
        raise I1Refused("the registry is not in a legal pre-approval state: " + "; ".join(problems))
    utc = datetime.now(timezone.utc).isoformat()
    design = design_digest(prereg)
    who = approver
    prereg.update(status="APPROVED", approved_by=who, approved_utc=utc,
                  preflight_manifest_sha256=basis["manifest_sha256"], approved_design_sha256=design,
                  preflight_run=basis["preflight_run"], approval_basis=basis)
    entry = _i1_entries(central)[0]
    entry.update(status="APPROVED", approved_by=who, approved_utc=utc)
    entry["notes"] = (entry.get("notes", "") + f" APPROVED by {who} at {utc}; design {design}; preflight "
                      f"{basis['preflight_run']} manifest {basis['manifest_sha256']}.").strip()
    problems = lifecycle_problems(prereg, central, reg, root)
    if problems:
        raise I1Refused("the approved state would violate the lifecycle invariants: " + "; ".join(problems))
    _journaled_write(root, {prereg_path: prereg, central_path: central})
    after = lifecycle_problems(json.loads(prereg_path.read_text()), json.loads(central_path.read_text()),
                               SeedRegistry.load(regdir), root)
    if after:
        raise I1Refused("after writing, the registry violates the lifecycle invariants: " + "; ".join(after))
    return {"approved_by": who, "approved_utc": utc, "approved_design_sha256": design,
            "preflight_manifest_sha256": basis["manifest_sha256"], "preflight_run": basis["preflight_run"],
            "approval_basis": basis}


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
    regdir = registry_root(REPO)                # this checkout's ledger only, in every mode
    reg = SeedRegistry.load(regdir)
    prereg = json.loads((REPO / PREREG_FILE).read_text())
    if mode in ("preflight", "science") and (not label or label in (".", "..") or Path(label).name != label):
        raise I1Refused(f"the {mode} label must be a plain directory name, not {label!r}")
    man = manifest(threads)
    if mode == "dev":
        fam, sids = _dev_sids(reg, prereg, family, start, n)
    elif mode == "preflight":
        fam = None
    elif mode == "science":
        if confirm_open != PREREG_ID:
            raise I1Refused(f"scientific I1 requires --confirm-open {PREREG_ID}")
        PreregistrationRegistry.load(regdir).require(PREREG_ID, item="I1")
        if prereg.get("status") not in ("APPROVED", "FROZEN") or not prereg.get("approved_by"):
            raise I1Refused(f"{PREREG_FILE} is not APPROVED")
        if prereg.get("approved_design_sha256") != design_digest(prereg):
            raise I1Refused("the approval does not name this design (approved_design_sha256 != design_digest)")
        frozen = prereg.get("preflight_manifest_sha256")
        if frozen != man["manifest_sha256"]:
            raise I1Refused(f"manifest {man['manifest_sha256']} differs from the frozen preflight manifest {frozen}")
        # the whole frozen chain again, from the preflight's ledger spend to the recorded approval basis
        verify_preflight_chain(REPO / "runs" / "i1" / str(prereg.get("preflight_run")), root=REPO, threads=threads,
                               stage="science")
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
        fam = open_family(mode, label, prereg=prereg, man=man, run_dir=run_dir, root=REPO)
        sids = list(fam.seeds())
    else:
        usage = REPO / "runs/dev/seed_usage.jsonl"
        usage.parent.mkdir(parents=True, exist_ok=True)
        with open(usage, "a") as f:
            f.write(json.dumps({"family": fam.name, "start": sids[0], "stop": sids[-1], "mechanism": "I1-integration",
                                "label": label, "purpose": "ENGINEERING_ONLY",
                                "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    quiet = mode == "preflight"
    try:
        before = _tree_metadata(legacy)
        _run_arm("LEGACY_REF", sids, run_dir, legacy, core, threads, None, quiet)
        normal = {r["sid"]: r for r in _run_arm("NORMAL", sids, run_dir, legacy, core, threads, None, quiet)}
        plans = _plans(normal)
        for arm, p in plans.items():
            (run_dir / f"plan_{arm}.json").write_text(json.dumps(p, sort_keys=True))
        for arm in LATER_ARMS:
            _run_arm(arm, sids, run_dir, legacy, core, threads, run_dir / f"plan_{arm}.json" if arm in plans else None,
                     quiet)
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
        opened = json.loads((run_dir / "seed_opened.json").read_text())
        at_end = _registry_binding(json.loads((REPO / PREREG_FILE).read_text()), REPO)   # incl. the seed ledger
        at_open = {k: opened.get(k) for k in at_end}
        rep["registry_binding"] = {"at_open": at_open, "at_end": at_end, "unchanged": at_open == at_end}
        if at_open != at_end:          # the approval records must not change while the science run is in flight
            rep["verdict"] = "BLOCKED"
            rep["blocked_reason"] = "registry records changed between opening and the end of the run"
        (run_dir / "i1_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    _e.print_report(rep)
    return run_dir
