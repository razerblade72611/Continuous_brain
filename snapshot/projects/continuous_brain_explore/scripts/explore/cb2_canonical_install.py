#!/usr/bin/env python3
"""Install the sealed-accepted r3h-B skill into the canonical individual (decision A, DAV-90/DAV-96).

Run on David's machine, from the repository root, with the explore venv:

  python scripts/explore/cb2_canonical_install.py --weights <ab2baa67....pt> --backup-dir <a Windows folder> \\
      --sealed-report evidence/explore/cb2/r3h-sealed-report.json [--finish-life]

What it does, stopping at the first thing that is not as expected (nothing is changed before step 4):

 1. the frozen r3h-B weights: the file's SHA-256 is the sealed candidate's (ab2baa67...), and the sealed report says
    PASS for exactly those weights, that state hash and that event;
 2. the canonical individual: restores to its recorded state hash; still has the routine skill (nothing installed
    yet) and its original vision organ.  If it was saved in the middle of a life (in the viewer), --finish-life lets
    it live that life to its end headless, with its routine skill, every tick archived as LIVED (what the viewer
    would have done); without the flag it stops here;
 3. rollback point: the pre-install snapshot's name, state hash and per-file SHA-256s are recorded, and the whole
    home is copied to a .tgz in --backup-dir (hashed);
 4. the install: one normal consolidation event (cb2_evaluate._consolidate), chained in consolidations.jsonl, with
    the sealed acceptance as its lineage; then r3h-B is marked as the stable skill (the S2 rollback point);
 5. verification, reloading from disk: the new state hash restores; the consolidation log verifies; the installed
    weights are ab2baa67...; every channel outside the skills slot, the tick and the life are exactly as before.

Gate 4 (prediction / persistent memory) stays open: this installs the accepted controller, nothing more.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))
from cb2_collect import ARCHIVE, INDIVIDUALS, git_rev, vision  # noqa: E402
from cb2_evaluate import _consolidate, load_individual  # noqa: E402

FROZEN = {"weights_sha256": "ab2baa674be37970d2f0016f1e10427ac3be785e9dd907e2097449ae95004ee5",
          "candidate_state_hash": "1b401fc81301f8d263928b4b774c403e170b84afb4bfe254830e9dfc41ac075d",
          "candidate_event_sha": "a782200e4b22e4b14aef1e0ed85854ff4117db83c1f4f869f1385989746dc59c",
          "prereg_sha256": "fc8da59d"}
V1 = "0c46b2d1ace00c027d45e23ae01b00da8b802d10660122da49f9727a1cfe9999"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def channels(ind, skip_slot="skills") -> dict:
    return {n: ind.rt.store.value_digest(n) for n in ind.rt.store.names() if n.split(".")[0] != skip_slot}


def stop(msg: str, rep: dict, out: Path) -> int:
    rep["verdict"] = "STOPPED: " + msg
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print("STOP:", msg)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="the frozen r3h-B skill weights (ab2baa67....pt)")
    ap.add_argument("--sealed-report", default=str(REPO / "evidence/explore/cb2/r3h-sealed-report.json"))
    ap.add_argument("--individuals", default=str(INDIVIDUALS))
    ap.add_argument("--individual", default="cb2-ind-0001")
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--finish-life", action="store_true",
                    help="if it was saved mid-life, live that life to its end first (routine skill, archived)")
    ap.add_argument("--out", default=str(REPO / "evidence/explore/cb2/canonical-install-r3hB.json"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    home = Path(a.individuals) / a.individual
    rep = {"format": "cb2-canonical-install/1", "decision": "A (David, 2026-10-09): r3h-B with vision v1",
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **git_rev(), "home": str(home),
           "gate4": "OPEN: installing the accepted controller does not solve prediction / persistent memory"}

    # 1. the frozen weights and the sealed acceptance -----------------------------------------------------------
    wpath = Path(a.weights)
    rep["weights_file_sha256"] = sha_file(wpath)
    if rep["weights_file_sha256"] != FROZEN["weights_sha256"]:
        return stop("the weights file is not the frozen r3h-B (sha256 differs)", rep, out)
    sealed = json.loads(Path(a.sealed_report).read_text())
    c = sealed["prereg"]["candidate"]
    rep["sealed"] = {"verdict": sealed["verdict"], "prereg_sha256": sealed["prereg_sha256"],
                     "weights_sha256": c["weights_sha256"], "state_hash": c["state_hash"], "event_sha": c["event_sha"]}
    if not (sealed["verdict"] == "PASS" and c["weights_sha256"] == FROZEN["weights_sha256"]
            and c["state_hash"] == FROZEN["candidate_state_hash"] and c["event_sha"] == FROZEN["candidate_event_sha"]
            and sealed["prereg_sha256"].startswith(FROZEN["prereg_sha256"])):
        return stop("the sealed report does not certify these weights", rep, out)

    # 2. the canonical individual --------------------------------------------------------------------------------
    v = vision()
    ind, meta = load_individual(home, v)                    # refuses if it does not restore to its recorded hash
    rep["before"] = {"snapshot": meta["snapshot"], "state_hash": meta["state_hash"], "life": meta["life"],
                     "lives_lived": meta["lives_lived"], "in_life": bool(meta.get("in_life")), "skills": meta.get("skills")}
    if meta.get("skills", "routine") != "routine" or meta.get("consolidations"):
        return stop(f"it already has a learned skill installed ({meta.get('weights_sha256')})", rep, out)
    if ind.vision.weights_sha256 != V1 or (meta.get("organs") or {}).get("vision", {}).get("weights_sha256", V1) != V1:
        return stop("its vision organ is not v1", rep, out)
    if meta.get("in_life"):
        if not a.finish_life:
            return stop("it was saved in the middle of a life: re-run with --finish-life (or finish it in the viewer)",
                        rep, out)
        import torch
        torch.set_num_threads(1)
        from continuous_brain_individual.viewer import LiveSession
        s = LiveSession(v, {"backend": "dev"}, REPO / "runs/explore/viewer/lives")
        s.live_as(home, a.archive)
        n0 = int(s.obs["step"])
        while s.alive:
            s.step()
        rep["finished_life"] = {"life": int(s.ind.life), "world": int(s.seed), "from_step": n0,
                                "to_step": int(s.obs["step"]), "death": s.death}
        print(json.dumps({"finished_life": rep["finished_life"]}), flush=True)
        ind, meta = load_individual(home, v)
        if meta.get("in_life"):
            return stop("the life was lived but it is still marked mid-life", rep, out)
    pre = {"snapshot": meta["snapshot"], "state_hash": meta["state_hash"], "life": meta["life"],
           "lives_lived": meta["lives_lived"], "tick": int(ind.rt.tick)}

    # 3. the rollback point -------------------------------------------------------------------------------------
    snapdir = home / "checkpoints" / meta["snapshot"]
    pre["snapshot_files_sha256"] = {p.name: sha_file(p) for p in sorted(snapdir.iterdir()) if p.is_file()}
    bdir = Path(a.backup_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    tgz = bdir / f"{a.individual}-preinstall-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(home, arcname=a.individual)
    pre["backup"] = {"path": str(tgz), "sha256": sha_file(tgz), "bytes": tgz.stat().st_size}
    rep["rollback_point"] = pre
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"rollback_point": {k: pre[k] for k in ("snapshot", "state_hash", "backup")}}, indent=1), flush=True)
    before_channels, before_tick, before_life = channels(ind), int(ind.rt.tick), int(ind.life)

    # 4. the install: one recorded consolidation, then r3h-B marked stable ---------------------------------------
    lineage = {"decision": rep["decision"], "sealed_acceptance": rep["sealed"],
               "candidate": {"state_hash": FROZEN["candidate_state_hash"], "event_sha": FROZEN["candidate_event_sha"]},
               "weights_file": wpath.name, "code": git_rev(),
               "note": "r3h-B: learned organ-command selector (r2 core frozen) + scripted executor organ; vision v1"}
    rc = _consolidate(home, ind, meta, v, wpath.read_bytes(), lineage)
    if rc != 0:
        return stop("the consolidation did not complete", rep, out)
    meta2 = json.loads((home / "individual.json").read_text())
    meta2["stable"] = {"weights_sha256": meta2["weights_sha256"], "event_sha": meta2.get("event_sha"),
                       "snapshot": meta2["snapshot"], "note": "r3h-B, sealed-accepted (DAV-96); S2 rollback point",
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (home / "individual.json").write_text(json.dumps(meta2, indent=1) + "\n")

    # 5. verification, from disk ----------------------------------------------------------------------------------
    from continuous_brain_cb2.consolidation import ConsolidationLog
    after, meta3 = load_individual(home, v)
    events = ConsolidationLog(home / "consolidations.jsonl").verify()
    changed = sorted(n for n, d in channels(after).items() if before_channels.get(n) != d)
    missing = sorted(set(before_channels) - set(channels(after)))
    rep["after"] = {"snapshot": meta3["snapshot"], "state_hash": meta3["state_hash"], "event_sha": meta3.get("event_sha"),
                    "weights_sha256": meta3.get("weights_sha256"), "skills": meta3.get("skills"),
                    "consolidation_events": events, "stable": meta3.get("stable"),
                    "non_skill_channels": len(before_channels), "non_skill_channels_changed": changed,
                    "non_skill_channels_missing": missing, "tick_kept": int(after.rt.tick) == before_tick,
                    "life_kept": int(after.life) == before_life, "vision": after.vision.weights_sha256}
    ok = (meta3.get("weights_sha256") == FROZEN["weights_sha256"] and meta3.get("skills") == "cb2" and events >= 1
          and not changed and not missing and rep["after"]["tick_kept"] and rep["after"]["life_kept"]
          and after.vision.weights_sha256 == V1)
    rep["verdict"] = "INSTALLED and verified" if ok else "INSTALLED but verification FAILED: roll back (see rollback_point)"
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"verdict": rep["verdict"], "after": rep["after"]}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
