#!/usr/bin/env python3
"""Gate 7 (DAV-90): swap an organ of an individual, keep the same individual (exploration lane, ENGINEERING_ONLY).

  swap      python scripts/explore/cb2_organ_swap.py swap --individuals runs/explore/cb2/candidates/X \\
                --organ vision --weights runs/explore/vision/tile_decoder_v2.pt \\
                --card continuous_brain_organs/cards/tile_decoder_v2.json --note "..."
  rollback  python scripts/explore/cb2_organ_swap.py rollback --individuals ... --to <weights sha256> --note "..."
  verify    python scripts/explore/cb2_organ_swap.py verify --individuals ...

Each swap is one recorded event (continuous_brain_cb2/organ_swap.py): every channel, the clock and the life carried
over exactly (the state hash is unchanged), the organ's record in the binding changed, the event hash-chained in
<home>/organ_swaps.jsonl, both organs' weights and cards kept by hash in <home>/organs/, a new snapshot saved, and
individual.json updated (organs.vision names the organ it now lives with).  A rollback is a swap back.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/explore"))
from cb2_collect import CARD, WEIGHTS, git_rev, vision  # noqa: E402
from cb2_evaluate import load_individual  # noqa: E402


def _keep(home: Path, weights: Path, card: Path, sha: str) -> None:
    d = home / "organs"
    d.mkdir(exist_ok=True)
    for src, dst in ((weights, d / f"{sha}.pt"), (card, d / f"{sha}.card.json")):
        if not dst.exists():
            shutil.copyfile(src, dst)


def _do(home: Path, new_weights: Path, new_card: Path, lineage: dict) -> int:
    from continuous_brain_cb2.organ_swap import OrganSwapLog, identity_digest, organ_provenance, swap_organ
    from continuous_brain_individual.individual import Individual, IndividualConfig
    from continuous_brain_organs.language import LanguageOrgan
    from continuous_brain_organs.vision import VisionOrgan
    from continuous_brain_snn.snapshot import SnapshotStore
    src, meta = load_individual(home, vision())
    if meta.get("in_life") and not lineage.get("candidate"):
        print("note: saved mid-life (a candidate copy); the swap applies at this moment of that life")
    old = src.vision
    kept = home / "organs" / f"{old.weights_sha256}.pt"
    if not kept.exists():                               # its first swap: keep the organ it was born with too
        if old.weights_sha256 != vision().weights_sha256:
            raise SystemExit("its current organ is neither kept in its home nor the default one")
        _keep(home, WEIGHTS, CARD, old.weights_sha256)
    new = VisionOrgan(new_weights, new_card)
    if new.weights_sha256 == old.weights_sha256:
        raise SystemExit("that is the organ it already has")
    _keep(home, new_weights, new_card, new.weights_sha256)
    log = OrganSwapLog(home / "organ_swaps.jsonl")
    prev = log.events()[-1]["event"]["event_sha"] if log.events() else None
    if prev != meta.get("organ_event_sha"):
        raise SystemExit("organ_swaps.jsonl and individual.json disagree (an interrupted swap?): resolve it first")
    before_digest = identity_digest(src)
    dst = Individual(IndividualConfig(lifelong=True, skills=meta.get("skills", "routine")), new,
                     LanguageOrgan.from_config({"backend": "dev"}))
    ev = swap_organ(src, dst, "vision", {**lineage, "code": git_rev(), "card": new.card.get("name")}, prev_event_sha=prev)
    ev["identity_digest_before"], ev["identity_digest_after"] = before_digest, identity_digest(dst)
    if ev["identity_digest_before"] != ev["identity_digest_after"]:
        raise SystemExit("identity channels changed: refusing to record the swap")
    rec = log.append(ev)
    snap = dst.rt.save(SnapshotStore(home), label=f"organswap{rec['seq']}")
    upd = {**meta, "snapshot": snap.name, "state_hash": dst.state_hash(), "life": dst.life,
           "organs": {**(meta.get("organs") or {}), "vision": {"name": new.card.get("name"), "weights_sha256": new.weights_sha256}},
           "organ_swaps": rec["seq"] + 1, "organ_event_sha": ev["event_sha"], "parent_snapshot": Path(meta["snapshot"]).name,
           **git_rev()}
    Path(str(snap) + ".meta.json").write_text(json.dumps(upd) + "\n")
    (home / "individual.json").write_text(json.dumps(upd, indent=1) + "\n")
    assert organ_provenance(dst, "vision")["weights_sha256"] == new.weights_sha256
    print(json.dumps({"event": ev["kind"], "event_sha": ev["event_sha"], "from": ev["from"], "to": ev["to"],
                      "parent_state_hash": ev["parent_state_hash"], "state_hash": ev["state_hash"],
                      "identity_digest": ev["identity_digest_after"], "snapshot": str(snap)}, indent=1))
    return 0


def swap_cmd(a) -> int:
    return _do(Path(a.individuals) / a.individual, Path(a.weights), Path(a.card),
               {"kind": "organ_swap", "note": a.note, "candidate": True})


def rollback_cmd(a) -> int:
    home = Path(a.individuals) / a.individual
    d = home / "organs"
    return _do(home, d / f"{a.to}.pt", d / f"{a.to}.card.json", {"kind": "organ_rollback", "note": a.note, "candidate": True})


def verify_cmd(a) -> int:
    from continuous_brain_cb2.organ_swap import OrganSwapLog
    home = Path(a.individuals) / a.individual
    log = OrganSwapLog(home / "organ_swaps.jsonl")
    n = log.verify()
    ind, meta = load_individual(home, vision())
    last = log.events()[-1]["event"]["event_sha"] if n else None
    ok = last == meta.get("organ_event_sha") and (not n or log.events()[-1]["event"]["to"]["weights_sha256"] == ind.vision.weights_sha256)
    print(json.dumps({"organ_swaps": n, "log_matches_individual": ok, "state_hash": ind.state_hash(),
                      "organs": meta.get("organs"), "vision_weights_sha256": ind.vision.weights_sha256}, indent=1))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("swap")
    s.add_argument("--organ", default="vision", choices=["vision"])
    s.add_argument("--weights", required=True)
    s.add_argument("--card", required=True)
    s.add_argument("--note", default="")
    s.set_defaults(func=swap_cmd)
    r = sub.add_parser("rollback")
    r.add_argument("--to", required=True, help="the earlier organ's weights sha256 (kept in <home>/organs/)")
    r.add_argument("--note", default="")
    r.set_defaults(func=rollback_cmd)
    v = sub.add_parser("verify")
    v.set_defaults(func=verify_cmd)
    for p in (s, r, v):
        p.add_argument("--individuals", required=True)
        p.add_argument("--individual", default="cb2-ind-0001")
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
