#!/usr/bin/env python3
"""CLS-0C sealed: objective validity check of one Grok B raw reply (the pre-registered retry rule, AUDIT_LOG event 15).
Never inspects whether labels agree with anything; only form.

  python scripts/cls0c_adjcheck.py PACKAGE RAW_REPLY world N9020db|templates
Exit 0 = valid (prints the parsed JSON's sha256), 1 = invalid (prints reasons -> retry in a fresh chat).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

LABELS = {"SUPPORTED", "TEMPORAL", "IMPLICATION", "NEAR-MISS", "UNFAMILIAR", "AMBIGUOUS"}
TLABELS = {"SUPPORTED", "IMPLICATION", "AMBIGUOUS"}


def parse(raw):
    s = raw.strip()
    m = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", s, re.S)
    return json.loads(m.group(1) if m else s)


def check(pkg, raw, kind, wid=None):
    E = []
    try:
        j = parse(raw)
    except Exception as e:  # noqa: BLE001
        return None, [f"not parseable JSON: {e}"]
    if kind == "templates":
        got = j.get("templates") if isinstance(j, dict) else None
        want = {t["id"] for t in pkg["templates"]}
        if not isinstance(got, dict):
            return j, ["no 'templates' object"]
        if set(got) != want:
            E.append(f"ids: missing {sorted(want - set(got))[:5]} extra {sorted(set(got) - want)[:5]}")
        E += [f"{k}: bad label {v!r}" for k, v in got.items() if v not in TLABELS]
        return j, E
    w = next(w for w in pkg["worlds"] if w["world"] == wid)
    refs = {x["ref"] for x in w["experiences"]}
    want = {s["id"] for s in w["statements"]}
    got = j.get("statements") if isinstance(j, dict) else None
    if not isinstance(got, dict):
        return j, ["no 'statements' object"]
    if set(got) != want:
        E.append(f"ids: missing {sorted(want - set(got))[:5]} extra {sorted(set(got) - want)[:5]}")
    for k, v in got.items():
        lab = v.get("label") if isinstance(v, dict) else None
        if lab not in LABELS:
            E.append(f"{k}: bad label {lab!r}")
            continue
        rr = v.get("refs") or []
        if any(r not in refs for r in rr):
            E.append(f"{k}: ref not in this world {rr}")
        if lab in ("SUPPORTED", "TEMPORAL") and (not (v.get("answer") or "").strip() or not rr):
            E.append(f"{k}: {lab} without answer and >=1 valid ref")
    return j, E


if __name__ == "__main__":
    pkg = json.loads(Path(sys.argv[1]).read_text())
    raw = Path(sys.argv[2]).read_text()
    j, E = check(pkg, raw, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    if E:
        print("INVALID\n" + "\n".join(E))
        sys.exit(1)
    from collections import Counter
    vals = j["templates"].values() if sys.argv[3] == "templates" else [v["label"] for v in j["statements"].values()]
    print("VALID", hashlib.sha256(raw.rstrip("\n").encode()).hexdigest(), dict(Counter(vals)))
