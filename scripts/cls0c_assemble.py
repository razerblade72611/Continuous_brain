#!/usr/bin/env python3
"""CLS-0C sealed: assemble the Grok-A author dataset from the raw captures, byte-for-byte reproducibly.

Transformations: a leading ```json / trailing ``` markdown fence is stripped before json parsing, and each
author-supplied replacement in FIXES substitutes the rejected template in place. Nothing else is changed. Prints the sha256 of each raw input and of the output.

  python scripts/cls0c_assemble.py results/cls0c_sealed_work/raw results/cls0c_sealed_work/dataset_author.json
"""
import hashlib
import json
import re
import sys
from pathlib import Path

WORLD_FILES = ["grokA_batch1.json"] + [f"grokA_W{n:02d}.json" for n in range(4, 10)] + ["grokA_W10.raw.txt"]
TEMPLATES = "grokA_templates.raw.txt"
# Author corrections requested after validate errors (AUDIT_LOG event 11): (attr, kind, rejected wording, fix file).
FIXES = [("pet species", "supported", "At home {e} looks after a", "grokA_templates_fix1.raw.txt")]


def load(p):
    b = p.read_bytes()
    print(f"{hashlib.sha256(b.rstrip(b'\\n')).hexdigest()}  {p.name}")
    s = b.decode("utf-8").strip()
    m = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", s, re.S)
    return json.loads(m.group(1) if m else s)


def main(raw, out):
    raw = Path(raw)
    worlds = []
    for f in WORLD_FILES:
        worlds += load(raw / f)["worlds"]
    ids = [w["world_id"] for w in worlds]
    assert ids == [f"W{n:02d}" for n in range(1, 11)], ids
    tpl = load(raw / TEMPLATES)["attribute_templates"]
    for attr, kind, old, f in FIXES:
        new = load(raw / f)[attr][f"{kind}_replacement"]
        ts = tpl[attr][kind]
        ts[ts.index(old)] = new  # same position; raises if the rejected wording is absent
    data = json.dumps({"worlds": worlds, "attribute_templates": tpl}, indent=1, sort_keys=True, ensure_ascii=False)
    Path(out).write_text(data + "\n")
    print(f"{hashlib.sha256(data.encode()).hexdigest()}  {Path(out).name}  (worlds={len(worlds)}, "
          f"items={sum(len(w['items']) for w in worlds)}, attrs={len(tpl)})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
