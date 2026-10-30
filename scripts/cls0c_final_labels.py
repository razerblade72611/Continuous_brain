#!/usr/bin/env python3
"""CLS-0C sealed: build FINAL_LABELS (the `seal` input) from reconcile's disputes.json by class rules.

The frozen `seal` command supports exclusions only; every kept item keeps its author label. Rules are applied per
dispute class (author label -> adjudicator label), never per item, plus an explicit list of integrity exclusions.

  python scripts/cls0c_final_labels.py DISPUTES.json RULES.json OUT_final_labels.json
RULES.json: {"keep_classes": [["IMPLICATION","NEAR-MISS"], ["TEMPORAL","SUPPORTED"], ...],
             "integrity_exclude_templates": ["attr|kind|template", ...], "note": "..."}
Every dispute not in a keep class is excluded (protocol default: unresolved -> excluded).
A same-label dispute (answer/ref mismatch) is class [L, L]. Kept classes require identical gold for SUPPORTED/TEMPORAL
on both sides, or no gold on either side; otherwise the item is excluded regardless of class.
"""
import json
import re
import sys
from collections import Counter


def norm(s):
    s = re.sub(r"[^\w\s'-]", " ", (s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(a|an|the)\s+", "", s)


GOLD = ("SUPPORTED", "TEMPORAL")


def gold_compatible(x):
    a, j = x["author"], x["adjudicator"]
    ag, jg = a["label"] in GOLD, j.get("label") in GOLD
    if not ag and not jg:
        return True
    if ag and jg:
        return norm(a.get("answer")) == norm(j.get("answer")) and bool(set(a.get("xids") or []) & set(j.get("refs") or []))
    return False


def build(disputes, rules):
    keep = {tuple(c) for c in rules.get("keep_classes", [])}
    ex_i, ex_t, kept, log = [], list(rules.get("integrity_exclude_templates", [])), [], Counter()
    for x in disputes:
        if "iid" in x:
            cls = (x["author"]["label"], x["adjudicator"].get("label"))
            if cls in keep and gold_compatible(x):
                kept.append(x["iid"]); log[("kept",) + cls] += 1
            else:
                ex_i.append(x["iid"]); log[("excluded",) + cls] += 1
        else:
            ex_t.append(x["template"]); log[("excluded", "template", x["adjudicator"])] += 1
    return {"exclude_items": sorted(ex_i), "exclude_templates": sorted(set(ex_t)), "kept_disputed_items": sorted(kept),
            "rules": rules, "class_log": {" ".join(k): v for k, v in sorted(log.items())}}


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))
    out = build(d, json.load(open(sys.argv[2])))
    json.dump(out, open(sys.argv[3], "w"), indent=1, sort_keys=True)
    print(json.dumps(out["class_log"], indent=1), len(out["exclude_items"]), "items /", len(out["exclude_templates"]),
          "templates excluded")
