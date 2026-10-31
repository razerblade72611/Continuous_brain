#!/usr/bin/env python3
"""CONS-0A run 1: POST-HOC diagnostics (not a primary verdict). Value-prefix acquisition (the answer BEGINS with the
gold value: run 1's QA targets had no end-of-answer terminator, so answers ran on) and teacher-forced log p of V vs N and
vs the competitor W.        python scripts/cons0a_posthoc.py results/cons0a_run1"""
import json
import re
import sys
from pathlib import Path


def norm(s):
    s = re.sub(r"[^\w\s'-]", " ", (s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(a|an|the)\s+", "", s)


def main(d):
    d, agg = Path(d), {}
    pre = lambda r: norm(r["answer"]).startswith(norm(r["value"]))
    for f in sorted(d.glob("seed*/run_seed*.json")):
        r = json.loads(f.read_text())
        arms = {"N": {int(L): v for L, v in r["N"].items()}} | \
               {a: {int(L): e for L, e in v["eval"].items()} for a, v in r["arms"].items()}
        for a, evs in arms.items():
            for L, ev in evs.items():
                x = agg.setdefault(f"{a}|{L}", {"new_tmpl": [0, 0], "new_para": [0, 0], "life1_para": [0, 0],
                                                "nearmiss_answered": [0, 0], "dlogpV_vs_N": [], "margin_V_minus_W": []})
                for k, sel in (("new_tmpl", lambda q: q["kind"] == "template" and q["life"] == L),
                               ("new_para", lambda q: q["kind"] == "paraphrase" and q["life"] == L),
                               ("life1_para", lambda q: q["kind"] == "paraphrase" and q["life"] == 1)):
                    rs = [q for q in ev["rows"] if sel(q)]
                    x[k][0] += sum(map(pre, rs)); x[k][1] += len(rs)
                nm = [q for q in ev["rows"] if q["kind"] == "nearmiss"]
                x["nearmiss_answered"][0] += sum(not q["unknown"] and not q["empty"] for q in nm)
                x["nearmiss_answered"][1] += len(nm)
                nI = {q["eid"]: q for q in arms["N"][L]["intrusion"]}
                for q in ev["intrusion"]:
                    x["dlogpV_vs_N"].append(q["V"] - nI[q["eid"]]["V"]); x["margin_V_minus_W"].append(q["V"] - q["W"])
    out = {k: {m: (v if isinstance(v, list) and len(v) == 2 and isinstance(v[0], int) and m != "dlogpV_vs_N"
                   and m != "margin_V_minus_W" else round(sum(v) / len(v), 2)) for m, v in x.items()}
           for k, x in sorted(agg.items())}
    (d / "POSTHOC.json").write_text(json.dumps(out, indent=1))
    for k, x in out.items():
        print(k, x)


if __name__ == "__main__":
    main(sys.argv[1])
