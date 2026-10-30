#!/usr/bin/env python3
"""CLS-0C sealed: DESCRIPTIVE analyses of the one sealed run (VALIDITY_AMENDMENT.md). Nothing here changes a primary
verdict. Post-hoc analyses are labelled post_hoc.
  python scripts/cls0c_sealed_descriptive.py results/cls0c_sealed results/cls0c_sealed_run
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def norm(s):
    s = re.sub(r"[^\w\s'-]", " ", (s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(a|an|the)\s+", "", s)


def contains(text, v):
    v, t = norm(v), re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", text.lower()))
    return bool(v) and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", t) is not None


def main(sd, rd):
    sd, rd = Path(sd), Path(rd)
    S, C = json.loads((sd / "sealed.json").read_text()), json.loads((sd / "COHORTS.json").read_text())
    R = [r for r in json.loads((rd / "sealed_rows.json").read_text()) if r["method"] == "B1_K1"]
    gold = {f"N|{w['world_id']}|{it['iid']}": (it, {x["xid"]: x for x in w["experiences"]})
            for w in S["worlds"] for it in w["items"]}
    nar = {r["key"]: r for r in R if r["fmt"] == "narrative"}
    out = {"cohorts": {}, "miss_breakdown": {}, "wrong_episode": [], "post_hoc": {}, "per_world": {}}
    for name, keys in C["cohorts"].items():
        rs = [nar[k] for k in keys if k in nar]
        pos = name.startswith(("TEMPORAL", "disputed_kept_TEMPORAL", "disputed_kept_SUPPORTED"))
        f = (lambda r: r["correct"]) if pos else (lambda r: r["answered"])
        out["cohorts"][name] = {"metric": "correct" if pos else "commit", "n": len(rs),
                                "value": round(sum(map(f, rs)) / len(rs), 3) if rs else None}
    for cat in ("narrative_supported", "temporal"):
        c = Counter()
        for r in (r for r in R if r["fmt"] == "narrative" and r["cat"] == cat):
            it, xs = gold[r["key"]]
            g, a = norm(it["answer"]), norm(r["answer"])
            k = ("correct" if r["correct"] else "abstain" if not r["answered"] else
                 "wrong_episode" if r["eid"] not in it["xids"] else
                 "right_episode_span_boundary" if a and (a in g or g in a) else "right_episode_other_value")
            c[k] += 1
            if k == "wrong_episode":
                t = xs.get(r["eid"], {}).get("text", "")
                out["wrong_episode"].append({"key": r["key"], "cat": cat, "gold": it["answer"], "answer": r["answer"],
                                             "cited": r["eid"], "cited_contains_gold": contains(t, it["answer"]),
                                             "cited_life": xs.get(r["eid"], {}).get("life"),
                                             "gold_lives": sorted(xs[x]["life"] for x in it["xids"])})
        out["miss_breakdown"][cat] = dict(c)
        rs = [r for r in R if r["fmt"] == "narrative" and r["cat"] == cat]
        ok = sum(1 for r in rs if r["answered"] and r["eid"] in gold[r["key"]][0]["xids"]
                 and norm(gold[r["key"]][0]["answer"]) in norm(r["answer"]))
        out["post_hoc"][f"{cat}_lenient_contains_right_episode"] = {"value": round(ok / len(rs), 3), "n": len(rs)}
    pw = defaultdict(Counter)
    for r in R:
        if r["fmt"] != "narrative" or r["cat"] == "natural_question":
            continue
        w = r["key"].split("|")[1]
        pw[w][r["cat"] + "_n"] += 1
        pw[w][r["cat"] + "_ok"] += int(r["correct"] if r["cat"] in ("narrative_supported", "temporal") else not r["answered"])
    out["per_world"] = {w: {c: f"{d[c + '_ok']}/{d[c + '_n']}" for c in
                            ("narrative_supported", "temporal", "implication", "near_miss", "unfamiliar")}
                        for w, d in sorted(pw.items())}
    sup = [r for r in R if r["fmt"] == "narrative" and r["cat"] == "narrative_supported" and not r["key"].startswith("N|W07|")]
    out["post_hoc"]["narrative_supported_excluding_W07"] = {"value": round(sum(r["correct"] for r in sup) / len(sup), 3),
                                                            "n": len(sup)}
    mech = Counter()
    for r in R:
        if r["fmt"] != "narrative" or r["cat"] not in ("narrative_supported", "temporal"):
            continue
        g, cands, m = set(gold[r["key"]][0]["xids"]), set(r["candidates"]), r["margins"]
        if not r["answered"]:
            k = ("abstain: gold not in K1 top-5" if not g & cands else
                 "abstain: gold retrieved, VERIFY rejected all candidates" if not any(v > 0 for v in m.values()) else
                 "abstain: gold VERIFY-rejected, another passed, extraction/contract failed" if not any(m.get(e, 0) > 0 for e in g) else
                 "abstain: gold verified, extraction/evidence contract failed")
        elif r["eid"] not in g:
            k = ("wrong episode: gold not retrieved" if not g & cands else
                 "wrong episode: gold also verified, most-recent selection chose another" if any(m.get(e, 0) > 0 for e in g) else
                 "wrong episode: gold VERIFY-rejected, another accepted")
        else:
            continue
        mech[k] += 1
    out["mechanisms_supported_plus_temporal"] = dict(mech)
    (rd / "DESCRIPTIVE.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("cohorts", "miss_breakdown", "post_hoc")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
