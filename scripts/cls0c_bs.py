"""CLS-0C Phase B-strict: the ONE bounded strict-entailment revision of B1 (Claude + Sol, 2026-10-29).
Protocol: lab/CLS0C_B_STRICT_PROTOCOL.md. Identical to Phase B's B1_K1 (scripts/cls0c_b.py) except the VERIFY question
line, which asks whether the memory, taken literally, establishes the completion without assumptions. Deterministic
promotion rule; the original B1_K1 numbers come from the committed Phase B results (hash recorded).

  python scripts/cls0c_bs.py --results results/cls0b_final --phase-b results/cls0c_b --model /workspace/hf/Qwen3-8B \\
      --external /workspace/hf/Qwen3-Embedding-0.6B --out results/cls0c_bs
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


B = _load("cls0c_b_for_bs", "scripts/cls0c_b.py")
SEEDS = (2, 3, 4, 5)
QS = ("Taken literally, does the memory above establish what completes this statement, without assuming anything the "
      "memory does not say? Answer yes or no.\nAnswer:")
assert B.VERIFY.count(B._Q) == 4
VERIFY_STRICT = B.VERIFY.replace(B._Q, QS)               # the ONLY change: the question line (3 examples + the item)
PROMOTE = {"not_entailed_commit_max": 0.30, "template_min": 0.99, "para_entailed_min": 0.97, "near_max": 0.05,
           "unseen_max": 0.0, "temporal_min": 1.0, "provenance_min": 0.95}
NOT_ENTAILED_FORMS = [k for k, v in B.AUDIT.items() if v == "not_entailed"]


# ---------------------------------------------------------------- pure parts (unit-tested) -------------------------
def b1_answer(cue, cands, verify, extract_value, texts, times):
    """B1 exactly as in Phase B run_life: supported = yes-no margin > 0; temporal rule; extraction; evidence contract."""
    sup = [{"eid": e, "time": times[e], "margin": verify(cue, e)} for e in cands]
    sup = [s for s in sup if s["margin"] > 0]
    ch = B.select(sup, B.time_constraint(cue))
    out = B.commit(ch["eid"] if ch else None, extract_value(cue, ch["eid"]) if ch else None, texts)
    out["n_supported"] = len(sup)
    return out, (ch["eid"] if ch else None)


def not_entailed_commitments(rows, form_of):
    """Commitment = ANY answer given on a NOT ENTAILED paraphrase. Overall and per (attribute, paraphrase index)."""
    ne = [r for r in rows if r["cat"] == "para_not_entailed"]
    per = {f"{a}|p{i}": [r for r in ne if form_of(r["key"]) == (a, i)] for a, i in NOT_ENTAILED_FORMS}
    rate = lambda rs: (sum(r["answered"] for r in rs) / len(rs)) if rs else None
    return {"n": len(ne), "committed": sum(r["answered"] for r in ne), "rate": rate(ne),
            "per_form": {k: {"n": len(v), "committed": sum(r["answered"] for r in v), "rate": rate(v)}
                         for k, v in per.items()}}


def promote(s, ne):
    checks = {"not_entailed_commit": ne["rate"] is not None and ne["rate"] <= PROMOTE["not_entailed_commit_max"],
              "template": s["template"] >= PROMOTE["template_min"],
              "para_entailed": s["para_entailed"] >= PROMOTE["para_entailed_min"],
              "near_unsupported": s["near_unsupported"] <= PROMOTE["near_max"],
              "unseen_unsupported": s["unseen_unsupported"] <= PROMOTE["unseen_max"],
              "temporal": s["temporal"] is not None and s["temporal"] >= PROMOTE["temporal_min"],
              "provenance": (s["provenance"] or 0) >= PROMOTE["provenance_min"]}
    return all(checks.values()), checks


# ---------------------------------------------------------------- run ------------------------------------------------
def main():
    import torch
    from being.cls.keys import ExternalKeys
    from being.v3 import lifegen
    c0b = _load("cls0b_for_cls0c_bs", "scripts/cls0b.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--phase-b", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    t0 = time.time()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    dev, dtype = ("cuda", torch.bfloat16) if cuda else ("cpu", torch.float32)
    pb_dir = ROOT / a.phase_b
    run = {"which": "CLS-0C-B-strict", "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                               capture_output=True, text=True).stdout.strip(),
           "gpu": torch.cuda.get_device_name(0) if cuda else None, "test_mode": a.test_mode, "inputs": {},
           "verify_strict_sha256": hashlib.sha256(VERIFY_STRICT.encode()).hexdigest(), "promotion": PROMOTE}
    for p in sorted(pb_dir.glob("*.json")):
        run["inputs"]["phase_b/" + p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    pb = json.loads((pb_dir / "summary.json").read_text())
    assert pb["selected"] == "B1_K1" and pb["valid"], "Phase B summary is not the committed valid B1_K1 selection"
    if not a.test_mode:
        man_model, man_ext = B.manifest(a.model), B.manifest(a.external)
    ek = ExternalKeys(a.external, dev, dtype)
    S = {}
    for seed in a.seeds:
        p = ROOT / a.results / f"seed{seed}.json"
        run["inputs"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        d = json.loads(p.read_text())
        assert d["status"] == "complete" and d["seed"] == seed
        if not a.test_mode and (d["model_manifest"]["files"] != man_model or d["external_manifest"]["files"] != man_ext):
            sys.exit(f"seed {seed}: model or embedder files differ from the ones CLS-0B used")
        eps = [{k: v for k, v in e.items() if k != "text_sha256"} for e in d["episodes"]]
        for e, r in zip(eps, d["episodes"]):
            assert hashlib.sha256(e["text"].encode()).hexdigest() == r["text_sha256"]
        stored = [e for e in eps if d["writes"].get(e["eid"], {}).get("ok")]
        cues, cal, neg, temporal = c0b.build_cues(seed, eps, lifegen)
        assert d["n_cues"] == {"fact": len(cues), "calibration": len(cal), "test_negative": len(neg),
                               "temporal": len(temporal)}
        texts = {e["eid"]: e["text"] for e in stored}
        qs = sorted({c["cue"] for c in cues + neg + temporal})
        S[seed] = dict(stored=stored, cues=cues, neg=neg, temporal=temporal, texts=texts,
                       times={e["eid"]: e["time"] for e in stored}, attr_of={e["eid"]: e["attr"] for e in eps},
                       ev=dict(zip(texts, ek.embed(list(texts.values())))), qv=dict(zip(qs, ek.embed(qs, query=True))))

    def items_at(seed, L):                                   # identical to Phase B (scripts/cls0c_b.py main)
        s = S[seed]
        it = [{"key": f"F|{c['eid']}|{c['form']}", "cue": c["cue"], "cat": B.cue_category(c, s["attr_of"]),
               "gold_value": c["value"], "gold_eids": {c["eid"]}} for c in s["cues"] if c["time"] <= L]
        for j, c in enumerate(s["neg"]):
            if not (c["form"] == "near_miss" and c["observed_since"] > L):
                it.append({"key": f"N|{j}|{c['form']}", "cue": c["cue"], "cat": B.cue_category(c, s["attr_of"])})
        tq = {e["value"]: e["eid"] for e in s["stored"] if e["kind"] == "temporal"}
        for c in c0b.temporal_items(s["temporal"], L):
            it.append({"key": f"T|{c['form']}", "cue": c["cue"], "cat": "temporal", "gold_value": c["expect"],
                       "gold_eids": {tq[c["expect"]]}})
        return it

    def cands(seed, L, cue):
        s = S[seed]
        bank = [e["eid"] for e in s["stored"] if e["time"] <= L]
        return sorted(bank, key=lambda e: -float(s["qv"][cue] @ s["ev"][e]))[:B.K_CAND]
    need_v = sorted({(sd, it["cue"], e) for sd in a.seeds for L in range(1, 6) for it in items_at(sd, L)
                     for e in cands(sd, L, it["cue"])})
    # ---- preflight (Phase B method: representative actual prompts, slower of a median and a longest batch) ----
    reader = B.Reader(a.model, dev, dtype)
    if not reader.single_token_answers and not a.test_mode:
        sys.exit('" yes" / " no" are not single tokens for this reader')
    mem = lambda sd, e: B.MEM.format(t=S[sd]["times"][e], text=S[sd]["texts"][e])
    vp = [VERIFY_STRICT.format(mem=mem(sd, e), cue=c) for sd, c, e in need_v]
    xp = [B.EXTRACT_VALUE.format(mem=mem(sd, e), cue=c) for sd, c, e in need_v]

    def rate(fn, prompts):
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        mid = len(order) // 2
        worst = 0.0
        for bt in ([prompts[i] for i in order[max(0, mid - B.BATCH // 2):mid + B.BATCH // 2]],
                   [prompts[i] for i in order[-B.BATCH:]]):
            if cuda:
                torch.cuda.synchronize()
            t = time.time()
            fn(bt)
            if cuda:
                torch.cuda.synchronize()
            worst = max(worst, (time.time() - t) / len(bt))
        return worst
    t_v, t_x = rate(reader.yesno, vp), rate(lambda b: reader.generate(b, B.MAX_NEW_VALUE), xp)
    proj = len(vp) * t_v + len(xp) * t_x
    run["preflight"] = {"s_per_verify": t_v, "s_per_extract_value": t_x, "unique_verify": len(vp),
                        "extract_upper": len(xp), "projected_s": round(proj, 1), "budget_s": B.BUDGET_S}
    print("preflight", json.dumps(run["preflight"]), flush=True)
    if proj > B.BUDGET_S and not a.test_mode:
        run["status"] = "preflight_stop"
        (out_dir / "summary.json").write_text(json.dumps(run, indent=1))
        sys.exit(4)
    # ---- strict verification, then the extractions actually needed ----
    tv = time.time()
    V = dict(zip(need_v, reader.yesno(vp)))
    secs = {"verify": time.time() - tv}
    print(f"strict verify done ({len(V)})", flush=True)
    want = set()
    for sd in a.seeds:
        for L in range(1, 6):
            for it in items_at(sd, L):
                _, ch = b1_answer(it["cue"], cands(sd, L, it["cue"]), lambda c, e: V[(sd, c, e)],
                                  lambda c, e: "", S[sd]["texts"], S[sd]["times"])
                if ch:
                    want.add((sd, it["cue"], ch))
    xkeys = sorted(want)
    tx = time.time()
    X = dict(zip(xkeys, map(B.cut_value, reader.generate([B.EXTRACT_VALUE.format(mem=mem(sd, e), cue=c)
                                                           for sd, c, e in xkeys], B.MAX_NEW_VALUE))))
    secs["extract"] = time.time() - tx
    print(f"value extraction done ({len(X)})", flush=True)
    # ---- evaluate ----
    pooled5, per_seed = [], {}
    for sd in a.seeds:
        s = S[sd]
        form_of = lambda key: (s["attr_of"][key.split("|")[1]], int(key.split("|")[2][-1]))
        lives = []
        for L in range(1, 6):
            rows = []
            for it in items_at(sd, L):
                o, _ = b1_answer(it["cue"], cands(sd, L, it["cue"]), lambda c, e: V[(sd, c, e)],
                                 lambda c, e: X[(sd, c, e)], s["texts"], s["times"])
                g = B.grade(o, it["gold_value"], it["gold_eids"], s["texts"]) if "gold_value" in it else \
                    {"answered": o["answer"] is not None, "correct": False, "provenance_ok": False}
                rows.append({"key": it["key"], "cat": it["cat"], **g, **o, "candidates": cands(sd, L, it["cue"]),
                             "margins": {e: V[(sd, it["cue"], e)] for e in cands(sd, L, it["cue"])}})
            summ = B.summarize(rows)
            lives.append({"life": L, "summary": summ, "rows": rows})
            print(f"seed {sd} life {L}: T{summ['template']:.2f} P{summ['para_entailed']:.2f} "
                  f"N{summ['near_unsupported']:.2f} U{summ['unseen_unsupported']:.2f}", flush=True)
        pooled5 += [dict(r, seed=sd) for r in lives[-1]["rows"]]
        per_seed[sd] = {"summary": lives[-1]["summary"],
                        "not_entailed": not_entailed_commitments(lives[-1]["rows"], form_of)}
        (out_dir / f"seed{sd}.json").write_text(json.dumps({"seed": sd, "lives": lives}, indent=1, allow_nan=False))
    attr_all = {(sd, e): a_ for sd in a.seeds for e, a_ in S[sd]["attr_of"].items()}
    form_pooled = lambda key_seed: (attr_all[(key_seed[1], key_seed[0].split("|")[1])],
                                    int(key_seed[0].split("|")[2][-1]))
    summ = B.summarize(pooled5)
    ne = not_entailed_commitments([dict(r, key=(r["key"], r["seed"])) for r in pooled5], form_pooled)
    ok, checks = promote(summ, ne)
    # the original B1_K1 (Phase B, committed) on the same measures, for the report
    base_rows = []
    for sd in a.seeds:
        pbs = json.loads((pb_dir / f"seed{sd}.json").read_text())
        base_rows += [dict(r, key=(r["key"], sd)) for r in pbs["lives"][-1]["methods"]["B1_K1"]["rows"]]
    base_ne = not_entailed_commitments(base_rows, form_pooled)
    run.update({"status": "complete", "seconds": secs, "pooled_life5": summ, "not_entailed_life5": ne,
                "per_seed_life5": per_seed, "promotion_checks": checks,
                "selected_for_sealed": "B1_STRICT" if ok else "B1_K1",
                "baseline_B1_K1": {"pooled_life5": pb["pooled_life5"]["B1_K1"], "not_entailed_life5": base_ne},
                "wall_s": round(time.time() - t0, 1), "valid": not a.test_mode})
    (out_dir / "summary.json").write_text(json.dumps(run, indent=1, allow_nan=False))
    print("CLS-0C-BS", json.dumps({"B1_STRICT": {k: summ[k] for k in ("template", "para_entailed", "near_unsupported",
                                                                     "unseen_unsupported", "provenance", "temporal",
                                                                     "errors")},
                                   "not_entailed": {"rate": ne["rate"], "per_form": {k: v["rate"] for k, v in
                                                                                     ne["per_form"].items()}},
                                   "baseline_not_entailed_rate": base_ne["rate"], "checks": checks}, indent=1))
    print(f"CLS-0C-BS SELECTED FOR SEALED TEST: {run['selected_for_sealed']}")
    print("CLS-0C-BS DONE valid=%s" % run["valid"])
    sys.exit(0 if run["valid"] else 3)


if __name__ == "__main__":
    main()
