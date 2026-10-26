"""BIND-D1: rival-attribution and paired-entity diagnostic (Claude + Sol, 2026-10-26). DEV seeds only.

Runs the FROZEN Gate v3 life loop (being/v3/run.py live_stream, unchanged) for one arm, and replaces the checkpoint
probe with an instrumented copy that (1) computes every historical field exactly as run.probe_facts does, using the same
Being3.candidates_logprob scorer, and THEN (2) records, read-only:
  - all 50 forced-choice candidates: value, summed log-prob (the historical metric), token count, and whether the
    prompt's token ids are an exact prefix of prompt+" "+candidate (flagged, never repaired), plus explicit ties;
  - paired-entity queries for EVERY same-attribute pair of facts in lives 1..lives with different entities and
    different values, fixed from the generated stream before any scoring and oriented by the stream alone (A = the
    earlier fact; same life -> lower fact id): l(V|A), l(W|A), l(V|B), l(W|B). Cohorts (Sol, 2026-10-26): primary =
    A in life 1, B later; simultaneous = both in life 1; secondary = A after life 1. Identical-value pairs are excluded
    and counted. Stored per checkpoint (and for the wiped backbone) under "paired".
Probing is side-effect free for L and S (no memory update in forward), so training is identical to the archived runs.
A reference result (the archived A0 job for the same config) can be given to check reproduction.

  python scripts/bind_d1.py --arm L --set lr=1e-4 kl_weight=1.0 night_passes=4 --seeds 0,1 --lives 5 \\
      --out results/dev3/bind_d1/L_kl1.0_lr1e-4.json --reference results/dev3/stageA2_A0/L_kl_weight1.0_lr0.0001_night_passes4.json
"""
import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEV = set(range(10))
CHECKPOINTS = (1, 2, 3, 5)


def candidate_meta(tok, prompt, cands):
    """Token count of each candidate continuation and whether the standalone prompt ids are an exact prefix of the
    combined ids (the assumption Being3.candidates_logprob makes). Tokenizer only -- no model call."""
    p = tok(prompt).input_ids
    out = []
    for c in cands:
        full = tok(prompt + " " + c).input_ids
        out.append({"ntok": len(full) - len(p), "prefix_ok": full[:len(p)] == p})
    return out


def pairs_for(stream, n_lives):
    """All same-attribute fact pairs in lives 1..n_lives with different entities, oriented by the stream only
    (A earlier; same life -> lower fid). Returns (pairs, n_identical_value_excluded). No model output involved."""
    facts = sorted((f for life in stream.lives[:n_lives] for f in life), key=lambda f: (f.life, f.fid))
    pairs, same_value = [], 0
    for i, a in enumerate(facts):
        for b in facts[i + 1:]:
            if b.attr != a.attr or b.entity == a.entity:
                continue
            if b.value == a.value:
                same_value += 1
                continue
            cohort = "primary" if a.life == 1 and b.life > 1 else "simultaneous" if a.life == 1 else "secondary"
            pairs.append((a, b, cohort))
    return pairs, same_value


def make_instrumented(r3, tok, ctx):
    """Return a drop-in replacement for run.probe_facts. ctx['stream'], ctx['pairs'] are set per seed."""
    from being.v3.core import exact, first_answer
    from being.v3.lifegen import probes_for

    def probe_facts(b, facts, pools=None, with_paraphrase=True, with_logprob=True):
        rows, cand_lists = [], []
        for f in facts:                                    # (1) historical fields, identical computation and order
            p = probes_for(f)
            g = b.generate(p["template"])
            row = {"fid": f.fid, "attr": f.attr, "answer": f.value, "template": exact(g, f.value),
                   "template_gen": first_answer(g)}
            if with_paraphrase:
                row["paraphrase"] = [exact(b.generate(q), f.value) for q in p["paraphrase"]]
            if pools is not None:
                cands = r3.candidates_for(f, pools)
                lps = b.candidates_logprob(p["template"], cands)
                true, rest = lps[0], lps[1:]
                row["fc_rank"] = sum(x > true for x in rest)
                row["fc"] = row["fc_rank"] == 0
                row["margin"] = true - sum(rest) / len(rest)
                row["margin_best"] = true - max(rest)
                row["logprob"] = true
                cand_lists.append((row, p["template"], cands, lps))
            elif with_logprob:
                row["logprob"] = b.answer_logprob(p["template"], f.value)
            rows.append(row)
        with __import__("torch").no_grad():               # (2) read-only extras, after all historical measurements
            for row, prompt, cands, lps in cand_lists:
                meta = candidate_meta(tok, prompt, cands)
                row["fc_ties"] = sum(x == lps[0] for x in lps[1:])
                row["cands"] = [{"v": v, "lp": lp, **m} for v, lp, m in zip(cands, lps, meta)]
                row["prefix_fail"] = sum(not m["prefix_ok"] for m in meta)
            if ctx.get("pairs") is not None:
                paired = []
                for a, bf, cohort in ctx["pairs"]:
                    pa, pb = probes_for(a)["template"], probes_for(bf)["template"]
                    la = b.candidates_logprob(pa, [a.value, bf.value])
                    lb = b.candidates_logprob(pb, [a.value, bf.value])
                    paired.append({"a_fid": a.fid, "a_life": a.life, "b_fid": bf.fid, "b_life": bf.life, "attr": a.attr,
                                   "cohort": cohort, "V": a.value, "W": bf.value,
                                   "lV_A": la[0], "lW_A": la[1], "lV_B": lb[0], "lW_B": lb[1]})
                ctx["calls"].append(paired)
        return rows
    return probe_facts


def reproduction(result, ref_path):
    """Compare historical fields with an archived run of the same config: fact ids/values, generation, forced choice,
    margin_best and logprob (tolerance), checkpoint memory hashes."""
    ref = json.loads(Path(ref_path).read_text())
    rep = {"reference": str(ref_path), "reference_commit": ref.get("commit"), "seeds": {}}
    for r in result["per_seed"]:
        rr = next((x for x in ref["per_seed"] if x.get("probe_seed") == r["probe_seed"]), None)
        if rr is None:
            rep["seeds"][r["probe_seed"]] = "no reference seed"
            continue
        out = {"facts_match": True, "template_diff": 0, "fc_diff": 0, "max_abs_margin_diff": 0.0,
               "max_abs_logprob_diff": 0.0, "memory_hash_match": []}
        rcp = {c["after_life"]: c for c in rr["checkpoints"]}
        for c in r["checkpoints"]:
            L = c["after_life"]
            if L not in rcp:
                continue
            ref_rows = {x["fid"]: x for x in rcp[L]["life1"]}
            for x in c["life1"]:
                y = ref_rows.get(x["fid"])
                if y is None or y["answer"] != x["answer"]:
                    out["facts_match"] = False
                    continue
                out["template_diff"] += x["template"] != y["template"]
                out["fc_diff"] += x["fc"] != y["fc"]
                out["max_abs_margin_diff"] = max(out["max_abs_margin_diff"], abs(x["margin_best"] - y["margin_best"]))
                out["max_abs_logprob_diff"] = max(out["max_abs_logprob_diff"], abs(x["logprob"] - y["logprob"]))
            out["memory_hash_match"].append((L, c.get("memory_sha256") == rcp[L].get("memory_sha256")))
        rep["seeds"][r["probe_seed"]] = out
    return rep


def main():
    from being.competence import load_sentences, mean_loss
    from being.v3.core import V3Config, backbone_hash, load_qwen
    from being.v3.lifegen import make_stream
    from being.v3 import run as r3
    spec = importlib.util.spec_from_file_location("run_v3_dev", ROOT / "scripts/run_v3_dev.py")
    dev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev)

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["L", "S"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lives", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if not set(seeds) <= DEV:
        sys.exit("dev seeds 0-9 only")
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = V3Config(arm=a.arm, **{k: dev.parse_val(v) for k, v in (s.split("=", 1) for s in a.set)})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    model, tok = load_qwen()
    comp = {k: load_sentences(k) for k in ("eval", "modern")}
    base = {k: mean_loss(model, tok, v) for k, v in comp.items()}
    meta = {"which": "BIND-D1", "commit": commit, "config": asdict(cfg), "lives": a.lives,
            "device": str(next(model.parameters()).device), "backbone_sha256": backbone_hash(model),
            "base_ppl_loss": base, "per_seed": []}
    dev.write_atomic(out, {**meta, "status": "in_progress"})
    ctx = {}
    original = r3.probe_facts
    r3.probe_facts = make_instrumented(r3, tok, ctx)
    try:
        for seed in seeds:
            t0 = time.time()
            st = make_stream(seed, n_lives=a.lives)
            ctx["pairs"], n_same = pairs_for(st, a.lives)
            ctx["calls"] = []
            wiped = {"life1": r3.probe_facts(dev._Wiped(model, tok, cfg), st.lives[0], pools=st.pools)}
            row = r3.live_stream(model, tok, cfg, seed, n_lives=a.lives, checkpoints=CHECKPOINTS, probe_lives=(1,),
                                 competence=comp)
            if len(ctx["calls"]) != 1 + len(row["checkpoints"]):     # one probe call for wiped + one per checkpoint
                raise RuntimeError(f"probe calls {len(ctx['calls'])} != 1 + {len(row['checkpoints'])} checkpoints")
            wiped["paired"] = ctx["calls"][0]
            for cp, paired in zip(row["checkpoints"], ctx["calls"][1:]):
                cp["paired"] = paired
            row["wiped"] = wiped
            row["probe_seed"] = seed
            row["seconds"] = round(time.time() - t0, 1)
            row["pairs_frozen"] = [[x.fid, y.fid, c] for x, y, c in ctx["pairs"]]
            row["pairs_identical_value_excluded"] = n_same
            meta["per_seed"].append(row)
            dev.write_atomic(out, {**meta, "status": "in_progress"})
    finally:
        r3.probe_facts = original
    res = {**meta, "status": "complete"}
    if a.reference:
        res["reproduction"] = reproduction(res, ROOT / a.reference)
    dev.write_atomic(out, res)
    pf = sum(x.get("prefix_fail", 0) for r in res["per_seed"] for c in r["checkpoints"] for x in c["life1"])
    print("BIND-D1 DONE", out.name, "prefix_fail", pf, json.dumps(res.get("reproduction", {}), default=str)[:600])


if __name__ == "__main__":
    main()
