"""CLS-0C sealed test harness (Claude + Sol, 2026-10-29). Protocol: lab/CLS0C_SEALED_PROTOCOL.md.
Frozen and committed BEFORE any sealed item exists; tested only on public fixtures (tests/fixtures/cls0c_public.json).

Subcommands
  validate  DATASET                      schema, verbatim answers/evidence, checkpoints, counts, dev-wording duplicates
  blind     DATASET OUTDIR               adjudication package for Grok B: opaque ids, shuffled, no author labels
  reconcile DATASET ADJ MAPPING OUTDIR   author vs blind labels -> agreed / disputes (for David) / exclusions
  seal      DATASET FINAL_LABELS OUTDIR  apply resolutions, freeze sealed.json + SEALED.sha256
  bridge-write --seed 101 ...            write the adapter bank for the bridge seed (CLS-0B storage, resumable)
  run       SEALED_DIR ...               the ONE evaluation: B1_K1 primary, B0 descriptive, natural-question challenge,
                                         adapter bridge on seed 101 (identical synthetic questions)
Synthetic streams: fresh seeds 101-104 from the frozen BIND-I1/CLS-0B generator (condition S, no tokenizer matching).
"""
import argparse
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEALED_SEEDS = (101, 102, 103, 104)
BRIDGE_SEED = 101
N_SYN_SUPPORTED, N_SYN_IMPLICATION = 200, 150
LABELS = ("SUPPORTED", "IMPLICATION", "NEAR-MISS", "UNFAMILIAR", "TEMPORAL")
PER_WORLD = {"SUPPORTED": 10, "IMPLICATION": 10, "NEAR-MISS": 5, "UNFAMILIAR": 5, "TEMPORAL": 4}
N_WORLDS = 10
TEMPLATES_PER_ATTR = {"supported": 4, "implication": 3}
GATES = {"template": ("min", 0.95), "para_entailed": ("min", 0.90), "narrative_supported": ("min", 0.90),
         "near_unsupported": ("max", 0.10), "unfamiliar_unsupported": ("max", 0.10),
         "implication_commit": ("max", 0.10), "temporal": ("min", 0.90), "provenance": ("min", 0.95)}
PREDICTION = "B1_K1 fails implication_commit in both formats (development: 89/110 committed)"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


B = _load("cls0c_b_for_sealed", "scripts/cls0c_b.py")
C0B = _load("cls0b_for_sealed", "scripts/cls0b.py")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


# ---------------------------------------------------------------- dataset checks (pure, tested) ----------------------
def dev_wordings(lifegen):
    """Every development probe wording as a normalised template prefix (for duplicate rejection, never shown to the
    author)."""
    out = set()
    for a in lifegen.TRAIN:
        for t in list(lifegen.TRAIN[a]) + list(lifegen.PARAPHRASE[a]):
            pre = t.split("{v}")[0] if "{v}" in t else t
            out.add(B.norm(pre.replace("{e}", "E")))
    return out


def _tokens(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def validate(ds, lifegen, strict_counts=True):
    """Returns (errors, warnings). Errors block sealing; warnings (near-duplicates) go to adjudication."""
    E, W = [], []
    worlds = ds.get("worlds", [])
    if strict_counts and len(worlds) != N_WORLDS:
        E.append(f"expected {N_WORLDS} worlds, got {len(worlds)}")
    ids = set()
    for w in worlds:
        wid = w.get("world_id")
        xs = {x["xid"]: x for x in w.get("experiences", [])}
        for x in w.get("experiences", []):
            if x["xid"] in ids:
                E.append(f"duplicate id {x['xid']}")
            ids.add(x["xid"])
            if not (isinstance(x.get("life"), int) and 1 <= x["life"] <= 5) or not x.get("text", "").strip():
                E.append(f"{x['xid']}: bad life/text")
        counts = {lab: 0 for lab in LABELS}
        world_text = " ".join(x["text"] for x in xs.values()).lower()
        for it in w.get("items", []):
            iid, lab, stem = it.get("iid"), it.get("label"), it.get("stem", "")
            if iid in ids:
                E.append(f"duplicate id {iid}")
            ids.add(iid)
            if lab not in LABELS:
                E.append(f"{iid}: unknown label {lab}")
                continue
            counts[lab] += 1
            if not stem.strip() or stem.rstrip().endswith(("?", ".")):
                E.append(f"{iid}: stem must be an unfinished sentence")
            if lab in ("SUPPORTED", "TEMPORAL"):
                xids = it.get("xids") or []
                if not xids or any(x not in xs for x in xids):
                    E.append(f"{iid}: xids missing/unknown")
                    continue
                for x in xids:
                    if B.evidence(xs[x]["text"], it.get("answer", "")) is None:
                        E.append(f"{iid}: answer {it.get('answer')!r} not verbatim (whole words) in {x}")
                ev = it.get("evidence", "")
                if ev and not any(ev.strip() in xs[x]["text"] for x in xids):
                    E.append(f"{iid}: evidence sentence not verbatim in its experience")
                if lab == "TEMPORAL":
                    cp = it.get("checkpoint")
                    if not isinstance(cp, int) or not (1 <= cp <= 5) or any(xs[x]["life"] > cp for x in xids):
                        E.append(f"{iid}: bad checkpoint")
                    req = B.time_constraint(stem)
                    if req is not None and any(xs[x]["life"] != req for x in xids):
                        E.append(f"{iid}: 'At life {req}' item cites evidence from another life")
            if lab == "UNFAMILIAR":
                caps = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?", stem)
                if caps and all(c.lower() in world_text for c in caps):
                    W.append(f"{iid}: unfamiliar stem names someone who appears in the world")
        if strict_counts and counts != PER_WORLD:
            E.append(f"{wid}: label counts {counts} != {PER_WORLD}")
    dev = dev_wordings(lifegen)
    devtok = [_tokens(d) for d in dev]
    tpl = ds.get("attribute_templates", {})
    if strict_counts and set(tpl) != set(lifegen.TRAIN):
        E.append("attribute_templates must cover exactly the ten attributes")
    for a, d in tpl.items():
        for kind, n in TEMPLATES_PER_ATTR.items():
            ts = d.get(kind, [])
            if strict_counts and len(ts) != n:
                E.append(f"{a}/{kind}: expected {n} templates, got {len(ts)}")
            for t in ts:
                if t.count("{e}") != 1 or t.rstrip().endswith(("?", ".")):
                    E.append(f"{a}/{kind}: bad template {t!r}")
                nt = B.norm(t.replace("{e}", "E"))
                if nt in dev:
                    E.append(f"{a}/{kind}: duplicates a development wording: {t!r}")
                else:
                    tt = _tokens(nt)
                    if any(len(tt & d_) / max(1, len(tt | d_)) >= 0.8 for d_ in devtok):
                        W.append(f"{a}/{kind}: near-duplicate of a development wording: {t!r}")
    return E, W


def blind(ds, seed_str):
    """Adjudication package: every item and template gets an opaque id; order shuffled; author labels, answers,
    evidence, plausible completions and explanations removed; templates not split by kind."""
    rng = random.Random(f"CLS0C-blind-{seed_str}")
    mapping, worlds = {}, []
    for w in ds["worlds"]:
        items = list(w["items"])
        rng.shuffle(items)
        out = []
        for it in items:
            oid = "S" + sha(f"{seed_str}|{it['iid']}")[:10]
            mapping[oid] = it["iid"]
            out.append({"id": oid, "stem": it["stem"]})
        exps = []
        for x in w["experiences"]:
            ref = "E" + sha(f"{seed_str}|{x['xid']}")[:8]
            mapping[ref] = x["xid"]
            exps.append({"ref": ref, "life": x["life"], "text": x["text"]})
        exps.sort(key=lambda x: (x["life"], x["ref"]))
        worlds.append({"world": "N" + sha(f"{seed_str}|{w['world_id']}")[:6], "experiences": exps,
                       "statements": out})
        mapping["N" + sha(f"{seed_str}|{w['world_id']}")[:6]] = w["world_id"]
    rng.shuffle(worlds)
    tpls = []
    for a, d in ds["attribute_templates"].items():
        for kind in ("supported", "implication"):
            for t in d[kind]:
                oid = "T" + sha(f"{seed_str}|{a}|{kind}|{t}")[:10]
                mapping[oid] = {"attr": a, "kind": kind, "template": t}
                tpls.append({"id": oid, "attribute": a, "template": t})
    rng.shuffle(tpls)
    return {"worlds": worlds, "templates": tpls}, mapping


def reconcile(ds, adj, mapping):
    """adj: {"statements": {oid: {"label": L, "answer": str|None, "refs": [...]}}, "templates": {oid: "SUPPORTED"|
    "IMPLICATION"|"AMBIGUOUS"}}. Returns agreed, disputes (for David), per item."""
    author = {it["iid"]: it for w in ds["worlds"] for it in w["items"]}
    agreed, disputes = {}, []
    for oid, j in adj.get("statements", {}).items():
        iid = mapping[oid]
        a = author[iid]
        j = dict(j, refs=[mapping.get(r, r) for r in (j.get("refs") or [])])      # opaque refs -> author xids
        same = j.get("label") == a["label"]
        if same and a["label"] in ("SUPPORTED", "TEMPORAL"):
            same = B.norm(j.get("answer") or "") == B.norm(a.get("answer", "")) and \
                set(j.get("refs") or []) & set(a.get("xids") or [])
        if j.get("label") == "AMBIGUOUS" or not same:
            disputes.append({"iid": iid, "author": {k: a.get(k) for k in ("label", "answer", "xids", "stem")},
                             "adjudicator": j})
        else:
            agreed[iid] = a["label"]
    for oid, lab in adj.get("templates", {}).items():
        m = mapping[oid]
        key = f"{m['attr']}|{m['kind']}|{m['template']}"
        want = "SUPPORTED" if m["kind"] == "supported" else "IMPLICATION"
        if lab == want:
            agreed[key] = want
        else:
            disputes.append({"template": key, "author": want, "adjudicator": lab})
    missing = [i for i in author if i not in agreed and not any(d.get("iid") == i for d in disputes)]
    return agreed, disputes, missing


def sample_synthetic(seed, facts, templates, n, salt):
    """Deterministic, performance-blind allocation: (fact, template) pairs ordered by a fixed hash; the first n."""
    pairs = [(f, t) for f in facts for t in templates.get(f["attr"], [])]
    pairs.sort(key=lambda p: sha(f"{salt}|{seed}|{p[0]['eid']}|{p[1]}"))
    return pairs[:n]


def split_counts(total, k):
    return [total // k + (1 if i < total % k else 0) for i in range(k)]


def gate_report(summ):
    """Every gate is applied separately to each format in which the category exists (no averaging)."""
    rep, ok = {}, True
    for g, (kind, thr) in GATES.items():
        for fmt in ("synthetic", "narrative"):
            v = summ.get(fmt, {}).get(g)
            if v is None:
                continue
            p = v >= thr if kind == "min" else v <= thr
            rep[f"{fmt}:{g}"] = {"value": v, "threshold": f"{'>=' if kind == 'min' else '<='}{thr}", "pass": p}
            ok &= p
    return ok, rep


def score_rows(rows):
    """rows: dicts {fmt, cat, answered, correct, provenance_ok}. cat in template / para_entailed / implication /
    near_miss / unfamiliar / temporal / narrative_supported / natural_question."""
    out = {}
    for fmt in ("synthetic", "narrative"):
        R = [r for r in rows if r["fmt"] == fmt]
        by = lambda c: [r for r in R if r["cat"] == c]
        rate = lambda rs, f: (sum(map(f, rs)) / len(rs)) if rs else None
        pos = by("template") + by("para_entailed") + by("narrative_supported") + by("temporal")
        ans = [r for r in pos if r["answered"]]
        out[fmt] = {"n": {c: len(by(c)) for c in ("template", "para_entailed", "narrative_supported", "implication",
                                                  "near_miss", "unfamiliar", "temporal", "natural_question")},
                    "template": rate(by("template"), lambda r: r["correct"]),
                    "para_entailed": rate(by("para_entailed"), lambda r: r["correct"]),
                    "narrative_supported": rate(by("narrative_supported"), lambda r: r["correct"]),
                    "implication_commit": rate(by("implication"), lambda r: r["answered"]),
                    "near_unsupported": rate(by("near_miss"), lambda r: r["answered"]),
                    "unfamiliar_unsupported": rate(by("unfamiliar"), lambda r: r["answered"]),
                    "temporal": rate(by("temporal"), lambda r: r["correct"]),
                    "provenance": rate(ans, lambda r: r["provenance_ok"]),
                    "natural_question_challenge": rate(by("natural_question"), lambda r: r["correct"])}
    return out


# ---------------------------------------------------------------- evaluation items -----------------------------------
def synthetic_items(seed, lifegen, final_templates):
    """Fresh stream (frozen generator, condition S, no tokenizer matching) and its items at life 5."""
    i1 = _load("bind_i1_for_sealed", "scripts/bind_i1.py")
    eps, _ = C0B.build_episodes(seed, lifegen, i1, None)
    cues, cal, neg, temporal = C0B.build_cues(seed, eps, lifegen)
    facts = [e for e in eps if e["kind"] != "temporal"]
    k = SEALED_SEEDS.index(seed)
    n_sup = split_counts(N_SYN_SUPPORTED, len(SEALED_SEEDS))[k]
    n_imp = split_counts(N_SYN_IMPLICATION, len(SEALED_SEEDS))[k]
    items = []
    for c in cues:
        if c["form"] == "template":
            items.append({"key": f"S|{seed}|{c['eid']}|template", "cue": c["cue"], "cat": "template",
                          "gold_value": c["value"], "gold_eids": {c["eid"]}})
    sup = {a: d["supported"] for a, d in final_templates.items()}
    imp = {a: d["implication"] for a, d in final_templates.items()}
    for f, t in sample_synthetic(seed, facts, sup, n_sup, "sup"):
        items.append({"key": f"S|{seed}|{f['eid']}|sup|{sha(t)[:8]}", "cue": t.format(e=f["entity"]),
                      "cat": "para_entailed", "gold_value": f["value"], "gold_eids": {f["eid"]}})
    for f, t in sample_synthetic(seed, facts, imp, n_imp, "imp"):
        items.append({"key": f"S|{seed}|{f['eid']}|imp|{sha(t)[:8]}", "cue": t.format(e=f["entity"]),
                      "cat": "implication"})
    for j, c in enumerate(neg):
        items.append({"key": f"S|{seed}|N{j}|{c['form']}", "cue": c["cue"],
                      "cat": "near_miss" if c["form"] == "near_miss" else "unfamiliar"})
    tq = {e["value"]: e["eid"] for e in eps if e["kind"] == "temporal"}
    for c in C0B.temporal_items(temporal, 5):
        items.append({"key": f"S|{seed}|T|{c['form']}", "cue": c["cue"], "cat": "temporal",
                      "gold_value": c["expect"], "gold_eids": {tq[c["expect"]]}})
    return eps, cal, items


def narrative_items(world):
    cat = {"SUPPORTED": "narrative_supported", "IMPLICATION": "implication", "NEAR-MISS": "near_miss",
           "UNFAMILIAR": "unfamiliar", "TEMPORAL": "temporal"}
    out = []
    for it in world["items"]:
        d = {"key": f"N|{world['world_id']}|{it['iid']}", "cue": it["stem"], "cat": cat[it["label"]],
             "checkpoint": it.get("checkpoint", 5)}
        if it["label"] in ("SUPPORTED", "TEMPORAL"):
            d.update(gold_value=it["answer"], gold_eids=set(it["xids"]))
        out.append(d)
        if it["label"] == "SUPPORTED" and it.get("question"):
            out.append({**d, "key": d["key"] + "|q", "cue": it["question"], "cat": "natural_question"})
    return out


# ---------------------------------------------------------------- GPU stages ----------------------------------------
def bridge_write(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from being.cls.episodic import BudgetExceeded, EpisodicMemory
    from being.v3 import lifegen
    i1 = _load("bind_i1_for_bridge", "scripts/bind_i1.py")
    eps, _ = C0B.build_episodes(a.seed, lifegen, i1, None)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    bank, log = out / f"bridge_seed{a.seed}_bank.pt", out / f"bridge_seed{a.seed}_writes.json"
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if cuda else torch.float32,
                                                 attn_implementation="eager").to("cuda" if cuda else "cpu").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = dict(C0B.CFG) | (json.loads(a.test_cfg) if a.test_mode else {})
    mem = EpisodicMemory(model, tok, seed=a.seed, **cfg)
    writes = json.loads(log.read_text()) if log.exists() else {}
    if bank.exists():
        mem.load(bank)
    for e in eps[: (a.test_limit or None)]:
        if e["eid"] in mem.episodes or e["eid"] in writes:          # recorded outcomes are terminal
            continue
        t = time.time()
        try:
            ep = mem.write(e["eid"], e["text"], e["time"], e["source"], deadline=time.time() + C0B.EPISODE_CAP_S)
            writes[e["eid"]] = {"ok": True, "mask": ep.mask}
            mem.save(bank)
        except BudgetExceeded:
            writes[e["eid"]] = {"ok": False, "budget_censored": True}
        except RuntimeError as err:
            writes[e["eid"]] = {"ok": False, "error": str(err)}
        writes[e["eid"]]["seconds"] = round(time.time() - t, 1)
        log.write_text(json.dumps(writes, indent=1))
        print(time.strftime("%H:%M:%S"), "bridge write", e["eid"], writes[e["eid"]], flush=True)
    recon = {e["eid"]: mem.reconstruct(e["eid"], max_new=len(tok(e["text"]).input_ids) + 16).strip()
             for e in eps if e["eid"] in mem.episodes}
    (out / f"bridge_seed{a.seed}_recon.json").write_text(json.dumps(
        {"recon": recon, "exact": {k: v == next(x["text"] for x in eps if x["eid"] == k).strip()
                                   for k, v in recon.items()}, "config": cfg}, indent=1))
    print("BRIDGE WRITE DONE", sum(w.get("ok", False) for w in writes.values()), "/", len(eps), flush=True)


def run(a):
    import torch
    from being.cls.keys import ExternalKeys
    from being.v3 import lifegen
    sd = ROOT / a.sealed
    man = (sd / "SEALED.sha256").read_text().split()
    sealed_bytes = (sd / "sealed.json").read_bytes()
    if man[0] != sha(sealed_bytes):
        sys.exit("sealed.json does not match SEALED.sha256")
    S = json.loads(sealed_bytes)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "RUN_STARTED"
    if marker.exists() and not a.test_mode:
        sys.exit("the sealed run was already started once; results are never regenerated")
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    cuda = torch.cuda.is_available()
    dev, dtype = ("cuda", torch.bfloat16) if cuda else ("cpu", torch.float32)
    if not a.test_mode:
        d2 = json.loads((ROOT / "results/cls0b_final/seed2.json").read_text())
        if d2["model_manifest"]["files"] != B.manifest(a.model) or \
                d2["external_manifest"]["files"] != B.manifest(a.external):
            sys.exit("model or embedder files differ from the ones used in development")
    ek = ExternalKeys(a.external, dev, dtype)
    reader = B.Reader(a.model, dev, dtype)
    if not reader.single_token_answers and not a.test_mode:
        sys.exit('" yes" / " no" are not single tokens')
    banks = []                                              # (fmt, bank_id, texts, times, items, cal)
    for seed in SEALED_SEEDS:
        eps, cal, items = synthetic_items(seed, lifegen, S["templates"])
        banks.append(("synthetic", f"seed{seed}", {e["eid"]: e["text"] for e in eps},
                      {e["eid"]: e["time"] for e in eps}, items, cal))
    for w in S["worlds"]:
        banks.append(("narrative", w["world_id"], {x["xid"]: x["text"] for x in w["experiences"]},
                      {x["xid"]: x["life"] for x in w["experiences"]}, narrative_items(w), None))
    rb = json.loads((ROOT / a.bridge).read_text()) if a.bridge else None
    if rb is not None and not rb["recon"]:
        print("bridge: no reconstructed episodes - bridge not evaluated", flush=True)
        rb = None
    if rb is not None:
        bi = next(b for b in banks if b[1] == f"seed{BRIDGE_SEED}")
        banks.append(("bridge", f"seed{BRIDGE_SEED}", rb["recon"], {k: bi[3][k] for k in rb["recon"]},
                      [dict(it) for it in bi[4]], bi[5]))
    rows, t0 = [], time.time()
    for fmt, bid, texts, times, items, cal in banks:
        eids = list(texts)
        ev = dict(zip(eids, ek.embed([texts[e] for e in eids])))
        qs = sorted({it["cue"] for it in items} | ({c["cue"] for c in cal} if cal else set()))
        qv = dict(zip(qs, ek.embed(qs, query=True)))
        bank_at = lambda cp: [e for e in eids if times[e] <= cp]
        rank = lambda cue, cp: sorted(bank_at(cp), key=lambda e: -float(qv[cue] @ ev[e]))
        thr = C0B.threshold([max(float(qv[c["cue"]] @ ev[e]) for e in eids) for c in cal]) if cal else None
        need = sorted({(it["cue"], e) for it in items for e in rank(it["cue"], it.get("checkpoint", 5))[:B.K_CAND]})
        mem_of = lambda e: B.MEM.format(t=times[e], text=texts[e])
        V = dict(zip(need, reader.yesno([B.VERIFY.format(mem=mem_of(e), cue=c) for c, e in need])))
        plan = []
        for it in items:
            cp = it.get("checkpoint", 5)
            cands = rank(it["cue"], cp)[:B.K_CAND]
            sup = [{"eid": e, "time": times[e], "margin": V[(it["cue"], e)]} for e in cands]
            ch = B.select([s_ for s_ in sup if s_["margin"] > 0], B.time_constraint(it["cue"]))
            b0 = None
            if thr is not None:
                top = rank(it["cue"], cp)[0]
                b0 = top if float(qv[it["cue"]] @ ev[top]) >= thr else None
            plan.append((it, cands, ch, b0))
        xk = sorted({(it["cue"], e) for it, _, ch, b0 in plan for e in ((ch or {}).get("eid"), b0) if e})
        X = dict(zip(xk, map(B.cut_value, reader.generate([B.EXTRACT_VALUE.format(mem=mem_of(e), cue=c)
                                                             for c, e in xk], B.MAX_NEW_VALUE))))
        for it, cands, ch, b0 in plan:
            for meth, e in (("B1_K1", (ch or {}).get("eid")), ("B0", b0)):
                if meth == "B0" and thr is None:
                    continue
                o = B.commit(e, X[(it["cue"], e)] if e else None, texts)
                g = B.grade(o, it["gold_value"], it["gold_eids"], texts) if "gold_value" in it else \
                    {"answered": o["answer"] is not None, "correct": False, "provenance_ok": False}
                rows.append({"method": meth, "fmt": fmt, "bank": bid, "key": it["key"], "cat": it["cat"],
                             "cue": it["cue"], **g, **o, "candidates": cands,
                             "margins": {e_: V[(it["cue"], e_)] for e_ in cands}})
        print(f"{fmt} {bid}: {len(items)} items, {len(need)} verifications", flush=True)
    prim = [r for r in rows if r["method"] == "B1_K1" and r["fmt"] in ("synthetic", "narrative")]
    summ = score_rows(prim)
    passed, rep = gate_report(summ)
    bridge = None
    if rb is not None:
        raw = {r["key"]: r for r in prim if r["bank"] == f"seed{BRIDGE_SEED}"}
        br = [r for r in rows if r["fmt"] == "bridge" and r["method"] == "B1_K1"]
        bridge = {"reconstruction_exact": sum(rb["exact"].values()), "episodes_reconstructed": len(rb["exact"]),
                  "episodes_in_stream": len(bi[2]),
                  "answer_agreement": sum(B.norm(r["answer"] or "") == B.norm(raw[r["key"]]["answer"] or "")
                                          for r in br) / len(br),
                  "provenance_agreement": sum(r["eid"] == raw[r["key"]]["eid"] for r in br) / len(br),
                  "bridge_scores": score_rows([dict(r, fmt="synthetic") for r in br])["synthetic"]}
    res = {"which": "CLS-0C sealed", "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                             capture_output=True, text=True).stdout.strip(),
           "sealed_sha256": man[0], "gpu": torch.cuda.get_device_name(0) if cuda else None, "test_mode": a.test_mode,
           "primary": "B1_K1", "prediction": PREDICTION, "scores_B1_K1": summ, "gates": rep, "passed": passed,
           "scores_B0_descriptive": score_rows([r for r in rows if r["method"] == "B0"]), "bridge": bridge,
           "wall_s": round(time.time() - t0, 1)}
    (out / "sealed_rows.json").write_text(json.dumps(rows, indent=1, default=sorted, allow_nan=False))
    (out / "sealed_summary.json").write_text(json.dumps(res, indent=1, allow_nan=False))
    print("CLS-0C SEALED", json.dumps({"passed": passed, "gates": rep, "bridge": bridge}, indent=1))
    print(f"CLS-0C SEALED DONE passed={passed}")


def main():
    from being.v3 import lifegen
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate"); v.add_argument("dataset")
    b = sp.add_parser("blind"); b.add_argument("dataset"); b.add_argument("outdir")
    r = sp.add_parser("reconcile"); r.add_argument("dataset"); r.add_argument("adj"); r.add_argument("mapping")
    r.add_argument("outdir")
    s = sp.add_parser("seal"); s.add_argument("dataset"); s.add_argument("final_labels"); s.add_argument("outdir")
    bw = sp.add_parser("bridge-write")
    for p in (bw,):
        p.add_argument("--seed", type=int, default=BRIDGE_SEED); p.add_argument("--model", required=True)
        p.add_argument("--out", required=True); p.add_argument("--test-mode", action="store_true")
        p.add_argument("--test-cfg", default="{}"); p.add_argument("--test-limit", type=int, default=0)
    rn = sp.add_parser("run"); rn.add_argument("sealed"); rn.add_argument("--model", required=True)
    rn.add_argument("--external", required=True); rn.add_argument("--out", required=True)
    rn.add_argument("--bridge", default=None); rn.add_argument("--test-mode", action="store_true")
    a = ap.parse_args()
    if a.cmd == "validate":
        E, W = validate(json.loads(Path(a.dataset).read_text()), lifegen)
        print(json.dumps({"errors": E, "warnings": W}, indent=1))
        sys.exit(1 if E else 0)
    if a.cmd == "blind":
        ds = json.loads(Path(a.dataset).read_text())
        pkg, mapping = blind(ds, sha(Path(a.dataset).read_bytes())[:16])
        o = Path(a.outdir); o.mkdir(parents=True, exist_ok=True)
        (o / "adjudication_package.json").write_text(json.dumps(pkg, indent=1))
        (o / "PRIVATE_mapping.json").write_text(json.dumps(mapping, indent=1))
        print("blind package written")
    if a.cmd == "reconcile":
        ag, dis, miss = reconcile(json.loads(Path(a.dataset).read_text()), json.loads(Path(a.adj).read_text()),
                                  json.loads(Path(a.mapping).read_text()))
        o = Path(a.outdir); o.mkdir(parents=True, exist_ok=True)
        (o / "agreed.json").write_text(json.dumps(ag, indent=1))
        (o / "disputes.json").write_text(json.dumps(dis, indent=1))
        print(json.dumps({"agreed": len(ag), "disputes": len(dis), "unjudged": miss}, indent=1))
    if a.cmd == "seal":
        ds = json.loads(Path(a.dataset).read_text())
        fl = json.loads(Path(a.final_labels).read_text())          # {"exclude_items": [...], "exclude_templates": [...]}
        ex_i, ex_t = set(fl.get("exclude_items", [])), set(fl.get("exclude_templates", []))
        worlds = [{**w, "items": [it for it in w["items"] if it["iid"] not in ex_i]} for w in ds["worlds"]]
        tpl = {a_: {k: [t for t in d[k] if f"{a_}|{k}|{t}" not in ex_t] for k in ("supported", "implication")}
               for a_, d in ds["attribute_templates"].items()}
        E, _ = validate({"worlds": worlds, "attribute_templates": tpl}, lifegen, strict_counts=False)
        if E:
            sys.exit("seal refused: " + "; ".join(E[:5]))
        sealed = json.dumps({"worlds": worlds, "templates": tpl, "exclusions": fl}, indent=1, sort_keys=True)
        o = Path(a.outdir); o.mkdir(parents=True, exist_ok=True)
        (o / "sealed.json").write_text(sealed)
        (o / "SEALED.sha256").write_text(f"{sha(sealed)}  sealed.json\n")
        print("SEALED", sha(sealed))
    if a.cmd == "bridge-write":
        bridge_write(a)
    if a.cmd == "run":
        run(a)


if __name__ == "__main__":
    main()
