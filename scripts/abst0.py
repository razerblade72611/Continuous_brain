"""ABST-0 (lab/ABST0_PROTOCOL.md): learning a support boundary during consolidation. Development; cortex-only
(verbatim experience archive). Stages:
  prep      seed -> stream, QA corpus, negative corpora for every sleep (arm-independent, before any training),
             name-swapped G items, transition-exposure categories, blind audit sample           -> prep_seed{S}.json
  train     seed, E -> arms P, U, G (budget-matched; equal positive exposure), checkpoint per arm per sleep
  evalck    seed, sleep, ckpt|base, mode merged|unmerged -> fresh-process fp32 evaluation        -> eval_*.json
  calibrate seed 16: prep + train P sleep 1 for E in {4,8,16} (evaluation by evalck) -> choose E (analyze-cal)
  analyze   integrity (merged vs unmerged, zero mismatches), measures, D1-D4, mechanism gate, verdict
Pure functions (stream, probes, selection, classification, aggregation, criteria) are tested in tests/test_abst0.py.
"""
import argparse, copy, hashlib, importlib.util, json, math, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("cons0a_for_abst0", ROOT / "scripts/cons0a.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

DEV_SEEDS, CAL_SEED = (12, 13, 14, 15), 16
ATTRS9 = [a for a in ("city", "occupation", "favorite object", "pet species", "childhood fear", "workplace", "hobby",
                      "birth month", "allergy")]
THIRD_RATIO, NEG_PER_ENTITY, N_HNM, N_FRESH, MEM_MAX_TOK, T_POOL, T_PER_SLEEP = 0.5, 3, 40, 20, 1024, 400, 12
ARMS = ("P", "U", "G")
TH = {"pos": 0.50, "pos_vs_p": 0.10, "first": 0.50, "conf": 0.10, "unk": 0.90, "tpre_unk": 0.80, "tpost": 0.50,
      "tpost_vs_p": 0.15, "ret_vs_p": 0.10, "mech": 0.30, "seeds": 3, "ppl_mean": 1.02, "ppl_seed": 1.05}
NEG_GEN = ("Memory: {memory}\n\nWrite three different questions about {e} that this memory does NOT answer. Each "
           "question must ask for one specific fact about {e} that the memory does not mention.\n"
           "Write each on its own line as: Q: <question>\n")
NEG_GEN_FRESH = ("Write three different questions about a person named {t}. Each question must ask for one specific "
                 "fact about {t}.\nWrite each on its own line as: Q: <question>\n")
VERIFY3 = ("Memory: Mara Lind plays the cello every evening.\nQuestion: What instrument does Mara Lind play?\n"
           "Does the memory answer the question? Answer SUPPORTED, UNSUPPORTED or AMBIGUOUS.\nAnswer: SUPPORTED\n\n"
           "Memory: Mara Lind plays the cello every evening.\nQuestion: Which city does Mara Lind live in?\n"
           "Does the memory answer the question? Answer SUPPORTED, UNSUPPORTED or AMBIGUOUS.\nAnswer: UNSUPPORTED\n\n"
           "Memory: Mara Lind plays the cello every evening.\nQuestion: What about Mara Lind?\n"
           "Does the memory answer the question? Answer SUPPORTED, UNSUPPORTED or AMBIGUOUS.\nAnswer: AMBIGUOUS\n\n"
           "Memory: {memory}\nQuestion: {q}\n"
           "Does the memory answer the question? Answer SUPPORTED, UNSUPPORTED or AMBIGUOUS.\nAnswer:")
LABELS = ("SUPPORTED", "UNSUPPORTED", "AMBIGUOUS")


def sha(x):
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()


def code_hash():
    return sha(b"".join((ROOT / p).read_bytes() for p in ("scripts/abst0.py", "scripts/cons0a.py", "being/v3/lifegen.py")))


# ------------------------------------------------------------------ streams (pure) ---------------------------------
def _names(rng, lg, n, used):
    out = []
    while len(out) < n:
        f, l = lg._word(rng, 2).capitalize(), lg._word(rng, rng.choice((1, 2))).capitalize()
        if len(f) < 4 or len(l) < 3 or f.lower() == "david" or f.lower() in used or l.lower() in used:
            continue
        used.update((f.lower(), l.lower()))
        out.append(f"{f} {l}")
    return out


def make_stream(seed, lg):
    """Deterministic ABST-0 stream (protocol 'Streams'). Returns a dict of plain data."""
    rng = random.Random(f"ABST0-stream-{seed}")
    used = set()
    core = _names(rng, lg, 16, used)
    transient = _names(rng, lg, 13, used)
    T = _names(rng, lg, T_POOL, used)
    V = _names(rng, lg, N_FRESH, used)
    vals = {}
    for a in ATTRS9:
        pool = set()
        while len(pool) < 80:
            w = lg._word(rng, rng.choice((1, 2))) + rng.choice(lg.SUFFIX[a])
            if not (set(re.findall(r"[a-z]+", w.lower())) & used):
                pool.add(w)
        vals[a] = sorted(pool)
        rng.shuffle(vals[a])
    vi = {a: 0 for a in ATTRS9}

    def val(a, avoid=()):
        while True:
            v = vals[a][vi[a]]; vi[a] += 1
            if v not in avoid:
                return v
    order = list(range(16)); rng.shuffle(order)
    A = sorted(core[i] for i in order[:8]); H = sorted(core[i] for i in order[8:])
    l1 = A[:4] + H[:4]; l2 = A[4:] + H[4:]
    intro = {e: 1 for e in l1} | {e: 2 for e in l2}
    facts, cells = [], {}

    def add(life, e, a, role, v=None):
        assert (e, a) not in cells or role == "update_new", (e, a)
        v = v or val(a)
        f = {"fid": f"X{seed}L{life}F{len([x for x in facts if x['life'] == life]):02d}", "life": life, "entity": e,
             "attr": a, "value": v, "role": role, "text": " ".join(t.format(e=e, v=v) for t in lg.TRAIN[a])}
        facts.append(f)
        cells.setdefault((e, a), []).append(f)
        return f
    free = {e: list(ATTRS9) for e in core}
    for e in core:
        rng.shuffle(free[e])
    # planned cells first (reserve attributes): transitions 4A/4H (2 on life-1, 2 on life-2 entities each); updates 1A/1H
    trans = [(e, free[e].pop()) for e in A[:2] + A[4:6] + H[:2] + H[4:6]]
    upd = [(e, free[e].pop()) for e in (A[2], H[2])]
    for life, group in ((1, l1), (2, l2)):
        for e in group:
            add(life, e, free[e].pop(), "core")
        if life == 1:
            for e in transient[:2]:
                add(1, e, rng.choice(ATTRS9), "transient")
        else:
            for e, a in upd:
                add(2, e, a, "update_init")
    for e in transient[2:5]:
        add(3, e, rng.choice(ATTRS9), "transient")
    second = [e for e in core if e not in {x for x, _ in trans}][:7]
    for e in second:
        add(3, e, free[e].pop(), "second")
    for e, a in trans:
        add(4, e, a, "transition")
    for e, a in upd:
        old = cells[(e, a)][0]["value"]
        add(4, e, a, "update_new", v=val(a, avoid={old}))
    for e in transient[5:8]:
        add(4, e, rng.choice(ATTRS9), "transient")
    for e in transient[8:11]:
        add(5, e, rng.choice(ATTRS9), "transient")
    extra = [e for e in core if free[e]][:7]
    for e in extra:
        add(5, e, free[e].pop(), "extra")
    for e in transient:
        intro.setdefault(e, min(f["life"] for f in facts if f["entity"] == e) if any(f["entity"] == e for f in facts) else 99)
    fresh = [(v, ATTRS9[i % len(ATTRS9)]) for i, v in enumerate(V)]
    return {"seed": seed, "facts": facts, "A": A, "H": H, "transient": [e for e in transient if intro.get(e, 99) < 99],
            "intro": intro, "T": T, "V": V, "fresh": fresh, "transitions": [list(x) for x in trans],
            "updates": [list(x) for x in upd]}


def check_stream(st):
    S_words = {w.lower() for e in st["A"] + st["H"] + st["transient"] for w in e.split()}
    T_words = {w.lower() for e in st["T"] for w in e.split()}
    V_words = {w.lower() for e in st["V"] for w in e.split()}
    assert not (S_words & T_words) and not (S_words & V_words) and not (T_words & V_words), "name pools overlap"
    val_words = {w.lower() for f in st["facts"] for w in re.findall(r"[A-Za-z]+", f["value"])}
    assert not (val_words & (S_words | T_words | V_words)), "values overlap names"
    assert len(st["A"]) == len(st["H"]) == 8 and len(st["transitions"]) == 8 and len(st["updates"]) == 2
    tr = {tuple(x) for x in st["transitions"]}
    for f in st["facts"]:
        if (f["entity"], f["attr"]) in tr:
            assert f["life"] == 4 and f["role"] == "transition", f
    for e, a in st["transitions"]:
        assert st["intro"][e] in (1, 2)
    up = {tuple(x) for x in st["updates"]}
    assert not (up & tr)
    for f in st["facts"]:
        if (f["entity"], f["attr"]) in up:
            assert (f["life"], f["role"]) in ((2, "update_init"), (4, "update_new")), f
    for c in up:
        assert sum((f["entity"], f["attr"]) == c for f in st["facts"]) == 2
    for c in tr:
        assert sum((f["entity"], f["attr"]) == c for f in st["facts"]) == 1
    for life in range(1, 6):
        assert any(f["life"] == life for f in st["facts"])
    return True


def episodes(st, upto=None, life=None):
    return [f for f in st["facts"] if (upto is None or f["life"] <= upto) and (life is None or f["life"] == life)]


def memory(st, e, L):
    """Exact-name string index of the checkpoint's own acquired episodes (supervision scaffolding)."""
    eps = [f for f in episodes(st, upto=L) if re.search(r"(?<![\w])" + re.escape(e) + r"(?![\w])", f["text"])]
    assert all(f in eps for f in episodes(st, upto=L) if f["entity"] == e), "index incomplete"
    return " ".join(f["text"] for f in eps)


def known(st, L):
    """(entity, attr) -> current value, from facts with life <= L (the latest value wins)."""
    k = {}
    for f in sorted(episodes(st, upto=L), key=lambda f: f["life"]):
        k[(f["entity"], f["attr"])] = f["value"]
    return k


# ------------------------------------------------------------------ probes (pure) ----------------------------------
def probes(st, L, lg):
    """Every probe at checkpoint L (protocol 'Probes'); negatives use the template stem, positives template + paraphrases."""
    P, kn = [], known(st, L)
    upd = {tuple(x) for x in st["updates"]}
    tr = {tuple(x) for x in st["transitions"]}
    planned = upd | tr

    def pos(cat, f, value):
        P.append({"cat": cat, "key": f"{f['entity']}|{f['attr']}", "kind": "template", "value": value,
                  "item": C.statement(C.template_stem(lg, f["attr"], f["entity"]))})
        for k, s in C.paraphrase_stems(lg, f["attr"], f["entity"]):
            P.append({"cat": cat, "key": f"{f['entity']}|{f['attr']}", "kind": f"para{k}", "value": value,
                      "item": C.statement(s)})

    def neg(cat, e, a):
        P.append({"cat": cat, "key": f"{e}|{a}", "kind": "template", "value": None,
                  "item": C.statement(C.template_stem(lg, a, e))})
    seen = set()
    for f in episodes(st, upto=L):
        c = (f["entity"], f["attr"])
        if c in upd or c in seen:
            continue
        seen.add(c)
        pos("POS", f, kn[c])
        if st["intro"].get(f["entity"]) == L:
            pos("FIRST", f, kn[c])
        if f["life"] == 1:
            pos("RET", f, kn[c])
        if c in tr and L >= 4:
            pos("TPOST", f, kn[c])
    rng = random.Random(f"ABST0-nm-{st['seed']}-{L}")
    for cat, ents in (("HNM", st["H"]), ("ANM", st["A"])):
        cand = [(e, a) for e in ents if st["intro"][e] <= L for a in ATTRS9 if (e, a) not in kn and (e, a) not in planned]
        rng.shuffle(cand)
        for e, a in sorted(cand[:N_HNM]):
            neg(cat, e, a)
    for v, a in st["fresh"]:
        neg("FRESH", v, a)
    if L <= 3:
        for e, a in st["transitions"]:
            neg("TPRE" if st["intro"][e] <= L else "TPRE_UNFAM", e, a)
    if L >= 2:
        for e, a in st["updates"]:
            f0 = next(f for f in st["facts"] if f["entity"] == e and f["attr"] == a and f["role"] == "update_init")
            f1 = next(f for f in st["facts"] if f["entity"] == e and f["attr"] == a and f["role"] == "update_new")
            P.append({"cat": "UPD", "key": f"{e}|{a}", "kind": "template", "value": kn[(e, a)], "old": f0["value"],
                      "new": f1["value"] if L >= 4 else None, "item": C.statement(C.template_stem(lg, a, e))})
    return P


def outcome(ans, value):
    """correct / unknown / empty / substantive (incorrect). Exact UNKNOWN only (CONS-0A contract)."""
    if not C.norm(ans or ""):
        return "empty"
    if C.is_unknown(ans):
        return "unknown"
    if value is not None and C.norm(ans) == C.norm(value):
        return "correct"
    return "substantive"


def measures(rows, st=None, exposure=None):
    """Per-seed measures of one evaluation. Positive measures on paraphrase probes; negatives on templates. With the
    stream, transition measures are also split by group: A (all), A exposed, A unexposed, H (protocol D3)."""
    def frac(cat, cls, kinds=("para",), keys=None):
        rs = [r for r in rows if r["cat"] == cat and any(r["kind"].startswith(k) for k in kinds)
              and (keys is None or r["key"] in keys)]
        return (sum(r["outcome"] == cls for r in rs) / len(rs)) if rs else None
    m = {}
    for cat in ("POS", "FIRST", "RET", "TPOST"):
        m[f"{cat}_correct"] = frac(cat, "correct")
        m[f"{cat}_tmpl_correct"] = frac(cat, "correct", ("template",))
    for cat in ("HNM", "ANM", "FRESH", "TPRE", "TPRE_UNFAM"):
        for cls in ("substantive", "unknown", "empty"):
            m[f"{cat}_{cls}"] = frac(cat, cls, ("template",))
    for cat in ("POS",):
        rs = [r for r in rows if r["cat"] == cat and r["kind"].startswith("para")]
        ans = [r for r in rs if r["outcome"] in ("correct", "substantive")]
        m["POS_answer_rate"] = len(ans) / len(rs) if rs else None
        m["POS_selective_acc"] = (sum(r["outcome"] == "correct" for r in ans) / len(ans)) if ans else None
    if st is not None:
        exposure = exposure or {}
        tk = {f"{e}|{a}": e for e, a in st["transitions"]}
        groups = {"A": {k for k, e in tk.items() if e in st["A"]}, "H": {k for k, e in tk.items() if e in st["H"]}}
        if exposure is not None:
            groups["Aexp"] = {k for k in groups["A"] if exposure.get(k, {}).get("exposed")}
            groups["Aunexp"] = groups["A"] - groups["Aexp"]
        for g, keys in groups.items():
            m[f"TPRE_unknown_{g}"] = frac("TPRE", "unknown", ("template",), keys)
            m[f"TPOST_correct_{g}"] = frac("TPOST", "correct", ("para",), keys)
    ups = [r for r in rows if r["cat"] == "UPD"]
    m["UPD"] = [{"key": r["key"], "answer": r["answer"], "is_old": C.norm(r["answer"]) == C.norm(r["old"]),
                 "is_new": bool(r.get("new")) and C.norm(r["answer"]) == C.norm(r["new"]), "outcome": r["outcome"]}
                for r in ups]
    return m


# ------------------------------------------------------------------ negatives (pure parts) --------------------------
def parse_q(gen):
    out = []
    for line in (gen or "").split("\n"):
        m = re.match(r"\s*Q\s*[:.]\s*(.+)", line)
        if m:
            q = m.group(1).strip()
            if q.endswith("?") and 3 <= len(q.split()) <= 30 and "|" not in q:
                out.append(q)
    return out


def name_words(names):
    return {w.lower() for n in names for w in n.split()}


def neg_filters(q, target, other_names, wordings):
    """Pre-verifier acceptance (protocol 'Acceptance' 1-3, 5). Returns a reject reason or None."""
    if not re.search(r"(?<![\w])" + re.escape(target) + r"(?![\w])", q):
        return "no_target"
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", q)}
    if words & (name_words(other_names) - name_words([target])):
        return "other_name"
    if C.leaks(q, target, wordings):
        return "leak"
    return None


def select(items, n, rng):
    """n items cycling `items` in seeded order; returns (selected, exposure counts by index)."""
    out, exp, order, k = [], {}, [], 0
    if not items:
        return [], {}
    while len(out) < n:
        if k % len(items) == 0:
            order = list(range(len(items))); rng.shuffle(order)
        i = order[k % len(items)]
        out.append(items[i]); exp[i] = exp.get(i, 0) + 1; k += 1
    return out, exp


def third_slot(neg_sleep, n3, seed, L):
    """U's and G's third slot for one sleep (protocol 'Third slot'): ceil(n3/2) A-negatives, floor(n3/2) T-negatives;
    G = U with each A-negative replaced by its pre-verified name swap. Returns (U, G, log)."""
    nA, nT = math.ceil(n3 / 2), n3 // 2
    A_items, T_items = neg_sleep["A_accepted"], neg_sleep["T_accepted"]
    if not A_items or not T_items:                       # Sol: no silent substitution - corpus construction STOP
        raise SystemExit(f"NEGATIVE SHORTAGE seed {seed} sleep {L}: A {len(A_items)} / T {len(T_items)} accepted")
    a_sel, a_exp = select(A_items, nA, random.Random(f"ABST0-selA-{seed}-{L}"))
    t_sel, t_exp = select(T_items, nT, random.Random(f"ABST0-selT-{seed}-{L}"))
    U = [{"q": x["q"], "src": "A", "entity": x.get("entity")} for x in a_sel] + [{"q": x["q"], "src": "T"} for x in t_sel]
    G = [{"q": x["swap"], "src": "G"} for x in a_sel] + [{"q": x["q"], "src": "T"} for x in t_sel]
    return U, G, {"nA": nA, "nT": nT, "A_exposure": {A_items[i]["q"]: k for i, k in a_exp.items()},
                  "A_exposure_entity": {A_items[i].get("entity"): 0 for i in a_exp} | {
                      e: sum(k for i, k in a_exp.items() if A_items[i].get("entity") == e) for e in
                      {A_items[i].get("entity") for i in a_exp}},
                  "T_exposure": {T_items[i]["q"]: k for i, k in t_exp.items()},
                  "A_pool": len(A_items), "T_pool": len(T_items)}


def assemble(base, third_U, third_G, third_P, seed, L):
    """Same permutation for every arm; third-slot items occupy identical positions."""
    perm = list(range(len(base) + len(third_U)))
    random.Random(f"ABST0-perm-{seed}-{L}").shuffle(perm)
    return {arm: [(base + t)[p] for p in perm] for arm, t in (("U", third_U), ("G", third_G), ("P", third_P))}


# ------------------------------------------------------------------ aggregation and criteria (pure) ------------------
def pooled(vals):
    """Equal-weight seed mean; None if any seed is missing (a criterion needing it then fails)."""
    return None if (not vals or any(v is None for v in vals)) else sum(vals) / len(vals)


def criteria(M, seeds, ppl):
    """M[arm][seed][L] = measures; ppl[arm][seed][L] = {set: ratio}. Returns the D1-D4 table, mechanism gate, verdict."""
    get = lambda arm, L, k: pooled([M[arm][s][L].get(k) for s in seeds])
    ge = lambda x, y: x is not None and y is not None and x >= y
    le = lambda x, y: x is not None and y is not None and x <= y
    minus = lambda x, dlt: None if x is None else x - dlt          # a missing P measure never lets U pass
    d = {}
    d["D1"] = all(ge(get("U", L, "POS_correct"), TH["pos"]) and
                  ge(get("U", L, "POS_correct"), minus(get("P", L, "POS_correct"), TH["pos_vs_p"]))
                  for L in range(1, 6)) and all(ge(get("U", L, "FIRST_correct"), TH["first"]) for L in range(2, 6))
    d["D2"] = all(le(get("U", L, f"{c}_substantive"), TH["conf"]) and ge(get("U", L, f"{c}_unknown"), TH["unk"])
                  for L in range(1, 6) for c in ("HNM", "FRESH"))
    d["D3"] = all(ge(get("U", L, f"TPRE_unknown_{g}"), TH["tpre_unk"]) for L in (2, 3) for g in ("A", "H")) and \
        all(ge(get("U", L, f"TPOST_correct_{g}"), TH["tpost"]) and
            ge(get("U", L, f"TPOST_correct_{g}"), minus(get("P", L, f"TPOST_correct_{g}"), TH["tpost_vs_p"]))
            for L in (4, 5) for g in ("A", "H"))
    d["D4"] = ge(get("U", 5, "RET_correct"), minus(get("P", 5, "RET_correct"), TH["ret_vs_p"])) and all(
        all(sum(ppl["U"][s][L][k] for s in seeds) / len(seeds) <= TH["ppl_mean"] and
            max(ppl["U"][s][L][k] for s in seeds) <= TH["ppl_seed"] for k in ppl["U"][seeds[0]][L]) for L in range(1, 6))
    u5, g5 = get("U", 5, "HNM_substantive"), get("G", 5, "HNM_substantive")
    d["mechanism"] = u5 is not None and g5 is not None and u5 <= g5 - TH["mech"] and \
        sum(M["U"][s][5]["HNM_substantive"] < M["G"][s][5]["HNM_substantive"] for s in seeds
            if M["U"][s][5]["HNM_substantive"] is not None and M["G"][s][5]["HNM_substantive"] is not None) >= TH["seeds"]
    d["functional"] = all(d[k] for k in ("D1", "D2", "D3", "D4"))
    d["verdict"] = ("ABST0_FUNCTIONAL_AND_GROUNDED" if d["functional"] and d["mechanism"] else
                    "ABST0_FUNCTIONAL" if d["functional"] else "ABST0_NO_ADVANCE")
    return d


def integrity(merged, unmerged):
    """Zero answer and outcome-class mismatches, identical probes, |dppl| <= 1e-4, all finite."""
    ident = lambda ev: [(r["cat"], r["key"], r["kind"], r["item"]) for r in ev["rows"]]
    ok_probes = ident(merged) == ident(unmerged)
    am = sum(a["answer"] != b["answer"] for a, b in zip(merged["rows"], unmerged["rows"]))
    om = sum(a["outcome"] != b["outcome"] for a, b in zip(merged["rows"], unmerged["rows"]))
    dp = {k: abs(merged["ppl_ratio"][k] - unmerged["ppl_ratio"][k]) for k in merged["ppl_ratio"]}
    finite = all(math.isfinite(v) for v in list(merged["ppl_ratio"].values()) + list(unmerged["ppl_ratio"].values()))
    return {"same_probes": ok_probes, "answer_mismatches": am, "outcome_mismatches": om, "ppl_diff": dp,
            "ok": ok_probes and am == 0 and om == 0 and finite and max(dp.values()) <= 1e-4 and
            len(merged["rows"]) == len(unmerged["rows"])}


# ------------------------------------------------------------------ model side --------------------------------------
def labels3(model, tok, memory_text, qs, dev):
    """VERIFY3: label with the highest continuation log-prob (SUPPORTED / UNSUPPORTED / AMBIGUOUS)."""
    out = []
    for q in qs:
        p = VERIFY3.format(memory=memory_text or "(nothing)", q=q)
        lp = {lab: C.logp_cont(model, tok, p, " " + lab, dev) for lab in LABELS}
        out.append({"label": max(lp, key=lp.get), "lp": lp})
    return out


def stage_prep(a):
    import torch
    from being.v3 import lifegen as lg
    out = Path(a.out)
    prov = {"code": code_hash(), "seed": a.seed, "model_manifest": a.model_manifest_sha, "test_mode": a.test_mode}
    if out.exists() and json.loads(out.read_text()).get("provenance") == prov:
        print(f"PREP seed {a.seed}: up to date - skipped"); return
    st = make_stream(a.seed, lg); check_stream(st)
    dev = C._dev()
    model, tok = C.load_model(a.model, dev)
    wordings = C.eval_wordings(lg)
    R = [{"eid": f["fid"], "time": f["life"], "attr": f["attr"], "entity": f["entity"], "recon": f["text"]} for f in st["facts"]]
    Q = C.qa_build(model, tok, R, dev, lg)
    S_names = st["A"] + st["H"] + st["transient"]
    neg, audit, stops = {}, [], []
    t_cursor = 0
    for L in range(1, 6):
        cand_A = []
        for e in st["A"]:
            if st["intro"][e] > L:
                continue
            mem = memory(st, e, L)
            n_tok = len(tok(mem, add_special_tokens=False).input_ids)
            if n_tok >= MEM_MAX_TOK:
                sys.exit(f"MEMORY TOO LONG seed {a.seed} {e} L{L}: {n_tok} tokens")
            g = C.generate(model, tok, [NEG_GEN.format(memory=mem, e=e)], dev, max_new=120, bs=1)[0]
            for q in parse_q(g):
                r = neg_filters(q, e, [n for n in S_names if n != e], wordings)
                cand_A.append({"entity": e, "q": q, "memory": mem, "reject": r})
        todo = [c for c in cand_A if c["reject"] is None]
        for c, v in zip(todo, [labels3(model, tok, c["memory"], [c["q"]], dev)[0] for c in todo]):
            c["verify"] = v
            if v["label"] != "UNSUPPORTED":
                c["reject"] = f"verify_{v['label'].lower()}"
        acc, per = [], {}
        for c in cand_A:
            if c["reject"] is None:
                if per.get(c["entity"], 0) >= NEG_PER_ENTITY:
                    c["reject"] = "cap"; continue
                per[c["entity"]] = per.get(c["entity"], 0) + 1
                acc.append(c)
        # pre-verified name swaps for G: identical QA-id length, T name never in memory, VERIFY3(empty) == UNSUPPORTED
        for c in acc:
            n0 = len(C.qa_ids(tok, c["q"], "UNKNOWN")[0])
            for k in range(T_POOL):
                t = st["T"][(t_cursor + k) % T_POOL]
                q2 = re.sub(r"(?<![\w])" + re.escape(c["entity"]) + r"(?![\w])", t, c["q"])
                if len(C.qa_ids(tok, q2, "UNKNOWN")[0]) == n0 and labels3(model, tok, "", [q2], dev)[0]["label"] == "UNSUPPORTED":
                    c["swap"], c["swap_name"] = q2, t
                    t_cursor = (t_cursor + k + 1) % T_POOL
                    break
            else:
                sys.exit(f"NO NAME SWAP seed {a.seed} L{L}: {c['q']}")
        cand_T = []
        assert t_cursor < T_POOL - 5 * T_PER_SLEEP, "swap names reached the T-negative name range"
        for t in [st["T"][(T_POOL - 1 - (L - 1) * T_PER_SLEEP - j)] for j in range(T_PER_SLEEP)]:
            g = C.generate(model, tok, [NEG_GEN_FRESH.format(t=t)], dev, max_new=120, bs=1)[0]
            for q in parse_q(g):
                cand_T.append({"name": t, "q": q, "reject": neg_filters(q, t, S_names, wordings)})
        for c in [c for c in cand_T if c["reject"] is None]:
            v = labels3(model, tok, "", [c["q"]], dev)[0]
            c["verify"] = v
            if v["label"] != "UNSUPPORTED":
                c["reject"] = f"verify_{v['label'].lower()}"
        accT = [c for c in cand_T if c["reject"] is None]
        neg[L] = {"A_candidates": cand_A, "A_accepted": acc, "T_candidates": cand_T, "T_accepted": accT,
                  "rejects_A": {r: sum(c["reject"] == r for c in cand_A) for r in sorted({str(c["reject"]) for c in cand_A})},
                  "rejects_T": {r: sum(c["reject"] == r for c in cand_T) for r in sorted({str(c["reject"]) for c in cand_T})}}
        if (not acc or not accT) and a.test_mode:
            stops.append(L)                              # tiny-model plumbing tests only; recorded in the output
        elif not acc or not accT:
            sys.exit(f"NEGATIVE CORPUS STOP seed {a.seed} sleep {L}: A {len(acc)} / T {len(accT)} accepted "
                     "(protocol: zero eligible negatives stops corpus construction; a generator change needs an amendment)")
        rs = random.Random(f"ABST0-audit-{a.seed}-{L}")
        audit += [{"sleep": L, "entity": c["entity"], "memory": c["memory"], "q": c["q"]}
                  for c in rs.sample(acc, min(20, len(acc)))]
        print(f"NEG seed {a.seed} L{L}: A {len(acc)}/{len(cand_A)} accepted, T {len(accT)}/{len(cand_T)}", flush=True)
    res = {"which": "ABST-0 prep", "provenance": prov, "stream": st, "stream_sha256": sha(json.dumps(st, sort_keys=True)),
           "qa": Q, "neg": neg, "test_mode_corpus_stops": stops if a.test_mode else None}
    out.write_text(json.dumps(res, indent=1))
    out.with_name(f"audit_sample_seed{a.seed}.json").write_text(json.dumps(
        {"seed": a.seed, "note": "blind audit: memory + question only; label each supported / unsupported / ambiguous",
         "denominators": {L: sum(x["sleep"] == L for x in audit) for L in range(1, 6)}, "items": audit}, indent=1))
    print(f"PREP seed {a.seed}: QA {Q['n_accepted']} accepted; audit sample {len(audit)}", flush=True)


def qa_items(prep, tok, life=None, upto=None):
    st = prep["stream"]
    lives = {f["fid"]: f["life"] for f in st["facts"]}
    out = []
    for f in st["facts"]:
        L = lives[f["fid"]]
        if (life is not None and L != life) or (upto is not None and L > upto):
            continue
        for qa in prep["qa"]["accepted"].get(f["fid"], []) or []:
            ids, k = C.qa_ids(tok, qa["q"], qa["a"])
            out.append({"text": None, "ids": ids, "loss_tail": k, "life": L, "eid": f["fid"]})
    return out


def neg_items(tok, slot):
    out = []
    for x in slot:
        ids, k = C.qa_ids(tok, x["q"], "UNKNOWN")
        out.append({"text": None, "ids": ids, "loss_tail": k, "life": -1, "src": x["src"], "q": x["q"]})
    return out


def stage_train(a):
    import torch
    prep = json.loads(Path(a.prep).read_text())
    seed = prep["provenance"]["seed"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prov = {"code": code_hash(), "prep_sha256": sha(Path(a.prep).read_bytes()), "E": a.passes, "arms": a.arms,
            "sleeps": a.sleeps}
    done = out / f"train_seed{seed}.json"
    if done.exists() and json.loads(done.read_text()).get("provenance") == prov and all(
            (out / f"cortex_{arm}_s{L}.pt").exists() for arm in a.arms.split(",") for L in range(1, a.sleeps + 1)):
        print(f"TRAIN seed {seed}: up to date - skipped"); return
    dev = C._dev()
    torch.manual_seed(seed)
    model, tok = C.load_model(a.model, dev)
    chunk = C.background_chunker(tok, (ROOT / "data/background.txt").read_text().splitlines(), seed)
    tl = lambda s: len(tok(s, add_special_tokens=False).input_ids[:256])
    logs, schedules = {}, {}
    for L in range(1, a.sleeps + 1):
        new, old = qa_items(prep, tok, life=L), qa_items(prep, tok, upto=L - 1)
        base, _, exp_ = C.schedule(new, old, a.passes, random.Random(f"ABST0-sched-{seed}-{L}"), tl, chunk)
        n3 = math.ceil(THIRD_RATIO * a.passes * len(new))
        U3, G3, lg_ = third_slot(prep["neg"][str(L)], n3, seed, L)
        U3, G3 = neg_items(tok, U3), neg_items(tok, G3)
        P3 = [{"text": None, "ids": chunk(10_000 + i, len(x["ids"])), "loss_tail": x["loss_tail"], "life": 0,
               "background": True} for i, x in enumerate(U3)]
        sched = assemble(base, U3, G3, P3, seed, L)
        bud = {arm: C.budget(v) for arm, v in sched.items()}
        if len(set(bud.values())) != 1:
            sys.exit(f"BUDGET MISMATCH seed {seed} sleep {L}: {bud}")
        schedules[L] = sched
        logs[L] = {"budget": list(bud["U"]), "n_new": len(new), "n_third": n3, "replay_exposure": exp_, "third": lg_}
    # (Sol) verifier-inferred transition exposure from UNKNOWN items actually DELIVERED to U before teaching (sleeps
    # 1-3), computed after the schedule is fixed; written to a separate reporting file the selector never reads.
    st = prep["stream"]
    expo = {}
    for e, at in st["transitions"]:
        if e not in st["A"]:
            continue
        f4 = next(f for f in st["facts"] if f["entity"] == e and f["attr"] == at and f["role"] == "transition")
        delivered = [(L, q, k) for L in (1, 2, 3) if L <= a.sleeps for q, k in logs[L]["third"]["A_exposure"].items()
                     if re.search(r"(?<![\w])" + re.escape(e) + r"(?![\w])", q)]
        labs = labels3(model, tok, f4["text"], [q for _, q, _ in delivered], dev) if delivered else []
        hit = [{"sleep": L, "q": q, "exposures": k} for (L, q, k), x in zip(delivered, labs) if x["label"] == "SUPPORTED"]
        expo[f"{e}|{at}"] = {"exposed": bool(hit), "delivered_unknown_items_about_entity": len(delivered),
                             "delivered_exposures_about_entity": sum(k for _, _, k in delivered),
                             "matched": hit, "matched_exposures_by_sleep": {L: sum(h["exposures"] for h in hit
                                                                                  if h["sleep"] == L) for L in (1, 2, 3)},
                             "life4_episode": f4["text"]}
    (out / f"exposure_seed{seed}.json").write_text(json.dumps(
        {"note": "verifier-inferred exposure (VERIFY3 vs the life-4 episode) of DELIVERED U items; reporting only",
         "cells": expo}, indent=1))
    res = {"which": "ABST-0 train", "seed": seed, "provenance": prov, "sleeps": logs, "arms": {}}
    for arm in a.arms.split(","):
        cx = C.Cortex(model, seed=seed) if not hasattr(model, "_abst0_cx") else C._fresh_cortex(model._abst0_cx, seed)
        model._abst0_cx = cx
        opt = torch.optim.AdamW(cx.params(), lr=C.LR, betas=(0.9, 0.999), weight_decay=0.0)
        res["arms"][arm] = {}
        for L in range(1, a.sleeps + 1):
            lg_ = {}
            C.cx_model_train(model)
            C.train_sleep(model, tok, cx, opt, schedules[L][arm], dev, lg_)
            model.eval()
            torch.save(cx.state(), out / f"cortex_{arm}_s{L}.pt")
            lg_["ckpt_sha256"] = sha((out / f"cortex_{arm}_s{L}.pt").read_bytes())
            res["arms"][arm][L] = lg_
            print(time.strftime("%H:%M:%S"), f"TRAIN seed {seed} {arm} sleep {L}: loss {lg_['loss_first']:.3f}->"
                  f"{lg_['loss_last']:.3f}", flush=True)
    done.write_text(json.dumps(res, indent=1))
    print(f"TRAIN seed {seed} DONE", flush=True)


def load_fp32(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = C._dev()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, dev


def stage_evalck(a):
    import torch
    from being.v3 import lifegen as lg
    from being.competence import mean_loss
    prep = json.loads(Path(a.prep).read_text())
    st = prep["stream"]
    out = Path(a.out)
    ck = Path(a.ckpt) if a.ckpt else None
    prov = {"code": code_hash(), "prep_sha256": sha(Path(a.prep).read_bytes()), "sleep": a.sleep, "mode": a.mode,
            "ckpt_sha256": sha(ck.read_bytes()) if ck else None, "model_manifest": a.model_manifest_sha}
    if out.exists() and json.loads(out.read_text()).get("provenance") == prov:
        print(f"EVAL {out.name}: up to date - skipped"); return
    model, tok, dev = load_fp32(a.model)
    comp_sets, base_loss = C._comp(tok, model, a.test_mode)        # fp32 base denominator, same in every process
    if ck is not None:
        state = torch.load(ck)
        scale = C.ALPHA / C.RANK
        if a.mode == "merged":                                      # adapter-free merged fp32 model
            with torch.no_grad():
                for name, mod in model.named_modules():
                    if name in state:
                        A, B = (x.to(mod.weight.device, torch.float32) for x in state[name])
                        mod.weight += (B @ A) * scale
            assert all(not hasattr(m, "_cx") for m in model.modules())
        elif a.mode == "unmerged":
            cx = C.Cortex(model, seed=0); cx.load_state(state)
        else:
            sys.exit("mode must be merged or unmerged with --ckpt")
    P = probes(st, a.sleep, lg)
    gens = C.generate(model, tok, [C.frame(p["item"]) for p in P], dev)
    rows = []
    for p, g in zip(P, gens):
        ans = C.first_line(g)
        rows.append({k: v for k, v in p.items()} | {"answer": ans, "outcome": outcome(ans, p["value"])})
    ppl = {k: math.exp(mean_loss(model, tok, s)) / math.exp(base_loss[k]) for k, s in comp_sets.items()}
    ev = {"which": "ABST-0 evalck", "provenance": prov, "sleep": a.sleep, "rows": rows, "ppl_ratio": ppl,
          "measures": measures(rows, st), "env": {"torch": torch.__version__, "tf32": torch.backends.cuda.matmul.allow_tf32,
                                               "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}}
    out.write_text(json.dumps(ev, indent=1))
    m = ev["measures"]
    print(f"EVAL {out.name}: POS {m['POS_correct']} HNM conf {m['HNM_substantive']} unk {m['HNM_unknown']} "
          f"FRESH conf {m['FRESH_substantive']} ppl {ppl}", flush=True)


def stage_analyze(a):
    d = Path(a.dir)
    seeds = [int(s) for s in a.seeds.split(",")]
    M, ppl, integ, invalid = {}, {}, {}, set()
    for arm in ARMS + ("N",):
        M[arm], ppl[arm] = {}, {}
        for s in seeds:
            M[arm][s], ppl[arm][s] = {}, {}
            for L in range(1, 6):
                f = d / f"seed{s}" / (f"eval_N_s{L}.json" if arm == "N" else f"eval_{arm}_s{L}_merged.json")
                if not f.exists():
                    invalid.add(arm); M[arm][s][L] = {}; ppl[arm][s][L] = {"eval": float("inf")}; continue
                ev = json.loads(f.read_text())
                pf, xf = d / f"seed{s}" / f"prep_seed{s}.json", d / f"seed{s}" / f"exposure_seed{s}.json"
                stx = json.loads(pf.read_text())["stream"] if pf.exists() else None
                expo = json.loads(xf.read_text())["cells"] if xf.exists() else None
                M[arm][s][L] = measures(ev["rows"], stx, expo) if stx is not None else ev["measures"]
                ppl[arm][s][L] = ev["ppl_ratio"]
                if arm != "N":
                    fu = d / f"seed{s}" / f"eval_{arm}_s{L}_unmerged.json"
                    chk = integrity(ev, json.loads(fu.read_text())) if fu.exists() else {"ok": False, "missing": True}
                    integ[f"{s}:{arm}:{L}"] = chk
                    if not chk["ok"]:
                        invalid.add(arm)
    crit = criteria(M, seeds, ppl) if not ({"U", "P"} & invalid) else None
    verdict = "ABST0_INVALID" if ({"U", "P"} & invalid) else crit["verdict"]
    if crit and "G" in invalid:                          # Sol: functional assessable; mechanism NOT assessable
        crit["mechanism"] = "NOT_ASSESSABLE"
        verdict = "ABST0_FUNCTIONAL_MECHANISM_NOT_ASSESSABLE" if crit["functional"] else \
            "ABST0_NO_ADVANCE_MECHANISM_NOT_ASSESSABLE"
    keys = ("POS_correct", "FIRST_correct", "RET_correct", "TPOST_correct", "HNM_substantive", "HNM_unknown",
            "HNM_empty", "FRESH_substantive", "FRESH_unknown", "FRESH_empty", "TPRE_unknown", "ANM_substantive",
            "POS_answer_rate", "POS_selective_acc", "TPRE_unknown_A", "TPRE_unknown_H", "TPRE_unknown_Aexp",
            "TPRE_unknown_Aunexp", "TPOST_correct_A", "TPOST_correct_H", "TPOST_correct_Aexp", "TPOST_correct_Aunexp",
            "TPRE_UNFAM_unknown")
    table = {arm: {L: {k: pooled([M[arm][s][L].get(k) for s in seeds]) for k in keys} for L in range(1, 6)}
             for arm in ARMS + ("N",) if arm not in invalid}
    res = {"which": "ABST-0 analyze", "seeds": seeds, "invalid_arms": sorted(invalid), "integrity": integ,
           "criteria": crit, "verdict": verdict, "pooled": table}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"verdict": verdict, "criteria": crit, "invalid_arms": sorted(invalid)}, indent=1))
    print(f"verdict={verdict}")


def stage_calchoose(a):
    """Calibration decision from fp32 evaluations of P sleep 1 for each E, and of N sleep 1."""
    d = Path(a.dir)
    res = {"which": "ABST-0 calibration", "passes": {}}
    n = json.loads((d / "eval_N_s1.json").read_text())["measures"]
    res["N_FRESH_unknown"], res["N_HNM_unknown"] = n["FRESH_unknown"], n["HNM_unknown"]
    chosen = None
    for E in C.PASSES:
        ev = json.loads((d / f"E{E}" / "eval_P_s1_merged.json").read_text())
        ok = (ev["measures"]["POS_tmpl_correct"] or 0) >= 0.80 and all(v <= TH["ppl_mean"] for v in ev["ppl_ratio"].values())
        res["passes"][E] = {"measures": ev["measures"], "ppl_ratio": ev["ppl_ratio"], "ok": ok}
        if ok and chosen is None:
            chosen = E
    res["chosen_E"] = chosen
    res["contract_revision_needed"] = (n["FRESH_unknown"] or 0) < TH["unk"]
    res["stop"] = chosen is None or res["contract_revision_needed"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"CALIBRATION chosen E={chosen} N FRESH unknown={n['FRESH_unknown']} stop={res['stop']}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--seed", type=int, required=True); p.add_argument("--model", required=True)
    p.add_argument("--out", required=True); p.add_argument("--model-manifest-sha", default="")
    p.add_argument("--test-mode", action="store_true")
    t = sp.add_parser("train"); t.add_argument("--prep", required=True); t.add_argument("--model", required=True)
    t.add_argument("--passes", type=int, required=True); t.add_argument("--out", required=True)
    t.add_argument("--arms", default="P,U,G"); t.add_argument("--sleeps", type=int, default=5)
    e = sp.add_parser("evalck"); e.add_argument("--prep", required=True); e.add_argument("--model", required=True)
    e.add_argument("--sleep", type=int, required=True); e.add_argument("--ckpt"); e.add_argument("--mode", default="base")
    e.add_argument("--out", required=True); e.add_argument("--model-manifest-sha", default="")
    e.add_argument("--test-mode", action="store_true")
    n = sp.add_parser("analyze"); n.add_argument("--dir", required=True); n.add_argument("--out", required=True)
    n.add_argument("--seeds", default=",".join(map(str, DEV_SEEDS)))
    c = sp.add_parser("calchoose"); c.add_argument("--dir", required=True); c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prep": stage_prep, "train": stage_train, "evalck": stage_evalck, "analyze": stage_analyze,
     "calchoose": stage_calchoose}[a.cmd](a)


if __name__ == "__main__":
    main()
