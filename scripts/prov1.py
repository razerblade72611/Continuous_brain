"""PROV-1 (lab/PROV1_PROTOCOL.md; Sol design GO on v1 32eb9a1): can the being use an explicit, support-only record of
its own experience to know what it doesn't know? Development.

Built on SUPPORT-1's frozen code (imported, never modified; SUPPORT-1's CODE_HASH 38b8e705... stays valid):
streams, strata, split, T partition, fork record, oracle, cyclers, base blocks and the probe inventory are SUPPORT-1's,
so phase A can evaluate SUPPORT-1's own checkpoints and every number is comparable.

Stages:
  prep    (phase 0, CPU) seed -> PROV-1 schedules (RP/RU/RC, all sleeps, record-bearing token ids and loss boundaries),
          the RU/RC withheld fork schedules, and the probe inventory expanded into record conditions -> prov1_prep_seed{S}.json
  train   one prep -> cortex checkpoints for RP/RU/RC (+ RU/RC sleep-4 withheld branch)
  evalck  one checkpoint (or base) x history x sleep x conditions -> eval json (truth + compliance outcomes)
  codehash
"""
import argparse, collections, json, math, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import support1 as S                                      # frozen SUPPORT-1 (CODE_HASH 38b8e705...)
C, A0 = S.C, S.A0

SCHEMA, EXPERIMENT = "PROV1_SCHEMA_V1", "PROV1"
DEV_SEEDS, CAL_SEED = S.DEV_SEEDS, S.CAL_SEED
ATTRS9, AIDX, QT = S.ATTRS9, S.AIDX, S.QT
ARMS = ("RP", "RU", "RC")
FORK_ARMS = ("RU", "RC")
NEG_REPS = 3                                              # ordinary negative block x3 (6n); U4 was x4 (8n)
E_PASSES, SLEEPS, MAX_SEQ = S.E_PASSES, S.SLEEPS, S.MAX_SEQ
CONDITIONS = ("FAITHFUL", "ABSENT", "DEL-Q", "DEL-X", "ADD", "FULL")
PHASE_A_CONDITIONS = ("FAITHFUL", "DEL-Q", "DEL-X", "ADD", "FULL")       # ABSENT = SUPPORT-1's archived evaluation
PHASE_B_CONDITIONS = ("FAITHFUL", "ABSENT", "DEL-Q", "DEL-X", "ADD")
FORK_INPUT_TOL = 0.01                                     # withheld vs observed sleep-4 input tokens (records differ)
CODE_FILES = ("scripts/prov1.py",) + tuple(f for f in S.CODE_FILES)


def code_hash():                                          # SUPPORT-1's scheme over PROV-1's file set
    return S.sha(b"".join(S.sha((ROOT / p).read_bytes()).encode() for p in CODE_FILES))


sha, chash, cell_key = S.sha, S.chash, S.cell_key


# ------------------------------------------------------------------ the support record (§3) -----------------------------
def record_set(facts, e, L):
    """Core attributes a with a fact (e, a) at life <= L -> life of the latest such fact, in ATTRS9 order."""
    got = {}
    for f in sorted((f for f in facts if f["entity"] == e and f["life"] <= L and f["attr"] in AIDX),
                    key=lambda f: f["life"]):
        got[f["attr"]] = f["life"]
    return {a: got[a] for a in ATTRS9 if a in got}


def render(e, rs, values=None):
    """One fixed wording. `values` (FULL only) maps attr -> value."""
    if not rs:
        return f"Memory record: {e} - no experiences."
    parts = [f"{a} (life {l})" + (f": {values[a]}" if values else "") for a, l in rs.items()]
    return f"Memory record: {e} - experienced: {', '.join(parts)}. Nothing else was experienced about {e}."


def with_record(rec, item):
    return item if rec is None else f"{rec}\n{item}"


def rq_ids(tok, rec, q, label):
    """C.qa_ids with the record line before the Question line: identical answer ids, terminator and loss boundary."""
    p = tok(C.frame(with_record(rec, C.question(q))), add_special_tokens=False).input_ids
    ans = tok(" " + label, add_special_tokens=False).input_ids + tok(C.TERMINATOR, add_special_tokens=False).input_ids
    return p + ans, len(ans)


_WORD = re.compile(r"[a-z]+")


def leak_words(rec, e, facts):
    """Value words of e that appear in its (support-only) record; must be empty."""
    name = set(_WORD.findall(e.lower()))
    vals = {w for f in facts if f["entity"] == e for w in _WORD.findall(f["value"].lower())} - name
    return sorted(vals & (set(_WORD.findall(rec.lower())) - name))


# ------------------------------------------------------------------ schedules (§4 phase B) --------------------------------
def _spec(kind, slot, cell, wording, label, rec_mode, item_id=None, src=None):
    """A training item before rendering: rendering depends on the history (records follow the lived branch)."""
    return {"kind": kind, "slot": slot, "cell": list(cell), "wording": wording, "label": label, "rec_mode": rec_mode,
            "item_id": item_id, "source_item_id": src}


def _render_item(tok, sp, facts, L):
    e, a = sp["cell"]
    rs = record_set(facts, e, L)
    if sp["rec_mode"] == "delq":
        rs = {k: v for k, v in rs.items() if k != a}
    rec = render(e, rs)
    ids, nl = rq_ids(tok, rec, QT[a][sp["wording"]].format(e=e), sp["label"])
    x = S.entry(sp["kind"], sp["slot"], ids, nl, cell=sp["cell"], wording=sp["wording"], label=sp["label"],
                item_id=sp["item_id"], source=sp["source_item_id"])
    x["rec_mode"], x["record"] = sp["rec_mode"], rec
    return x


def _bg(chunk, key, like, slot):
    return [S.entry("bg", slot, chunk(key + j, len(x["input_ids"])), x["n_loss"]) for j, x in enumerate(like)]


def build_schedules(st, fork, strat, s1, tok, chunk, test_mode=False):
    """PROV-1 schedules for every arm and sleep (observed history) and the RU/RC withheld sleep-4 branch.
    s1 = SUPPORT-1's (positives, sched, swap_map, blocks, fork_sched), from its frozen build_schedules."""
    seed = st["seed"]
    positives, s1_sched, _, blocks, s1_fork = s1
    facts_obs = list(st["facts"])
    facts_wh = S.history_facts(st, fork, "withheld")
    orc_all = {L: S.oracle(facts_obs, L) for L in range(1, SLEEPS + 1)}
    rev = {tuple(c) for c in strat["rev"]}
    upd = {tuple(c) for c in st["updates"]}
    # counterfactual population: A entities' cells with any fact, minus REV (all sleeps, Sol) and update cells
    cf_cells = sorted({(f["entity"], f["attr"]) for f in facts_obs if f["entity"] in st["A"]} - rev - upd, key=cell_key)
    cf_cyc = S.Cycler([(c, k) for c in cf_cells for k in (0, 1)], f"PROV1-cfcyc-{seed}")
    specs, sched, fork_sched, cf_log = {}, {arm: {} for arm in ARMS}, {}, {}
    for L in range(1, SLEEPS + 1):
        orc = orc_all[L]
        pos = positives[L]
        n = len(pos)
        new = [_spec("pos", "new", p["cell"], p["wording"], p["label"], "faithful", item_id=p["item_id"])
               for _ in range(E_PASSES) for p in pos]
        replay = None if L == 1 else [
            _spec("replay", "replay", x["cell"], x["wording"], x["label"], "faithful", src=x["source_item_id"])
            for x in s1_sched["P"][L] if x["slot"] == "replay"]                         # SUPPORT-1's exact replay draw
        base = [_spec("neg_a", "third", c, k, "UNKNOWN", "faithful") for c, k in blocks[L]["A"]] + \
               [_spec("neg_t", "third", c, k, "UNKNOWN", "faithful") for c, k in blocks[L]["T"]]
        if len(base) != 2 * n:
            raise SystemExit(f"PREP STOP: base block {len(base)} != 2n {2 * n} at sleep {L}")
        neg = [dict(x) for _ in range(NEG_REPS) for x in base]
        cf = []
        for _ in range(8 * n - len(neg)):
            c, k = cf_cyc.take(lambda ck: ck[0] in orc)
            cf.append(_spec("cf", "third", c, k, "UNKNOWN", "delq"))
        cf_log[L] = [[list(x["cell"]), x["wording"]] for x in cf]
        specs[L] = {"new": new, "replay": replay, "neg": neg, "cf": cf}

    def render_sleep(L, facts, sp_new):
        new_r = [_render_item(tok, x, facts, L) for x in sp_new]
        sp = specs[L]
        replay_r = (_bg(chunk, 1_000_000, new_r, "replay") if sp["replay"] is None
                    else [_render_item(tok, x, facts, L) for x in sp["replay"]])
        neg_r = [_render_item(tok, x, facts, L) for x in sp["neg"]]
        cf_r = [_render_item(tok, x, facts, L) for x in sp["cf"]]
        reserve = _bg(chunk, 3_000_000 + L * 10_000, cf_r, "third")                     # RU's reserve, matched to RC's cf
        third_rc = neg_r + cf_r
        bg_rp = _bg(chunk, 2_000_000 + L * 10_000, third_rc, "third")                  # RP's third, matched to RC's
        third = {"RP": bg_rp, "RU": neg_r + reserve, "RC": third_rc}
        perm = list(range(len(new_r) + len(replay_r) + len(third_rc)))
        random.Random(f"PROV1-perm-{seed}-{L}").shuffle(perm)
        out = {}
        for arm in ARMS:
            content = new_r + replay_r + third[arm]
            out[arm] = [dict(content[p], pos=i) for i, p in enumerate(perm)]
        return out

    for L in range(1, SLEEPS + 1):
        r = render_sleep(L, facts_obs, specs[L]["new"])
        for arm in ARMS:
            sched[arm][L] = r[arm]
    # withheld sleep-4 branch: target positives replaced by the token-matched distractor positives (SUPPORT-1 §7),
    # every record rendered from the withheld history (Sol: induced record changes are legitimate; budgets re-validated)
    dist = {tuple(f["target"]): f for f in fork["facts"]}
    sp_new_wh = []
    for x in specs[4]["new"]:
        if tuple(x["cell"]) in dist:
            f = dist[tuple(x["cell"])]
            sp_new_wh.append(_spec("pos", "new", (f["entity"], f["attr"]), x["wording"], f["value"], "faithful",
                                   item_id=f"{f['fid']}-q{x['wording']}"))
        else:
            sp_new_wh.append(dict(x))
    wh = render_sleep(4, facts_wh, sp_new_wh)
    fork_sched = {arm: wh[arm] for arm in FORK_ARMS}
    return sched, fork_sched, cf_log


def budget(rows):
    return S.budget(rows)


# ------------------------------------------------------------------ probes (§3 conditions) ------------------------------
def expand_probes(st, fork, base_rows, history, L, seed):
    """SUPPORT-1's probe rows for (history, L) -> one row per applicable record condition, with truth label (the
    oracle of the lived history; corruption never changes it) and compliance target (what the shown record implies)."""
    facts = S.history_facts(st, fork, history)
    orc = S.oracle(facts, L)
    out = []
    for r in base_rows:
        e, a = r["cell"]
        rs = record_set(facts, e, L)
        vals = {k: orc[(e, k)] for k in rs if (e, k) in orc}
        truth = r["gold_current"]
        supported = a in rs

        def add(cond, rec, compliance, **kw):
            out.append({"probe_id": f"{r['probe_id']}-{cond}", "base_probe_id": r["probe_id"], "condition": cond,
                        "cat": r["cat"], "construction": r["construction"], "kind": r["kind"], "cell": [e, a],
                        "item": with_record(rec, r["item"]), "record": rec, "gold_truth": truth,
                        "gold_compliance": compliance, "supported": supported, "n_entries": len(rs), **kw})
        add("FAITHFUL", render(e, rs), truth)
        add("ABSENT", None, truth)
        add("FULL", render(e, rs, values=vals), truth)
        if supported:
            add("DEL-Q", render(e, {k: v for k, v in rs.items() if k != a}), "UNKNOWN")
            others = [k for k in rs if k != a]                                  # canonical (ATTRS9) order
            if others:
                d = random.Random(f"PROV1-delx-{seed}-{history}-{L}-{e}-{a}").choice(others)
                add("DEL-X", render(e, {k: v for k, v in rs.items() if k != d}), truth, delx_attr=d)
        else:
            rs_add = dict(rs); rs_add[a] = L
            add("ADD", render(e, {k: rs_add[k] for k in ATTRS9 if k in rs_add}), None)
    return out


# ------------------------------------------------------------------ prep stage ------------------------------------------
def prep(seed, tok, model_manifest="", test_mode=False):
    st = S.make_stream(seed)
    fork = S.fork_record(st, tok, test_mode)
    strat = S.strata(st)
    spl = S.split(st, [tuple(c) for c in strat["pm"]])
    tp = S.tpart(st)
    lines = (ROOT / "data/background.txt").read_text().splitlines()
    chunk1 = C.background_chunker(tok, lines, f"SUPPORT1-{seed}")                 # SUPPORT-1's own, for its schedules
    s1 = S.build_schedules(st, fork, strat, spl, tp, tok, chunk1, test_mode)
    positives, s1_sched, swap_map, blocks, s1_fork = s1
    chunk = C.background_chunker(tok, lines, f"PROV1-{seed}")
    sched, fork_sched, cf_log = build_schedules(st, fork, strat, s1, tok, chunk, test_mode)
    bud = {arm: {L: budget(sched[arm][L]) for L in sched[arm]} for arm in ARMS}
    for L in range(1, SLEEPS + 1):
        if len({tuple(bud[arm][L]) for arm in ARMS}) != 1:
            raise SystemExit(f"PREP STOP: budget mismatch at sleep {L}: { {a: bud[a][L] for a in ARMS} }")
    fb = {arm: budget(fork_sched[arm]) for arm in FORK_ARMS}
    for arm, (n_it, n_in, n_loss) in fb.items():
        o = bud[arm][4]
        if n_it != o[0] or n_loss != o[2] or abs(n_in - o[1]) > FORK_INPUT_TOL * o[1]:
            if not test_mode:
                raise SystemExit(f"PREP STOP: fork budget {arm} {fb[arm]} vs observed sleep 4 {o}")
        if len({tuple(fb[x]) for x in FORK_ARMS}) != 1 and not test_mode:
            raise SystemExit(f"PREP STOP: RU/RC fork budgets differ: {fb}")
    too_long = [len(x["input_ids"]) for d in (sched, {a: {4: fork_sched[a]} for a in FORK_ARMS})
                for arm in d for L in d[arm] for x in d[arm][L] if len(x["input_ids"]) > MAX_SEQ]
    if too_long and not test_mode:
        raise SystemExit(f"PREP STOP: {len(too_long)} sequences over {MAX_SEQ} tokens")
    leaks = [(x["cell"], w) for d in (sched, {a: {4: fork_sched[a]} for a in FORK_ARMS}) for arm in d for L in d[arm]
             for x in d[arm][L] if x.get("record")
             for w in leak_words(x["record"], x["cell"][0], S.history_facts(st, fork, "observed") + fork["facts"])]
    s1_probes = {"observed": {L: S.build_probes(st, fork, strat, spl, s1_sched, swap_map, "observed", L)
                              for L in range(1, SLEEPS + 1)},
                 "withheld": {4: S.build_probes(st, fork, strat, spl, s1_sched, swap_map, "withheld", 4)}}
    probes = {h: {L: expand_probes(st, fork, rows, h, L, seed) for L, rows in d.items()} for h, d in s1_probes.items()}
    leaks += [(r["cell"], w) for d in probes.values() for rows in d.values() for r in rows
              if r["record"] and r["condition"] != "FULL"
              for w in leak_words(r["record"], r["cell"][0], S.history_facts(st, fork, "observed") + fork["facts"])]
    if leaks:
        raise SystemExit(f"PREP STOP: {len(leaks)} value words inside support-only records, e.g. {leaks[:3]}")
    long_probe = [r["probe_id"] for d in probes.values() for rows in d.values() for r in rows
                  if len(tok(C.frame(r["item"]), add_special_tokens=False).input_ids) > MAX_SEQ]
    if long_probe and not test_mode:
        raise SystemExit(f"PREP STOP: {len(long_probe)} probe prompts over {MAX_SEQ} tokens")
    keys = {"cfcyc": f"PROV1-cfcyc-{seed}", "perm": f"PROV1-perm-{seed}-{{L}}", "delx": f"PROV1-delx-{seed}-{{history}}-{{L}}-{{e}}-{{a}}",
            "background": f"CONS0A-bg-PROV1-{seed}-{{i}}", "support1": "all SUPPORT-1 keys, unchanged (via support1.prep pieces)"}
    return {"schema": SCHEMA, "experiment": EXPERIMENT, "seed": seed, "test_mode": test_mode, "code_hash": code_hash(),
            "support1_code_hash": S.code_hash(), "model_manifest": model_manifest, "stream_sha256": chash(st),
            "stream": st, "fork": fork, "strata": {k: [list(c) for c in v] for k, v in strat.items()},
            "split": {"r": spl["r"], "sel": [list(c) for c in spl["sel"]], "how": [list(c) for c in spl["how"]]},
            "support1_probes_sha256": chash({h: {str(L): v for L, v in d.items()} for h, d in s1_probes.items()}),
            "schedule": {arm: {str(L): v for L, v in sched[arm].items()} for arm in ARMS},
            "fork_schedule": {"L": 4, **fork_sched}, "counterfactuals": {str(L): v for L, v in cf_log.items()},
            "probes": {h: {str(L): v for L, v in d.items()} for h, d in probes.items()},
            "sampler_keys": keys, "budget": {arm: {str(L): v for L, v in d.items()} for arm, d in bud.items()},
            "fork_budget": fb}


def stage_prep(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if not a.test_mode and not a.model_manifest_sha:
        sys.exit("PREP: production prep needs --model-manifest-sha")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        if not a.test_mode and s not in DEV_SEEDS + (CAL_SEED,):
            sys.exit(f"PREP: seed {s} is not a PROV-1 seed")
        p = prep(s, tok, a.model_manifest_sha, a.test_mode)
        f = out / f"prov1_prep_seed{s}.json"
        f.write_text(json.dumps(p, ensure_ascii=False))
        cnt = collections.Counter(r["condition"] for rows in p["probes"]["observed"].values() for r in rows)
        print(f"PREP seed {s}: budget/sleep {[p['budget']['RC'][str(L)] for L in range(1, 6)]}; fork {p['fork_budget']}; "
              f"observed rows by condition {dict(cnt)}; support1 probes {p['support1_probes_sha256'][:12]}; "
              f"sha256 {sha(f.read_bytes())}", flush=True)


# ------------------------------------------------------------------ train stage -----------------------------------------
def stage_train(a):
    import torch
    prep = json.loads(Path(a.prep).read_text())
    psha = sha(Path(a.prep).read_bytes())
    if not a.test_mode and (prep["test_mode"] or prep["code_hash"] != code_hash()
                            or prep["model_manifest"] != a.model_manifest_sha):
        sys.exit("TRAIN: production training needs a non-test prep built by this code and model")
    S._setup_determinism()
    seed, arms = prep["seed"], a.arms.split(",")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prov = {"code_hash": code_hash(), "prep_sha256": psha, "model_manifest": a.model_manifest_sha,
            "test_mode": a.test_mode, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "arms": arms, "sleeps": a.sleeps, "fork": a.fork}
    done = out / f"prov1_train_seed{seed}.json"
    if done.exists():
        old = json.loads(done.read_text())
        if old["provenance"] == prov and all(
                (out / f"cortex_{arm}_s{L}.pt").exists() and sha((out / f"cortex_{arm}_s{L}.pt").read_bytes()) ==
                old["arms"][arm][str(L)]["ckpt_sha256"] for arm in arms for L in range(1, a.sleeps + 1)):
            print(f"TRAIN seed {seed}: up to date (checkpoint hashes verified) - skipped"); return
    dev = C._dev()
    torch.manual_seed(seed); random.seed(seed)
    model, tok = C.load_model(a.model, dev)
    res = {"schema": SCHEMA, "experiment": EXPERIMENT, "seed": seed, "provenance": prov, "arms": {},
           "fork": {"admitted": bool(a.fork), "parent_full_state_hash": {}, "withheld": {}}}
    cx = None
    for arm in arms:
        cx = C.Cortex(model, seed=seed) if cx is None else C._fresh_cortex(cx, seed)
        opt = torch.optim.AdamW(cx.params(), lr=C.LR, betas=(0.9, 0.999), weight_decay=0.0)
        step, res["arms"][arm], parent = 0, {}, None
        for L in range(1, a.sleeps + 1):
            rows = prep["schedule"][arm][str(L)]
            lg_ = {}
            C.cx_model_train(model)
            C.train_sleep(model, tok, cx, opt, S._items(rows), dev, lg_)
            model.eval()
            step += lg_["updates"]
            ck = out / f"cortex_{arm}_s{L}.pt"
            torch.save(cx.state(), ck)
            res["arms"][arm][str(L)] = {"ckpt_sha256": sha(ck.read_bytes()), "full_state_hash": S.full_state_hash(cx, opt, step),
                                        "delivered": S._delivered(rows), "budget": budget(rows),
                                        "loss_first": lg_["loss_first"], "loss_last": lg_["loss_last"], "seconds": lg_["seconds"]}
            print(time.strftime("%H:%M:%S"), f"TRAIN seed {seed} {arm} sleep {L}: loss {lg_['loss_first']:.3f}->"
                  f"{lg_['loss_last']:.3f} ({lg_['seconds']}s)", flush=True)
            if a.fork and arm in FORK_ARMS and L == 3:
                parent = S.capture(cx, opt, step)
                res["fork"]["parent_full_state_hash"][arm] = res["arms"][arm]["3"]["full_state_hash"]
        if parent is not None and a.sleeps >= 4:
            step = S.restore(cx, opt, parent)
            assert S.full_state_hash(cx, opt, step) == res["fork"]["parent_full_state_hash"][arm], "fork parent not restored"
            rows = prep["fork_schedule"][arm]
            lg_ = {}
            C.cx_model_train(model)
            C.train_sleep(model, tok, cx, opt, S._items(rows), dev, lg_)
            model.eval()
            step += lg_["updates"]
            ck = out / f"cortex_{arm}_s4_withheld.pt"
            torch.save(cx.state(), ck)
            res["fork"]["withheld"][arm] = {"ckpt_sha256": sha(ck.read_bytes()),
                                            "full_state_hash": S.full_state_hash(cx, opt, step), "delivered": S._delivered(rows)}
    done.write_text(json.dumps(res, indent=1))
    print(f"TRAIN seed {seed} DONE", flush=True)


# ------------------------------------------------------------------ evaluation ------------------------------------------
def outcome(ans, gold):
    """SUPPORT-1's grading; gold None (ADD compliance) -> None."""
    return None if gold is None else S.outcome(ans, gold)


def stage_evalck(a):
    import torch
    from being.competence import mean_loss
    prep = json.loads(Path(a.prep).read_text())
    psha = sha(Path(a.prep).read_bytes())
    if not a.test_mode and (prep["test_mode"] or prep["model_manifest"] != a.model_manifest_sha or prep["code_hash"] != code_hash()):
        sys.exit("EVAL: production evaluation needs a non-test prep built by this code and model")
    conds = a.conditions.split(",")
    if any(c not in CONDITIONS for c in conds):
        sys.exit(f"EVAL: unknown condition in {conds}")
    out = Path(a.out)
    ck = Path(a.ckpt) if a.ckpt else None
    mode = "base" if ck is None else a.mode
    prov = {"code_hash": code_hash(), "prep_sha256": psha, "ckpt_sha256": sha(ck.read_bytes()) if ck else None,
            "model_manifest": a.model_manifest_sha, "test_mode": a.test_mode, "conditions": conds, "monitors": a.monitors,
            "source": a.source}
    if out.exists() and json.loads(out.read_text()).get("provenance") == prov:
        print(f"EVAL {out.name}: up to date - skipped"); return
    model, tok, dev = A0.load_fp32(a.model)
    comp_sets, base_loss = C._comp(tok, model, a.test_mode) if a.monitors else ({}, {})
    if ck is not None:
        state = torch.load(ck)
        if mode == "merged":
            scale = C.ALPHA / C.RANK
            with torch.no_grad():
                for name, mod in model.named_modules():
                    if name in state:
                        Am, Bm = (x.to(mod.weight.device, torch.float32) for x in state[name])
                        mod.weight += (Bm @ Am) * scale
            assert all(not hasattr(m, "_cx") for m in model.modules())
        elif mode == "unmerged":
            cx = C.Cortex(model, seed=0); cx.load_state(state)
        else:
            sys.exit("mode must be merged or unmerged with --ckpt")
    P = [r for r in prep["probes"][a.history][str(a.sleep)] if r["condition"] in conds]
    gens = C.generate(model, tok, [C.frame(r["item"]) for r in P], dev)
    rows = []
    for r, g in zip(P, gens):
        ans = C.first_line(g)
        rows.append({"probe_id": r["probe_id"], "answer": ans, "outcome_truth": outcome(ans, r["gold_truth"]),
                     "outcome_compliance": outcome(ans, r["gold_compliance"])})
    ppl = {k: math.exp(mean_loss(model, tok, s)) / math.exp(base_loss[k]) for k, s in comp_sets.items()}
    ev = {"schema": SCHEMA, "experiment": EXPERIMENT, "seed": prep["seed"], "arm": a.arm, "sleep": a.sleep, "mode": mode,
          "history": a.history, "provenance": prov, "ppl_ratio": ppl or None,
          "base_denominator_sha256": chash(base_loss) if base_loss else None, "rows": rows,
          "env": {"torch": torch.__version__, "tf32": torch.backends.cuda.matmul.allow_tf32,
                  "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}}
    out.write_text(json.dumps(ev, indent=1))
    by = {r["probe_id"]: r for r in P}

    def rate(cat, cond, cls):
        xs = [x for x in rows if by[x["probe_id"]]["cat"] == cat and by[x["probe_id"]]["condition"] == cond]
        return round(sum(x["outcome_truth"] == cls for x in xs) / len(xs), 3) if xs else None
    print(f"EVAL {out.name}: FAITHFUL POS {rate('POS', 'FAITHFUL', 'correct')} HNM unk {rate('HNM', 'FAITHFUL', 'unknown')} "
          f"FRESH unk {rate('FRESH', 'FAITHFUL', 'unknown')} SEL-D unk {rate('SEL-D', 'FAITHFUL', 'unknown')} "
          f"HOW unk {rate('HOW', 'FAITHFUL', 'unknown')} | DEL-Q POS unk {rate('POS', 'DEL-Q', 'unknown')} "
          f"DEL-X POS {rate('POS', 'DEL-X', 'correct')} | ppl {ppl or '-'}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--seeds", default="20,21,22,23,24"); p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True); p.add_argument("--model-manifest-sha", default=""); p.add_argument("--test-mode", action="store_true")
    t = sp.add_parser("train"); t.add_argument("--prep", required=True); t.add_argument("--model", required=True)
    t.add_argument("--out", required=True); t.add_argument("--arms", default=",".join(ARMS)); t.add_argument("--sleeps", type=int, default=SLEEPS)
    t.add_argument("--fork", action="store_true"); t.add_argument("--model-manifest-sha", default=""); t.add_argument("--test-mode", action="store_true")
    e = sp.add_parser("evalck"); e.add_argument("--prep", required=True); e.add_argument("--model", required=True)
    e.add_argument("--ckpt"); e.add_argument("--mode", default="merged"); e.add_argument("--arm", required=True)
    e.add_argument("--sleep", type=int, required=True); e.add_argument("--history", default="observed")
    e.add_argument("--conditions", default=",".join(PHASE_B_CONDITIONS)); e.add_argument("--monitors", action="store_true")
    e.add_argument("--source", default="prov1", choices=("prov1", "support1"))
    e.add_argument("--out", required=True); e.add_argument("--model-manifest-sha", default=""); e.add_argument("--test-mode", action="store_true")
    sp.add_parser("codehash")
    a = ap.parse_args()
    if a.cmd == "codehash":
        print(code_hash()); return
    {"prep": stage_prep, "train": stage_train, "evalck": stage_evalck}[a.cmd](a)


if __name__ == "__main__":
    main()
