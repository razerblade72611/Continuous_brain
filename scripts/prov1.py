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
def _guard(prep, psha, a, kind):
    """Production training needs: a non-test prep built by this code and model, a PROV-1 seed, and for development
    seeds Sol's validator report (PASS, covering this prep, this code and model) and a passing phase-1 calibration
    bound to this code, model and the seed-24 prep."""
    if a.test_mode:
        return
    if prep["test_mode"] or prep["code_hash"] != code_hash() or not a.model_manifest_sha or prep["model_manifest"] != a.model_manifest_sha:
        sys.exit(f"{kind}: production needs a non-test prep built by this code and model")
    if prep["seed"] not in DEV_SEEDS + (CAL_SEED,):
        sys.exit(f"{kind}: seed {prep['seed']} is not a PROV-1 seed")
    if prep["seed"] in DEV_SEEDS:
        v = json.loads(Path(a.validator_report).read_text()) if getattr(a, "validator_report", None) else None
        if (not v or v.get("verdict") != "PASS" or v.get("errors") or psha not in v.get("prep_sha256", [])
                or v.get("code_hash") != code_hash() or v.get("model_manifest") != a.model_manifest_sha):
            sys.exit(f"{kind}: needs Sol's validator report (PASS, no errors) covering this prep, code and model")
        c = json.loads(Path(a.calibration_result).read_text()) if getattr(a, "calibration_result", None) else None
        cp = Path(a.calibration_prep).read_bytes() if getattr(a, "calibration_prep", None) else None
        if (not c or c.get("pass") is not True or c.get("seed") != CAL_SEED or c.get("errors") or c.get("test_mode")
                or c.get("code_hash") != code_hash() or c.get("model_manifest") != a.model_manifest_sha
                or cp is None or c.get("prep_sha256") != sha(cp)):
            sys.exit(f"{kind}: needs a passing PROV-1 calibration bound to this code, model and the seed-24 prep")


def _train_cache_ok(old, prov, out, arms, a):
    """Reuse only a COMPLETE, verified log (Sol's HOLD at b88ca93): same provenance; every main checkpoint present with
    its logged hash; and when the fork is requested, every required withheld checkpoint present with its logged hash,
    plus the logged fork parent equal to that arm's logged sleep-3 full-state hash."""
    def file_ok(path, want):
        return isinstance(want, str) and path.exists() and sha(path.read_bytes()) == want
    try:
        if old.get("provenance") != prov:
            return False
        for arm in arms:
            for L in range(1, a.sleeps + 1):
                if not file_ok(out / f"cortex_{arm}_s{L}.pt", old["arms"][arm][str(L)]["ckpt_sha256"]):
                    return False
        if a.fork and a.sleeps >= 4:
            fk = old["fork"]
            for arm in (x for x in arms if x in FORK_ARMS):
                if not file_ok(out / f"cortex_{arm}_s4_withheld.pt", fk["withheld"][arm]["ckpt_sha256"]):
                    return False
                parent = fk["parent_full_state_hash"][arm]
                if not isinstance(parent, str) or parent != old["arms"][arm]["3"]["full_state_hash"]:
                    return False
        return True
    except (KeyError, TypeError):
        return False


def stage_train(a):
    import torch
    prep = json.loads(Path(a.prep).read_text())
    psha = sha(Path(a.prep).read_bytes())
    _guard(prep, psha, a, "TRAIN")
    S._setup_determinism()
    seed, arms = prep["seed"], a.arms.split(",")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prov = {"code_hash": code_hash(), "prep_sha256": psha, "model_manifest": a.model_manifest_sha,
            "test_mode": a.test_mode, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "arms": arms, "sleeps": a.sleeps, "fork": a.fork}
    done = out / f"prov1_train_seed{seed}.json"
    if done.exists() and _train_cache_ok(json.loads(done.read_text()), prov, out, arms, a):
        print(f"TRAIN seed {seed}: up to date (main + fork checkpoint hashes and fork parent verified) - skipped"); return
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


def _eval_cache_ok(old, prov, prep, a, mode, conds):
    """Reuse only an evaluation of exactly this request (Sol's HOLD at b88ca93): provenance (which now includes arm,
    sleep, history and mode), header fields, and the complete expected probe inventory in prep order."""
    try:
        if old.get("provenance") != prov:
            return False
        if (old.get("schema"), old.get("experiment"), old.get("seed"), old.get("arm"), old.get("sleep"), old.get("mode"),
                old.get("history")) != (SCHEMA, EXPERIMENT, prep["seed"], a.arm, a.sleep, mode, a.history):
            return False
        want = [r["probe_id"] for r in prep["probes"][a.history][str(a.sleep)] if r["condition"] in conds]
        return [r["probe_id"] for r in old["rows"]] == want and len(want) > 0
    except (KeyError, TypeError, AttributeError):
        return False


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
            "source": a.source, "arm": a.arm, "sleep": a.sleep, "history": a.history, "mode": mode}
    if out.exists() and _eval_cache_ok(json.loads(out.read_text()), prov, prep, a, mode, conds):
        print(f"EVAL {out.name}: up to date (full request identity and probe inventory verified) - skipped"); return
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


_HEX64 = re.compile(r"[0-9a-f]{64}")


def _hex64(x):
    return isinstance(x, str) and _HEX64.fullmatch(x) is not None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def _load_ev(path, prep, psha, arm, L, mode, conds, ckpt, errs):
    ev = json.loads(Path(path).read_text())
    pv = ev.get("provenance") or {}
    if (mode == "base") != (ckpt is None) or (ckpt is not None and not _hex64(ckpt)):
        errs.append(f"{Path(path).name}: expected checkpoint identity is not a valid sha256 (or base has one)")
    if not _hex64(ev.get("base_denominator_sha256")):
        errs.append(f"{Path(path).name}: missing or invalid base-denominator identity")
    want = {"code_hash": code_hash(), "prep_sha256": psha, "ckpt_sha256": ckpt, "model_manifest": prep["model_manifest"],
            "test_mode": prep["test_mode"], "conditions": list(conds), "monitors": True, "source": "prov1", "arm": arm,
            "sleep": L, "history": "observed", "mode": mode}
    if {k: pv.get(k) for k in want} != want:
        errs.append(f"{Path(path).name}: provenance does not bind to this prep/code/model/checkpoint/request")
    if (ev.get("schema"), ev.get("experiment"), ev.get("seed"), ev.get("arm"), ev.get("sleep"), ev.get("mode"), ev.get("history")) != \
            (SCHEMA, EXPERIMENT, CAL_SEED, arm, L, mode, "observed"):
        errs.append(f"{Path(path).name}: identity fields wrong")
    inv = [r for r in prep["probes"]["observed"][str(L)] if r["condition"] in conds]
    if [r["probe_id"] for r in ev.get("rows", [])] != [r["probe_id"] for r in inv]:
        errs.append(f"{Path(path).name}: rows are not the complete inventory in prep order")
    elif any(outcome(x["answer"], r["gold_truth"]) != x["outcome_truth"] or
             outcome(x["answer"], r["gold_compliance"]) != x["outcome_compliance"] for x, r in zip(ev["rows"], inv)):
        errs.append(f"{Path(path).name}: stored outcomes disagree with independent grading")
    ppl = ev.get("ppl_ratio") or {}
    if set(ppl) != {"eval", "modern"} or not all(_num(v) for v in ppl.values()):
        errs.append(f"{Path(path).name}: both competence monitors must be present, numeric (not boolean), finite and positive")
    if (ev.get("env") or {}).get("tf32") is not False and not prep["test_mode"]:
        errs.append(f"{Path(path).name}: TF32 not recorded as disabled")
    return ev, inv


def stage_calcheck(a):
    """Phase-1 calibration gate (seed 24, sleep 1, RP and RC, E = 4). Plumbing and learnability, not the hypothesis:
      - provenance, identity, complete inventory and independently recomputed outcomes for RP/RC twins and base;
      - twins: 0 answer mismatches, |dppl| <= 1e-4; one base denominator across all five evaluations;
      - competence of RP and RC finite and <= 1.02;
      - RP (positives with records) FAITHFUL POS template acquisition >= 0.80; base ABSENT FRESH UNKNOWN >= 0.90.
    RC's values are reported only (its sleep-1 behaviour is a result, not a calibration condition)."""
    prep_b = Path(a.prep).read_bytes(); prep = json.loads(prep_b); psha = sha(prep_b)
    tr = json.loads(Path(a.train).read_text())
    errs, conds = [], PHASE_B_CONDITIONS
    if prep["seed"] != CAL_SEED or prep["code_hash"] != code_hash():
        errs.append("prep is not the calibration seed built by this code")
    tp = tr.get("provenance", {})
    if (tr.get("schema"), tr.get("experiment"), tr.get("seed")) != (SCHEMA, EXPERIMENT, CAL_SEED) or tp.get("prep_sha256") != psha \
            or tp.get("code_hash") != code_hash() or tp.get("arms") != ["RP", "RC"] or tp.get("sleeps") != 1 or tp.get("fork") \
            or tp.get("model_manifest") != prep["model_manifest"] or tp.get("test_mode") != prep["test_mode"]:
        errs.append("train log is not RP,RC sleep 1 of this prep/code/model")
    evs = {}
    for arm in ("RP", "RC"):
        ck = tr.get("arms", {}).get(arm, {}).get("1", {}).get("ckpt_sha256")
        for mode in ("merged", "unmerged"):
            evs[(arm, mode)] = _load_ev(getattr(a, f"{arm.lower()}_{mode}"), prep, psha, arm, 1, mode, conds, ck, errs)
    evs[("N", "base")] = _load_ev(a.base, prep, psha, "N", 1, "base", conds, None, errs)
    gpus = {(ev.get("env") or {}).get("gpu") for ev, _ in evs.values()} | {tp.get("gpu")}
    if len(gpus) != 1 or not all(isinstance(g, str) and g for g in gpus) or \
            (not prep["test_mode"] and gpus != {"NVIDIA RTX 6000 Ada Generation"}):
        errs.append(f"GPU identity not one recorded RTX 6000 Ada across train and evaluations: {sorted(map(str, gpus))}")
    nb = evs[("N", "base")][0].get("ppl_ratio") or {}
    if not all(_num(nb.get(k)) and abs(nb[k] - 1.0) <= 1e-9 for k in ("eval", "modern")):
        errs.append("base evaluation's competence ratios are not unit ratios against its own denominator")
    if not errs:
        for arm in ("RP", "RC"):
            m, u = evs[(arm, "merged")][0], evs[(arm, "unmerged")][0]
            if any(x["answer"] != y["answer"] for x, y in zip(m["rows"], u["rows"])):
                errs.append(f"{arm}: merged/unmerged answer mismatch")
            if any(abs(m["ppl_ratio"][k] - u["ppl_ratio"][k]) > 1e-4 for k in ("eval", "modern")):
                errs.append(f"{arm}: merged/unmerged ppl mismatch > 1e-4")
        if len({ev.get("base_denominator_sha256") for ev, _ in evs.values()}) != 1:
            errs.append("base denominators differ")

    def rate(key, cond, cat, kind_pref, cls):
        if key not in evs or errs:
            return None
        ev, inv = evs[key]
        xs = [x["outcome_truth"] for x, r in zip(ev["rows"], inv) if r["condition"] == cond and r["cat"] == cat
              and r["kind"].startswith(kind_pref)]
        return sum(o == cls for o in xs) / len(xs) if xs else None
    acq = rate(("RP", "merged"), "FAITHFUL", "POS", "template", "correct")
    fresh = rate(("N", "base"), "ABSENT", "FRESH", "template", "unknown")
    comp = {arm: evs[(arm, "merged")][0]["ppl_ratio"] for arm in ("RP", "RC")} if not errs else None
    comp_ok = bool(comp) and all(v <= 1.02 for d in comp.values() for v in d.values())
    ok = not errs and acq is not None and acq >= 0.80 and fresh is not None and fresh >= 0.90 and comp_ok
    res = {"schema": SCHEMA, "experiment": EXPERIMENT, "which": "calcheck", "seed": CAL_SEED, "pass": bool(ok), "errors": errs,
           "RP_pos_template_acquisition": acq, "N_absent_fresh_unknown": fresh, "competence": comp, "competence_in_band": comp_ok,
           "RC_reported": {"pos_template_correct": rate(("RC", "merged"), "FAITHFUL", "POS", "template", "correct"),
                           "fresh_unknown": rate(("RC", "merged"), "FAITHFUL", "FRESH", "template", "unknown"),
                           "delq_pos_unknown": rate(("RC", "merged"), "DEL-Q", "POS", "template", "unknown")},
           "prep_sha256": psha, "code_hash": code_hash(), "model_manifest": prep["model_manifest"], "test_mode": prep["test_mode"],
           "gpu": sorted(map(str, gpus)),
           "inputs_sha256": {"train": sha(Path(a.train).read_bytes()), "base": sha(Path(a.base).read_bytes()),
                             **{f"{arm}_{mode}": sha(Path(getattr(a, f"{arm.lower()}_{mode}")).read_bytes())
                                for arm in ("RP", "RC") for mode in ("merged", "unmerged")}}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"CALCHECK pass={ok} RP acquisition={acq} N FRESH unknown={fresh} competence={comp} errors={errs}", flush=True)
    if not ok:
        sys.exit(4)


def stage_project(a):
    """Phase-1 timings -> projected phase-A and phase-B cost from the ACTUAL inventories of the four development preps.
    Exit 5 if phase A alone exceeds the remaining budget; phase B affordability is reported for the phase-A review."""
    t = json.loads(Path(a.timings).read_text())   # {train_items, train_seconds, eval_rows, eval_seconds}
    s_item, s_row = t["train_seconds"] / t["train_items"], t["eval_seconds"] / t["eval_rows"]
    s1 = [json.loads(Path(f).read_text()) for f in a.support1_preps.split(",")]
    pv = [json.loads(Path(f).read_text()) for f in a.preps.split(",")]
    rows = lambda p, h, L, conds: sum(1 for r in p["probes"][h][str(L)] if r["condition"] in conds)
    regen = sum(len(x) for p in s1 for arm in p["schedule"] for x in p["schedule"][arm].values()) + \
        sum(len(p["fork_schedule"][arm]) for p in s1 for arm in ("U4", "G4"))
    ea = sum(rows(p, "observed", L, PHASE_A_CONDITIONS) * 2 * 2 for p in pv for L in range(1, SLEEPS + 1)) + \
        sum(rows(p, "withheld", 4, PHASE_A_CONDITIONS) * 2 for p in pv)
    tb = sum(len(x) for p in pv for arm in ARMS for x in p["schedule"][arm].values()) + \
        sum(len(p["fork_schedule"][arm]) for p in pv for arm in FORK_ARMS)
    eb = sum(rows(p, "observed", L, PHASE_B_CONDITIONS) * (len(ARMS) * 2 + 1) for p in pv for L in range(1, SLEEPS + 1)) + \
        sum(rows(p, "withheld", 4, PHASE_B_CONDITIONS) * len(FORK_ARMS) * 2 for p in pv)
    plan = lambda items, erows: (lambda h: {"hours": round(h, 2), "usd": round(h * a.usd_per_hour, 2),
                                            "train_items": items, "eval_rows": erows})(
        (s_item * items + s_row * erows) / 3600 + a.boot_hours)
    A_, B_ = plan(regen, ea), plan(tb, eb)
    left = a.ceiling - a.spent
    res = {"schema": SCHEMA, "experiment": EXPERIMENT, "which": "projection", "s_per_item": s_item, "s_per_row": s_row,
           "phase_A": A_, "phase_B": B_, "total_usd": round(A_["usd"] + B_["usd"], 2), "ceiling_usd": a.ceiling,
           "spent_usd": a.spent, "phase_A_affordable": A_["usd"] <= left,
           "A_and_B_affordable": A_["usd"] + B_["usd"] <= left, "code_hash": code_hash()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"PROJECTION phase A {A_} phase B {B_} remaining ${left:.2f} -> A affordable {res['phase_A_affordable']}, "
          f"A+B affordable {res['A_and_B_affordable']}", flush=True)
    if not res["phase_A_affordable"]:
        sys.exit(5)


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
    for x in (t,):
        x.add_argument("--validator-report"); x.add_argument("--calibration-result"); x.add_argument("--calibration-prep")
    c = sp.add_parser("calcheck"); c.add_argument("--prep", required=True); c.add_argument("--train", required=True)
    for arm in ("rp", "rc"):
        for mode in ("merged", "unmerged"):
            c.add_argument(f"--{arm}-{mode}", required=True)
    c.add_argument("--base", required=True); c.add_argument("--out", required=True)
    j = sp.add_parser("project"); j.add_argument("--timings", required=True); j.add_argument("--preps", required=True)
    j.add_argument("--support1-preps", required=True); j.add_argument("--usd-per-hour", type=float, default=0.99)
    j.add_argument("--boot-hours", type=float, default=0.5); j.add_argument("--ceiling", type=float, default=30.0)
    j.add_argument("--spent", type=float, default=0.0); j.add_argument("--out", required=True)
    sp.add_parser("codehash")
    a = ap.parse_args()
    if a.cmd == "codehash":
        print(code_hash()); return
    {"prep": stage_prep, "train": stage_train, "evalck": stage_evalck, "calcheck": stage_calcheck,
     "project": stage_project}[a.cmd](a)


if __name__ == "__main__":
    main()
