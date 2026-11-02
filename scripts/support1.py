"""SUPPORT-1 (lab/SUPPORT1_PROTOCOL.md v2.1, Sol GO at 20d3c3a): can consolidation learn a support rule from exact
oracle supervision? Development; cortex-only.

Stages:
  prep   (phase 0, CPU) seed -> stream, strata, split, T partition, fork record, every schedule (all arms, all sleeps,
         token ids and loss boundaries) and every probe set (both histories)                    -> prep_seed{S}.json
  (train / evalck / restore-test / analyze are added in later commits; analysis is also implemented independently by Sol)

Every random operation follows protocol §2.1: random.Random(key), canonical population order, shuffle/sample/coin.
"""
import argparse, base64, collections, hashlib, inspect, json, math, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import abst0 as A0                                        # frozen ABST-0 stream construction (make_stream, _names)
C = A0.C                                                  # cons0a (FRAME, qa_ids, statement, stems, background)
from being.v3 import lifegen as LG

SCHEMA, EXPERIMENT = "SUPPORT1_SCHEMA_V1", "SUPPORT1"
DEV_SEEDS, CAL_SEED = (20, 21, 22, 23), 24
ATTRS9 = list(A0.ATTRS9)
AIDX = {a: i for i, a in enumerate(ATTRS9)}
ARMS = ("P", "U1", "G1", "U4", "G4")
E_PASSES, SLEEPS, N_HNM, MAX_SEQ = 4, 5, 40, 256
REV_QUOTA = {"U1": 2, "U4": 8}                            # presentations per familiar, not-yet-acquired REV cell per sleep
QT = {
    "city": ["Which city does {e} live in?", "What is the name of the city {e} calls home?"],
    "occupation": ["What does {e} do for a living?", "What is {e}'s line of work?"],
    "favorite object": ["What object does {e} like best?", "Which possession does {e} value above all others?"],
    "pet species": ["What kind of animal does {e} keep as a pet?", "What species is the pet that {e} looks after?"],
    "childhood fear": ["What frightened {e} as a child?", "What was {e} scared of when young?"],
    "workplace": ["Where does {e} work?", "What is the name of the place where {e} is employed?"],
    "hobby": ["What does {e} do for fun?", "Which pastime does {e} enjoy?"],
    "birth month": ["In which month does {e} have a birthday?", "What month was {e} born in?"],
    "allergy": ["What is {e} allergic to?", "What substance causes {e} an allergic reaction?"],
}
QDIAG_FULL = {"POS-Q": 10, "SEL-Q": 12, "HOW-Q": 12, "HNM-Q": 10, "REV-Q": 4, "GTQ": 16}
QDIAG_FALLBACK = {"POS-Q": 4, "SEL-Q": 8, "HOW-Q": 6, "HNM-Q": 4, "REV-Q": 4, "GTQ": 6}   # = the first k of each group
CODE_FILES = ("scripts/support1.py", "scripts/abst0.py", "scripts/cons0a.py", "scripts/cls0c_b.py",
              "being/v3/lifegen.py", "being/competence.py", "data/background.txt", "data/competence_eval.txt",
              "data/competence_modern.txt", "lab/support1_norm_vectors.json")


# ------------------------------------------------------------------ hashing / canonical forms -------------------------
def sha(x):
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def chash(obj):
    return sha(canon(obj))


def code_hash():
    return sha(b"".join(sha((ROOT / p).read_bytes()).encode() for p in CODE_FILES))


def cell_key(c):                                          # canonical cell order: (entity, ATTRS9 index)
    return (c[0], AIDX[c[1]])


# ------------------------------------------------------------------ stream (§2) ----------------------------------------
_SRC = inspect.getsource(A0.make_stream)
assert _SRC.count('f"ABST0-stream-{seed}"') == 1


def make_stream(seed):
    """ABST-0's make_stream, byte for byte, with only the RNG key string changed to "SUPPORT1-stream-{seed}"."""
    ns = dict(vars(A0))
    exec(_SRC.replace('f"ABST0-stream-{seed}"', 'f"SUPPORT1-stream-{seed}"'), ns)
    st = ns["make_stream"](seed, LG)
    A0.check_stream(st)
    return st


def fork_record(st, tok, test_mode=False, max_tries=200_000):
    """§2/§7: for each H transition target (canonical order) one reserved transient entity, drawn after the stream, and
    one distractor fact on the same attribute with a value new to the stream. Both QT wordings of the distractor must
    be token- and loss-matched to the target's (§7). Stored in prep.fork, never in the stream."""
    rng = random.Random(f"SUPPORT1-fork-{st['seed']}")
    used = {w.lower() for e in st["A"] + st["H"] + st["transient"] + st["T"] + st["V"] for w in e.split()}
    used |= {w.lower() for f in st["facts"] for w in re.findall(r"[A-Za-z]+", f["value"])}
    stream_vals = {f["value"] for f in st["facts"]}
    tval = {(f["entity"], f["attr"]): f["value"] for f in st["facts"] if f["role"] == "transition"}
    htr = sorted((tuple(c) for c in st["transitions"] if c[0] in st["H"]), key=cell_key)
    facts, ents = [], []
    for i, (e0, a) in enumerate(htr):
        want = [(len(x[0]), x[1]) for x in (C.qa_ids(tok, QT[a][k].format(e=e0), tval[(e0, a)]) for k in (0, 1))]
        for _ in range(max_tries):
            trial = set(used)
            e = A0._names(rng, LG, 1, trial)[0]
            v = LG._word(rng, rng.choice((1, 2))) + rng.choice(LG.SUFFIX[a])
            if v in stream_vals or set(re.findall(r"[a-z]+", v.lower())) & trial:
                continue
            got = [(len(x[0]), x[1]) for x in (C.qa_ids(tok, QT[a][k].format(e=e), v) for k in (0, 1))]
            if got == want or test_mode:
                used = trial | {w.lower() for w in re.findall(r"[A-Za-z]+", v)}
                break
        else:
            raise SystemExit(f"PREP STOP: no token-matched fork distractor for {(e0, a)}")
        ents.append(e)
        facts.append({"fid": f"F{st['seed']}L4D{i}", "life": 4, "entity": e, "attr": a, "value": v,
                      "role": "fork_distractor", "target": [e0, a],
                      "text": " ".join(t.format(e=e, v=v) for t in LG.TRAIN[a])})
    return {"entities": ents, "facts": facts, "targets": [list(c) for c in htr]}


def history_facts(st, fork, history):
    if history == "observed":
        return list(st["facts"])
    tg = {tuple(c) for c in fork["targets"]}
    return [f for f in st["facts"] if (f["entity"], f["attr"]) not in tg] + list(fork["facts"])


def oracle(facts, L):
    """§3: (entity, attr) -> value of the latest fact with life <= L."""
    k = {}
    for f in sorted((f for f in facts if f["life"] <= L), key=lambda f: f["life"]):
        k[(f["entity"], f["attr"])] = f["value"]
    return k


def strata(st):
    core = st["A"] + st["H"]
    have = {(f["entity"], f["attr"]) for f in st["facts"]}
    planned = {tuple(c) for c in st["transitions"]} | {tuple(c) for c in st["updates"]}
    pm = sorted(((e, a) for e in core for a in ATTRS9 if (e, a) not in have and (e, a) not in planned), key=cell_key)
    rev = sorted((tuple(c) for c in st["transitions"] if c[0] in st["A"]), key=cell_key)
    hnm = [c for c in pm if c[0] in st["H"]]
    late = {(f["entity"], f["attr"]) for f in st["facts"] if f["life"] in (3, 5) and f["entity"] in core}
    htr = {tuple(c) for c in st["transitions"] if c[0] in st["H"]}
    excluded = sorted(late | htr | {tuple(c) for c in st["updates"]}, key=cell_key)
    return {"pm": pm, "rev": rev, "hnm": hnm, "excluded": excluded}


def split(st, pm, max_r=100_000):
    """§2 S4: deterministic retry until every attribute has |#SEL - #HOW| <= 1 across A entities."""
    A_pm = {e: [a for (x, a) in pm if x == e] for e in sorted(st["A"])}
    if any(len(v) < 2 for v in A_pm.values()):
        raise SystemExit("PREP STOP: an A entity has fewer than 2 permanently missing cells")
    for r in range(max_r):
        rng = random.Random(f"SUPPORT1-split-{st['seed']}-{r}")
        sel, how = [], []
        for e in sorted(st["A"]):
            atts = sorted(A_pm[e], key=lambda a: AIDX[a])
            rng.shuffle(atts)
            heads = rng.random() < 0.5
            h = math.ceil(len(atts) / 2)
            s_, h_ = (atts[:h], atts[h:]) if heads else (atts[h:], atts[:h])
            sel += [(e, a) for a in s_]; how += [(e, a) for a in h_]
        if all(abs(sum(a == x for _, x in sel) - sum(a == x for _, x in how)) <= 1 for a in ATTRS9):
            return {"r": r, "sel": sorted(sel, key=cell_key), "how": sorted(how, key=cell_key)}
    raise SystemExit(f"PREP STOP: no balanced split within {max_r} retries")


def tpart(st):
    t = sorted(st["T"])
    random.Random(f"SUPPORT1-tpart-{st['seed']}").shuffle(t)
    return {"t_neg": t[:len(t) // 2], "t_swap": t[len(t) // 2:]}


# ------------------------------------------------------------------ cyclers (§2.1) ---------------------------------------
class Cycler:
    """Shuffled-once canonical population; consumed in order with wraparound. An ineligible entry advances the cursor
    without consuming quota; a full traversal with nothing eligible is a STOP (Sol's clarification at GO)."""

    def __init__(self, population, key):
        self.items = list(population)
        random.Random(key).shuffle(self.items)
        self.i = 0

    def take(self, eligible):
        for _ in range(len(self.items)):
            x = self.items[self.i % len(self.items)]
            self.i += 1
            if eligible(x):
                return x
        raise SystemExit("PREP STOP: cycler found no eligible entry in a full traversal")


# ------------------------------------------------------------------ schedules (§4) ---------------------------------------
def frame_ids(tok, q, label):
    ids, n_loss = C.qa_ids(tok, q, label)
    return ids, n_loss


def entry(pos_kind, slot, ids, n_loss, cell=None, src_cell=None, wording=None, label=None, item_id=None, source=None):
    return {"slot": slot, "kind": pos_kind, "item_id": item_id, "source_item_id": source,
            "cell": list(cell) if cell else None, "src_cell": list(src_cell) if src_cell else None,
            "wording": wording, "label": label, "input_ids": list(ids), "loss_start": len(ids) - n_loss, "n_loss": n_loss}


def build_schedules(st, fork, strat, spl, tp, tok, chunk, test_mode=False):
    """Every arm's schedule for every sleep (main history), the fork's withheld sleep-4 schedule (U4, G4), the swap
    map and the per-sleep base blocks. Pure given the tokenizer and background chunker."""
    seed = st["seed"]
    intro = st["intro"]
    facts = st["facts"]
    sel = [tuple(c) for c in spl["sel"]]
    rev = [tuple(c) for c in strat["rev"]]
    sel_cyc = Cycler(sorted(sel, key=cell_key), f"SUPPORT1-selcyc-{seed}")
    sel_toggle = {c: 0 for c in sel}
    t_pop = sorted(((t, AIDX[a], k) for t in tp["t_neg"] for a in ATTRS9 for k in (0, 1)))
    t_cyc = Cycler(t_pop, f"SUPPORT1-tcyc-{seed}")
    swap_pool = sorted(tp["t_swap"]); random.Random(f"SUPPORT1-swap-{seed}").shuffle(swap_pool)   # §2.1: sorted, then shuffled
    # Amendment A1 (proposed to Sol): ONE fixed T_swap name per A entity, the first in swap_pool order whose framed
    # (length, n_loss) equals the entity's for ALL 18 question forms (9 attributes x 2 wordings), so G repeats the same
    # fake person exactly as U repeats the real one.
    sig = lambda name: [tuple((len(x[0]), x[1])) for x in (C.qa_ids(tok, QT[a][k].format(e=name), "UNKNOWN")
                                                            for a in ATTRS9 for k in (0, 1))]
    swap_of, taken = {}, set()
    for e in sorted(st["A"]):
        want = sig(e)
        for t in swap_pool:
            if t not in taken and (sig(t) == want or test_mode):     # test mode: tiny BPE cannot length-match
                swap_of[e] = t; taken.add(t); break
        else:
            raise SystemExit(f"PREP STOP: no T_swap name matches {e} on all 18 question forms")
    positives, sched, swap_map, blocks, fork_sched = {}, {arm: {} for arm in ARMS}, {}, {}, None
    for L in range(1, SLEEPS + 1):
        orc = oracle(facts, L)
        newf = sorted((f for f in facts if f["life"] == L), key=lambda f: f["fid"])
        pos = [{"item_id": f"{f['fid']}-q{k}", "cell": [f["entity"], f["attr"]], "wording": k, "label": f["value"],
                "life": L} for f in newf for k in (0, 1)]
        positives[L] = pos
        n = len(pos)

        def pos_entry(p, slot, label=None, source=None):
            q = QT[p["cell"][1]][p["wording"]].format(e=p["cell"][0])
            lab = label if label is not None else p["label"]
            ids, nl = frame_ids(tok, q, lab)
            return entry("pos" if slot == "new" else "replay", slot, ids, nl, cell=p["cell"], wording=p["wording"],
                         label=lab, item_id=p["item_id"] if slot == "new" else None, source=source)
        new = [pos_entry(p, "new") for _ in range(E_PASSES) for p in pos]
        # replay (S2): unique acquired (cell, wording) of lives < L, relabelled with the oracle at L
        if L == 1:
            replay = [entry("bg", "replay", chunk(1_000_000 + i, len(x["input_ids"])), x["n_loss"])
                      for i, x in enumerate(new)]
        else:
            latest = {}
            for Lp in range(1, L):
                for p in positives[Lp]:
                    latest[(tuple(p["cell"]), p["wording"])] = p["item_id"]
            pop = sorted(latest, key=lambda cw: (cell_key(cw[0]), cw[1]))
            rcyc = Cycler(pop, f"SUPPORT1-replay-{seed}-{L}")
            replay = []
            for _ in range(E_PASSES * n):
                c, k = rcyc.take(lambda x: True)
                replay.append(pos_entry({"cell": list(c), "wording": k, "item_id": None}, "replay",
                                        label=orc[c], source=latest[(c, k)]))
        # nested base block B_L (S1)
        A_part = []
        for c in rev:
            if intro[c[0]] <= L and c not in orc:
                A_part += [(c, 0), (c, 1)]
        if len(A_part) > n:                              # base block = U1 dose: 2 per REV cell; U4 = 4x = 8
            raise SystemExit(f"PREP STOP: REV quota {len(A_part)} > A positions {n} at sleep {L}")
        while len(A_part) < n:
            c = sel_cyc.take(lambda c: intro[c[0]] <= L)
            A_part.append((c, sel_toggle[c])); sel_toggle[c] ^= 1
        T_part = []
        for _ in range(n):
            t, ai, k = t_cyc.take(lambda x: True)
            T_part.append(((t, ATTRS9[ai]), k))
        base_U, base_G, smap = [], [], []
        for c, k in A_part:
            q = QT[c[1]][k].format(e=c[0])
            ids, nl = frame_ids(tok, q, "UNKNOWN")
            base_U.append(entry("neg_a", "third", ids, nl, cell=c, wording=k, label="UNKNOWN"))
            t = swap_of[c[0]]
            ids2, nl2 = frame_ids(tok, QT[c[1]][k].format(e=t), "UNKNOWN")
            base_G.append(entry("neg_swap", "third", ids2, nl2, cell=(t, c[1]), src_cell=c, wording=k, label="UNKNOWN"))
            smap.append({"src_cell": list(c), "wording": k, "t_name": t})
        for (t, a), k in T_part:
            ids, nl = frame_ids(tok, QT[a][k].format(e=t), "UNKNOWN")
            e_ = entry("neg_t", "third", ids, nl, cell=(t, a), wording=k, label="UNKNOWN")
            base_U.append(e_); base_G.append(dict(e_))
        swap_map[L] = smap
        blocks[L] = {"A": [[list(c), k] for c, k in A_part], "T": [[[t, a], k] for (t, a), k in T_part]}
        U4 = [dict(x) for _ in range(4) for x in base_U]
        G4 = [dict(x) for _ in range(4) for x in base_G]
        bg = [entry("bg", "third", chunk(2_000_000 + L * 10_000 + j, len(x["input_ids"])), x["n_loss"])
              for j, x in enumerate(U4)]
        third = {"U4": U4, "G4": G4, "P": bg,
                 "U1": [dict(x) for x in base_U] + bg[len(base_U):],
                 "G1": [dict(x) for x in base_G] + bg[len(base_G):]}
        perm = list(range(len(new) + len(replay) + len(U4)))
        random.Random(f"SUPPORT1-perm-{seed}-{L}").shuffle(perm)
        for arm in ARMS:
            content = new + replay + third[arm]
            sched[arm][L] = [dict(content[p], pos=i) for i, p in enumerate(perm)]
        if L == 4:                                         # §7: withheld history, all 32 target presentations replaced
            dist = {tuple(f["target"]): f for f in fork["facts"]}
            fork_sched = {}
            for arm in ("U4", "G4"):
                rows = []
                for x in sched[arm][L]:
                    if x["kind"] == "pos" and x["cell"] and tuple(x["cell"]) in dist:
                        f = dist[tuple(x["cell"])]
                        ids, nl = frame_ids(tok, QT[f["attr"]][x["wording"]].format(e=f["entity"]), f["value"])
                        y = entry("pos", "new", ids, nl, cell=(f["entity"], f["attr"]), wording=x["wording"],
                                  label=f["value"], item_id=f"{f['fid']}-q{x['wording']}")
                        if (len(ids), nl) != (len(x["input_ids"]), x["n_loss"]) and not test_mode:
                            raise SystemExit(f"PREP STOP: fork distractor {f['fid']} not token/loss-matched to {x['cell']}")
                        rows.append(dict(y, pos=x["pos"]))
                    else:
                        rows.append(dict(x))
                fork_sched[arm] = rows
    return positives, sched, swap_map, blocks, fork_sched


def budget(rows):
    return [len(rows), sum(len(x["input_ids"]) for x in rows), sum(x["n_loss"] for x in rows)]


# ------------------------------------------------------------------ probes (§5) ------------------------------------------
def build_probes(st, fork, strat, spl, sched, swap_map, history, L):
    """All probes of one (history, sleep), with stable IDs and gold from that history's oracle at L."""
    seed, intro = st["seed"], dict(st["intro"])
    facts = history_facts(st, fork, history)
    if history == "withheld":
        for e in fork["entities"]:
            intro[e] = 4
    orc = oracle(facts, L)
    upd = {tuple(c) for c in st["updates"]}
    tr = {tuple(c) for c in st["transitions"]}
    counters, rows = {}, []

    def add(cat, construction, kind, cell, item, gold, **kw):
        n = counters.get(cat, 0); counters[cat] = n + 1
        r = {"probe_id": f"{seed}-{history}-{L}-{cat}-{n:04d}", "cat": cat, "construction": construction, "kind": kind,
             "cell": list(cell), "item": item, "gold_current": gold, "gold_old": None, "gold_new": None,
             "delivered_eligible": None, "pair_probe_id": None}
        r.update(kw); rows.append(r); return r["probe_id"]

    def stmt_pos(cat, c, gold):
        add(cat, "statement", "template", c, C.statement(C.template_stem(LG, c[1], c[0])), gold)
        for k, s in C.paraphrase_stems(LG, c[1], c[0]):
            add(cat, "statement", f"para{k}", c, C.statement(s), gold)

    def stmt_neg(cat, c, gold="UNKNOWN"):
        return add(cat, "statement", "template", c, C.statement(C.template_stem(LG, c[1], c[0])), gold)

    def ques(cat, c, k, gold, **kw):
        return add(cat, "question", f"qt{k}", c, C.question(QT[c[1]][k].format(e=c[0])), gold, **kw)
    seen = set()
    for f in sorted((f for f in facts if f["life"] <= L), key=lambda f: (f["life"], f["fid"])):
        c = (f["entity"], f["attr"])
        if c in upd or c in seen:
            continue
        seen.add(c)
        stmt_pos("POS", c, orc[c])
        if intro.get(f["entity"]) == L:
            stmt_pos("FIRST", c, orc[c])
        if f["life"] == 1:
            stmt_pos("RET", c, orc[c])
        if c in tr and L >= 4 and c in orc:
            stmt_pos("TPOST-A" if c[0] in st["A"] else "TPOST-H", c, orc[c])
    if L >= 2:
        for c in sorted(upd, key=cell_key):
            f0 = next(f for f in facts if (f["entity"], f["attr"]) == c and f["role"] == "update_init")
            f1 = next(f for f in facts if (f["entity"], f["attr"]) == c and f["role"] == "update_new")
            add("UPD", "statement", "template", c, C.statement(C.template_stem(LG, c[1], c[0])), orc[c],
                gold_old=f0["value"], gold_new=f1["value"] if L >= 4 else None)
    if L <= 3:
        for c in sorted(tr, key=cell_key):
            grp = "A" if c[0] in st["A"] else "H"
            stmt_neg(f"TPRE-{grp}" if intro[c[0]] <= L else f"TPRE-{grp}-UNFAM", c)
    # SEL delivered / pending: delivered = scheduled in U1 at any sleep <= L (main history schedule)
    delivered = {tuple(x["cell"]) for Lp in range(1, L + 1) for x in sched["U1"][Lp] if x["kind"] == "neg_a"}
    stmt_ids = {}
    for c in [tuple(x) for x in spl["sel"]]:
        if intro[c[0]] <= L:
            stmt_ids[c] = stmt_neg("SEL-D" if c in delivered else "SEL-P", c)
    for c in [tuple(x) for x in spl["how"]]:
        if intro[c[0]] <= L:
            stmt_ids[c] = stmt_neg("HOW", c)
    hpop = [tuple(c) for c in strat["hnm"] if intro[c[0]] <= L]
    hsel = sorted(random.Random(f"SUPPORT1-hnm-{seed}-{L}").sample(hpop, min(N_HNM, len(hpop))), key=cell_key)
    for c in hsel:
        stmt_neg("HNM", c)
    for v, a in st["fresh"]:
        stmt_neg("FRESH", (v, a))
    if L == 4:                                             # §7 fork targets, asked in both histories
        for c in [tuple(x) for x in fork["targets"]]:
            stmt_neg("FORK-H", c, gold=orc.get(c, "UNKNOWN"))
            ques("FORK-H-Q", c, 0, orc.get(c, "UNKNOWN"))
    # QDIAG (S3): fixed rows, identical across arms; fallback-32 = the first k of each group
    rng = random.Random(f"SUPPORT1-qdiag-{seed}-{L}")
    posc = sorted(seen, key=cell_key)
    for i, c in enumerate(rng.sample(posc, min(QDIAG_FULL["POS-Q"], len(posc)))):
        ques("POS-Q", c, i % 2, orc[c])
    dpairs = sorted({(tuple(x["cell"]), x["wording"]) for Lp in range(1, L + 1) for x in sched["U1"][Lp]
                     if x["kind"] == "neg_a" and tuple(x["cell"]) in {tuple(s) for s in spl["sel"]}},
                    key=lambda cw: (cell_key(cw[0]), cw[1]))
    for c, k in rng.sample(dpairs, min(QDIAG_FULL["SEL-Q"], len(dpairs))):
        ques("SEL-Q", c, k, "UNKNOWN", delivered_eligible=True, pair_probe_id=stmt_ids[c])
    howc = sorted((tuple(x) for x in spl["how"] if intro[x[0]] <= L), key=cell_key)
    for i, c in enumerate(rng.sample(howc, min(QDIAG_FULL["HOW-Q"], len(howc)))):
        ques("HOW-Q", c, i % 2, "UNKNOWN", pair_probe_id=stmt_ids[c])
    for i, c in enumerate(rng.sample(hsel, min(QDIAG_FULL["HNM-Q"], len(hsel)))):
        ques("HNM-Q", c, i % 2, "UNKNOWN")
    for c in [tuple(x) for x in strat["rev"]]:
        ques("REV-Q", c, 0, orc.get(c, "UNKNOWN"))
    gpairs = sorted({(m["t_name"], m["src_cell"][1], m["wording"]) for Lp in range(1, L + 1) for m in swap_map[Lp]},
                    key=lambda x: (x[0], AIDX[x[1]], x[2]))
    for t, a, k in rng.sample(gpairs, min(QDIAG_FULL["GTQ"], len(gpairs))):
        ques("GTQ", (t, a), k, "UNKNOWN")
    return rows


# ------------------------------------------------------------------ prep stage ------------------------------------------
def prep(seed, tok, model_manifest="", test_mode=False):
    st = make_stream(seed)
    fork = fork_record(st, tok, test_mode)
    strat = strata(st)
    spl = split(st, [tuple(c) for c in strat["pm"]])
    tp = tpart(st)
    chunk = C.background_chunker(tok, (ROOT / "data/background.txt").read_text().splitlines(), f"SUPPORT1-{seed}")
    positives, sched, swap_map, blocks, fork_sched = build_schedules(st, fork, strat, spl, tp, tok, chunk, test_mode)
    bud = {arm: {L: budget(sched[arm][L]) for L in sched[arm]} for arm in ARMS}
    for L in range(1, SLEEPS + 1):
        arms_chk = ARMS if not test_mode else ("P", "U1", "U4")        # test mode: G unmatched by design (tiny BPE)
        if len({tuple(bud[arm][L]) for arm in arms_chk}) != 1:
            raise SystemExit(f"PREP STOP: budget mismatch at sleep {L}: { {a: bud[a][L] for a in ARMS} }")
    fb = {arm: budget(fork_sched[arm]) for arm in fork_sched}
    if not test_mode and any(v != bud["U4"][4] for v in fb.values()):
        raise SystemExit(f"PREP STOP: fork budget differs from main sleep 4: {fb}")
    too_long = [len(x["input_ids"]) for arm in ARMS for L in sched[arm] for x in sched[arm][L]
                if len(x["input_ids"]) > MAX_SEQ]
    if too_long and not test_mode:
        raise SystemExit(f"PREP STOP: {len(too_long)} sequences over {MAX_SEQ} tokens")
    probes = {"observed": {L: build_probes(st, fork, strat, spl, sched, swap_map, "observed", L)
                           for L in range(1, SLEEPS + 1)},
              "withheld": {4: build_probes(st, fork, strat, spl, sched, swap_map, "withheld", 4)}}
    vec = json.loads((ROOT / "lab/support1_norm_vectors.json").read_text())
    if any(C.B.norm(a) != b for a, b in vec):
        raise SystemExit("PREP STOP: normaliser test vectors fail")
    keys = {"stream": f"SUPPORT1-stream-{seed}", "fork": f"SUPPORT1-fork-{seed}", "split": f"SUPPORT1-split-{seed}-{{r}}",
            "tpart": f"SUPPORT1-tpart-{seed}", "selcyc": f"SUPPORT1-selcyc-{seed}", "tcyc": f"SUPPORT1-tcyc-{seed}",
            "replay": f"SUPPORT1-replay-{seed}-{{L}}", "swap": f"SUPPORT1-swap-{seed}", "perm": f"SUPPORT1-perm-{seed}-{{L}}",
            "hnm": f"SUPPORT1-hnm-{seed}-{{L}}", "qdiag": f"SUPPORT1-qdiag-{seed}-{{L}}",
            "background": f"CONS0A-bg-SUPPORT1-{seed}-{{i}}"}
    return {"schema": SCHEMA, "experiment": EXPERIMENT, "seed": seed, "test_mode": test_mode, "code_hash": code_hash(),
            "model_manifest": model_manifest, "stream": st, "stream_sha256": chash(st), "tpart": tp,
            "split": {"r": spl["r"], "sel": [list(c) for c in spl["sel"]], "how": [list(c) for c in spl["how"]]},
            "strata": {k: [list(c) for c in v] for k, v in strat.items()}, "fork": fork,
            "positives": {str(L): v for L, v in positives.items()},
            "schedule": {arm: {str(L): v for L, v in sched[arm].items()} for arm in ARMS},
            "fork_schedule": {"L": 4, **fork_sched}, "swap_map": {str(L): v for L, v in swap_map.items()},
            "base_blocks": {str(L): v for L, v in blocks.items()},
            "probes": {h: {str(L): v for L, v in d.items()} for h, d in probes.items()},
            "sampler_keys": keys, "qdiag_fallback": QDIAG_FALLBACK,
            "normalisation": {"code": "cls0c_b.norm", "vectors_sha256": sha((ROOT / "lab/support1_norm_vectors.json").read_bytes())},
            "budget": {arm: {str(L): v for L, v in d.items()} for arm, d in bud.items()}}


def stage_prep(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if not a.test_mode and not a.model_manifest_sha:
        sys.exit("PREP: production prep needs --model-manifest-sha")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        if not a.test_mode and s not in DEV_SEEDS + (CAL_SEED,):
            sys.exit(f"PREP: seed {s} is not a SUPPORT-1 seed")
        p = prep(s, tok, a.model_manifest_sha, a.test_mode)
        f = out / f"prep_seed{s}.json"
        f.write_text(json.dumps(p, ensure_ascii=False))
        n_rows = sum(len(v) for L, v in p["probes"]["observed"].items())
        print(f"PREP seed {s}: split r={p['split']['r']} SEL {len(p['split']['sel'])} HOW {len(p['split']['how'])} "
              f"HNM-eligible {len(p['strata']['hnm'])}; budget/sleep {[p['budget']['P'][str(L)] for L in range(1, 6)]}; "
              f"observed probe rows {n_rows}; sha256 {sha(f.read_bytes())}", flush=True)


# ------------------------------------------------------------------ outcome (§3) -----------------------------------------
def outcome(ans, gold):
    n = C.B.norm(ans)
    if n == "":
        return "empty"
    if n == "unknown":
        return "unknown"
    if gold != "UNKNOWN" and n == C.B.norm(gold):
        return "correct"
    return "substantive"


# ------------------------------------------------------------------ full-state identity (§7 E2) ------------------------
def _named_tensors(cx, opt):
    """LoRA tensors by sorted site name, then optimiser state as opt/{param_name}/{state_key} (sorted)."""
    pname = {}
    out = []
    for name, _, A, Bm in sorted(cx.sites, key=lambda s: s[0]):
        out += [(f"{name}.A", A), (f"{name}.B", Bm)]
        pname[id(A)], pname[id(Bm)] = f"{name}.A", f"{name}.B"
    for group in opt.param_groups:
        for p in group["params"]:
            for k, v in sorted(opt.state.get(p, {}).items()):
                if hasattr(v, "detach"):
                    out.append((f"opt/{pname[id(p)]}/{k}", v))
    return sorted(out[:2 * len(cx.sites)], key=lambda x: x[0]) + sorted(out[2 * len(cx.sites):], key=lambda x: x[0])


def tensor_hash(cx, opt):
    import numpy as np
    h = hashlib.sha256()
    for name, t in _named_tensors(cx, opt):
        a = t.detach().to("cpu").contiguous()
        h.update(name.encode() + b"\0" + str(a.dtype).encode() + b"\0" + json.dumps(list(a.shape)).encode() + b"\0")
        h.update(a.view(-1).view(__import__("torch").uint8).numpy().astype(np.uint8).tobytes() if a.numel() else b"")
    return h.hexdigest()


def rng_hash():
    import numpy as np, torch
    ns = np.random.get_state()
    st = {"python": repr(random.getstate()),
          "numpy": [ns[0], base64.b64encode(ns[1].tobytes()).decode(), ns[2], ns[3], ns[4]],
          "torch_cpu": base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode(),
          "cuda": [base64.b64encode(s.numpy().tobytes()).decode() for s in torch.cuda.get_rng_state_all()]
          if torch.cuda.is_available() else []}
    return chash(st)


def full_state_hash(cx, opt, step):
    return chash({"tensors": tensor_hash(cx, opt), "rng": rng_hash(), "step": step})


def capture(cx, opt, step):
    import numpy as np, torch
    return {"cx": cx.state(), "opt": __import__("copy").deepcopy(opt.state_dict()), "step": step,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}


def restore(cx, opt, s):
    import numpy as np, torch
    cx.load_state(s["cx"])
    opt.load_state_dict(__import__("copy").deepcopy(s["opt"]))
    random.setstate(s["rng"]["python"]); np.random.set_state(s["rng"]["numpy"]); torch.set_rng_state(s["rng"]["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["rng"]["cuda"])
    return s["step"]


# ------------------------------------------------------------------ train (§4, §7) ---------------------------------------
def _items(rows):
    return [{"text": None, "ids": x["input_ids"], "loss_tail": x["n_loss"]} for x in rows]


def _delivered(rows):
    c = collections.Counter((x["kind"], tuple(x["cell"]), tuple(x["src_cell"]) if x["src_cell"] else None, x["wording"],
                             x["label"]) for x in rows if x["kind"] != "bg")
    return [{"kind": k, "cell": list(cell), "src_cell": list(src) if src else None, "wording": w, "label": lab, "count": n}
            for (k, cell, src, w, lab), n in sorted(c.items(), key=lambda kv: json.dumps(kv[0], default=str))]


def _setup_determinism():
    import os, torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _guard(prep, psha, a, kind):
    if a.test_mode:
        return
    if prep["test_mode"] or not a.model_manifest_sha or prep["model_manifest"] != a.model_manifest_sha:
        sys.exit(f"{kind}: production needs a non-test prep built with this model (--model-manifest-sha)")
    if prep["code_hash"] != code_hash():
        sys.exit(f"{kind}: prep was built by different code")
    if prep["seed"] not in DEV_SEEDS + (CAL_SEED,):
        sys.exit(f"{kind}: seed {prep['seed']} is not a SUPPORT-1 seed")
    if prep["seed"] in DEV_SEEDS:
        v = json.loads(Path(a.validator_report).read_text()) if getattr(a, "validator_report", None) else None
        if not v or v.get("verdict") != "PASS" or psha not in v.get("prep_sha256", []):
            sys.exit(f"{kind}: needs Sol's validator report with verdict PASS covering this prep's sha256")
        c = json.loads(Path(a.calibration_result).read_text()) if getattr(a, "calibration_result", None) else None
        cp = Path(a.calibration_prep).read_bytes() if getattr(a, "calibration_prep", None) else None
        if (not c or c.get("pass") is not True or c.get("seed") != CAL_SEED or c.get("errors") or c.get("test_mode")
                or c.get("code_hash") != code_hash() or c.get("model_manifest") != a.model_manifest_sha
                or cp is None or c.get("prep_sha256") != sha(cp)):
            sys.exit(f"{kind}: needs a passing calibration result bound to this code, model and the seed-24 prep")


def stage_train(a):
    import torch
    prep = json.loads(Path(a.prep).read_text())
    psha = sha(Path(a.prep).read_bytes())
    _guard(prep, psha, a, "TRAIN")
    _setup_determinism()
    seed = prep["seed"]
    arms = a.arms.split(",")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prov = {"code_hash": code_hash(), "prep_sha256": psha, "model_manifest": a.model_manifest_sha,
            "test_mode": a.test_mode, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "arms": arms, "sleeps": a.sleeps, "fork": a.fork}
    done = out / f"train_seed{seed}.json"
    if done.exists():
        old = json.loads(done.read_text())
        ok = old["provenance"] == prov and all(
            (out / f"cortex_{arm}_s{L}.pt").exists() and sha((out / f"cortex_{arm}_s{L}.pt").read_bytes()) ==
            old["arms"][arm][str(L)]["ckpt_sha256"] for arm in arms for L in range(1, a.sleeps + 1))
        if ok:
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
            C.train_sleep(model, tok, cx, opt, _items(rows), dev, lg_)
            model.eval()
            step += lg_["updates"]
            ck = out / f"cortex_{arm}_s{L}.pt"
            torch.save(cx.state(), ck)
            res["arms"][arm][str(L)] = {"ckpt_sha256": sha(ck.read_bytes()), "full_state_hash": full_state_hash(cx, opt, step),
                                        "delivered": _delivered(rows), "budget": budget(rows),
                                        "loss_first": lg_["loss_first"], "loss_last": lg_["loss_last"], "seconds": lg_["seconds"]}
            print(time.strftime("%H:%M:%S"), f"TRAIN seed {seed} {arm} sleep {L}: loss {lg_['loss_first']:.3f}->"
                  f"{lg_['loss_last']:.3f} ({lg_['seconds']}s)", flush=True)
            if a.fork and arm in ("U4", "G4") and L == 3:
                parent = capture(cx, opt, step)
                res["fork"]["parent_full_state_hash"][arm] = res["arms"][arm]["3"]["full_state_hash"]
        if parent is not None and a.sleeps >= 4:            # §7 withheld branch from the exact sleep-3 state
            step = restore(cx, opt, parent)
            assert full_state_hash(cx, opt, step) == res["fork"]["parent_full_state_hash"][arm], "fork parent not restored"
            rows = prep["fork_schedule"][arm]
            lg_ = {}
            C.cx_model_train(model)
            C.train_sleep(model, tok, cx, opt, _items(rows), dev, lg_)
            model.eval()
            step += lg_["updates"]
            ck = out / f"cortex_{arm}_s4_withheld.pt"
            torch.save(cx.state(), ck)
            res["fork"]["withheld"][arm] = {"ckpt_sha256": sha(ck.read_bytes()),
                                            "full_state_hash": full_state_hash(cx, opt, step), "delivered": _delivered(rows)}
    done.write_text(json.dumps(res, indent=1))
    print(f"TRAIN seed {seed} DONE", flush=True)


def stage_restoretest(a):
    """§7/§8 admission: sleep 4 run uninterrupted vs resumed from the saved sleep-3 state -> canonical state equality
    (every LoRA and optimiser tensor torch.equal, step equal, RNG byte-equal)."""
    import torch
    prep = json.loads(Path(a.prep).read_text())
    _setup_determinism()
    seed, arm = prep["seed"], "U4"
    dev = C._dev()
    torch.manual_seed(seed); random.seed(seed)
    model, tok = C.load_model(a.model, dev)
    cx = C.Cortex(model, seed=seed)
    opt = torch.optim.AdamW(cx.params(), lr=C.LR, betas=(0.9, 0.999), weight_decay=0.0)
    step = 0
    for L in (1, 2, 3):
        lg_ = {}; C.cx_model_train(model); C.train_sleep(model, tok, cx, opt, _items(prep["schedule"][arm][str(L)]), dev, lg_)
        step += lg_["updates"]
    s3 = capture(cx, opt, step)

    def sleep4():
        lg_ = {}; C.cx_model_train(model); C.train_sleep(model, tok, cx, opt, _items(prep["schedule"][arm]["4"]), dev, lg_)
        return lg_["updates"]
    s_a = step + sleep4()
    A = {n: t.detach().cpu().clone() for n, t in _named_tensors(cx, opt)}; ha = full_state_hash(cx, opt, s_a)
    s_b = restore(cx, opt, s3) + sleep4()
    B = {n: t.detach().cpu().clone() for n, t in _named_tensors(cx, opt)}; hb = full_state_hash(cx, opt, s_b)
    eq = set(A) == set(B) and all(torch.equal(A[n], B[n]) for n in A)
    res = {"schema": SCHEMA, "experiment": EXPERIMENT, "which": "restoretest", "seed": seed, "arm": arm,
           "tensors_equal": eq, "step_equal": s_a == s_b, "full_state_hash": [ha, hb], "admitted": bool(eq and s_a == s_b and ha == hb),
           "code_hash": code_hash(), "prep_sha256": sha(Path(a.prep).read_bytes())}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"RESTORETEST admitted={res['admitted']} tensors_equal={eq} hashes_equal={ha == hb}", flush=True)


# ------------------------------------------------------------------ evaluation (§5) --------------------------------------
def stage_evalck(a):
    import torch
    from being.competence import mean_loss
    prep = json.loads(Path(a.prep).read_text())
    psha = sha(Path(a.prep).read_bytes())
    if not a.test_mode and (prep["test_mode"] or prep["model_manifest"] != a.model_manifest_sha or prep["code_hash"] != code_hash()):
        sys.exit("EVAL: production evaluation needs a non-test prep built by this code and model")
    out = Path(a.out)
    ck = Path(a.ckpt) if a.ckpt else None
    mode = "base" if ck is None else a.mode
    prov = {"code_hash": code_hash(), "prep_sha256": psha, "ckpt_sha256": sha(ck.read_bytes()) if ck else None,
            "model_manifest": a.model_manifest_sha, "test_mode": a.test_mode, "qdiag": a.qdiag}
    if out.exists() and json.loads(out.read_text()).get("provenance") == prov:
        print(f"EVAL {out.name}: up to date - skipped"); return
    model, tok, dev = A0.load_fp32(a.model)
    comp_sets, base_loss = C._comp(tok, model, a.test_mode)
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
    P = prep["probes"][a.history][str(a.sleep)]
    if a.qdiag == 32:                                      # fallback: the first k rows of each QDIAG group
        keep, seen = set(), collections.Counter()
        for r in P:
            if r["cat"] in QDIAG_FALLBACK:
                seen[r["cat"]] += 1
                if seen[r["cat"]] <= QDIAG_FALLBACK[r["cat"]]:
                    keep.add(r["probe_id"])
            else:
                keep.add(r["probe_id"])
        P = [r for r in P if r["probe_id"] in keep]
    gens = C.generate(model, tok, [C.frame(r["item"]) for r in P], dev)
    rows = [{"probe_id": r["probe_id"], "answer": C.first_line(g), "outcome": outcome(C.first_line(g), r["gold_current"])}
            for r, g in zip(P, gens)]
    ppl = {k: math.exp(mean_loss(model, tok, s)) / math.exp(base_loss[k]) for k, s in comp_sets.items()}
    ev = {"schema": SCHEMA, "experiment": EXPERIMENT, "seed": prep["seed"], "arm": a.arm, "sleep": a.sleep, "mode": mode,
          "history": a.history, "provenance": prov, "ppl_ratio": ppl, "base_denominator_sha256": chash(base_loss),
          "rows": rows, "env": {"torch": torch.__version__, "tf32": torch.backends.cuda.matmul.allow_tf32,
                                "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}}
    out.write_text(json.dumps(ev, indent=1))
    by = {r["probe_id"]: r for r in P}
    summ = lambda cat, cls: (lambda xs: round(sum(x["outcome"] == cls for x in xs) / len(xs), 3) if xs else None)(
        [x for x in rows if by[x["probe_id"]]["cat"] == cat])
    print(f"EVAL {out.name}: POS {summ('POS', 'correct')} SEL-Q unk {summ('SEL-Q', 'unknown')} SEL-D unk {summ('SEL-D', 'unknown')} "
          f"HOW unk {summ('HOW', 'unknown')} HNM conf {summ('HNM', 'substantive')} unk {summ('HNM', 'unknown')} "
          f"FRESH unk {summ('FRESH', 'unknown')} ppl {ppl}", flush=True)


def stage_calcheck(a):
    """§8 phase-1 calibration gate (Sol's HOLD at fba75d5): seed 24, P sleep 1, E=4. Binds every input to the current
    artifacts and grades independently:
      - complete sleep-1 probe inventory, in prep order, for merged, unmerged and base;
      - provenance: prep, code, model, checkpoint (from the train log), arm/sleep/mode/history/qdiag;
      - outcomes RECOMPUTED from answers + prep gold;
      - twins: 0 answer/outcome mismatches, |dppl| <= 1e-4 on both monitors, one shared base denominator;
      - competence: both monitors finite and <= 1.02;
      - POS template acquisition >= 0.80 over ALL POS template probes; N FRESH exact-UNKNOWN >= 0.90 over ALL FRESH."""
    prep_b = Path(a.prep).read_bytes(); prep = json.loads(prep_b); psha = sha(prep_b)
    tr_b = Path(a.train).read_bytes(); tr = json.loads(tr_b)
    evs = {k: (Path(f).read_bytes(), json.loads(Path(f).read_text())) for k, f in
           (("merged", a.merged), ("unmerged", a.unmerged), ("base", a.base))}
    errs, code = [], code_hash()
    inv = prep["probes"]["observed"]["1"]
    gold = {r["probe_id"]: r for r in inv}
    if prep["seed"] != CAL_SEED:
        errs.append(f"prep seed {prep['seed']} is not the calibration seed")
    if prep["code_hash"] != code:
        errs.append("prep built by different code")
    tp = tr.get("provenance", {})
    if (tr.get("seed") != CAL_SEED or tp.get("prep_sha256") != psha or tp.get("code_hash") != code or tp.get("arms") != ["P"]
            or tp.get("sleeps") != 1 or tp.get("model_manifest") != prep["model_manifest"]
            or tp.get("test_mode") != prep["test_mode"]):
        errs.append("train log is not P sleep 1 of this prep/code/model")
    ck = tr.get("arms", {}).get("P", {}).get("1", {}).get("ckpt_sha256")
    for mode, (_, ev) in evs.items():
        pv = ev.get("provenance", {})
        want = {"code_hash": code, "prep_sha256": psha, "ckpt_sha256": None if mode == "base" else ck,
                "model_manifest": prep["model_manifest"], "test_mode": prep["test_mode"], "qdiag": 64}
        if {k: pv.get(k) for k in want} != want:
            errs.append(f"{mode}: provenance does not bind to this prep/code/model/checkpoint")
        if (ev.get("schema"), ev.get("experiment"), ev.get("seed"), ev.get("sleep"), ev.get("mode"), ev.get("history"),
                ev.get("arm")) != (SCHEMA, EXPERIMENT, CAL_SEED, 1, mode, "observed", "N" if mode == "base" else "P"):
            errs.append(f"{mode}: identity fields wrong")
        if [r["probe_id"] for r in ev.get("rows", [])] != [r["probe_id"] for r in inv]:
            errs.append(f"{mode}: rows are not the complete sleep-1 inventory in prep order")
        elif any(outcome(r["answer"], gold[r["probe_id"]]["gold_current"]) != r["outcome"] for r in ev["rows"]):
            errs.append(f"{mode}: stored outcomes disagree with independent grading")
        ppl = ev.get("ppl_ratio") or {}
        if set(ppl) != {"eval", "modern"} or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in ppl.values()):
            errs.append(f"{mode}: both competence monitors must be present and finite")
    m = n = None
    if not errs:
        m, mu, n = evs["merged"][1], evs["unmerged"][1], evs["base"][1]
        if any(x["answer"] != y["answer"] or x["outcome"] != y["outcome"] for x, y in zip(m["rows"], mu["rows"])):
            errs.append("merged/unmerged answer or outcome mismatch")
        if any(abs(m["ppl_ratio"][k] - mu["ppl_ratio"][k]) > 1e-4 for k in ("eval", "modern")):
            errs.append("merged/unmerged ppl mismatch > 1e-4")
        if len({ev.get("base_denominator_sha256") for _, ev in evs.values()}) != 1:
            errs.append("base denominators differ")

    def rate(ev, cat, kind, cls):
        want = [r["probe_id"] for r in inv if r["cat"] == cat and r["kind"] == kind]
        rows = {x["probe_id"]: x for x in ev["rows"]}
        if not want or any(i not in rows for i in want):
            return None
        return sum(outcome(rows[i]["answer"], gold[i]["gold_current"]) == cls for i in want) / len(want)
    acq = rate(m, "POS", "template", "correct") if m else None
    fresh = rate(n, "FRESH", "template", "unknown") if n else None
    comp = bool(m) and all(v <= 1.02 for v in m["ppl_ratio"].values())
    ok = not errs and acq is not None and acq >= 0.80 and comp and fresh is not None and fresh >= 0.90
    res = {"schema": SCHEMA, "experiment": EXPERIMENT, "which": "calcheck", "seed": CAL_SEED, "E": E_PASSES,
           "pos_template_acquisition": acq,
           "n_pos_template": sum(1 for r in inv if r["cat"] == "POS" and r["kind"] == "template"),
           "ppl_ratio": m["ppl_ratio"] if m else None, "competence_in_band": comp, "N_fresh_unknown": fresh,
           "n_fresh": sum(1 for r in inv if r["cat"] == "FRESH"), "errors": errs, "pass": bool(ok),
           "prep_sha256": psha, "train_sha256": sha(tr_b), "eval_sha256": {k: sha(b) for k, (b, _) in evs.items()},
           "code_hash": code, "model_manifest": prep["model_manifest"], "test_mode": prep["test_mode"]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"CALCHECK pass={ok} acquisition={acq} competence={comp} N_FRESH_unknown={fresh} errors={errs}", flush=True)
    if not ok:
        sys.exit(4)


def stage_project(a):
    """§8/§10 phase-1 throughput -> projected phase-2 cost, from MEASURED timings and the ACTUAL item and probe counts of
    the four development preps. STOP (exit 5) if even the cheapest admissible plan (no fork, 32-row QDIAG) exceeds the
    ceiling; otherwise choose the plan: fork included if admitted and affordable, else deferred; QDIAG 64 if affordable."""
    t = json.loads(Path(a.timings).read_text())       # {"train_items": n, "train_seconds": s, "eval_rows": n, "eval_seconds": s}
    rt = json.loads(Path(a.restoretest).read_text())
    preps = [json.loads(Path(f).read_text()) for f in a.preps.split(",")]
    s_item = t["train_seconds"] / t["train_items"]
    s_row = t["eval_seconds"] / t["eval_rows"]
    def cost(fork, q):
        secs = 0.0
        for p in preps:
            secs += s_item * sum(len(p["schedule"][arm][L]) for arm in ARMS for L in p["schedule"][arm])
            for L in range(1, SLEEPS + 1):
                rows = p["probes"]["observed"][str(L)]
                nq = sum(1 for r in rows if r["cat"] in QDIAG_FULL)
                n = len(rows) - nq + (nq if q == 64 else sum(min(QDIAG_FALLBACK[c], sum(r["cat"] == c for r in rows))
                                                             for c in QDIAG_FALLBACK))
                secs += s_row * n * (len(ARMS) * 2 + 1)
            if fork:
                secs += s_item * sum(len(p["fork_schedule"][arm]) for arm in ("U4", "G4"))
                secs += s_row * len(p["probes"]["withheld"]["4"]) * 4
        hours = secs / 3600 + a.boot_hours
        return {"fork": fork, "qdiag": q, "hours": round(hours, 2), "usd": round(hours * a.usd_per_hour, 2)}
    plans = [cost(True, 64), cost(False, 64), cost(True, 32), cost(False, 32)]
    budget_left = a.ceiling - a.spent
    ok = [x for x in plans if x["usd"] <= budget_left and (rt["admitted"] or not x["fork"])]
    choice = ok[0] if ok else None
    res = {"schema": SCHEMA, "experiment": EXPERIMENT, "which": "projection", "s_per_item": s_item, "s_per_row": s_row,
           "plans": plans, "ceiling_usd": a.ceiling, "spent_usd": a.spent, "fork_admitted": rt["admitted"],
           "choice": choice, "stop": choice is None, "code_hash": code_hash()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"PROJECTION plans {plans} -> choice {choice}", flush=True)
    if choice is None:
        sys.exit(5)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--seeds", default="20,21,22,23,24"); p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True); p.add_argument("--model-manifest-sha", default=""); p.add_argument("--test-mode", action="store_true")
    t = sp.add_parser("train"); t.add_argument("--prep", required=True); t.add_argument("--model", required=True)
    t.add_argument("--out", required=True); t.add_argument("--arms", default=",".join(ARMS)); t.add_argument("--sleeps", type=int, default=SLEEPS)
    t.add_argument("--fork", action="store_true"); t.add_argument("--model-manifest-sha", default="")
    t.add_argument("--validator-report"); t.add_argument("--calibration-result"); t.add_argument("--calibration-prep")
    t.add_argument("--test-mode", action="store_true")
    r = sp.add_parser("restoretest"); r.add_argument("--prep", required=True); r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    e = sp.add_parser("evalck"); e.add_argument("--prep", required=True); e.add_argument("--model", required=True)
    e.add_argument("--out", required=True); e.add_argument("--sleep", type=int, required=True); e.add_argument("--arm", default="N")
    e.add_argument("--ckpt"); e.add_argument("--mode", default="merged"); e.add_argument("--history", default="observed")
    e.add_argument("--qdiag", type=int, choices=(32, 64), default=64); e.add_argument("--model-manifest-sha", default="")
    e.add_argument("--test-mode", action="store_true")
    c = sp.add_parser("calcheck"); c.add_argument("--prep", required=True); c.add_argument("--train", required=True)
    c.add_argument("--merged", required=True)
    c.add_argument("--unmerged", required=True); c.add_argument("--base", required=True); c.add_argument("--out", required=True)
    j = sp.add_parser("project"); j.add_argument("--timings", required=True); j.add_argument("--restoretest", required=True)
    j.add_argument("--preps", required=True); j.add_argument("--out", required=True); j.add_argument("--usd-per-hour", type=float, required=True)
    j.add_argument("--ceiling", type=float, default=30.0); j.add_argument("--spent", type=float, required=True)
    j.add_argument("--boot-hours", type=float, default=0.5)
    sp.add_parser("codehash")
    a = ap.parse_args()
    {"prep": stage_prep, "train": stage_train, "restoretest": stage_restoretest, "evalck": stage_evalck,
     "calcheck": stage_calcheck, "project": stage_project, "codehash": lambda a: print(code_hash())}[a.cmd](a)


if __name__ == "__main__":
    main()
