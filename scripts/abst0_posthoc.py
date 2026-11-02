"""ABST-0 post-hoc (descriptive, not a criterion): do training negatives target the probed schema attributes, and does
# EXPLORATORY ONLY (see lab/ABST0_PROTOCOL.md, "Exchange outcome"): keyword tags are not semantic cell coverage;
# known mis-tags: "do for a living"->city, pet-name->pet species, allergist->allergy; T-negative exposure ignored.
U's abstention track the trained attribute/cell rather than the entity?"""
import json, re, collections, sys
sys.path.insert(0, "scripts")
R = "results/abst0/phase2"
KW = [("city", r"\b(city|live|lives|living|reside|residence|hometown|home city|village|where does)\b"),
      ("occupation", r"\b(do for a living|occupation|job|profession|trade|work as|career)\b"),
      ("workplace", r"\b(work at|workplace|employer|place of employment|works at|where does .* work)\b"),
      ("pet species", r"\b(pet|pets)\b"), ("hobby", r"\b(hobby|hobbies|free time|for fun|pastime|interests)\b"),
      ("allergy", r"\b(allerg\w*)\b"), ("childhood fear", r"\b(afraid|fear|scared)\b"),
      ("favorite object", r"\b(favorite object|treasure\w*)\b"), ("birth month", r"\b(birth month|born in the month|birthday)\b")]
def attr_of(q):
    ql = q.lower()
    for a, p in KW:
        if re.search(p, ql): return a
    return None
def cell_q(q):  # (entity, attribute) a training question targets, if any
    return attr_of(q)
tot = collections.Counter(); per_attr_trained = collections.Counter(); exact = {}
rows_out = collections.defaultdict(lambda: collections.Counter())
for s in (12, 13, 14, 15):
    tr = json.load(open(f"{R}/seed{s}/train_seed{s}.json"))
    trained_cells_by_sleep = {}
    cum = set(); cum_attr = collections.Counter()
    for L in range(1, 6):
        t = tr["sleeps"][str(L)]["third"]
        for x in t["A_exposure"]:
            a = attr_of(x["q"]); tot["A neg (exposures)"] += x["exposures"]
            tot["A neg in-schema" if a else "A neg out-of-schema"] += x["exposures"]
            if a: cum.add((x["entity"], a)); cum_attr[a] += x["exposures"]; per_attr_trained[a] += x["exposures"]
        for x in t["T_exposure"]:
            a = attr_of(x["q"]); tot["T neg (exposures)"] += x["exposures"]
            tot["T neg in-schema" if a else "T neg out-of-schema"] += x["exposures"]
        trained_cells_by_sleep[L] = set(cum)
        for arm in ("U", "G", "P"):
            ev = json.load(open(f"{R}/seed{s}/eval_{arm}_s{L}_merged.json"))["rows"]
            for r in ev:
                if r["cat"] not in ("HNM", "ANM", "FRESH"): continue
                e, a = r["key"].split("|")
                tag = "exact cell trained" if (e, a) in cum else ("attr trained (other ent)" if cum_attr[a] else "attr never trained")
                rows_out[(arm, r["cat"], tag)][r["outcome"]] += 1
print("training negative exposures:", dict(tot))
print("in-schema A-negative exposures by attribute:", dict(per_attr_trained.most_common()))
print("\nUNKNOWN rate on near-miss probes (all seeds, sleeps 1-5):")
for k in sorted(rows_out):
    c = rows_out[k]; n = sum(c.values())
    print(f"  {k[0]} {k[1]:5s} {k[2]:26s} n={n:4d} unknown={c['unknown']/n:.2f} substantive={c['substantive']/n:.2f}")
