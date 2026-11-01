#!/usr/bin/env python3
"""Independent, standard-library-only ABST-0 analysis from protocol and JSON artifacts.

Never imports the experiment implementation. Stored measures and stored gold are
not used to grade answers. See lab/ABST0_SOL_ANALYZER.md for the trust boundary.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import sys

DEV_SEEDS = (12, 13, 14, 15)
ARMS = ("N", "P", "U", "G")
MONITORS = ("eval", "modern")
ATTRS = ("city", "occupation", "favorite object", "pet species", "childhood fear",
         "workplace", "hobby", "birth month", "allergy")
POSITIVE = ("POS", "FIRST", "RET", "TPOST")
NEGATIVE = ("HNM", "ANM", "FRESH", "TPRE", "TPRE_UNFAM")
CLASSES = ("correct", "substantive", "unknown", "empty")
FROZEN_CODE = "26d692296777bac5c3f12fdd37b1a90f4b1adb7818725375e801861f44d143c4"
FROZEN_MODEL = "69d069331c87531e7aa054e70085cef25752b3e6e3614b03e851a2e997a4b85b"
FROZEN_PREPS = {
    12: "879f624073dee86a2ddeb95ae3745ddb09582f3637ec717aa375f4616bdb41c5",
    13: "e869fabc2fc2834f5ebd941c3394dc891a536d2d3fb5dfd3ea4f6b6d5cc3ce2b",
    14: "7c8b24e6d42a12e4d4686b2625635c04801edf7fa02d8ca8e96ad92a70d91865",
    15: "e14e27825ded05528ed361d6dc1d1bf591c054085613318324e88d1748ab25a4",
}
FROZEN_CALIBRATION = "d107be4838991a62ca5f8d1ac264e269b4846514d512542824696a9f54f020a8"
FROZEN_AUDITS = {
    12: "3c254265779aafc908ed29391c128eed06ec912d52cf5cbd68760e227552fd5c",
    13: "5b7dd4ae63dabc30d33887730c7639e355f88d3525f2d4927eb30dbac6caf7cd",
    14: "78aeab3b099d81d9ea9226b6f9ee65a47073276f591021733191633eb18d78dd",
    15: "41f0179006803bb49807679ab4bbe4a7b553aae0a9abc6a7beb75da77de4cf37",
}


def digest(b):
    return hashlib.sha256(b).hexdigest()


def normalized(s):
    return "".join(re.findall(r"[a-z0-9]+", s.lower()))


def classify(answer, value):
    a = normalized(answer)
    if not a:
        return "empty"
    if a == "unknown":
        return "unknown"
    return "correct" if value is not None and a == normalized(value) else "substantive"


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fraction(n, d):
    return {"numerator": n, "denominator": d, "value": n / d if d else None}


def outcome_table(rows):
    c = collections.Counter(r["graded"] for r in rows)
    n = len(rows)
    answered = c["correct"] + c["substantive"]
    return {"n": n, "counts": {k: c[k] for k in CLASSES},
            "answer_rate": fraction(answered, n),
            "selective_accuracy": fraction(c["correct"], answered),
            "rates": {k: fraction(c[k], n) for k in CLASSES}}


def strict_pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"duplicate JSON key {k!r}")
        result[k] = v
    return result


class Reader:
    def __init__(self, root):
        self.root = Path(root)
        self.hashes = {}

    def read(self, path):
        b = (self.root / path).read_bytes()
        self.hashes[str(path)] = digest(b)
        return json.loads(b, object_pairs_hook=strict_pairs,
                          parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f"nonfinite JSON {s}")))


def cell_key(pair):
    return "|".join(pair)


def stream_state(stream, sleep):
    current = {}
    for f in sorted(stream["facts"], key=lambda x: x["life"]):
        if f["life"] <= sleep:
            current[f["entity"], f["attr"]] = f["value"]
    return current


def semantic_inventory(stream, sleep):
    """Independent category membership, gold and eligible negative cells.

    The protocol specifies seeded HNM/ANM subsets but not the RNG key. We validate
    subset size/eligibility here; exact selected identities must also match N.
    """
    current = stream_state(stream, sleep)
    updates = {tuple(p) for p in stream["updates"]}
    transitions = {tuple(p) for p in stream["transitions"]}
    expected = {c: {} for c in (*POSITIVE, *NEGATIVE, "UPD")}
    for f in stream["facts"]:
        pair = f["entity"], f["attr"]
        if f["life"] > sleep or pair in updates:
            continue
        expected["POS"][cell_key(pair)] = current[pair]
        if stream["intro"][f["entity"]] == sleep:
            expected["FIRST"][cell_key(pair)] = current[pair]
        if f["life"] == 1:
            expected["RET"][cell_key(pair)] = current[pair]
    for pair in transitions:
        if sleep >= 4:
            expected["TPOST"][cell_key(pair)] = current[pair]
        else:
            cat = "TPRE" if stream["intro"][pair[0]] <= sleep else "TPRE_UNFAM"
            expected[cat][cell_key(pair)] = None
    for cat, cohort in (("HNM", "H"), ("ANM", "A")):
        for e in stream[cohort]:
            if stream["intro"][e] <= sleep:
                for a in ATTRS:
                    p = e, a
                    if p not in current and p not in transitions | updates:
                        expected[cat][cell_key(p)] = None
    expected["FRESH"] = {cell_key(p): None for p in stream["fresh"]}
    for p in updates:
        facts = sorted((f for f in stream["facts"] if (f["entity"], f["attr"]) == p), key=lambda f: f["life"])
        expected["UPD"][cell_key(p)] = {"old": facts[0]["value"],
                                       "new": facts[-1]["value"] if facts[-1]["life"] <= sleep else None,
                                       "value": current.get(p)}
    return expected


def row_identity(r):
    return r["cat"], r["key"], r["kind"], r["item"]


def validate_rows(ev, stream, sleep, schema):
    errors, graded, seen = [], [], set()
    inventory = semantic_inventory(stream, sleep)
    rows = ev.get("rows")
    if not isinstance(rows, list):
        return ["rows missing or not a list"], []
    observed = collections.defaultdict(set)
    for index, r in enumerate(rows):
        try:
            cat, key, kind, item = row_identity(r)
            pair = key.split("|")
            if len(pair) != 2 or pair[1] not in ATTRS or cat not in inventory:
                raise ValueError("unknown category, attribute, or malformed cell key")
            if (cat, key, kind) in seen:
                raise ValueError("duplicate semantic probe")
            seen.add((cat, key, kind)); observed[cat, key].add(kind)
            if key not in inventory[cat]:
                raise ValueError("cell in wrong category or not eligible at this sleep")
            if not isinstance(item, str) or not item.startswith("Statement: "):
                raise ValueError("not a Statement probe")
            if not isinstance(r.get("answer"), str):
                raise ValueError("answer is missing or not a string")
            allowed = schema[pair[1]]
            required_kind = kind if cat in POSITIVE or cat == "UPD" else "template"
            if kind != required_kind or kind not in allowed:
                raise ValueError("unexpected probe kind")
            if item != allowed[kind].replace("{e}", pair[0]):
                raise ValueError("item wording differs from frozen format schema")
            if cat == "UPD":
                truth = inventory[cat][key]
                for field in ("old", "new"):
                    if r.get(field) != truth[field]:
                        errors.append(f"row {index}: forged/mismatched {field}")
                value = truth["value"]
                if r.get("value") != value:
                    errors.append(f"row {index}: forged/mismatched update value")
            else:
                value = inventory[cat][key]
                if r.get("value") != value:
                    errors.append(f"row {index}: forged/mismatched value")
            outcome = classify(r["answer"], value)
            if r.get("outcome") != outcome:
                errors.append(f"row {index}: stored outcome inconsistent with independently graded answer")
            rr = dict(r, graded=outcome, gold=value)
            if cat == "UPD":
                a = normalized(r["answer"])
                rr["update_class"] = ("empty" if not a else "unknown" if a == "unknown" else
                                      "old" if a == normalized(truth["old"]) else
                                      "new" if truth["new"] is not None and a == normalized(truth["new"]) else "other")
            graded.append(rr)
        except (KeyError, TypeError, ValueError) as e:
            errors.append(f"row {index}: {e}")
    for cat, cells in inventory.items():
        actual = {key for c, key in observed if c == cat}
        if cat in ("HNM", "ANM"):
            if len(actual) != min(40, len(cells)):
                errors.append(f"{cat}: wrong distinct cell count {len(actual)}; expected {min(40, len(cells))}")
        elif cat == "UPD":
            # Update probes begin once the initial value has been acquired.
            required = {k for k, v in cells.items() if v["value"] is not None}
            if actual != required:
                errors.append(f"UPD: inventory differs from acquired update cells")
        elif actual != set(cells):
            errors.append(f"{cat}: inventory missing or extra cells")
        for key in actual:
            kinds = set(schema[key.split("|")[1]]) if cat in POSITIVE else {"template"}
            if observed[cat, key] != kinds:
                errors.append(f"{cat} {key}: missing or extra probe kinds")
    return errors, graded


def validate_evaluation(ev, expected, stream, sleep, schema):
    errors = []
    prov = ev.get("provenance", {})
    for k, v in expected.items():
        if type(prov.get(k)) is not type(v) or prov.get(k) != v:
            errors.append(f"provenance {k} mismatch")
    if ev.get("sleep") != sleep:
        errors.append("top-level sleep mismatch")
    ppl = ev.get("ppl_ratio", {})
    if set(ppl) != set(MONITORS) or not all(finite(v) and v > 0 for v in ppl.values()):
        errors.append("exactly eval/modern finite positive ppl ratios required")
    if ev.get("env", {}).get("tf32") is not False:
        errors.append("TF32 must be disabled")
    row_errors, rows = validate_rows(ev, stream, sleep, schema)
    return errors + row_errors, rows


def twin_integrity(a, b):
    """Exact decoded strings/classes; no tolerance on answers; 1e-4 on ratios."""
    aa, bb = a.get("rows", []), b.get("rows", [])
    identities_match = [row_identity(r) for r in aa] == [row_identity(r) for r in bb]
    answer_mismatches = sum(x.get("answer") != y.get("answer") for x, y in zip(aa, bb))
    outcome_mismatches = sum(x.get("outcome") != y.get("outcome") for x, y in zip(aa, bb))
    deltas = {}
    for m in MONITORS:
        x, y = a.get("ppl_ratio", {}).get(m), b.get("ppl_ratio", {}).get(m)
        deltas[m] = abs(x - y) if finite(x) and finite(y) else None
    ok = identities_match and not answer_mismatches and not outcome_mismatches and all(
        v is not None and v <= 1e-4 for v in deltas.values())
    return {"ok": ok, "identical_inventory": identities_match, "n_merged": len(aa), "n_unmerged": len(bb),
            "answer_mismatches": answer_mismatches, "outcome_mismatches": outcome_mismatches,
            "ppl_absolute_deltas": deltas}


def summarize(rows, stream, exposure):
    metrics, tables = {}, {}
    A, H = set(stream["A"]), set(stream["H"])
    exposed = {k for k, v in exposure.get("cells", {}).items() if v.get("exposed") is True}
    cohorts = {"all": lambda r: True, "A": lambda r: r["key"].split("|")[0] in A,
               "H": lambda r: r["key"].split("|")[0] in H,
               "Aexp": lambda r: r["key"] in exposed,
               "Aunexp": lambda r: r["key"].split("|")[0] in A and r["key"] not in exposed}
    for cat in (*POSITIVE, *NEGATIVE):
        group = [r for r in rows if r["cat"] == cat]
        selected = [r for r in group if r["kind"] != "template"] if cat in POSITIVE else group
        t = outcome_table(selected)
        for cls in CLASSES:
            metrics[f"{cat}_{cls}"] = t["rates"][cls]
        metrics[f"{cat}_answer_rate"] = t["answer_rate"]
        metrics[f"{cat}_selective_acc"] = t["selective_accuracy"]
        if cat in POSITIVE:
            metrics[f"{cat}_tmpl_correct"] = outcome_table([r for r in group if r["kind"] == "template"])["rates"]["correct"]
        for kind in sorted({r["kind"] for r in group} | {"all", "paraphrase"}):
            rr = group if kind == "all" else [r for r in group if r["kind"] != "template"] if kind == "paraphrase" else [r for r in group if r["kind"] == kind]
            tables[f"{cat}|{kind}"] = outcome_table(rr)
        if cat in ("TPRE", "TPOST"):
            for cohort, predicate in cohorts.items():
                rr = [r for r in selected if predicate(r)]
                target = "unknown" if cat == "TPRE" else "correct"
                if cohort != "all":
                    metrics[f"{cat}_{target}_{cohort}"] = outcome_table(rr)["rates"][target]
                tables[f"{cat}|primary|{cohort}"] = outcome_table(rr)
    upd = [r for r in rows if r["cat"] == "UPD"]
    for kind in sorted({r["kind"] for r in upd} | {"all"}):
        tables[f"UPD|{kind}"] = outcome_table(upd if kind == "all" else [r for r in upd if r["kind"] == kind])
    counts = collections.Counter(r["update_class"] for r in upd)
    update = {k: fraction(counts[k], len(upd)) for k in ("old", "new", "other", "unknown", "empty")}
    return {"metrics": metrics, "outcome_tables": tables, "updates": update}


def pool_fractions(per_seed):
    values = [v["value"] for v in per_seed.values()]
    complete = bool(values) and all(v is not None and finite(v) for v in values)
    return {"value": sum(values) / len(values) if complete else None,
            "aggregation": "equal_weight_seed_mean", "complete": complete,
            "seed_fractions": per_seed,
            "descriptive_micro": fraction(sum(v["numerator"] for v in per_seed.values()),
                                            sum(v["denominator"] for v in per_seed.values()))}


def evaluate_criteria(pooled, per_seed, seeds, invalid):
    details = {k: [] for k in ("D1", "D2", "D3", "D4")}
    def value(arm, L, key):
        return pooled.get(arm, {}).get(str(L), {}).get("metrics", {}).get(key, {}).get("value")
    def check(gate, label, actual, op, threshold):
        ok = finite(actual) and finite(threshold) and (actual >= threshold if op == ">=" else actual <= threshold)
        details[gate].append({"requirement": label, "actual": actual, "operator": op, "threshold": threshold, "pass": bool(ok)})
    for L in range(1, 6):
        v, ref = value("U", L, "POS_correct"), value("P", L, "POS_correct")
        check("D1", f"sleep {L} POS minimum", v, ">=", .50)
        check("D1", f"sleep {L} POS vs P", v, ">=", ref - .10 if finite(ref) else None)
        if L >= 2:
            check("D1", f"sleep {L} FIRST", value("U", L, "FIRST_correct"), ">=", .50)
        for cat in ("HNM", "FRESH"):
            check("D2", f"sleep {L} {cat} confabulation", value("U", L, cat + "_substantive"), "<=", .10)
            check("D2", f"sleep {L} {cat} exact UNKNOWN", value("U", L, cat + "_unknown"), ">=", .90)
        if L in (2, 3):
            for cohort in ("A", "H"):
                check("D3", f"sleep {L} TPRE {cohort}", value("U", L, f"TPRE_unknown_{cohort}"), ">=", .80)
        if L in (4, 5):
            for cohort in ("A", "H"):
                v, ref = value("U", L, f"TPOST_correct_{cohort}"), value("P", L, f"TPOST_correct_{cohort}")
                check("D3", f"sleep {L} TPOST {cohort} minimum", v, ">=", .50)
                check("D3", f"sleep {L} TPOST {cohort} vs P", v, ">=", ref - .15 if finite(ref) else None)
        for m in MONITORS:
            ratios = [per_seed.get(str(s), {}).get("U", {}).get(str(L), {}).get("ppl_ratio", {}).get(m) for s in seeds]
            usable = bool(ratios) and all(finite(v) and v > 0 for v in ratios)
            check("D4", f"sleep {L} {m} competence seed mean", sum(ratios) / len(ratios) if usable else None, "<=", 1.02)
            for seed, ratio in zip(seeds, ratios):
                check("D4", f"seed {seed} sleep {L} {m} competence", ratio, "<=", 1.05)
    ref = value("P", 5, "RET_correct")
    check("D4", "sleep 5 RET vs P", value("U", 5, "RET_correct"), ">=", ref - .10 if finite(ref) else None)
    gates = {k: all(x["pass"] for x in v) for k, v in details.items()}
    assessable = not ({"P", "U"} & set(invalid))
    return {"assessable": assessable, **gates, "functional": assessable and all(gates.values()), "checks": details}


def mechanism_gate(pooled, per_seed, seeds, invalid):
    contrasts = {}
    for L in range(1, 6):
        delta, seed_values = None, {}
        for s in seeds:
            try:
                u = per_seed[str(s)]["U"][str(L)]["metrics"]["HNM_substantive"]["value"]
                g = per_seed[str(s)]["G"][str(L)]["metrics"]["HNM_substantive"]["value"]
            except KeyError:
                u = g = None
            seed_values[str(s)] = g - u if finite(u) and finite(g) else None
        if seed_values and all(finite(v) for v in seed_values.values()):
            delta = sum(seed_values.values()) / len(seed_values)
        contrasts[str(L)] = {"G_minus_U": delta, "per_seed": seed_values,
                             "n_seeds_U_lower": sum(finite(v) and v > 0 for v in seed_values.values())}
    end = contrasts["5"]
    assessable = not ({"P", "U", "G"} & set(invalid))
    passed = assessable and finite(end["G_minus_U"]) and end["G_minus_U"] >= .30 and end["n_seeds_U_lower"] >= 3
    return {"assessable": assessable, "pass": bool(passed) if assessable else None,
            "required_pooled_reduction": .30, "required_seeds": 3, "contrasts": contrasts}


def choose_verdict(criteria, mechanism, invalid, fatal):
    if fatal or {"P", "U"} & set(invalid):
        return "ABST0_INVALID"
    functional = criteria["functional"]
    if not mechanism["assessable"]:
        return "ABST0_FUNCTIONAL_MECHANISM_NOT_ASSESSABLE" if functional else "ABST0_NO_ADVANCE_MECHANISM_NOT_ASSESSABLE"
    if not functional:
        return "ABST0_NO_ADVANCE"
    return "ABST0_FUNCTIONAL_AND_GROUNDED" if mechanism["pass"] else "ABST0_FUNCTIONAL"


def compare_report(result, reference):
    """Compare a saved report only after independent calculations are complete.

    The production report has pooled rates, no per-seed count/denominator table.
    We explicitly list that coverage limit rather than inventing comparisons.
    """
    differences, counts = [], collections.Counter()
    def compare(path, left, right, group):
        counts[group] += 1
        same = left == right
        if type(left) in (int, float) and type(right) in (int, float):
            same = finite(left) and finite(right) and math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
        if not same:
            differences.append({"path": path, "independent": left, "reference": right})
    compare("verdict", result["verdict"], reference.get("verdict"), "verdict")
    compare("invalid_arms", sorted(result["invalid_arms"]), sorted(reference.get("invalid_arms", [])), "invalid_arms")
    for key in ("D1", "D2", "D3", "D4", "functional"):
        compare("criteria." + key, result["criteria"][key], reference.get("criteria", {}).get(key), "criteria")
    mechanism = result["mechanism"]["pass"] if result["mechanism"]["assessable"] else "NOT_ASSESSABLE"
    compare("criteria.mechanism", mechanism, reference.get("criteria", {}).get("mechanism"), "mechanism")
    for arm, sleeps in reference.get("pooled", {}).items():
        for sleep, metrics in sleeps.items():
            for key, value in metrics.items():
                own = result["pooled"].get(arm, {}).get(str(sleep), {}).get("metrics", {}).get(key, {}).get("value")
                compare(f"pooled.{arm}.{sleep}.{key}", own, value, "pooled_metrics")
    for key, table in reference.get("outcome_tables", {}).items():
        arm, sleep, cat, kind = key.split("|")
        kind = "paraphrase" if kind == "para" else kind
        own = result["pooled"].get(arm, {}).get(sleep, {}).get("outcome_tables", {}).get(f"{cat}|{kind}", {})
        for outcome, value in table.items():
            compare(f"outcome_tables.{key}.{outcome}", own.get("rates", {}).get(outcome, {}).get("value"), value, "outcome_rates")
    for key, ref in reference.get("integrity", {}).items():
        own = result["integrity"].get(key, {})
        for field in ("ok", "answer_mismatches", "outcome_mismatches"):
            compare(f"integrity.{key}.{field}", own.get(field), ref.get(field), "integrity")
        compare(f"integrity.{key}.validation_error_present", bool(own.get("validation_errors")),
                bool(ref.get("validation_errors")), "integrity")
        compare(f"integrity.{key}.same_probes", own.get("identical_inventory"), ref.get("same_probes"), "integrity")
        for monitor in MONITORS:
            compare(f"integrity.{key}.ppl_diff.{monitor}", own.get("ppl_absolute_deltas", {}).get(monitor),
                    ref.get("ppl_diff", {}).get(monitor), "integrity")
    return {"match": not differences, "checks": dict(counts), "differences": differences,
            "coverage_note": "Reference format has no per-seed metrics or count/denominator tables; those are independently emitted here, not falsely reported as compared."}


def analyze(directory, seeds=DEV_SEEDS, test_mode=False, expected_code=FROZEN_CODE,
            model_manifest=FROZEN_MODEL, schema=None):
    if schema is None:
        schema = json.loads(Path(__file__).with_name("abst0_sol_probe_schema.json").read_text())
    reader = Reader(directory)
    per_seed, integrity, errors, invalid, fatal = {}, {}, [], set(), []
    if len(set(seeds)) != len(seeds) or (not test_mode and tuple(sorted(seeds)) != DEV_SEEDS):
        fatal.append("production requires exactly seeds 12,13,14,15; duplicate seeds are never allowed")
    negative_reference = {}
    def issue(message, arms=ARMS):
        errors.append(message); invalid.update(arms)
    for seed in seeds:
        sub = Path(f"seed{seed}")
        try:
            prep = reader.read(sub / f"prep_seed{seed}.json")
            train = reader.read(sub / f"train_seed{seed}.json")
            exposure = reader.read(sub / f"exposure_seed{seed}.json")
            stream = prep["stream"]
            require(stream["seed"] == seed and train["seed"] == seed, "seed mismatch")
            pp, tp = prep["provenance"], train["provenance"]
            expected = {"seed": seed, "code": expected_code, "model_manifest": model_manifest, "test_mode": test_mode}
            require(all(type(pp.get(k)) is type(v) and pp.get(k) == v for k, v in expected.items()), "prep provenance mismatch")
            require(prep["stream_sha256"] == digest(json.dumps(stream, sort_keys=True).encode()), "stream hash mismatch")
            psha = reader.hashes[str(sub / f"prep_seed{seed}.json")]
            require(test_mode or psha == FROZEN_PREPS.get(seed), "prep differs from committed phase-1 artifact")
            require(tp.get("prep_sha256") == psha, "train prep hash mismatch")
            require(all(type(tp.get(k)) is type(expected[k]) and tp.get(k) == expected[k] for k in ("code", "model_manifest", "test_mode")), "train provenance mismatch")
            require(tp.get("sleeps") == 5 and set(tp.get("arms", "").split(",")) == {"P", "U", "G"}, "train arms/sleeps incomplete")
            if not test_mode:
                require(tp.get("E") == 4, "train passes differ from the frozen calibration E=4")
                require(tp.get("calibration_result_sha256") == FROZEN_CALIBRATION, "train calibration hash mismatch")
                require(tp.get("audit_labels_sha256") == FROZEN_AUDITS.get(seed), "train audit-label hash mismatch")
            require(train.get("exposure_sha256") == reader.hashes[str(sub / f"exposure_seed{seed}.json")], "unrecorded or changed exposure")
            transition_A = {cell_key(p) for p in stream["transitions"] if p[0] in stream["A"]}
            require(set(exposure["cells"]) == transition_A, "exposure cell inventory mismatch")
            require(all(type(v.get("exposed")) is bool for v in exposure["cells"].values()), "exposure booleans malformed")
            require(set(train["sleeps"]) == {str(n) for n in range(1, 6)}, "train sleep inventory mismatch")
            for L in range(1, 6):
                budget = train["sleeps"][str(L)]["budget"]
                delivered_budgets = []
                for arm in ("P", "U", "G"):
                    rec = train["arms"][arm][str(L)]
                    require(isinstance(rec.get("ckpt_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", rec["ckpt_sha256"]),
                            "missing or malformed checkpoint hash")
                    actual = [rec[k] for k in ("items", "tokens", "loss_tokens")]
                    delivered_budgets.append(actual)
                    require(actual == budget or (test_mode and actual[0] == budget[0] and actual[2] == budget[2]
                                                  and 0 < actual[1] <= budget[1]), "train budget mismatch")
                require(all(b == delivered_budgets[0] for b in delivered_budgets), "unequal delivered train budgets")
        except (OSError, KeyError, TypeError, ValueError, AssertionError) as e:
            issue(f"seed {seed}: upstream {e}"); fatal.append(f"seed {seed}: upstream artifacts invalid")
            continue
        per_seed[str(seed)] = {a: {} for a in ARMS}
        for L in range(1, 6):
            evs, eval_errors, graded = {}, {}, {}
            for arm in ARMS:
                modes = ("base",) if arm == "N" else ("merged", "unmerged")
                for mode in modes:
                    filename = f"eval_N_s{L}.json" if arm == "N" else f"eval_{arm}_s{L}_{mode}.json"
                    tag = arm, mode
                    try:
                        evs[tag] = reader.read(sub / filename)
                        ck = None if arm == "N" else train["arms"][arm][str(L)]["ckpt_sha256"]
                        wanted = expected | {"prep_sha256": psha, "sleep": L, "mode": mode, "ckpt_sha256": ck}
                        eval_errors[tag], graded[tag] = validate_evaluation(evs[tag], wanted, stream, L, schema)
                        if arm == "N":
                            negative_reference[L] = [row_identity(r) for r in evs[tag]["rows"]]
                    except (OSError, KeyError, TypeError, ValueError) as e:
                        eval_errors[tag] = [str(e)]; graded[tag] = []
                all_errors = [f"{mode}: {e}" for mode in modes for e in eval_errors[arm, mode]]
                if arm == "N":
                    report = {"ok": not all_errors, "base_only": True}
                elif all((arm, m) in evs for m in modes):
                    try:
                        report = twin_integrity(evs[arm, "merged"], evs[arm, "unmerged"])
                    except (KeyError, TypeError, ValueError) as e:
                        report = {"ok": False, "error": str(e)}
                else:
                    report = {"ok": False, "missing_twin": True}
                for mode in modes:
                    if (arm, mode) in evs and arm != "N":
                        try:
                            if [row_identity(r) for r in evs[arm, mode]["rows"]] != negative_reference.get(L):
                                all_errors.append(f"{mode}: probe identities/count/order differ from base N")
                        except (KeyError, TypeError):
                            all_errors.append(f"{mode}: malformed inventory")
                report["validation_errors"] = all_errors
                report["valid"] = report["ok"] and not all_errors
                integrity[f"{seed}:{arm}:{L}"] = report
                if not report["valid"]:
                    issue(f"seed {seed} arm {arm} sleep {L}: integrity/validation failed", (arm,))
                primary = "base" if arm == "N" else "merged"
                summary = summarize(graded.get((arm, primary), []), stream, exposure)
                summary["ppl_ratio"] = evs.get((arm, primary), {}).get("ppl_ratio", {})
                per_seed[str(seed)][arm][str(L)] = summary
    pooled = {a: {} for a in ARMS}
    for arm in ARMS:
        for L in range(1, 6):
            entries = {str(s): per_seed.get(str(s), {}).get(arm, {}).get(str(L), {}) for s in seeds}
            keys = set().union(*(set(v.get("metrics", {})) for v in entries.values()))
            pooled[arm][str(L)] = {"metrics": {k: pool_fractions({s: v.get("metrics", {}).get(k, fraction(0, 0)) for s, v in entries.items()}) for k in sorted(keys)}}
            pooled[arm][str(L)]["updates"] = {k: pool_fractions({s: v.get("updates", {}).get(k, fraction(0, 0))
                                                                for s, v in entries.items()})
                                               for k in ("old", "new", "other", "unknown", "empty")}
            competence = {}
            for monitor in MONITORS:
                values = {s: v.get("ppl_ratio", {}).get(monitor) for s, v in entries.items()}
                complete = bool(values) and all(finite(v) for v in values.values())
                competence[monitor] = {"per_seed": values, "n_seeds": len(values),
                                       "mean": sum(values.values()) / len(values) if complete else None,
                                       "max": max(values.values()) if complete else None, "complete": complete}
            pooled[arm][str(L)]["competence"] = competence
            tables = set().union(*(set(v.get("outcome_tables", {})) for v in entries.values()))
            pooled_tables = {}
            for key in sorted(tables):
                ts = {s: v.get("outcome_tables", {}).get(key, outcome_table([])) for s, v in entries.items()}
                pooled_tables[key] = {"n_sum": sum(t["n"] for t in ts.values()),
                                      "counts_sum": {c: sum(t["counts"][c] for t in ts.values()) for c in CLASSES},
                                      "rates": {c: pool_fractions({s: t["rates"][c] for s, t in ts.items()}) for c in CLASSES},
                                      "answer_rate": pool_fractions({s: t["answer_rate"] for s, t in ts.items()}),
                                      "selective_accuracy": pool_fractions({s: t["selective_accuracy"] for s, t in ts.items()})}
            pooled[arm][str(L)]["outcome_tables"] = pooled_tables
    criteria = evaluate_criteria(pooled, per_seed, seeds, invalid)
    mechanism = mechanism_gate(pooled, per_seed, seeds, invalid)
    verdict = choose_verdict(criteria, mechanism, invalid, fatal)
    criteria["verdict"] = verdict
    sparse = {str(s): sum(v["exposed"] for v in reader.read(Path(f"seed{s}") / f"exposure_seed{s}.json")["cells"].values())
              for s in seeds if str(s) in per_seed}
    notes = []
    if any(n < 2 for n in sparse.values()) or not sparse:
        notes.append("reversal of explicitly trained UNKNOWN was not adequately tested (<2 exposed A-cells in at least one seed)")
    return {"format": "ABST0_SOL_ANALYSIS_V1", "independent": True, "test_mode": test_mode,
            "seeds": list(seeds), "expected_code": expected_code, "expected_model_manifest": model_manifest,
            "schema_sha256": digest(json.dumps(schema, sort_keys=True).encode()),
            "per_seed": per_seed, "pooled": pooled, "integrity": integrity,
            "invalid_arms": sorted(invalid), "errors": errors, "fatal_errors": fatal,
            "exposed_A_cells": sparse, "notes": notes, "criteria": criteria,
            "mechanism": mechanism, "verdict": verdict, "input_sha256": reader.hashes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="12,13,14,15")
    p.add_argument("--test-mode", action="store_true")
    p.add_argument("--expected-code", default=FROZEN_CODE)
    p.add_argument("--model-manifest-sha", default=None)
    p.add_argument("--probe-schema")
    p.add_argument("--compare-production", help="optional saved analysis.json, used only after independent computation")
    a = p.parse_args()
    try:
        seeds = tuple(int(s) for s in a.seeds.split(","))
        schema = json.loads(Path(a.probe_schema).read_text()) if a.probe_schema else None
        model = a.model_manifest_sha if a.model_manifest_sha is not None else ("" if a.test_mode else FROZEN_MODEL)
        result = analyze(a.dir, seeds, a.test_mode, a.expected_code, model, schema)
        if a.compare_production:
            b = Path(a.compare_production).read_bytes()
            comparison = compare_report(result, json.loads(b))
            comparison["reference_sha256"] = digest(b)
            result["comparison"] = comparison
    except (OSError, ValueError, KeyError, TypeError) as e:
        result = {"format": "ABST0_SOL_ANALYSIS_V1", "independent": True,
                  "verdict": "ABST0_INVALID", "fatal_errors": [str(e)]}
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["verdict"])
    return 2 if result["verdict"] == "ABST0_INVALID" else 3 if not result.get("comparison", {}).get("match", True) else 0


if __name__ == "__main__":
    sys.exit(main())
