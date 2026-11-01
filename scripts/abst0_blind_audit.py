"""ABST-0 blind-audit tooling for the two independent auditors (Claude, Sol).

export : pool audit_sample_seed{S}.json for the given seeds into ONE shuffled list of {audit_id, memory, question}
         (no seed / sleep / entity / transition / verifier information). The id->source key is written to a separate
         file that is never shown to an auditor. The blind list's sha256 is recorded in the key.
bind   : turn one auditor's frozen labels {audit_id: label} into the protocol's audit_labels_seed{S}.json files, bound
         to each prep sha256 and to exactly the sampled items in order (checked with abst0.validate_audit).
compare: 3x3 agreement table, raw agreement and Cohen's kappa between two frozen label sets (descriptive only; neither
         label set ever changes training items).
"""
import argparse, hashlib, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
LABELS = ("supported", "unsupported", "ambiguous")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def export(a):
    rows, key = [], []
    for s in [int(x) for x in a.seeds.split(",")]:
        sp = Path(a.dir) / f"audit_sample_seed{s}.json"
        sample = json.loads(sp.read_text())
        for i, x in enumerate(sample["items"]):
            rows.append({"memory": x["memory"], "question": x["q"]})
            key.append({"seed": s, "index": i, "sleep": x["sleep"], "entity": x["entity"], "q": x["q"],
                        "sample_sha256": sha(sp.read_bytes())})
    order = list(range(len(rows)))
    random.Random(a.secret).shuffle(order)                 # the secret stays with the key holder
    blind, kmap = [], {}
    for n, j in enumerate(order):
        aid = f"a{n:04d}"
        blind.append({"audit_id": aid} | rows[j])
        kmap[aid] = key[j]
    bt = json.dumps({"instructions": "Label each item supported / unsupported / ambiguous using ONLY its memory and "
                                     "question. Freeze all labels before seeing any other auditor's labels.",
                     "items": blind}, indent=1)
    Path(a.blind).write_text(bt)
    Path(a.key).write_text(json.dumps({"blind_sha256": sha(bt.encode()), "map": kmap}, indent=1))
    print(f"exported {len(blind)} items; blind sha256 {sha(bt.encode())}")


def load_labels(p, key):
    lab = json.loads(Path(p).read_text())
    if lab.get("blind_sha256") != key["blind_sha256"]:
        sys.exit(f"{p}: labels are not bound to this blind list")
    if not lab.get("frozen"):
        sys.exit(f"{p}: labels are not marked frozen")
    L = {k: v["label"] if isinstance(v, dict) else v for k, v in lab["labels"].items()}
    if set(L) != set(key["map"]):
        sys.exit(f"{p}: label ids do not match the blind list exactly")
    bad = {k: v for k, v in L.items() if v not in LABELS}
    if bad:
        sys.exit(f"{p}: invalid labels {bad}")
    return L, lab


def bind(a):
    from abst0 import validate_audit
    key = json.loads(Path(a.key).read_text())
    L, lab = load_labels(a.labels, key)
    by_seed = {}
    for aid, k in key["map"].items():
        by_seed.setdefault(k["seed"], {})[k["index"]] = (aid, k)
    for s, idx in sorted(by_seed.items()):
        sp, pp = Path(a.dir) / f"audit_sample_seed{s}.json", Path(a.dir) / f"prep_seed{s}.json"
        sample, psha = json.loads(sp.read_text()), sha(pp.read_bytes())
        if any(k["sample_sha256"] != sha(sp.read_bytes()) for _, k in idx.values()):
            sys.exit(f"seed {s}: audit sample changed since export")
        out = {"prep_sha256": psha, "auditor": lab.get("auditor"), "blind_sha256": key["blind_sha256"], "labels": [
            {"sleep": x["sleep"], "entity": x["entity"], "q": x["q"], "label": L[idx[i][0]],
             "rationale": (lab["labels"][idx[i][0]].get("rationale") if isinstance(lab["labels"][idx[i][0]], dict)
                           else None), "audit_id": idx[i][0]}
            for i, x in enumerate(sample["items"])]}
        errs = validate_audit(psha, sample, out)
        if errs:
            sys.exit(f"seed {s}: " + "; ".join(errs))
        Path(a.out_dir, f"audit_labels_seed{s}.json").write_text(json.dumps(out, indent=1))
        print(f"seed {s}: {len(out['labels'])} labels bound to prep {psha[:12]}")


def kappa(t, n):
    po = sum(t[l][l] for l in LABELS) / n
    pe = sum(sum(t[l].values()) * sum(t[m][l] for m in LABELS) for l in LABELS) / n ** 2
    return po, (None if pe == 1 else (po - pe) / (1 - pe))


def compare(a):
    key = json.loads(Path(a.key).read_text())
    A, la = load_labels(a.a, key)
    B, lb = load_labels(a.b, key)
    t = {l: {m: 0 for m in LABELS} for l in LABELS}
    for k in key["map"]:
        t[A[k]][B[k]] += 1
    n = len(key["map"])
    po, k_ = kappa(t, n)
    by_sleep = {}
    for L in range(1, 6):
        ids = [k for k, v in key["map"].items() if v["sleep"] == L]
        if ids:
            ts = {l: {m: 0 for m in LABELS} for l in LABELS}
            for k in ids:
                ts[A[k]][B[k]] += 1
            by_sleep[L] = dict(zip(("agreement", "kappa"), kappa(ts, len(ids)))) | {"n": len(ids)}
    res = {"which": "ABST-0 inter-auditor agreement (descriptive)", "blind_sha256": key["blind_sha256"],
           "rows": la.get("auditor"), "cols": lb.get("auditor"), "n": n, "table": t, "raw_agreement": po,
           "cohen_kappa": k_, "by_sleep": by_sleep,
           "disagreements": sorted(k for k in key["map"] if A[k] != B[k])}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "table", "raw_agreement", "cohen_kappa")}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export"); e.add_argument("--dir", required=True); e.add_argument("--seeds", default="12,13,14,15")
    e.add_argument("--secret", required=True); e.add_argument("--blind", required=True); e.add_argument("--key", required=True)
    b = sp.add_parser("bind"); b.add_argument("--dir", required=True); b.add_argument("--key", required=True)
    b.add_argument("--labels", required=True); b.add_argument("--out-dir", required=True)
    c = sp.add_parser("compare"); c.add_argument("--key", required=True); c.add_argument("--a", required=True)
    c.add_argument("--b", required=True); c.add_argument("--out", required=True)
    a = p.parse_args()
    {"export": export, "bind": bind, "compare": compare}[a.cmd](a)
