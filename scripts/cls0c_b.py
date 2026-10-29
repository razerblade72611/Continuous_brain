"""CLS-0C Phase B: evidence-aware addressing on the four CLS-0B development seeds (Claude + Sol, 2026-10-29).
Protocol: lab/CLS0C_B_PROTOCOL.md. Paraphrase labels: lab/CLS0C_PARAPHRASE_AUDIT.md. Read-only: raw experienced text,
no training, no writes to CLS-0B. Methods (same frozen Qwen3-8B reader for every LLM judgment):
  B0  frozen K1 routing (CLS-0B threshold, top-1) + value extraction, no verification
  B1  K1 top-5 (and K1 top-5 u K3 top-5) candidates -> yes/no verification -> temporal rule -> value extraction
  B2  open-vocabulary index: reader extracts subject|relation|value per episode, parses each cue into subject|relation,
      exact normalised relation match (B2) or entity-free relation-embedding cosine >= 0.85 (B2e); B2c closed-schema
      reference (descriptive only, never selectable)
Evaluator-only: hidden entity/attr/value fields, gold eids, audit labels.

  python scripts/cls0c_b.py --results results/cls0b_final --model /workspace/hf/Qwen3-8B \\
      --external /workspace/hf/Qwen3-Embedding-0.6B --out results/cls0c_b
"""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (2, 3, 4, 5)
K_CAND = 5
RECALL5_RULE = 0.98                 # B1 uses K1 u K3 candidates if pooled life-5 K1 recall@5 (entailed para) < this
TAU_REL = 0.85                      # B2e relation-embedding threshold, fixed a priori
TAU_SWEEP = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
GATES = {"template": 0.95, "para_entailed": 0.90, "near_unsupported": 0.10, "unseen_unsupported": 0.10,
         "provenance": 0.95}
BUDGET_S = 2.5 * 3600               # preflight: projected reader time must fit, else exit 4 before the main work
MAX_NEW_VALUE, MAX_NEW_RECORD, MAX_NEW_QUERY = 16, 40, 32
BATCH = 16

# ---------------------------------------------------------------- frozen labels and prompts --------------------------
AUDIT = {  # (attribute, paraphrase index) -> label; lab/CLS0C_PARAPHRASE_AUDIT.md
    ("allergy", 0): "not_entailed", ("allergy", 1): "not_entailed",
    ("birth month", 0): "entailed", ("birth month", 1): "entailed",
    ("childhood fear", 0): "borderline", ("childhood fear", 1): "entailed",
    ("city", 0): "entailed", ("city", 1): "entailed",
    ("favorite object", 0): "not_entailed", ("favorite object", 1): "entailed",
    ("hobby", 0): "entailed", ("hobby", 1): "borderline",
    ("occupation", 0): "entailed", ("occupation", 1): "entailed",
    ("pet species", 0): "entailed", ("pet species", 1): "not_entailed",
    ("sister", 0): "entailed", ("sister", 1): "entailed",
    ("workplace", 0): "entailed", ("workplace", 1): "entailed",
}
MEM = "Memory (from life {t}): {text}\n"
_Q = "Does the memory above state exactly what completes this statement? Answer yes or no.\nAnswer:"
VERIFY_SHOTS = (
    "Memory (from life 2): Mara Lind plays the cello every evening.\n"
    'Statement to complete: "The instrument Mara Lind plays is ..."\n' + _Q + " yes\n\n"
    "Memory (from life 2): Mara Lind plays the cello every evening.\n"
    "Statement to complete: \"Mara Lind's home city is ...\"\n" + _Q + " no\n\n"
    "Memory (from life 3): Mara Lind plays the cello every evening.\n"
    'Statement to complete: "At life 1, the instrument Mara Lind plays is ..."\n' + _Q + " no\n\n")
VERIFY = VERIFY_SHOTS + "{mem}" + 'Statement to complete: "{cue} ..."\n' + _Q
EXTRACT_VALUE = (
    "Memory (from life 2): Mara Lind plays the cello every evening.\n"
    "Using only the memory above, complete the statement with the exact words from the memory.\n"
    "The instrument Mara Lind plays is the cello.\n\n"
    "{mem}Using only the memory above, complete the statement with the exact words from the memory.\n{cue}")
EXTRACT_RECORD = (
    "Text: Mara Lind plays the cello every evening.\n"
    "Write the one fact stated in the text in the form: subject | relation | value\n"
    "Fact: Mara Lind | plays instrument | cello\n\n"
    "Text: {text}\nWrite the one fact stated in the text in the form: subject | relation | value\n{schema}Fact:")
PARSE_QUERY = (
    'Statement to complete: "The instrument Mara Lind plays is ..."\n'
    "Write what the statement asks for in the form: subject | relation\n"
    "Asked: Mara Lind | plays instrument\n\n"
    'Statement to complete: "{cue} ..."\nWrite what the statement asks for in the form: subject | relation\n'
    "{schema}Asked:")
SCHEMA = "Use one of these relations: {rels}\n"          # B2c only (closed-schema reference)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- pure parts (unit-tested) -------------------------
def norm(s):
    s = re.sub(r"[^\w\s'-]", " ", (s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(a|an|the)\s+", "", s)


def parse_fields(gen, n):
    """First line of a generation split on '|' into exactly n non-empty fields, else None."""
    line = (gen or "").strip().split("\n")[0]
    parts = [p.strip().strip('"').strip() for p in line.split("|")]
    return parts if len(parts) == n and all(parts) else None


def cut_value(gen):
    """Value from a completion: first line, up to the first sentence end."""
    line = (gen or "").strip().split("\n")[0]
    line = re.split(r"(?<=[^A-Z])\.(\s|$)", line)[0]
    return line.strip().strip('"').strip(" .,;:")


def evidence(text, value):
    """The sentence of `text` containing `value` verbatim (case-insensitive), or None."""
    v = (value or "").strip().lower()
    v = re.sub(r"^(a|an|the)\s+", "", v)
    if not v:
        return None
    for sent in re.findall(r"[^.]*\.?", text):
        if v and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", sent.lower()):
            return sent.strip()
    return None


def time_constraint(cue):
    m = re.match(r"^At life (\d+),", cue)
    return int(m.group(1)) if m else None


def select(cands, life_req):
    """Temporal rule. cands: dicts with eid, time, margin. Dated -> that life's candidates; unqualified -> most recent.
    Ties -> larger margin, then eid (deterministic)."""
    if life_req is not None:
        cands = [c for c in cands if c["time"] == life_req]
    if not cands:
        return None
    return sorted(cands, key=lambda c: (-c["time"], -c.get("margin", 0.0), c["eid"]))[0]


def commit(eid, value, texts):
    """Answer contract: a value only counts as SUPPORTED with a verbatim evidence span in its episode (fail closed)."""
    if eid is None or not value:
        return {"answer": None, "eid": None, "span": None}
    span = evidence(texts[eid], value)
    if span is None:
        return {"answer": None, "eid": None, "span": None, "rejected_value": value, "rejected_eid": eid}
    return {"answer": value, "eid": eid, "span": span}


def grade(out, gold_value, gold_eids, texts):
    """Evaluator: correct iff answered, normalised value equal, episode in gold_eids (it states the value), span
    verbatim in that episode."""
    if out["answer"] is None:
        return {"answered": False, "correct": False, "provenance_ok": False}
    prov = out["eid"] in gold_eids and out["span"] is not None and out["span"] in texts[out["eid"]]
    return {"answered": True, "correct": norm(out["answer"]) == norm(gold_value) and prov, "provenance_ok": prov}


def summarize(rows):
    """rows: dicts {cat, answered, correct, provenance_ok}. cat in template / para_entailed / para_borderline /
    para_not_entailed / near_miss / unseen / temporal."""
    by = lambda c: [r for r in rows if r["cat"] == c]
    rate = lambda rs, f: (sum(map(f, rs)) / len(rs)) if rs else None
    pos = by("template") + by("para_entailed") + by("temporal")
    ans = [r for r in pos if r["answered"]]
    s = {"n": {c: len(by(c)) for c in ("template", "para_entailed", "para_borderline", "para_not_entailed",
                                       "near_miss", "unseen", "temporal")},
         "template": rate(by("template"), lambda r: r["correct"]),
         "para_entailed": rate(by("para_entailed"), lambda r: r["correct"]),
         "temporal": rate(by("temporal"), lambda r: r["correct"]),
         "near_unsupported": rate(by("near_miss"), lambda r: r["answered"]),
         "unseen_unsupported": rate(by("unseen"), lambda r: r["answered"]),
         "provenance": rate(ans, lambda r: r["provenance_ok"]),
         "para_borderline_answered_stored": rate(by("para_borderline"), lambda r: r["correct"]),
         "para_borderline_abstained": rate(by("para_borderline"), lambda r: not r["answered"]),
         "para_not_entailed_answered_stored": rate(by("para_not_entailed"), lambda r: r["correct"]),
         "para_not_entailed_abstained": rate(by("para_not_entailed"), lambda r: not r["answered"])}
    s["errors"] = (sum(not r["correct"] for r in by("template") + by("para_entailed")) +
                   sum(r["answered"] for r in by("near_miss") + by("unseen")))
    s["passes"] = all(v is not None for v in (s["template"], s["para_entailed"], s["near_unsupported"],
                                              s["unseen_unsupported"])) and \
        s["template"] >= GATES["template"] and s["para_entailed"] >= GATES["para_entailed"] and \
        s["near_unsupported"] <= GATES["near_unsupported"] and \
        s["unseen_unsupported"] <= GATES["unseen_unsupported"] and \
        (s["provenance"] if s["provenance"] is not None else 0) >= GATES["provenance"]
    return s


def choose(summ, seconds, b1_name):
    """Frozen selection among B1 (per the candidate-set rule), B2, B2e."""
    b1 = summ[b1_name]
    b2s = [m for m in ("B2", "B2e") if summ[m]["passes"]]
    if b1["passes"]:
        better = [m for m in b2s if seconds[m] <= 0.5 * seconds[b1_name] or summ[m]["errors"] <= 0.5 * b1["errors"]]
        if better:
            return min(better, key=lambda m: (summ[m]["errors"], m)), "a B2 variant is substantially cheaper or more reliable"
        return b1_name, "B1 passes (preferred on a tie)"
    if b2s:
        return min(b2s, key=lambda m: (summ[m]["errors"], m)), "only a B2 variant passes"
    best = min((b1_name, "B2", "B2e"), key=lambda m: (summ[m]["errors"], m))
    return None, f"no method passes; best by total errors: {best}"


def cue_category(c, attr_of):
    if c["form"] == "template":
        return "template"
    if c["form"].startswith("paraphrase"):
        return "para_" + AUDIT[(attr_of[c["eid"]], int(c["form"][-1]))]
    return {"near_miss": "near_miss", "unseen_person": "unseen"}[c["form"]]


def run_life(L, bank, items, texts, times, k1_top, k3_top, k1_route, verify, extract_value, records, qparse,
             relsim, k_cand=K_CAND):
    """Every method on every item at checkpoint L. items: dicts {key, cue}. The callables are cached reader results:
    verify(cue, eid) -> yes-no margin; extract_value(cue, eid) -> str; records[m][eid] -> (subj, rel, val) or None;
    qparse[m][cue] -> (subj, rel) or None; relsim(rel_a, rel_b) -> cosine. Returns {method: {key: out}} plus
    descriptive candidate info."""
    in_bank = set(bank)
    out = {m: {} for m in ("B0", "B1_K1", "B1_K1K3", "B2", "B2e", "B2c")}
    info = {}
    for it in items:
        cue, key = it["cue"], it["key"]
        req = time_constraint(cue)
        r = k1_route(cue)                                                      # B0
        out["B0"][key] = commit(r, extract_value(cue, r) if r else None, texts)
        c1 = [e for e in k1_top(cue) if e in in_bank][:k_cand]
        c13 = list(dict.fromkeys(c1 + [e for e in k3_top(cue) if e in in_bank][:k_cand]))
        for name, cs in (("B1_K1", c1), ("B1_K1K3", c13)):
            sup = [{"eid": e, "time": times[e], "margin": verify(cue, e)} for e in cs]
            sup = [s for s in sup if s["margin"] > 0]
            ch = select(sup, req)
            out[name][key] = commit(ch["eid"] if ch else None, extract_value(cue, ch["eid"]) if ch else None, texts)
            out[name][key]["n_supported"] = len(sup)
        info[key] = {"k1_top": c1, "k1k3": c13}
        for name, src in (("B2", "B2"), ("B2e", "B2"), ("B2c", "B2c")):
            q = qparse[src].get(cue)
            match = []
            if q:
                for e in bank:
                    rec = records[src].get(e)
                    if not rec or norm(rec[0]) != norm(q[0]):
                        continue
                    if name == "B2e":
                        s_ = relsim(norm(rec[1]), norm(q[1]))
                        if s_ >= TAU_REL:
                            match.append({"eid": e, "time": times[e], "margin": s_, "value": rec[2]})
                    elif norm(rec[1]) == norm(q[1]):
                        match.append({"eid": e, "time": times[e], "margin": 0.0, "value": rec[2]})
            ch = select(match, req)
            out[name][key] = commit(ch["eid"] if ch else None, ch["value"] if ch else None, texts)
            if name == "B2e" and q:                                            # descriptive tau sweep input
                info[key]["b2e_subject_matches"] = [
                    {"eid": e, "time": times[e], "sim": relsim(norm(records["B2"][e][1]), norm(q[1]))}
                    for e in bank if records["B2"].get(e) and norm(records["B2"][e][0]) == norm(q[0])]
    return out, info


# ---------------------------------------------------------------- run ------------------------------------------------
class Reader:
    """Frozen Qwen3-8B reader: plain prompts, greedy, left-padded batches."""
    def __init__(self, path, dev, dtype):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()
        self.dev = dev
        self.yes = self.tok(" yes", add_special_tokens=False).input_ids[0]
        self.no = self.tok(" no", add_special_tokens=False).input_ids[0]
        self.calls = {"yesno": 0, "gen": 0}

    def yesno(self, prompts):
        res = []
        with self.torch.no_grad():
            for i in range(0, len(prompts), BATCH):
                b = self.tok(prompts[i:i + BATCH], return_tensors="pt", padding=True).to(self.dev)
                lp = self.torch.log_softmax(self.model(**b).logits[:, -1].float(), -1)
                res += (lp[:, self.yes] - lp[:, self.no]).tolist()
        self.calls["yesno"] += len(prompts)
        return res

    def generate(self, prompts, max_new):
        res = []
        with self.torch.no_grad():
            for i in range(0, len(prompts), BATCH):
                b = self.tok(prompts[i:i + BATCH], return_tensors="pt", padding=True).to(self.dev)
                g = self.model.generate(**b, max_new_tokens=max_new, do_sample=False,
                                        pad_token_id=self.tok.pad_token_id)
                res += self.tok.batch_decode(g[:, b.input_ids.shape[1]:], skip_special_tokens=True)
        self.calls["gen"] += len(prompts)
        return res


def manifest(d):
    d = Path(d)
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix in (".safetensors", ".json", ".txt"))
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()
    return {p.name: sha(p) for p in files}


def main():
    import torch
    from being.cls.keys import ExternalKeys
    from being.v3 import lifegen
    c0b = _load("cls0b_for_cls0c_b", "scripts/cls0b.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)   # CPU plumbing test, tiny models
    a = ap.parse_args()
    t_start = time.time()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    dev, dtype = ("cuda", torch.bfloat16) if cuda else ("cpu", torch.float32)
    run = {"which": "CLS-0C-B", "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                        text=True).stdout.strip(),
           "gpu": torch.cuda.get_device_name(0) if cuda else None, "test_mode": a.test_mode, "inputs": {},
           "frozen": {"k_cand": K_CAND, "recall5_rule": RECALL5_RULE, "tau_rel": TAU_REL, "gates": GATES,
                      "prompts_sha256": hashlib.sha256((VERIFY + EXTRACT_VALUE + EXTRACT_RECORD + PARSE_QUERY + SCHEMA)
                                                       .encode()).hexdigest()}}
    # ---- stage 1: data, cues, embeddings, candidates (embedder only) ----
    ek = ExternalKeys(a.external, dev, dtype)
    if not a.test_mode:                                                     # hash the 16 GB reader once, not per seed
        man_model, man_ext = manifest(a.model), manifest(a.external)
    S = {}
    for seed in a.seeds:
        p = ROOT / a.results / f"seed{seed}.json"
        run["inputs"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        d = json.loads(p.read_text())
        assert d["status"] == "complete" and d["seed"] == seed
        if not a.test_mode:
            if d["model_manifest"]["files"] != man_model or d["external_manifest"]["files"] != man_ext:
                sys.exit(f"seed {seed}: model or embedder files differ from the ones CLS-0B used")
        eps = [{k: v for k, v in e.items() if k != "text_sha256"} for e in d["episodes"]]
        for e, r in zip(eps, d["episodes"]):
            assert hashlib.sha256(e["text"].encode()).hexdigest() == r["text_sha256"]
        stored = [e for e in eps if d["writes"].get(e["eid"], {}).get("ok")]
        cues, cal, neg, temporal = c0b.build_cues(seed, eps, lifegen)
        assert d["n_cues"] == {"fact": len(cues), "calibration": len(cal), "test_negative": len(neg),
                               "temporal": len(temporal)}, "rebuilt cue counts differ from CLS-0B"
        texts = {e["eid"]: e["text"] for e in stored}
        times = {e["eid"]: e["time"] for e in stored}
        attr_of = {e["eid"]: e["attr"] for e in eps}                       # evaluator-only (audit category)
        qs = sorted({c["cue"] for c in cues + cal + neg + temporal})
        ev = dict(zip(texts, ek.embed(list(texts.values()))))
        qv = dict(zip(qs, ek.embed(qs, query=True)))
        S[seed] = dict(d=d, eps=eps, stored=stored, cues=cues, cal=cal, neg=neg, temporal=temporal, texts=texts,
                       times=times, attr_of=attr_of, ev=ev, qv=qv)
    # candidates per life (cached by (seed, life, cue))
    def k1_rank(seed, L, cue):
        s = S[seed]
        bank = [e["eid"] for e in s["stored"] if e["time"] <= L]
        return sorted(bank, key=lambda e: -float(s["qv"][cue] @ s["ev"][e]))
    def k3_rank(seed, L, cue, _c={}):
        key = (seed, L)
        if key not in _c:
            s = S[seed]
            bank = [e["eid"] for e in s["stored"] if e["time"] <= L]
            allq = sorted(s["qv"])
            sims = c0b.tfidf_sims([s["texts"][e] for e in bank], allq)
            _c[key] = {q: sorted(range(len(bank)), key=lambda j: -row[j]) for q, row in zip(allq, sims)}, bank
        order, bank = _c[key]
        return [bank[j] for j in order[cue]]
    def k1_threshold(seed, L):
        s = S[seed]
        bank = [e["eid"] for e in s["stored"] if e["time"] <= L]
        return c0b.threshold([max(float(s["qv"][c["cue"]] @ s["ev"][e]) for e in bank) for c in s["cal"]])
    def items_at(seed, L):
        s = S[seed]
        it = []
        for c in s["cues"]:
            if c["time"] <= L:
                it.append({"key": f"F|{c['eid']}|{c['form']}", "cue": c["cue"], "cat": cue_category(c, s["attr_of"]),
                           "gold_value": c["value"], "gold_eids": {c["eid"]}})
        for j, c in enumerate(s["neg"]):
            if c["form"] == "near_miss" and c["observed_since"] > L:
                continue
            it.append({"key": f"N|{j}|{c['form']}", "cue": c["cue"], "cat": cue_category(c, s["attr_of"])})
        tq = {e["value"]: e["eid"] for e in s["stored"] if e["kind"] == "temporal"}
        for c in c0b.temporal_items(s["temporal"], L):
            it.append({"key": f"T|{c['form']}", "cue": c["cue"], "cat": "temporal", "gold_value": c["expect"],
                       "gold_eids": {tq[c["expect"]]}})
        return it
    # recall@5 rule (retrieval only): pooled life-5 K1 recall@5 on ENTAILED paraphrases
    hits = [it["key"].split("|")[1] in k1_rank(seed, 5, it["cue"])[:K_CAND]
            for seed in a.seeds for it in items_at(seed, 5) if it["cat"] == "para_entailed"]
    recall5 = sum(hits) / len(hits) if hits else None
    b1_name = "B1_K1" if recall5 is not None and recall5 >= RECALL5_RULE else "B1_K1K3"
    run["recall5_k1_para_entailed_life5"] = recall5
    run["b1_selectable"] = b1_name
    print(f"K1 recall@5 (entailed paraphrases, life 5, pooled) = {recall5:.4f} -> selectable B1 = {b1_name}", flush=True)
    # unique reader work (cached across lives)
    need_v, need_x = set(), set()
    for seed in a.seeds:
        s = S[seed]
        for L in range(1, 6):
            bank = {e["eid"] for e in s["stored"] if e["time"] <= L}
            thr = k1_threshold(seed, L)
            for it in items_at(seed, L):
                c1 = [e for e in k1_rank(seed, L, it["cue"]) if e in bank][:K_CAND]
                c3 = [e for e in k3_rank(seed, L, it["cue"]) if e in bank][:K_CAND]
                for e in set(c1) | set(c3):
                    need_v.add((seed, it["cue"], e))
                    need_x.add((seed, it["cue"], e))                      # upper bound: any candidate may be chosen
                top = c1[0]
                if float(s["qv"][it["cue"]] @ s["ev"][top]) >= thr:
                    need_x.add((seed, it["cue"], top))
    n_rec = sum(len(S[s]["stored"]) for s in a.seeds) * 2                  # B2 + B2c
    n_q = sum(len({it["cue"] for L in range(1, 6) for it in items_at(s, L)}) for s in a.seeds) * 2
    # ---- preflight: real reader latency ----
    reader = Reader(a.model, dev, dtype)
    s0 = S[a.seeds[0]]
    e0 = s0["stored"][0]
    c0 = s0["cues"][0]["cue"]
    mem0 = MEM.format(t=e0["time"], text=e0["text"])
    def timed(fn):
        ts = []
        for _ in range(2):
            if cuda:
                torch.cuda.synchronize()
            t = time.time()
            fn()
            if cuda:
                torch.cuda.synchronize()
            ts.append(time.time() - t)
        return min(ts)
    t_v = timed(lambda: reader.yesno([VERIFY.format(mem=mem0, cue=c0)] * BATCH)) / BATCH
    t_x = timed(lambda: reader.generate([EXTRACT_VALUE.format(mem=mem0, cue=c0)] * BATCH, MAX_NEW_VALUE)) / BATCH
    t_r = timed(lambda: reader.generate([EXTRACT_RECORD.format(text=e0["text"], schema="")] * BATCH,
                                        MAX_NEW_RECORD)) / BATCH
    t_q = timed(lambda: reader.generate([PARSE_QUERY.format(cue=c0, schema="")] * BATCH, MAX_NEW_QUERY)) / BATCH
    proj = len(need_v) * t_v + len(need_x) * t_x + n_rec * t_r + n_q * t_q
    run["preflight"] = {"s_per_verify": t_v, "s_per_extract_value": t_x, "s_per_record": t_r, "s_per_query": t_q,
                        "unique_verify": len(need_v), "unique_extract_upper": len(need_x), "records": n_rec,
                        "query_parses": n_q, "projected_s": round(proj, 1), "budget_s": BUDGET_S}
    print("preflight", json.dumps(run["preflight"]), flush=True)
    if proj > BUDGET_S and not a.test_mode:
        run["status"] = "preflight_stop"
        (out_dir / "run.json").write_text(json.dumps(run, indent=1))
        sys.exit(4)
    # ---- stage 2: reader work ----
    secs = {"verify": 0.0, "extract": 0.0, "records": 0.0, "query": 0.0}
    V, X = {}, {}
    def timed_call(kind, fn):
        t = time.time()
        r = fn()
        secs[kind] += time.time() - t
        return r
    keys = sorted(need_v)
    prompts = [VERIFY.format(mem=MEM.format(t=S[sd]["times"][e], text=S[sd]["texts"][e]), cue=c) for sd, c, e in keys]
    V.update(zip(keys, timed_call("verify", lambda: reader.yesno(prompts))))
    print(f"verify done ({len(keys)})", flush=True)
    rels = ", ".join(sorted(lifegen.TRAIN))                                 # B2c ONLY (descriptive reference)
    records, qparse = {}, {}
    for seed in a.seeds:
        s = S[seed]
        eids = list(s["texts"])
        records[seed], qparse[seed] = {}, {}
        for m, schema in (("B2", ""), ("B2c", SCHEMA.format(rels=rels))):
            gens = timed_call("records", lambda: reader.generate(
                [EXTRACT_RECORD.format(text=s["texts"][e], schema=schema) for e in eids], MAX_NEW_RECORD))
            records[seed][m] = {e: (tuple(f) if (f := parse_fields(g, 3)) else None) for e, g in zip(eids, gens)}
            records[seed][m + "_raw"] = dict(zip(eids, gens))
            cs = sorted({it["cue"] for L in range(1, 6) for it in items_at(seed, L)})
            gq = timed_call("query", lambda: reader.generate([PARSE_QUERY.format(cue=c, schema=schema) for c in cs],
                                                              MAX_NEW_QUERY))
            qparse[seed][m] = {c: (tuple(f) if (f := parse_fields(g, 2)) else None) for c, g in zip(cs, gq)}
            qparse[seed][m + "_raw"] = dict(zip(cs, gq))
    print("index built", flush=True)
    rel_strings = sorted({norm(r[1]) for sd in a.seeds for m in ("B2",) for r in records[sd][m].values() if r} |
                         {norm(q[1]) for sd in a.seeds for q in qparse[sd]["B2"].values() if q})
    rv = dict(zip(rel_strings, ek.embed(rel_strings))) if rel_strings else {}
    relsim = lambda x, y: float(rv[x] @ rv[y]) if x in rv and y in rv else -1.0
    # ---- stage 3: evaluate every checkpoint (dry pass collects the value extractions actually needed) ----
    def life_ctx(seed, L):
        s = S[seed]
        bank = [e["eid"] for e in s["stored"] if e["time"] <= L]
        thr = k1_threshold(seed, L)
        bset = set(bank)
        def k1_route(cue):
            top = [e for e in k1_rank(seed, L, cue) if e in bset][0]
            return top if float(s["qv"][cue] @ s["ev"][top]) >= thr else None
        return bank, thr, items_at(seed, L), k1_route
    want = set()
    for seed in a.seeds:
        for L in range(1, 6):
            bank, thr, its, k1_route = life_ctx(seed, L)
            def rec_x(c, e, seed=seed):
                want.add((seed, c, e))
                return ""
            run_life(L, bank, its, S[seed]["texts"], S[seed]["times"], lambda c: k1_rank(seed, L, c),
                     lambda c: k3_rank(seed, L, c), k1_route, lambda c, e: V[(seed, c, e)], rec_x,
                     records[seed], qparse[seed], relsim)
    xkeys = sorted(want)
    xprompts = [EXTRACT_VALUE.format(mem=MEM.format(t=S[sd]["times"][e], text=S[sd]["texts"][e]), cue=c)
                for sd, c, e in xkeys]
    X = dict(zip(xkeys, map(cut_value, timed_call("extract", lambda: reader.generate(xprompts, MAX_NEW_VALUE)))))
    print(f"value extraction done ({len(xkeys)})", flush=True)
    used = {m: {"v": set(), "x": set()} for m in ("B0", "B1_K1", "B1_K1K3")}
    per_seed, pooled5 = {}, {m: [] for m in ("B0", "B1_K1", "B1_K1K3", "B2", "B2e", "B2c")}
    sweep5 = {t: [] for t in TAU_SWEEP}
    for seed in a.seeds:
        s = S[seed]
        lives = []
        for L in range(1, 6):
            bank, thr, its, k1_route = life_ctx(seed, L)
            outs, info = run_life(L, bank, its, s["texts"], s["times"], lambda c: k1_rank(seed, L, c),
                                  lambda c: k3_rank(seed, L, c), k1_route,
                                  lambda c, e: V[(seed, c, e)], lambda c, e: X[(seed, c, e)],
                                  records[seed], qparse[seed], relsim)
            for it in its:                                                   # attributable reader calls
                for m, cand in (("B1_K1", "k1_top"), ("B1_K1K3", "k1k3")):
                    used[m]["v"] |= {(seed, it["cue"], e) for e in info[it["key"]][cand]}
                for m in used:
                    o = outs[m][it["key"]]
                    e_ = o.get("eid") or o.get("rejected_eid")
                    if e_:
                        used[m]["x"].add((seed, it["cue"], e_))
            graded = {}
            for m, o in outs.items():
                rows = []
                for it in its:
                    if "gold_value" in it:
                        g = grade(o[it["key"]], it["gold_value"], it["gold_eids"], s["texts"])
                    else:
                        g = {"answered": o[it["key"]]["answer"] is not None, "correct": False, "provenance_ok": False}
                    rows.append({"key": it["key"], "cat": it["cat"], **g, **o[it["key"]]})
                graded[m] = {"summary": summarize(rows), "rows": rows}
                if L == 5:
                    pooled5[m] += rows
            if L == 5:                                                       # descriptive B2e tau sweep
                for t in TAU_SWEEP:
                    for it in its:
                        ms = [x for x in info[it["key"]].get("b2e_subject_matches", []) if x["sim"] >= t]
                        ch = select([{**x, "margin": x["sim"]} for x in ms], time_constraint(it["cue"]))
                        o = commit(ch["eid"] if ch else None,
                                   records[seed]["B2"][ch["eid"]][2] if ch else None, s["texts"])
                        g = grade(o, it["gold_value"], it["gold_eids"], s["texts"]) if "gold_value" in it else \
                            {"answered": o["answer"] is not None, "correct": False, "provenance_ok": False}
                        sweep5[t].append({"cat": it["cat"], **g})
            lives.append({"life": L, "threshold_k1": thr, "methods": graded, "candidates": info})
            print(f"seed {seed} life {L}: " + " ".join(f"{m} T{(graded[m]['summary']['template'] or 0):.2f}"
                                                      f"/P{(graded[m]['summary']['para_entailed'] or 0):.2f}"
                                                      f"/N{(graded[m]['summary']['near_unsupported'] or 0):.2f}"
                                                      for m in graded), flush=True)
        per_seed[seed] = {m: lives[-1]["methods"][m]["summary"] for m in pooled5}
        (out_dir / f"seed{seed}.json").write_text(json.dumps(
            {"seed": seed, "lives": lives, "records": {m: {e: list(r) if r else None for e, r in records[seed][m].items()}
                                                       for m in ("B2", "B2c")},
             "records_raw": {m: records[seed][m + "_raw"] for m in ("B2", "B2c")},
             "queries": {m: {c: list(q) if q else None for c, q in qparse[seed][m].items()} for m in ("B2", "B2c")},
             "queries_raw": {m: qparse[seed][m + "_raw"] for m in ("B2", "B2c")}}, indent=1, allow_nan=False))
    summ = {m: summarize(r) for m, r in pooled5.items()}
    # attributable reader seconds: calls actually used by the method x measured mean seconds per call of that kind
    mv = secs["verify"] / max(len(V), 1)
    mx = secs["extract"] / max(len(X), 1)
    half_index = (secs["records"] + secs["query"]) / 2                       # B2 and B2c each built half
    seconds = {m: len(used[m]["v"]) * mv + len(used[m]["x"]) * mx for m in used}
    seconds.update({"B2": half_index, "B2e": half_index, "B2c": half_index})
    sel, why = choose(summ, seconds, b1_name)
    run.update({"status": "complete", "seconds": secs, "method_seconds_approx": seconds, "reader_calls": reader.calls,
                "pooled_life5": summ, "per_seed_life5": per_seed,
                "b2e_tau_sweep_life5": {str(t): summarize(r) for t, r in sweep5.items()},
                "selected": sel, "selection_reason": why, "wall_s": round(time.time() - t_start, 1),
                "valid": not a.test_mode})
    (out_dir / "summary.json").write_text(json.dumps(run, indent=1, allow_nan=False))
    print("CLS-0C-B", json.dumps({m: {k: summ[m][k] for k in ("template", "para_entailed", "near_unsupported",
                                                              "unseen_unsupported", "provenance", "temporal",
                                                              "errors", "passes")} for m in summ}, indent=1))
    print(f"CLS-0C-B SELECTED {sel} ({why})")
    print("CLS-0C-B DONE valid=%s" % run["valid"])
    sys.exit(0 if run["valid"] else 3)


if __name__ == "__main__":
    main()
