"""CLS-0B: does an independently written, keyed episodic memory preserve and retrieve an individual's experiences across
five lives? (Claude + Sol, 2026-10-28). Development seeds 2-5 on the BIND-I1 streams (condition S: same-attribute
competition), Qwen3-8B. Protocol: lab/CLS0B_PROTOCOL.md. Decision rules: scripts/cls0b_analyze.py.

Per seed:
 WRITE   every observed fact of lives 1-5 is one episode (its four observed phrasings, in order), written in
         chronological order into its own adapter with being/cls/episodic.py (published mechanism; initial mask 78 -
         a disclosed, measured deviation from the published 45 - growth x1.2, 10 attempts, 1000 epochs, exact
         reconstruction, per-episode wall-clock cap). Two extra TEMPORAL episodes (one entity, one attribute, different
         values at lives 1 and 3) are written at their lives. The bank is saved after every episode and a restarted run
         resumes from it (digest checked). Each adapter is written ONCE; all retrieval conditions read the same bank.
 EVAL    at the end of each life L (only episodes experienced by then), for each retrieval method
           K1 external Qwen3-Embedding keys | K2 internal layer-24 keys | K3 lexical TF-IDF over the experienced text |
           K4 K1 keys permuted across episodes (negative control)
         a null threshold per method and checkpoint is set from CALIBRATION negatives only (unobserved people) so that
         at most 10% of them activate a memory; then every cue is routed (top-1, or decline):
           fact cues: the template probe and two held-out paraphrases for every stored fact;
           test negatives: unobserved people (20) and familiar people asked about an attribute never observed for them (20);
           temporal cues: "At life 1/3, ..." and the unqualified probe, routed top-2.
         Answers are scored as 50-way forced choice (true value vs the other 49 values the individual experienced) by
         summed log-prob of the value after the prompt, under readouts:
           iRAG   - context = the episode RECONSTRUCTED from the routed adapter (primary)
           raw    - context = the ORIGINAL experienced text of the routed episode (episodic-text baseline)
           direct - the routed adapter active, no context (K1 only; secondary)
           none   - no memory (reference) | oracle - the correct episode's original text (reference)
 Context format: "Memory from life {t}: {text}\\n" lines, then the cue. Plain continuation, no chat template.

  python scripts/cls0b.py --seed 2 --model /workspace/hf/Qwen3-8B --external /workspace/hf/Qwen3-Embedding-0.6B \\
      --out results/cls0b/seed2.json
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = {2, 3, 4, 5}
CFG = {"layer": 10, "key_layer": 24, "r": 2, "init_mask": 78, "growth": 1.2, "retries": 10, "lr": 0.05,
       "max_epochs": 1000, "check_every": 100}
EPISODE_CAP_S = 1200
EVAL_BUDGET_S = 5400          # inference preflight: projected evaluation time must fit, else stop before any training
FALSE_ACTIVATION_BUDGET = 0.10
N_CAL_NEG, N_TEST_UNSEEN, N_TEST_NEARMISS = 40, 20, 20
METHODS = ("K1", "K2", "K3", "K4")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- pure parts (unit-tested) -------------------------
def build_episodes(seed, lifegen, i1, ntok):
    """Episodes = observed facts of condition S, chronological, plus 2 temporal-diagnostic episodes."""
    st = i1.build_stream(seed, lifegen, ntok)
    i1.check_stream(st, lifegen)
    d = st["design"]
    kinds = {f.fid: "target" for f in st["targets"]} | {f.fid: "counter" for f in st["counters"]} | \
            {f.fid: "treatment" for f in st["treatment"]["S"]}
    eps = []
    for L, life in enumerate(st["lives"]["S"], start=1):
        for f in life:
            eps.append({"eid": f.fid, "text": " ".join(f.train_texts), "time": L, "source": f"observed:life{L}",
                        "entity": f.entity, "attr": f.attr, "value": f.value, "kind": kinds.get(f.fid, "filler")})
    used = {w.lower() for e in eps for w in re.findall(r"[A-Za-z]+", e["text"])}
    rng = random.Random(f"CLS0B-temporal-{seed}")
    name, (q1, q3) = fresh_name(rng, lifegen, used), [fresh_word(rng, lifegen, used) for _ in range(2)]
    attr = d["extra"][0]
    for L, v in ((1, q1), (3, q3)):
        eps.append({"eid": f"I{seed}TQ{L}", "time": L, "source": f"observed:life{L}", "entity": name, "attr": attr,
                    "value": v, "kind": "temporal",
                    "text": " ".join(t.format(e=name, v=v) for t in lifegen.TRAIN[attr])})
    eps.sort(key=lambda e: (e["time"], e["kind"] == "temporal", e["eid"]))
    return eps, st


def fresh_word(rng, lifegen, used):
    while True:
        w = lifegen._word(rng, rng.choice((2, 3))).capitalize()
        if w.lower() not in used and len(w) >= 4:
            used.add(w.lower())
            return w


def fresh_name(rng, lifegen, used):
    return f"{fresh_word(rng, lifegen, used)} {fresh_word(rng, lifegen, used)}"


def template(lifegen, attr, entity):
    t = lifegen.TRAIN[attr][0]
    return t[: t.index("{v}")].format(e=entity).rstrip()


def build_cues(seed, eps, lifegen):
    facts = [e for e in eps if e["kind"] != "temporal"]
    values = [e["value"] for e in facts]
    attrs = sorted({e["attr"] for e in facts})
    cues = []
    for e in facts:
        pool = [e["value"]] + [v for v in values if v != e["value"]]
        base = {"eid": e["eid"], "time": e["time"], "kind": e["kind"], "value": e["value"], "pool": pool}
        cues.append({**base, "cue": template(lifegen, e["attr"], e["entity"]), "form": "template"})
        for i, p in enumerate(lifegen.PARAPHRASE[e["attr"]]):
            cues.append({**base, "cue": p.format(e=e["entity"]), "form": f"paraphrase{i}"})
    used = {w.lower() for e in eps for w in re.findall(r"[A-Za-z]+", e["text"])}
    rng = random.Random(f"CLS0B-negatives-{seed}")

    def unseen(n, tag):
        out = []
        for _ in range(n):
            ent = fresh_name(rng, lifegen, used)
            out.append({"cue": template(lifegen, rng.choice(attrs), ent), "form": tag, "entity": ent})
        return out
    cal = unseen(N_CAL_NEG, "calibration")
    neg = unseen(N_TEST_UNSEEN, "unseen_person")
    observed = {}
    for e in facts:
        observed.setdefault(e["entity"], set()).add(e["attr"])
    ents = sorted(observed)
    for _ in range(N_TEST_NEARMISS):
        ent = rng.choice(ents)
        free = [a for a in attrs if a not in observed[ent]]
        neg.append({"cue": template(lifegen, rng.choice(free), ent), "form": "near_miss", "entity": ent,
                    "observed_since": min(x["time"] for x in facts if x["entity"] == ent)})
    tq = [e for e in eps if e["kind"] == "temporal"]
    tpool = [tq[0]["value"], tq[1]["value"]] + values[:48]
    base = template(lifegen, tq[0]["attr"], tq[0]["entity"])
    temporal = [{"cue": f"At life 1, {base}", "expect": tq[0]["value"], "form": "life1"},
                {"cue": f"At life 3, {base}", "expect": tq[1]["value"], "form": "life3"},
                {"cue": base, "expect": tq[1]["value"], "form": "latest"}]
    for t in temporal:
        t["pool"] = tpool
    return cues, cal, neg, temporal


def tfidf_sims(episode_texts, cue_texts):
    """K3: cosine of TF-IDF vectors over lowercase word tokens; IDF from the stored episodes only."""
    tok = lambda s: re.findall(r"[a-z]+", s.lower())
    docs = [tok(t) for t in episode_texts]
    n = len(docs)
    df = {}
    for d_ in docs:
        for w in set(d_):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

    def vec(words):
        v = {}
        for w in words:
            if w in idf:
                v[w] = v.get(w, 0) + idf[w]
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / nrm for w, x in v.items()}
    ev = [vec(d_) for d_ in docs]
    out = []
    for c in cue_texts:
        cv = vec(tok(c))
        out.append([sum(cv.get(w, 0) * x for w, x in e.items()) for e in ev])
    return out


def threshold(cal_top1, budget=FALSE_ACTIVATION_BUDGET):
    """Lowest threshold at which at most floor(budget*n) calibration negatives activate (activation: sim >= thr).
    n=40, budget 0.10 -> at most 4 activate (ties at the boundary make it more conservative, never less)."""
    s = sorted(cal_top1)
    allowed = int(math.floor(len(s) * budget + 1e-9))
    idx = len(s) - allowed - 1                          # the highest value that must NOT activate
    return math.nextafter(s[idx], math.inf) if idx >= 0 else -math.inf


def permuted_keys(seed, life, eids, keys):
    """K4 at a checkpoint: K1 keys permuted ONLY among the memories experienced by then (seed+life-specific)."""
    perm = random.Random(f"CLS0B-perm-{seed}-life{life}").sample(list(eids), len(eids))
    return {e: keys[p] for e, p in zip(eids, perm)}


def temporal_items(temporal, life):
    """Chronological temporal questions: before life 3 only the life-1 state exists (and is the latest); from life 3
    the life-1, life-3 and latest questions are asked. Pools are evaluator-only comparison sets."""
    t1, t3, latest = temporal
    if life < 3:
        return [t1, {**latest, "expect": t1["expect"], "form": "latest_pre"}]
    return [t1, t3, latest]


def route(sim_row, eids, thr, k=1):
    order = sorted(range(len(eids)), key=lambda j: -sim_row[j])[:k]
    return [(eids[j], sim_row[j]) for j in order if sim_row[j] >= thr]


def context(memories):
    return "".join(f"Memory from life {t}: {text}\n" for t, text in sorted(memories))


# ---------------------------------------------------------------- run ------------------------------------------------
def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from being.cls.episodic import BudgetExceeded, EpisodicMemory
    from being.cls.keys import ExternalKeys
    from being.v3 import lifegen
    i1 = _load("bind_i1_for_cls0b", "scripts/bind_i1.py")

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--stream-tokenizer", default=str(ROOT / "models/qwen3-0.6b"),
                    help="tokenizer that built the BIND-I1 streams (token-matched values)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)       # CPU end-to-end test only
    ap.add_argument("--test-cfg", default="{}", help=argparse.SUPPRESS)
    ap.add_argument("--test-limit", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    cfg = dict(CFG) | (json.loads(a.test_cfg) if a.test_mode else {})
    if a.seed not in SEEDS:
        sys.exit("CLS-0B uses dev seeds 2-5 only")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    bank_path = out.with_name(out.stem + "_bank.pt")
    if out.exists() and json.loads(out.read_text()).get("status") == "complete":
        sys.exit(f"{out} is complete; results are never overwritten")
    stok = AutoTokenizer.from_pretrained(a.stream_tokenizer)
    eps, st = build_episodes(a.seed, lifegen, i1, lambda w: len(stok(" " + w).input_ids))
    ref = json.loads((ROOT / f"results/dev3/bind_i1/seed{a.seed}.json").read_text())
    design_check = "match" if json.loads(json.dumps(st["design"])) == ref["design"] else "MISMATCH"
    if design_check != "match" and not a.test_mode:
        sys.exit("stream differs from the committed BIND-I1 design")
    cues, cal, neg, temporal = build_cues(a.seed, eps, lifegen)
    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()

    def manifest(d):
        """sha256 of every weight/config/tokenizer file, plus the HF commit recorded at download time if present."""
        d = Path(d)
        files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix in (".safetensors", ".json", ".txt"))
        rev = None
        meta = list((d / ".cache" / "huggingface" / "download").glob("*.metadata"))
        if meta:
            rev = meta[0].read_text().splitlines()[0].strip()
        return {"revision": rev, "files": {p.name: sha(p) for p in files}}
    res = {"which": "CLS-0B", "seed": a.seed, "config": cfg, "episode_cap_s": EPISODE_CAP_S,
           "design_check": design_check, "test_mode": a.test_mode,
           "false_activation_budget": FALSE_ACTIVATION_BUDGET, "eval_budget_s": EVAL_BUDGET_S,
           "note_candidate_universe": "the 50-value pools are evaluator-only comparison sets (they include values "
                                      "experienced later); they never enter retrieved memories or prompt context",
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "model": a.model, "model_manifest": manifest(a.model), "external": a.external,
           "external_manifest": manifest(a.external), "gpu": torch.cuda.get_device_name(0) if cuda else None,
           "episodes": [{k: v for k, v in e.items()} | {"text_sha256": hashlib.sha256(e["text"].encode()).hexdigest()}
                        for e in eps],
           "n_cues": {"fact": len(cues), "calibration": len(cal), "test_negative": len(neg), "temporal": len(temporal)},
           "status": "writing"}

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=str))
        os.replace(tmp, out)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if cuda else torch.float32,
                                                 attn_implementation="eager").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mem = EpisodicMemory(model, tok, seed=a.seed, **cfg)
    res["writes"] = {}
    @torch.no_grad()
    def fc(prompt, cands, direct=None):
        """Summed log-prob of ' '+cand after prompt; lm_head only on the candidate positions."""
        mem.site.active = [direct] if direct else []
        try:
            p_ids = tok(prompt).input_ids
            seqs = [tok(prompt + " " + c).input_ids for c in cands]
            bad = sum(s[:len(p_ids)] != p_ids for s in seqs)
            L = max(len(s) for s in seqs)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=dev)
            att = torch.zeros((len(seqs), L), dtype=torch.long, device=dev)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s, device=dev)
                att[i, :len(s)] = 1
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
            n = len(p_ids)
            out = []
            for i, s in enumerate(seqs):
                rows = h[i, n - 1:len(s) - 1]
                lp = torch.log_softmax(model.lm_head(rows).float(), -1)
                out.append(float(lp.gather(-1, torch.tensor(s[n:], device=dev)[:, None]).sum()))
            return out, bad
        finally:
            mem.site.active = []

    # ---- inference preflight (resource qualification, before any training) ----
    rep = next(e for e in eps if e["kind"] == "target")
    rep_cue = next(c for c in cues if c["eid"] == rep["eid"] and c["form"] == "template")
    ts_ = []
    for _ in range(3):                                     # the REAL scorer (backbone + lm_head + log-softmax + gather)
        if cuda:
            torch.cuda.synchronize()
        t = time.time()
        fc(context([(1, rep["text"])]) + rep_cue["cue"], rep_cue["pool"])
        if cuda:
            torch.cuda.synchronize()
        ts_.append(time.time() - t)
    t_call = sorted(ts_)[1]
    t = time.time()
    with torch.no_grad():
        model.generate(tok(rep["text"], return_tensors="pt").input_ids.to(dev), max_new_tokens=100, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    t_gen = time.time() - t
    # upper bound from the real evaluation loops: reference prompts (none, oracle) are identical at every checkpoint,
    # so they are scored once per fact cue; routed readouts are counted at EVERY checkpoint ignoring the cache
    # (2 per method for iRAG and raw + 1 K1 direct = 9 per fact cue); temporal items 1 per method
    n_calls = 2 * len(cues) + sum(9 * sum(1 for c in cues if c["time"] <= L) + 4 * len(temporal_items(temporal, L))
                                  for L in range(1, 6))
    proj = n_calls * t_call + len(eps) * t_gen
    res["preflight"] = {"seconds_per_50way": t_call, "seconds_per_reconstruction": t_gen, "max_scoring_calls": n_calls,
                        "projected_eval_s": round(proj, 1), "eval_budget_s": EVAL_BUDGET_S,
                        "cuda_free_total_mib": [x // 2**20 for x in torch.cuda.mem_get_info()] if cuda else None,
                        "peak_alloc_mib": torch.cuda.max_memory_allocated() // 2**20 if cuda else None}
    print(time.strftime("%H:%M:%S"), f"seed {a.seed} preflight", json.dumps(res["preflight"]), flush=True)
    if proj > EVAL_BUDGET_S and not a.test_mode:
        res["status"] = "preflight_stop"
        save()
        sys.exit(4)
    if bank_path.exists():
        n = mem.load(bank_path)
        for e in eps:
            if e["eid"] in mem.episodes:
                assert mem.episodes[e["eid"]].text == e["text"], "resumed bank does not match the stream"
        prev = json.loads(out.read_text()) if out.exists() else {}
        if prev and prev.get("model_manifest") != res["model_manifest"]:
            sys.exit("resume refused: the model files differ from the run that wrote this bank")
        if prev and prev.get("config") != res["config"]:
            sys.exit("resume refused: the storage config differs from the run that wrote this bank")
        res["writes"] = prev.get("writes", {})
        res["resumed_with"] = n
        for k, ep_ in mem.episodes.items():                     # bank saved but the log write was interrupted
            if not res["writes"].get(k, {}).get("ok"):
                res["writes"][k] = {"ok": True, "mask": ep_.mask, "attempts": ep_.attempts, "recovered_from_bank": True,
                                    "seconds": sum(x.get("seconds", 0) for x in ep_.attempts),
                                    "epochs_total": sum(x.get("epochs", 0) for x in ep_.attempts)}
        assert {k for k, w in res["writes"].items() if w.get("ok")} == set(mem.episodes), \
            "write log claims an episode the bank does not hold"
    save()
    # ---- WRITE ----
    t_write = time.time()
    for j, e in enumerate(eps):
        # every recorded outcome is terminal: a failed or budget-censored write is never retried on resume
        if e["eid"] in mem.episodes or e["eid"] in res["writes"] or (a.test_limit and j >= a.test_limit):
            continue
        t = time.time()
        try:
            ep = mem.write(e["eid"], e["text"], e["time"], e["source"], deadline=time.time() + EPISODE_CAP_S)
            res["writes"][e["eid"]] = {"ok": True, "mask": ep.mask, "attempts": ep.attempts}
            mem.save(bank_path)
        except BudgetExceeded as err:
            res["writes"][e["eid"]] = {"ok": False, "budget_censored": True, "attempts": err.attempts}
        except torch.cuda.OutOfMemoryError:
            res["status"], res["error"] = "failed", "CUDA OOM during write"
            save()
            sys.exit(1)
        except RuntimeError as err:
            res["writes"][e["eid"]] = {"ok": False, "error": str(err), "attempts": getattr(err, "attempts", None)}
        w = res["writes"][e["eid"]]
        w["seconds"] = round(time.time() - t, 1)
        w["epochs_total"] = sum(x.get("epochs", 0) for x in w.get("attempts") or [])
        print(time.strftime("%H:%M:%S"), f"seed {a.seed} write {e['eid']} ok={w['ok']} mask={w.get('mask')} "
              f"{w['seconds']}s", flush=True)
        save()
    res["write_seconds"] = round(time.time() - t_write, 1)
    ep_total = sum(w["epochs_total"] for w in res["writes"].values())
    sec_total = sum(w["seconds"] for w in res["writes"].values())
    res["throughput"] = {"epochs_per_s": ep_total / sec_total if sec_total else None, "epochs": ep_total,
                         "seconds": sec_total}
    res["status"] = "evaluating"
    save()
    # ---- EVAL ----
    stored = [e for e in eps if e["eid"] in mem.episodes]
    texts = {e["eid"]: e["text"] for e in eps}
    times = {e["eid"]: e["time"] for e in eps}
    recon = {e["eid"]: mem.reconstruct(e["eid"], max_new=len(tok(e["text"]).input_ids) + 16).strip() for e in stored}
    res["reconstruction_exact"] = {k: v == texts[k].strip() for k, v in recon.items()}
    ek = ExternalKeys(a.external, dev, torch.bfloat16 if cuda else torch.float32)
    all_cue_texts = [c["cue"] for c in cues] + [c["cue"] for c in cal] + [c["cue"] for c in neg] + \
                    [c["cue"] for c in temporal]
    k1_ep = {e["eid"]: v for e, v in zip(stored, ek.embed([e["text"] for e in stored]))} if stored else {}
    k1_cue = dict(zip(all_cue_texts, ek.embed(all_cue_texts, query=True)))
    k2_cue = {c: mem.key(c) for c in all_cue_texts}
    del ek
    if cuda:
        torch.cuda.empty_cache()

    cache, prefix_bad = {}, [0]

    def score(prompt, pool, direct=None):
        key = (prompt, direct, tuple(pool))                     # the score depends on the whole candidate pool
        if key not in cache:
            lps, bad = fc(prompt, pool, direct)
            prefix_bad[0] += bad
            cache[key] = lps
        lps = cache[key]
        return {"correct": all(lps[0] > x for x in lps[1:]), "margin": lps[0] - max(lps[1:]),
                "winner": pool[max(range(len(pool)), key=lambda j: lps[j])]}

    def sims(method, cue_list, eids, ep_texts, life):
        if method == "K3":
            return tfidf_sims(ep_texts, cue_list)
        if method == "K2":
            M = torch.stack([mem.episodes[e].key for e in eids])
            return [(M @ k2_cue[c]).tolist() for c in cue_list]
        keys = k1_ep if method == "K1" else permuted_keys(a.seed, life, eids, k1_ep)
        M = torch.stack([keys[e] for e in eids])
        return [(M @ k1_cue[c]).tolist() for c in cue_list]

    res["checkpoints"] = []
    for L in range(1, 6):
        bank = [e for e in stored if e["time"] <= L]
        eids = [e["eid"] for e in bank]
        ep_texts = [texts[x] for x in eids]
        fact_cues = [c for c in cues if c["time"] <= L]
        cp = {"after_life": L, "stored": len(eids),
              "experienced": sum(1 for e in eps if e["time"] <= L), "methods": {}}
        refs = []
        for c in fact_cues:
            r = {"eid": c["eid"], "form": c["form"], "kind": c["kind"], "time": c["time"],
                 "none": score(c["cue"], c["pool"]),
                 "oracle": score(context([(times[c["eid"]], texts[c["eid"]])]) + c["cue"], c["pool"])}
            refs.append(r)
        cp["references"] = refs
        for m in METHODS:
            if not eids:
                cp["methods"][m] = None
                continue
            cal_s = sims(m, [c["cue"] for c in cal], eids, ep_texts, L)
            thr = threshold([max(r) for r in cal_s])
            mrow = {"threshold": thr, "facts": [], "negatives": [], "temporal": []}
            fs = sims(m, [c["cue"] for c in fact_cues], eids, ep_texts, L)
            for c, row in zip(fact_cues, fs):
                rt = route(row, eids, thr)
                got = rt[0][0] if rt else None
                rec = {"eid": c["eid"], "form": c["form"], "kind": c["kind"], "time": c["time"], "routed": got,
                       "sim": rt[0][1] if rt else max(row), "route_ok": got == c["eid"]}
                if got is None:
                    rec["irag"] = rec["raw"] = score(c["cue"], c["pool"])
                else:
                    rec["irag"] = score(context([(times[got], recon[got])]) + c["cue"], c["pool"])
                    rec["raw"] = score(context([(times[got], texts[got])]) + c["cue"], c["pool"])
                    if m == "K1":
                        rec["direct"] = score(c["cue"], c["pool"], direct=got)
                if m == "K1" and got is None:
                    rec["direct"] = {"correct": False, "declined": True}
                mrow["facts"].append(rec)
            ns = sims(m, [c["cue"] for c in neg], eids, ep_texts, L)
            for c, row in zip(neg, ns):
                if c["form"] == "near_miss" and c["observed_since"] > L:
                    continue                                     # that person has not been met yet
                rt = route(row, eids, thr)
                mrow["negatives"].append({"form": c["form"], "activated": bool(rt), "top": max(row)})
            titems = temporal_items(temporal, L)
            ts = sims(m, [c["cue"] for c in titems], eids, ep_texts, L)
            for c, row in zip(titems, ts):
                rt = route(row, eids, thr, k=2)
                ctx = context([(times[e_], recon[e_]) for e_, _ in rt])
                s_ = score(ctx + c["cue"], [c["expect"]] + [v for v in c["pool"] if v != c["expect"]])
                mrow["temporal"].append({"form": c["form"], "routed": [e_ for e_, _ in rt], **s_})
            cp["methods"][m] = mrow
        res["checkpoints"].append(cp)
        print(time.strftime("%H:%M:%S"), f"seed {a.seed} eval life {L} done ({len(cache)} scored prompts)", flush=True)
        save()
    res["prefix_failures"] = prefix_bad[0]
    res["status"] = "complete"
    save()
    print("CLS-0B SEED DONE", a.seed, json.dumps({"stored": len(stored), "experienced": len(eps),
                                                  "throughput": res["throughput"]}), flush=True)


if __name__ == "__main__":
    main()
