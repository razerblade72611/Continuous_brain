#!/usr/bin/env python3
"""CONS-0A: consolidation from the episodic store into a cortical LoRA (protocol: lab/CONS0A_PROTOCOL.md, frozen
2026-10-31, Claude + Sol). Development mechanism test, not a sealed test.

Stages (each one a separate process; the episodic adapters are never installed where the cortex trains or is tested):
  prep      --seed N --bank B --model M --out F       reconstruct every episode from its own adapter (exactness logged)
  qa        --replay F --model M --out Q              evidence-grounded QA distillation (generate -> span -> entail -> leak)
  calibrate --model M --out C                         seed 6, raw stream text, V format, sleep 1: choose E in {4, 8, 16}
  run       --replay F --qa Q --model M --passes E --out DIR
            arms N, S-V, I-V, S-QA, I-QA over 5 sleeps (shared sleep-1 branch); in-process evaluation every sleep;
            the cortical LoRA after sleeps 1 and 5 is saved for the fresh merged evaluation
  fresh     --replay F --model M --lora L --sleep K --out J    fresh process, LoRA MERGED into the base, same probes
  analyze   DIR...                                    advancement criteria A1-A5 (pooled over seeds)
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


B = _load("cls0c_b_for_cons0a", "scripts/cls0c_b.py")       # AUDIT, norm, manifest (frozen; imports stdlib only)
C0B = _load("cls0b_for_cons0a", "scripts/cls0b.py")          # build_episodes, CFG, fresh_name

SEEDS = (2, 3, 4, 5)
CAL_SEED = 6
PASSES = (4, 8, 16)
RANK, ALPHA = 16, 32
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LR, BATCH = 2e-4, 8
MAX_NEW = 8
N_NEARMISS, N_UNFAMILIAR = 40, 20
QA_PER_EPISODE = 3
LEAK_JACCARD = 0.8
MERGE_AGREE, MERGE_TOL, PPL_TOL = 0.97, 0.02, 0.01
STREAM_TOK = "/workspace/Continuous_brain/models/qwen3-0.6b"   # the CLS-0B stream tokenizer (token-matched streams)
GATES = {"acq": 0.50, "gain": 0.30, "seeds_pos": 3, "conf": 0.10, "ppl_mean": 1.02, "ppl_seed": 1.05}

FRAME = ("Answer from your own memory with the exact answer only, or UNKNOWN if you do not know it.\n\n"
         "Statement: The capital city of France is ...\nAnswer: Paris\n\n"
         "Statement: The middle name of a stranger called Pell Arno is ...\nAnswer: UNKNOWN\n\n"
         "{item}\nAnswer:")
QA_GEN = ("Memory: {text}\n\nWrite three different questions that this memory answers. After each question give the "
          "answer, copied exactly from the memory.\nWrite each on its own line as: Q: <question> | A: <answer>\n")
ENTAIL = ("Memory: Mara Lind plays the cello every evening.\nQuestion: What instrument does Mara Lind play?\n"
          "Proposed answer: cello\nTaken literally, does the memory establish this answer without assuming anything "
          "it does not say? Answer yes or no.\nAnswer: yes\n\n"
          "Memory: Mara Lind plays the cello every evening.\nQuestion: Where does Mara Lind live?\n"
          "Proposed answer: cello\nTaken literally, does the memory establish this answer without assuming anything "
          "it does not say? Answer yes or no.\nAnswer: no\n\n"
          "Memory: Mara Lind plays the cello every evening.\nQuestion: Which instrument will Mara Lind teach next "
          "year?\nProposed answer: cello\nTaken literally, does the memory establish this answer without assuming "
          "anything it does not say? Answer yes or no.\nAnswer: no\n\n"
          "Memory: {text}\nQuestion: {q}\nProposed answer: {a}\nTaken literally, does the memory establish this "
          "answer without assuming anything it does not say? Answer yes or no.\nAnswer:")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


norm = B.norm


# ---------------------------------------------------------------- probes and contract (pure, tested) -----------------
def template_stem(lifegen, attr, e):
    """First TRAIN template ending in {v} (not starting with it), truncated before {v}. Acquisition only."""
    for t in lifegen.TRAIN[attr]:
        s = t.rstrip().rstrip(".")
        if s.endswith("{v}") and not s.startswith("{v}"):
            return s[: -len("{v}")].rstrip().format(e=e)
    raise ValueError(f"no stem template for {attr}")


def paraphrase_stems(lifegen, attr, e):
    """The ENTAILED held-out paraphrases of the frozen CLS-0C audit (primary generalisation probes)."""
    return [(k, p.format(e=e)) for k, p in enumerate(lifegen.PARAPHRASE[attr]) if B.AUDIT.get((attr, k)) == "entailed"]


def frame(item):
    return FRAME.format(item=item)


def statement(stem):
    return f"Statement: {stem} ..."


def question(q):
    return f"Question: {q}"


def first_line(gen):
    s = (gen or "").strip().split("\n")[0].strip()
    return s[:-1].strip() if s.endswith(".") else s


def is_unknown(ans):
    """Abstention is ONLY the exact answer UNKNOWN (case/punctuation-insensitive). "UNKNOWN: Leston" is an answer."""
    return norm(ans) == "unknown"


def grade(ans, value):
    return {"answer": ans, "correct": bool(ans) and not is_unknown(ans) and norm(ans) == norm(value),
            "unknown": is_unknown(ans), "empty": not norm(ans)}


def parse_qa(gen):
    out = []
    for line in (gen or "").split("\n"):
        m = re.match(r"\s*Q:\s*(.+?)\s*\|\s*A:\s*(.+?)\s*$", line)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip().rstrip(".").strip()))
    return out


def span_ok(ans, text):
    a = ans.strip()
    return 1 <= len(a.split()) <= 4 and re.search(r"(?<![\w])" + re.escape(a.lower()) + r"(?![\w])", text.lower()) \
        is not None


def _toks(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def eval_wordings(lifegen):
    """Every evaluation stem wording (templates + all paraphrases, every attribute), entity replaced by E."""
    out = []
    for a in lifegen.TRAIN:
        out += [t.split("{v}")[0].replace("{e}", "E") for t in lifegen.TRAIN[a]]
        out += [p.replace("{e}", "E") for p in lifegen.PARAPHRASE[a]]
    return [_toks(w) for w in out if w.strip()]


def leaks(q, entity, wordings):
    t = _toks(q.replace(entity, "E"))
    return any(len(t & w) / max(1, len(t | w)) >= LEAK_JACCARD for w in wordings)


def negatives(eps, L, seed, lifegen, n=N_NEARMISS):
    """Near-miss: (entity experienced by L) x (attribute never observed for it by L); seeded; up to n."""
    attrs = sorted(a for a in lifegen.ATTRS if a not in lifegen.RELATIONAL)
    seen = {}
    for e in eps:
        if e["time"] <= L:
            seen.setdefault(e["entity"], set()).add(e["attr"])
    cands = sorted((ent, a) for ent, s in seen.items() for a in attrs if a not in s)
    random.Random(f"CONS0A-nm-{seed}-{L}").shuffle(cands)
    return cands[:n]


def unfamiliar(eps, seed, lifegen, n=N_UNFAMILIAR):
    rng = random.Random(f"CONS0A-unf-{seed}")
    used = {w.lower() for e in eps for w in re.findall(r"[A-Za-z]+", e["text"])}
    attrs = sorted(a for a in lifegen.ATTRS if a not in lifegen.RELATIONAL)
    return [(C0B.fresh_name(rng, lifegen, used), rng.choice(attrs)) for _ in range(n)]


def probe_set(eps, st, L, seed, lifegen):
    """Every probe at checkpoint L. Positives: facts with time <= L (template + entailed paraphrases)."""
    P = []
    for e in eps:
        if e["time"] > L:
            continue
        if e["kind"] == "temporal":
            continue                                         # scored separately (two values for one slot)
        P.append({"kind": "template", "eid": e["eid"], "life": e["time"], "value": e["value"],
                  "item": statement(template_stem(lifegen, e["attr"], e["entity"]))})
        for k, s in paraphrase_stems(lifegen, e["attr"], e["entity"]):
            P.append({"kind": "paraphrase", "eid": e["eid"], "life": e["time"], "value": e["value"], "pk": k,
                      "item": statement(s)})
    for ent, a in negatives(eps, L, seed, lifegen):
        P.append({"kind": "nearmiss", "entity": ent, "attr": a, "item": statement(template_stem(lifegen, a, ent))})
    for ent, a in unfamiliar(eps, seed, lifegen):
        P.append({"kind": "unfamiliar", "entity": ent, "attr": a, "item": statement(template_stem(lifegen, a, ent))})
    tq = sorted((e for e in eps if e["kind"] == "temporal"), key=lambda e: e["time"])
    if tq and L >= tq[-1]["time"]:
        P.append({"kind": "temporal", "old": tq[0]["value"], "value": tq[-1]["value"],
                  "item": statement(template_stem(lifegen, tq[0]["attr"], tq[0]["entity"]))})
    return P


def intrusion_items(st, lifegen):
    """BIND-I1 targets: under A's template stem, the values V (A's), W (the condition-S competitor), Z (counter)."""
    out = []
    for a, b, c in zip(st["targets"], st["treatment"]["S"], st["counters"]):
        assert a.attr == b.attr == c.attr
        out.append({"eid": a.fid, "item": statement(template_stem(lifegen, a.attr, a.entity)),
                    "V": a.value, "W": b.value, "Z": c.value})
    return out


# ---------------------------------------------------------------- budget-matched schedules (pure, tested) ---------
def schedule(new, old, E, rng, tok_len, background_chunk):
    """One sleep. new/old: lists of items {text, loss_tail (None = all tokens)}. Returns (I, S, exposure):
    both contain E passes over `new` plus an extra slot of len(new)*E items. I's slot cycles through `old` in seeded
    shuffled order (background if `old` is empty, i.e. sleep 1); S's slot is background chunks with IDENTICAL token
    length and loss tail. Shuffled with the same permutation."""
    core = [dict(x) for _ in range(E) for x in new]
    slot, exposure = [], {}
    if old:
        order, k = [], 0
        while len(slot) < len(core):
            if k % len(old) == 0:
                order = list(range(len(old)))
                rng.shuffle(order)
            x = old[order[k % len(old)]]
            slot.append(dict(x))
            exposure[x["life"]] = exposure.get(x["life"], 0) + 1
            k += 1
    bg = []
    for i in range(len(core)):
        ref = slot[i] if slot else core[i]
        n = len(ref["ids"]) if ref.get("ids") is not None else tok_len(ref["text"])
        ids = background_chunk(i, n)
        bg.append({"text": None, "ids": ids, "loss_tail": ref.get("loss_tail"), "life": 0, "background": True})
    i_slot = slot if slot else bg
    perm = list(range(2 * len(core)))
    rng.shuffle(perm)
    I = [(core + i_slot)[p] for p in perm]
    S = [(core + bg)[p] for p in perm]
    return I, S, exposure


def background_chunker(tok, lines, seed):
    """Deterministic background text of an exact token length (concatenated background lines, cut in token space)."""
    ids = tok(" ".join(lines), add_special_tokens=False).input_ids

    def chunk(i, n):
        start = (random.Random(f"CONS0A-bg-{seed}-{i}").randrange(max(1, len(ids) - n)))
        return list(ids[start:start + n])
    return chunk


def budget(items):
    """(items, tokens, loss-bearing tokens) of one sleep's training list; every item carries exact token ids."""
    return (len(items), sum(len(x["ids"]) for x in items),
            sum(len(x["ids"]) if x.get("loss_tail") is None else x["loss_tail"] for x in items))


def replay_errors(bank_texts, eps, recons, require_complete, require_exact):
    """Fail-closed identity of the replay source: the bank must hold THIS stream's episodes, text-identical, and (in a
    real run) every reconstruction must be exact."""
    E = []
    ids = {e["eid"] for e in eps}
    if not set(bank_texts) <= ids:
        E.append(f"bank holds episodes not in the stream: {sorted(set(bank_texts) - ids)[:3]}")
    if require_complete and set(bank_texts) != ids:
        E.append(f"bank lacks {len(ids - set(bank_texts))} stream episodes")
    bad = [e["eid"] for e in eps if e["eid"] in bank_texts and bank_texts[e["eid"]] != e["text"]]
    if bad:
        E.append(f"{len(bad)} bank texts differ from the stream, e.g. {bad[:3]}")
    if require_exact:
        inex = [e["eid"] for e in eps if (recons.get(e["eid"]) or "").strip() != e["text"].strip()]
        if inex:
            E.append(f"{len(inex)} reconstructions are not exact, e.g. {inex[:3]}")
    return E


# ---------------------------------------------------------------- analysis (pure, tested) ---------------------------
def acc(rows, pred):
    rs = [r for r in rows if pred(r)]
    return (sum(r["correct"] for r in rs) / len(rs), len(rs)) if rs else (None, 0)


def conf_rate(rows):
    rs = [r for r in rows if r["kind"] == "nearmiss"]
    return (sum((not r["unknown"]) and (not r["empty"]) for r in rs) / len(rs), len(rs)) if rs else (None, 0)


def summarise(rows, L):
    return {"new_para": acc(rows, lambda r: r["kind"] == "paraphrase" and r["life"] == L)[0],
            "life1_para": acc(rows, lambda r: r["kind"] == "paraphrase" and r["life"] == 1)[0],
            "new_tmpl": acc(rows, lambda r: r["kind"] == "template" and r["life"] == L)[0],
            "nearmiss_conf": conf_rate(rows)[0],
            "unfamiliar_conf": (lambda rs: sum((not r["unknown"]) and (not r["empty"]) for r in rs) / len(rs)
                                if rs else None)([r for r in rows if r["kind"] == "unfamiliar"])}


def criteria(per_seed):
    """per_seed: {seed: {"N": {L: summ}, arm: {L: summ, "ppl": {L: {set: ratio}}}}} (fresh-merged values substituted
    for sleeps 1 and 5 by the caller). Returns A1-A5 per interleaving arm and the advancing arm (or None)."""
    seeds = sorted(per_seed)
    out = {}
    v = lambda x: 0.0 if x is None else x                   # a missing measure counts as 0 (never as a pass)
    for arm, match in (("I-V", "S-V"), ("I-QA", "S-QA")):
        a = {}
        pooled = lambda k, L, x=arm: sum(v(per_seed[s][x][L][k]) for s in seeds) / len(seeds)
        a["A1"] = all(pooled("new_para", L) >= GATES["acq"] for L in range(1, 6))
        a["A2"] = all(pooled("new_para", L) - pooled("new_para", L, "N") >= GATES["gain"] and
                      sum(v(per_seed[s][arm][L]["new_para"]) > v(per_seed[s]["N"][L]["new_para"]) for s in seeds)
                      >= GATES["seeds_pos"] for L in range(1, 6))
        a["A3"] = pooled("life1_para", 5) > pooled("life1_para", 5, match) and \
            sum(v(per_seed[s][arm][5]["life1_para"]) > v(per_seed[s][match][5]["life1_para"]) for s in seeds) \
            >= GATES["seeds_pos"]
        a["A4"] = all(all(per_seed[s][arm][L]["nearmiss_conf"] is not None for s in seeds) and
                      pooled("nearmiss_conf", L) <= GATES["conf"] for L in range(1, 6))
        a["A5"] = all(all(sum(per_seed[s][arm]["ppl"][L][k] for s in seeds) / len(seeds) <= GATES["ppl_mean"] and
                          max(per_seed[s][arm]["ppl"][L][k] for s in seeds) <= GATES["ppl_seed"]
                          for k in per_seed[seeds[0]][arm]["ppl"][L]) for L in range(1, 6))
        a["advance"] = all(a[k] for k in ("A1", "A2", "A3", "A4", "A5"))
        out[arm] = a
    adv = "I-V" if out["I-V"]["advance"] else ("I-QA" if out["I-QA"]["advance"] else None)
    return out, adv


# ---------------------------------------------------------------- model side ----------------------------------------
class Cortex:
    """One slow LoRA over all attention/MLP projections of every layer. fp32 parameters, frozen base."""

    def __init__(self, model, r=RANK, alpha=ALPHA, seed=0):
        import torch
        self.torch, self.r, self.scale = torch, r, alpha / r
        g = torch.Generator().manual_seed(seed)
        self.sites = []
        for name, mod in model.named_modules():
            if name.split(".")[-1] in TARGETS and isinstance(mod, torch.nn.Linear):
                dev = mod.weight.device
                A = torch.nn.Parameter((torch.randn(r, mod.in_features, generator=g) / math.sqrt(mod.in_features))
                                       .to(dev))
                Bm = torch.nn.Parameter(torch.zeros(mod.out_features, r, device=dev))
                self.sites.append((name, mod, A, Bm))
                mod._cx = (A, Bm, self.scale)
                if not hasattr(mod, "_cx_orig"):
                    mod._cx_orig = mod.forward
                    mod.forward = (lambda m: (lambda x: m._cx_orig(x) + (
                        (x.to(m._cx[0].dtype) @ m._cx[0].T @ m._cx[1].T) * m._cx[2]).to(x.dtype)
                        if m._cx is not None else m._cx_orig(x)))(mod)
        assert self.sites, "no target projections found"

    def params(self):
        return [p for _, _, A, Bm in self.sites for p in (A, Bm)]

    def n_params(self):
        return sum(p.numel() for p in self.params())

    def state(self):
        return {name: (A.detach().cpu().clone(), Bm.detach().cpu().clone()) for name, _, A, Bm in self.sites}

    def load_state(self, s):
        with self.torch.no_grad():
            for name, _, A, Bm in self.sites:
                A.copy_(s[name][0])
                Bm.copy_(s[name][1])

    def merge_and_detach(self):
        """Fold the LoRA into the base weights IN PLACE and remove it (fresh-process evaluation only)."""
        with self.torch.no_grad():
            for _, mod, A, Bm in self.sites:
                mod.weight += ((Bm @ A) * self.scale).to(mod.weight.dtype)
                mod._cx = None


def build_inputs(tok, items, dev):
    """Batch of items {text, loss_tail}: labels -100 except the last `loss_tail` tokens (all tokens if None)."""
    import torch
    seqs = [list(x["ids"])[:256] if x.get("ids") is not None else tok(x["text"], add_special_tokens=False).input_ids[:256]
            for x in items]
    n = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(seqs), n), pad)
    lab = torch.full((len(seqs), n), -100)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, (s, x) in enumerate(zip(seqs, items)):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        k = len(s) if x.get("loss_tail") is None else min(len(s), x["loss_tail"])
        lab[i, len(s) - k:len(s)] = torch.tensor(s[len(s) - k:])
    return ids.to(dev), lab.to(dev), att.to(dev)


def train_sleep(model, tok, cortex, opt, items, dev, log):
    t, losses = time.time(), []
    for i in range(0, len(items), BATCH):
        ids, lab, att = build_inputs(tok, items[i:i + BATCH], dev)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    log.update({"updates": math.ceil(len(items) / BATCH), "items": len(items), "seconds": round(time.time() - t, 1),
                "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
                "tokens": sum(len(x["ids"][:256]) if x.get("ids") is not None else
                              len(tok(x["text"], add_special_tokens=False).input_ids[:256]) for x in items),
                "loss_tokens": sum(x["loss_tail"] if x.get("loss_tail") is not None else
                                   (len(x["ids"][:256]) if x.get("ids") is not None else
                                    len(tok(x["text"], add_special_tokens=False).input_ids[:256])) for x in items)})


def generate(model, tok, prompts, dev, max_new=MAX_NEW, bs=16):
    import torch
    out = []
    side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = model.generate(**b, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            out += tok.batch_decode(g[:, b.input_ids.shape[1]:], skip_special_tokens=True)
    tok.padding_side = side
    return out


def logp_cont(model, tok, prompt, cont, dev):
    import torch
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(cont, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device=dev)
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
    return float(sum(lp[len(p) - 1 + j, t] for j, t in enumerate(c)))


def yesno(model, tok, prompts, dev):
    import torch
    y, n = (tok(w, add_special_tokens=False).input_ids[0] for w in (" yes", " no"))
    res = []
    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([tok(p, add_special_tokens=False).input_ids], device=dev)
            lp = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
            res.append(float(lp[y] - lp[n]))
    return res


def evaluate(model, tok, eps, st, L, seed, lifegen, dev, comp_sets, base_loss):
    from being.competence import mean_loss
    P = probe_set(eps, st, L, seed, lifegen)
    gens = generate(model, tok, [frame(p["item"]) for p in P], dev)
    rows = []
    for p, g in zip(P, gens):
        ans = first_line(g)
        r = {k: v for k, v in p.items() if k != "item"} | {"item": p["item"]}
        if p["kind"] in ("template", "paraphrase", "temporal"):
            r |= grade(ans, p["value"])
            if p["kind"] == "temporal":
                r["gave_old"] = norm(ans) == norm(p["old"])
        else:
            r |= {"answer": ans, "correct": False, "unknown": is_unknown(ans), "empty": not norm(ans)}
        rows.append(r)
    intr = []
    for it in intrusion_items(st, lifegen):
        pr = frame(it["item"])
        intr.append({"eid": it["eid"], **{k: logp_cont(model, tok, pr, " " + it[k], dev) for k in ("V", "W", "Z")}})
    ppl = {k: math.exp(mean_loss(model, tok, s)) / math.exp(base_loss[k]) for k, s in comp_sets.items()}
    return {"L": L, "rows": rows, "intrusion": intr, "ppl_ratio": ppl, "summary": summarise(rows, L)}


def load_model(path, dev):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _stream(seed, stream_tok, test_mode=False):
    """The stream exactly as CLS-0B built it: token-matched with the CLS-0B stream tokenizer; for dev seeds the design
    must equal the committed BIND-I1 reference (fail closed)."""
    from transformers import AutoTokenizer
    from being.v3 import lifegen
    i1 = _load("bind_i1_for_cons0a", "scripts/bind_i1.py")
    stok = AutoTokenizer.from_pretrained(stream_tok)
    eps, st = C0B.build_episodes(seed, lifegen, i1, lambda w: len(stok(" " + w).input_ids))
    if seed in SEEDS and not test_mode:
        ref = json.loads((ROOT / f"results/dev3/bind_i1/seed{seed}.json").read_text())
        if json.loads(json.dumps(st["design"])) != ref["design"]:
            sys.exit(f"STREAM MISMATCH: seed {seed} design differs from results/dev3/bind_i1 (stream tokenizer?)")
    return eps, st, lifegen


def _check_replay_matches(R, eps):
    bad = [e["eid"] for e, r in zip(eps, R["episodes"]) if e["eid"] != r["eid"] or e["text"] != r["stream_text"]
           or e["value"] != r["value"]]
    if len(eps) != len(R["episodes"]) or bad:
        sys.exit(f"REPLAY MISMATCH: replay does not match the stream ({len(bad)} episodes differ)")


# ---------------------------------------------------------------- stages ------------------------------------------
def stage_prep(a):
    import torch
    from being.cls.episodic import EpisodicMemory
    eps, st, lifegen = _stream(a.seed, a.stream_tokenizer, a.test_mode)
    dev = _dev()
    model, tok = load_model(a.model, dev)
    cfg = dict(C0B.CFG) | (json.loads(a.test_cfg) if a.test_mode else {})
    mem = EpisodicMemory(model, tok, seed=a.seed, **cfg)
    bank = Path(a.bank)
    mem.load(bank)
    out = []
    for e in eps:
        rec = mem.reconstruct(e["eid"], max_new=len(tok(e["text"]).input_ids) + 16).strip() \
            if e["eid"] in mem.episodes else None
        out.append({k: e[k] for k in ("eid", "time", "entity", "attr", "value", "kind")} |
                   {"stream_text": e["text"], "recon": rec, "exact": rec == e["text"].strip()})
    errors = replay_errors({k: ep.text for k, ep in mem.episodes.items()}, eps, {e["eid"]: e["recon"] for e in out},
                           require_complete=not a.test_mode, require_exact=not a.test_mode)
    res = {"which": "CONS-0A prep", "seed": a.seed, "bank": str(bank), "bank_sha256": sha(bank.read_bytes()),
           "model_manifest": B.manifest(a.model) if not a.test_mode else None, "test_mode": a.test_mode,
           "episodes": out, "exact": sum(e["exact"] for e in out), "n": len(out), "errors": errors}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"PREP seed {a.seed}: {res['exact']}/{res['n']} exact; errors {errors}", flush=True)
    if errors:
        sys.exit("PREP FAILED: " + "; ".join(errors))
    del torch


def stage_qa(a):
    from being.v3 import lifegen
    R = json.loads(Path(a.replay).read_text())
    dev = _dev()
    model, tok = load_model(a.model, dev)
    wordings = eval_wordings(lifegen)
    eps = [e for e in R["episodes"] if e["recon"]]
    gens = generate(model, tok, [QA_GEN.format(text=e["recon"]) for e in eps], dev, max_new=160, bs=8)
    cands, entail_req = [], []
    for e, g in zip(eps, gens):
        for q, ans in parse_qa(g):
            c = {"eid": e["eid"], "life": e["time"], "attr": e["attr"], "q": q, "a": ans, "reject": None}
            if not span_ok(ans, e["recon"]):
                c["reject"] = "span"
            elif norm(ans) and norm(ans) in norm(q):
                c["reject"] = "answer_in_question"
            elif leaks(q, e["entity"], wordings):
                c["reject"] = "leak"
            cands.append(c)
            if c["reject"] is None:
                entail_req.append((len(cands) - 1, ENTAIL.format(text=e["recon"], q=q, a=ans)))
    margins = yesno(model, tok, [p for _, p in entail_req], dev)
    for (i, _), m in zip(entail_req, margins):
        cands[i]["entail_margin"] = m
        if m <= 0:
            cands[i]["reject"] = "entailment"
    acc_, per = {}, {}
    for c in cands:
        if c["reject"] is None and per.get(c["eid"], 0) < QA_PER_EPISODE:
            per[c["eid"]] = per.get(c["eid"], 0) + 1
            acc_.setdefault(c["eid"], []).append({"q": c["q"], "a": c["a"]})
        elif c["reject"] is None:
            c["reject"] = "cap"
    by_attr = {}
    for e in eps:
        d = by_attr.setdefault(e["attr"], [0, 0])
        d[0] += 1
        d[1] += bool(acc_.get(e["eid"]))
    res = {"which": "CONS-0A qa", "seed": R["seed"], "replay_sha256": sha(Path(a.replay).read_bytes()),
           "generations": gens, "candidates": cands, "accepted": acc_,
           "acceptance_by_attr": {k: {"episodes": v[0], "with_qa": v[1]} for k, v in sorted(by_attr.items())},
           "rejects": {r: sum(c["reject"] == r for c in cands) for r in
                       ("span", "answer_in_question", "leak", "entailment", "cap")},
           "n_accepted": sum(map(len, acc_.values())), "episodes_without_qa": [e["eid"] for e in eps
                                                                               if not acc_.get(e["eid"])]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"QA seed {R['seed']}: {res['n_accepted']} accepted, rejects {res['rejects']}, "
          f"{len(res['episodes_without_qa'])} episodes without QA", flush=True)


def qa_ids(tok, q, a):
    """Exact supervision boundary by construction: prompt ids (tokenised exactly as at evaluation) + answer ids; the
    loss falls on the answer ids only."""
    p = tok(frame(question(q)), add_special_tokens=False).input_ids
    ans = tok(" " + a, add_special_tokens=False).input_ids
    return p + ans, len(ans)


def _items(R, Q, fam, tok, upto=None, life=None):
    out = []
    for e in R["episodes"]:
        if not e["recon"] or (life is not None and e["time"] != life) or (upto is not None and e["time"] > upto):
            continue
        if fam == "V":
            out.append({"text": e["recon"], "ids": tok(e["recon"], add_special_tokens=False).input_ids[:256],
                        "loss_tail": None, "life": e["time"], "eid": e["eid"]})
        else:
            for qa in (Q["accepted"].get(e["eid"]) or []):
                ids, k = qa_ids(tok, qa["q"], qa["a"])
                out.append({"text": None, "ids": ids, "loss_tail": k, "life": e["time"], "eid": e["eid"],
                            "q": qa["q"], "answer": qa["a"]})
    return out


def _comp(tok, model, test_mode):
    from being.competence import DEV_MONITOR_SETS, load_sentences, mean_loss
    sets = {k: load_sentences(k)[: (8 if test_mode else None)] for k in DEV_MONITOR_SETS}
    return sets, {k: mean_loss(model, tok, s) for k, s in sets.items()}


def stage_run(a):
    import torch
    from being.v3 import lifegen
    R, Q = json.loads(Path(a.replay).read_text()), json.loads(Path(a.qa).read_text())
    seed = R["seed"]
    eps, st, _ = _stream(seed, a.stream_tokenizer, a.test_mode)
    _check_replay_matches(R, eps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = _dev()
    torch.manual_seed(seed)
    model, tok = load_model(a.model, dev)
    comp_sets, base_loss = _comp(tok, model, a.test_mode)
    bg_lines = (ROOT / "data/background.txt").read_text().splitlines()
    chunk = background_chunker(tok, bg_lines, seed)
    tl = lambda s: len(tok(s, add_special_tokens=False).input_ids[:256])
    res = {"which": "CONS-0A run", "seed": seed, "passes": a.passes, "rank": RANK, "alpha": ALPHA, "lr": LR,
           "batch": BATCH, "test_mode": a.test_mode, "replay_sha256": sha(Path(a.replay).read_bytes()),
           "qa_sha256": sha(Path(a.qa).read_bytes()), "frame_sha256": sha(FRAME), "arms": {}, "N": {}}
    for L in range(1, 6):                                     # N: the unchanged base at every checkpoint
        res["N"][L] = evaluate(model, tok, eps, st, L, seed, lifegen, dev, comp_sets, base_loss)
    cx = Cortex(model, seed=seed)
    res["cortex_params"] = cx.n_params()
    for fam in ("V", "QA"):
        cx = _fresh_cortex(cx, seed)
        opt = torch.optim.AdamW(cx.params(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
        rng = random.Random(f"CONS0A-sched-{seed}-{fam}")
        new = _items(R, Q, fam, tok, life=1)
        I, S, exp_ = schedule(new, [], a.passes, rng, tl, chunk)
        assert [(x["text"], x.get("ids")) for x in I] == [(x["text"], x.get("ids")) for x in S]  # sleep 1 shared
        log1 = {"exposure": exp_}
        cx_model_train(model)
        train_sleep(model, tok, cx, opt, I, dev, log1)
        model.eval()
        ev1 = evaluate(model, tok, eps, st, 1, seed, lifegen, dev, comp_sets, base_loss)
        snap = {"cx": cx.state(), "opt": copy.deepcopy(opt.state_dict()), "rng": rng.getstate()}
        torch.save(snap["cx"], out / f"cortex_{fam}_shared_s1.pt")
        for kind in ("S", "I"):
            arm = f"{kind}-{fam}"
            cx.load_state(snap["cx"])
            opt.load_state_dict(copy.deepcopy(snap["opt"]))
            rng = random.Random()
            rng.setstate(snap["rng"])
            res["arms"][arm] = {"train": {1: log1}, "eval": {1: ev1}}
            for L in range(2, 6):
                new = _items(R, Q, fam, tok, life=L)
                old = _items(R, Q, fam, tok, upto=L - 1)
                I_, S_, exp_ = schedule(new, old, a.passes, rng, tl, chunk)
                if budget(I_) != budget(S_):                          # fail closed: S and I budgets must be equal
                    sys.exit(f"BUDGET MISMATCH {fam} sleep {L}: I {budget(I_)} vs S {budget(S_)}")
                items = I_ if kind == "I" else S_
                lg = {"exposure": exp_ if kind == "I" else {}, "budget": list(budget(items))}
                cx_model_train(model)
                train_sleep(model, tok, cx, opt, items, dev, lg)
                model.eval()
                res["arms"][arm]["train"][L] = lg
                res["arms"][arm]["eval"][L] = evaluate(model, tok, eps, st, L, seed, lifegen, dev, comp_sets,
                                                       base_loss)
                print(time.strftime("%H:%M:%S"), arm, "sleep", L, res["arms"][arm]["eval"][L]["summary"], flush=True)
            torch.save(cx.state(), out / f"cortex_{arm}_s5.pt")
        print(time.strftime("%H:%M:%S"), fam, "sleep 1", ev1["summary"], flush=True)
    (out / f"run_seed{seed}.json").write_text(json.dumps(res, indent=1))
    print(f"RUN seed {seed} DONE", flush=True)


def _fresh_cortex(cx, seed):
    """Reset the cortex to its initial state (A re-drawn from the same seed, B zero) for the next family."""
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, mod, A, Bm in cx.sites:
            A.copy_(torch.randn(A.shape, generator=g) / math.sqrt(mod.in_features))
            Bm.zero_()
    return cx


def cx_model_train(model):
    model.train()
    for m in model.modules():                                  # no dropout anywhere: deterministic training
        if hasattr(m, "p") and isinstance(getattr(m, "p"), float):
            m.p = 0.0


def stage_fresh(a):
    import torch
    from being.v3 import lifegen
    R = json.loads(Path(a.replay).read_text())
    seed = R["seed"]
    eps, st, _ = _stream(seed, a.stream_tokenizer, a.test_mode)
    _check_replay_matches(R, eps)
    dev = _dev()
    model, tok = load_model(a.model, dev)
    comp_sets, base_loss = _comp(tok, model, a.test_mode)
    cx = Cortex(model, seed=seed)
    cx.load_state(torch.load(a.lora))
    cx.merge_and_detach()
    ev = evaluate(model, tok, eps, st, a.sleep, seed, lifegen, dev, comp_sets, base_loss)
    ev["lora_sha256"] = sha(Path(a.lora).read_bytes())
    Path(a.out).write_text(json.dumps(ev, indent=1))
    print(f"FRESH {a.lora} sleep {a.sleep}: {ev['summary']}", flush=True)


def merge_check(inproc, fresh):
    """Merge equivalence: identical probe identities, >= 97% identical answers, every summary metric within 0.02 and
    every competence ratio within 0.01."""
    ident = lambda ev: [(r["kind"], r.get("eid") or r.get("entity"), r["item"]) for r in ev["rows"]]
    same_probes = ident(inproc) == ident(fresh)
    a = [r["answer"] for r in inproc["rows"]]
    b = [r["answer"] for r in fresh["rows"]]
    agree = sum(x == y for x, y in zip(a, b)) / max(1, len(a))
    diffs = {k: abs((inproc["summary"][k] or 0) - (fresh["summary"][k] or 0)) for k in inproc["summary"]}
    ppl = {k: abs(inproc["ppl_ratio"][k] - fresh.get("ppl_ratio", {}).get(k, float("inf"))) for k in inproc["ppl_ratio"]}
    return {"same_probes": same_probes, "agree": agree, "max_diff": max(diffs.values()),
            "max_ppl_diff": max(ppl.values()) if ppl else 0.0,
            "ok": same_probes and agree >= MERGE_AGREE and max(diffs.values()) <= MERGE_TOL and
            (max(ppl.values()) if ppl else 0.0) <= PPL_TOL}


def project(timing, E, n_eps, seeds=len(SEEDS), overhead_s=900):
    """Conservative wall-clock projection of the whole 0A campaign from calibration timings. Per seed: 23 in-process
    evaluations (N x5, two families x (1 + 2 x 4)) whose probe count grows with the checkpoint (~ life-5 size, taken
    as 3x the life-1 count), 8 fresh evaluations, training of 2 families x (E x n_eps x 2 slot) x (1 + up to 2 arms),
    plus prep/QA/model loads (overhead_s)."""
    per_eval = timing["eval_s_life1"] * 3.0
    train = timing["train_s_per_item"] * E * n_eps * 2 * 2 * 2
    return round(seeds * (23 * per_eval + 8 * (per_eval + 120) + train + overhead_s))


def stage_calibrate(a):
    import torch
    from being.v3 import lifegen
    eps, st, _ = _stream(CAL_SEED, a.stream_tokenizer, a.test_mode)
    dev = _dev()
    model, tok = load_model(a.model, dev)
    comp_sets, base_loss = _comp(tok, model, a.test_mode)
    t0 = time.time()
    n_ev = evaluate(model, tok, eps, st, 1, CAL_SEED, lifegen, dev, comp_sets, base_loss)
    eval_s1 = time.time() - t0
    bg = background_chunker(tok, (ROOT / "data/background.txt").read_text().splitlines(), CAL_SEED)
    tl = lambda s: len(tok(s, add_special_tokens=False).input_ids[:256])
    new = [{"text": e["text"], "ids": tok(e["text"], add_special_tokens=False).input_ids[:256], "loss_tail": None,
            "life": 1, "eid": e["eid"]} for e in eps if e["time"] == 1]
    cx = Cortex(model, seed=CAL_SEED)
    res = {"which": "CONS-0A calibrate", "seed": CAL_SEED, "N_nearmiss_conf": n_ev["summary"]["nearmiss_conf"],
           "N": n_ev["summary"], "passes": {}}
    chosen = None
    for E in PASSES:
        _fresh_cortex(cx, CAL_SEED)
        opt = torch.optim.AdamW(cx.params(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
        I, _, _ = schedule(new, [], E, random.Random(f"CONS0A-cal-{E}"), tl, bg)
        lg = {}
        cx_model_train(model)
        train_sleep(model, tok, cx, opt, I, dev, lg)
        model.eval()
        ev = evaluate(model, tok, eps, st, 1, CAL_SEED, lifegen, dev, comp_sets, base_loss)
        ok = (ev["summary"]["new_tmpl"] or 0) >= 0.80 and all(v <= GATES["ppl_mean"] for v in ev["ppl_ratio"].values())
        res["passes"][E] = {"summary": ev["summary"], "ppl_ratio": ev["ppl_ratio"], "train": lg, "ok": ok}
        print(f"CAL E={E}: {ev['summary']} ppl {ev['ppl_ratio']} ok={ok}", flush=True)
        if ok and chosen is None:
            chosen = E
    res["chosen_E"] = chosen or max(PASSES)
    res["timing"] = {"eval_s_life1": round(eval_s1, 1), "n_probes_life1": len(n_ev["rows"]),
                     "train_s_per_item": round(sum(v["train"]["seconds"] for v in res["passes"].values()) /
                                               max(1, sum(v["train"]["items"] for v in res["passes"].values())), 4)}
    res["projection_s"] = project(res["timing"], res["chosen_E"], len(eps))
    res["contract_revision_needed"] = (res["N_nearmiss_conf"] or 0) > GATES["conf"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"CALIBRATE chosen E={res['chosen_E']} N_nearmiss_conf={res['N_nearmiss_conf']} "
          f"contract_revision_needed={res['contract_revision_needed']}", flush=True)


def stage_analyze(a):
    per_seed, merges = {}, {}
    for d in map(Path, a.dirs):
        for f in sorted(d.glob("run_seed*.json")):
            r = json.loads(f.read_text())
            s = r["seed"]
            ps = {"N": {int(L): v["summary"] for L, v in r["N"].items()}}
            for arm, v in r["arms"].items():
                ps[arm] = {int(L): e["summary"] for L, e in v["eval"].items()}
                ps[arm]["ppl"] = {int(L): e["ppl_ratio"] for L, e in v["eval"].items()}
                for K in (1, 5):                              # primary endpoints: fresh merged evaluation
                    fp = d / f"fresh_seed{s}_{arm}_s{K}.json"
                    if fp.exists():
                        fr = json.loads(fp.read_text())
                        merges[f"{s}:{arm}:{K}"] = merge_check(v["eval"][str(K)], fr)
                        ps[arm][K] = fr["summary"]
                        ps[arm]["ppl"][K] = fr["ppl_ratio"]          # A5 at sleeps 1 and 5 from the fresh process
                    else:
                        merges[f"{s}:{arm}:{K}"] = {"ok": False, "missing": True}
            per_seed[s] = ps
    crit, adv = criteria(per_seed)
    invalid = [k for k, m in merges.items() if not m.get("ok")]
    out = {"seeds": sorted(per_seed), "criteria": crit, "advancing_arm": None if invalid else adv,
           "merge_checks": merges, "invalid": invalid,
           "status": "CONS0A_INVALID" if invalid else ("CONS0A_ADVANCE_" + adv if adv else "CONS0A_NO_ADVANCE")}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("criteria", "advancing_arm", "invalid", "status")}, indent=1))
    print(f"status={out['status']}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--seed", type=int, required=True); p.add_argument("--bank", required=True)
    p.add_argument("--model", required=True); p.add_argument("--out", required=True)
    p.add_argument("--test-mode", action="store_true"); p.add_argument("--test-cfg", default="{}")
    p.add_argument("--stream-tokenizer", default=STREAM_TOK)
    q = sp.add_parser("qa")
    q.add_argument("--replay", required=True); q.add_argument("--model", required=True); q.add_argument("--out", required=True)
    c = sp.add_parser("calibrate")
    c.add_argument("--model", required=True); c.add_argument("--out", required=True)
    c.add_argument("--test-mode", action="store_true"); c.add_argument("--stream-tokenizer", default=STREAM_TOK)
    r = sp.add_parser("run")
    r.add_argument("--replay", required=True); r.add_argument("--qa", required=True); r.add_argument("--model", required=True)
    r.add_argument("--passes", type=int, required=True); r.add_argument("--out", required=True)
    r.add_argument("--test-mode", action="store_true"); r.add_argument("--stream-tokenizer", default=STREAM_TOK)
    f = sp.add_parser("fresh")
    f.add_argument("--replay", required=True); f.add_argument("--model", required=True); f.add_argument("--lora", required=True)
    f.add_argument("--sleep", type=int, required=True); f.add_argument("--out", required=True)
    f.add_argument("--test-mode", action="store_true"); f.add_argument("--stream-tokenizer", default=STREAM_TOK)
    an = sp.add_parser("analyze")
    an.add_argument("dirs", nargs="+"); an.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prep": stage_prep, "qa": stage_qa, "calibrate": stage_calibrate, "run": stage_run, "fresh": stage_fresh,
     "analyze": stage_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
