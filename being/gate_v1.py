"""Gate v1: Claude's reimplementation of Grok's "Being gate" (Linear doc, 19 Oct 2026) from its written record.

This is NOT Grok's code (that is still in Grok's sandbox). Details the record does not state are marked ASSUMED.

Record:
  frozen distilgpt2; memory MLP width 512; input layer a random basis from the birth seed; output layer born zero;
  surprise = grad of MSE(memory(key), target); S = 0.9 S - 0.2 grad; W = (1 - forget) W + S; forget 0;
  a life is 3 facts; target = frozen input embedding of the answer token; awake 1 pass, sleep 100 replay passes;
  hit = cosine to true answer >= 0.30 and >= 0.05 above every other candidate;
  pass: lived >= 12/15, wiped <= 3/15, lived > wiped, backbone hash unchanged, competence delta < 1e-5.
ASSUMED: key = final hidden state (after ln_f) at the last token of the question prefix, L2-normalised;
  input layer fixed (random features), only the output layer learns; prefixes and name/colour lists below.
Extra controls added by Claude: fresh seeds, and a stranger arm (memory from another life, scored on this life).
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .backbone import backbone_hash
from .competence import mean_loss
from .memory import TitansMemory

PREFIXES = ["My friend's name is", "My friend's favorite color is", "My friend's favorite word is"]
NAMES = [" Alice", " Bruno", " Clara", " David", " Elena", " Frank", " Grace", " Henry"]  # ASSUMED list
COLORS = [" red", " blue", " green", " yellow", " purple", " orange", " black", " white"]  # ASSUMED list
WORDS = [" fish", " bird", " tree", " rock", " ship", " king", " moon", " rain"]  # from the record
CATEGORIES = [NAMES, COLORS, WORDS]
COMPETENCE = ["The capital of France is Paris.", "Two plus two equals four."]


def single_token(tok, words):
    ids = [tok(w).input_ids for w in words]
    assert all(len(i) == 1 for i in ids), [w for w, i in zip(words, ids) if len(i) != 1]
    return [i[0] for i in ids]


@torch.no_grad()
def key_of(model, tok, prefix: str) -> torch.Tensor:
    ids = tok(prefix, return_tensors="pt").input_ids
    h = model(ids, output_hidden_states=True).hidden_states[-1][0, -1]
    return h / h.norm()


def life(seed: int):
    rng = random.Random(seed)
    return [rng.randrange(len(c)) for c in CATEGORIES]


def live(mem, keys, targets, awake=1, sleep=100, lr=0.2, momentum=0.9, forget=0.0):
    for _ in range(awake + sleep):
        for k, v in zip(keys, targets):
            loss = F.mse_loss(mem(k), v)
            loss.backward()
            mem.titans_step(lr, momentum, forget)


def hits(mem, keys, answers, cand_embs):
    n = 0
    with torch.no_grad():
        for c, (k, a) in enumerate(zip(keys, answers)):
            out = mem(k)
            cos = F.cosine_similarity(out.unsqueeze(0), cand_embs[c], dim=-1)
            others = torch.cat([cos[:a], cos[a + 1:]])
            n += int(cos[a] >= 0.30 and bool((cos[a] - others >= 0.05).all()))
    return n


def run(model, tok, seeds, stranger_offset=1000):
    h0 = backbone_hash(model)
    comp0 = mean_loss(model, tok, COMPETENCE)
    wte = model.get_input_embeddings().weight.detach()
    cand_ids = [single_token(tok, c) for c in CATEGORIES]
    cand_embs = [wte[ids] for ids in cand_ids]
    keys = [key_of(model, tok, p) for p in PREFIXES]
    d, rows = wte.shape[1], []
    for s in seeds:
        ans = life(s)
        targets = [cand_embs[c][a] for c, a in enumerate(ans)]
        lived = TitansMemory(keys[0].numel(), 512, d, birth_seed=s, train_input=False)
        live(lived, keys, targets)
        wiped = TitansMemory(keys[0].numel(), 512, d, birth_seed=s, train_input=False)
        st_ans = life(s + stranger_offset)
        stranger = TitansMemory(keys[0].numel(), 512, d, birth_seed=s, train_input=False)
        live(stranger, keys, [cand_embs[c][a] for c, a in enumerate(st_ans)])
        rows.append({
            "seed": s,
            "lived": hits(lived, keys, ans, cand_embs),
            "wiped": hits(wiped, keys, ans, cand_embs),
            "stranger_on_lived_answers": hits(stranger, keys, ans, cand_embs),
            "stranger_same_answers": sum(int(a == b) for a, b in zip(ans, st_ans)),
        })
    comp1 = mean_loss(model, tok, COMPETENCE)
    tot = {k: sum(r[k] for r in rows) for k in ("lived", "wiped", "stranger_on_lived_answers", "stranger_same_answers")}
    n = 3 * len(seeds)
    return {
        "seeds": list(seeds), "probes": n, "per_seed": rows, "totals": tot,
        "backbone_hash_before": h0, "backbone_hash_after": backbone_hash(model),
        "competence_loss_before": comp0, "competence_loss_after": comp1,
        "competence_delta": abs(comp1 - comp0),
        "passed_grok_rule": (tot["lived"] >= 0.8 * n and tot["wiped"] <= 0.2 * n and tot["lived"] > tot["wiped"]
                             and h0 == backbone_hash(model) and abs(comp1 - comp0) < 1e-5),
    }
