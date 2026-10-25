"""Gate v3 life loop: one being, one stream, checkpoints with probes. Used by dev and sealed runners."""
from __future__ import annotations

import random
import time

import torch

from being.competence import load_sentences, mean_loss
from being.v3.core import Being3, V3Config, backbone_hash, exact, first_answer
from being.v3.lifegen import ATTRS, COMPOSE, TRAIN, make_stream, probes_for

ROOT_BG = ("background", "background_modern")


def background_pool():
    from being.gate_v2 import _background
    return _background("mixed")


def candidates_for(f, pools, k=50):
    """Forced-choice candidates: the true value plus k-1 distractors from the SAME attribute's closed pool
    (all 49 others for the 50-value pools; a fixed per-fact sample of 49 names for 'sister'). Chance = 1/k."""
    others = [v for v in pools[f.attr] if v != f.value]
    if len(others) > k - 1:
        others = random.Random(f.fid).sample(sorted(others), k - 1)
    return [f.value] + others


@torch.no_grad()
def probe_facts(b: Being3, facts, pools=None, with_paraphrase=True, with_logprob=True):
    rows = []
    for f in facts:
        p = probes_for(f)
        g = b.generate(p["template"])
        row = {"fid": f.fid, "attr": f.attr, "answer": f.value, "template": exact(g, f.value), "template_gen": first_answer(g)}
        if with_paraphrase:
            row["paraphrase"] = [exact(b.generate(q), f.value) for q in p["paraphrase"]]
        if pools is not None:   # primary v3 measure (amendment A): fact-specific forced choice + specific margin
            lps = b.candidates_logprob(p["template"], candidates_for(f, pools))
            true, rest = lps[0], lps[1:]
            row["fc_rank"] = sum(x > true for x in rest)
            row["fc"] = row["fc_rank"] == 0
            row["margin"] = true - sum(rest) / len(rest)
            row["margin_best"] = true - max(rest)     # vs the strongest same-pool rival: > 0 exactly when forced choice is right
            row["logprob"] = true
        elif with_logprob:
            row["logprob"] = b.answer_logprob(p["template"], f.value)
        rows.append(row)
    return rows


@torch.no_grad()
def probe_composition(b: Being3, items):
    return [{"hops": c["hops"], "ok": exact(b.generate(c["prompt"]), c["answer"])} for c in items]


def self_recall_cues(b: Being3, n: int, rng: random.Random, gate: bool, met: list):
    """G: the being recalls OLD lives from its own memory. No stored text from past lives is used.
    Cue = the NAME of someone it met in an earlier life (names only, never attributes or values; Grok ruling E,
    DAV-105: inside the no-diary rule, like recognising a familiar face). It answers its own attribute cues; with the
    gate, an answer is rehearsed only if >=3 of 4 phrasings agree (self-consistency) and it is a short phrase."""
    out = []
    for _ in range(n if met else 0):
        name = rng.choice(met)
        a = rng.choice(ATTRS)
        cues = [t[: t.index("{v}")].format(e=name).rstrip() for t in TRAIN[a] if t.index("{v}") > 0]
        answers = [first_answer(b.generate(c, 8)) for c in cues]
        best = max(set(answers), key=answers.count)
        if gate and (answers.count(best) < min(3, len(answers)) or not best or len(best.split()) > 3):
            continue
        out.append(TRAIN[a][0].format(e=name, v=best))
    return out


def live_stream(model, tok, cfg: V3Config, seed: int, n_lives: int = 100, checkpoints=(1, 10, 25, 50, 100),
                probe_lives=(1, 10, 50), log=print, competence=None, probe_stream=None):
    st = make_stream(seed, n_lives=n_lives)
    ps = probe_stream or st          # stranger arm: lives `st`, is probed on another stream's facts
    rng = random.Random(seed * 7 + 3)
    bg = background_pool()
    b = Being3(model, tok, cfg, birth_seed=seed)
    b.birth(bg[:300])
    h0 = backbone_hash(model)
    rows = {"seed": seed, "arm": cfg.arm, "trainable_params": b.trainable_params(), "backbone_start": h0,
            "checkpoints": [], "replayed": 0, "replay_attempts": 0}
    met = []                                              # names of people met in earlier lives (G cue list)
    t0 = time.time()
    for life_no, life in enumerate(st.lives, start=1):
        day = [t for f in life for t in f.train_texts]
        rng.shuffle(day)
        for t in day:                                     # awake
            b.step(t, teacher_text=rng.choice(bg))        # teacher_text only used by L (KL on every step)
            for _ in range(cfg.rehearsal):
                b.step(rng.choice(bg), teacher_text=rng.choice(bg))
        for _ in range(cfg.night_passes):                 # night: today's buffer, then discarded
            order = list(day)
            rng.shuffle(order)
            replay = self_recall_cues(b, cfg.replay_cues, rng, cfg.g_gate, met) if cfg.arm == "G" and life_no > 1 else []
            rows["replayed"] += len(replay)
            rows["replay_attempts"] += cfg.replay_cues if cfg.arm == "G" and life_no > 1 and met else 0
            for i, t in enumerate(order):
                b.step(t, teacher_text=rng.choice(bg))
                for r in range(cfg.rehearsal):
                    if replay and r == 0 and i % 2 == 0:  # G: replay replaces half the background rehearsal
                        b.step(replay[(i // 2) % len(replay)])
                    else:
                        b.step(rng.choice(bg), teacher_text=rng.choice(bg))
        del day
        met.extend(sorted({f.entity for f in life} - set(met)))
        if life_no in checkpoints:
            cp = {"after_life": life_no, "seconds": round(time.time() - t0, 1), "memory_sha256": b.memory_hash()}
            for pl in probe_lives:
                if pl <= life_no:
                    cp[f"life{pl}"] = probe_facts(b, ps.lives[pl - 1], pools=ps.pools)
            done = [c for c in ps.composition if c["life_b"] <= life_no]
            cp["composition"] = probe_composition(b, done[:40])
            neg = [n for n in ps.negatives if int(n["sister_fid"][1:4]) <= life_no][:20]
            cp["negatives_answered"] = sum(bool(first_answer(b.generate(n["prompt"])) ) for n in neg)
            if competence:
                cp["ppl_loss"] = {k: mean_loss(model, tok, v) for k, v in competence.items()}   # memory active
            rows["checkpoints"].append(cp)
            log(f"[{cfg.arm} seed {seed}] life {life_no}: " + ", ".join(
                f"life{pl} t{sum(r['template'] for r in cp[f'life{pl}'])} fc{sum(r['fc'] for r in cp[f'life{pl}'])}"
                f" m{sum(r['margin'] for r in cp[f'life{pl}']) / 10:.1f}" for pl in probe_lives if f"life{pl}" in cp)
                + f", comp {sum(c['ok'] for c in cp['composition'])}/{len(cp['composition'])}, {cp['seconds']}s")
    rows["backbone_end"] = backbone_hash(model)
    rows["final_state_sha256"] = b.memory_hash()
    b.close()
    return rows
