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


@torch.no_grad()
def probe_facts(b: Being3, facts, with_paraphrase=True, with_logprob=True):
    rows = []
    for f in facts:
        p = probes_for(f)
        g = b.generate(p["template"])
        row = {"fid": f.fid, "attr": f.attr, "answer": f.value, "template": exact(g, f.value), "template_gen": first_answer(g)}
        if with_paraphrase:
            row["paraphrase"] = [exact(b.generate(q), f.value) for q in p["paraphrase"]]
        if with_logprob:
            row["logprob"] = b.answer_logprob(p["template"], f.value)
        rows.append(row)
    return rows


@torch.no_grad()
def probe_composition(b: Being3, items):
    return [{"hops": c["hops"], "ok": exact(b.generate(c["prompt"]), c["answer"])} for c in items]


def self_recall_cues(b: Being3, n: int, rng: random.Random, gate: bool):
    """G: the being recalls OLD lives from its own memory. No stored text from past lives is used.
    It names a person it knows, then answers its own attribute cues; with the gate, an answer is rehearsed only if
    >=3 of 4 phrasings of the cue agree (self-consistency) and the answer is a single short phrase (template filter)."""
    out = []
    for _ in range(n):
        name = first_answer(b.generate("Someone I know well is named", 6))
        if len(name.split()) != 2 or not all(w[:1].isupper() for w in name.split()):
            continue
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
            "checkpoints": [], "replayed": 0}
    t0 = time.time()
    for life_no, life in enumerate(st.lives, start=1):
        day = [t for f in life for t in f.train_texts]
        rng.shuffle(day)
        for t in day:                                     # awake
            b.step(t)
            for _ in range(cfg.rehearsal):
                b.step(rng.choice(bg), teacher_text=rng.choice(bg))
        for _ in range(cfg.night_passes):                 # night: today's buffer, then discarded
            order = list(day)
            rng.shuffle(order)
            replay = self_recall_cues(b, cfg.replay_cues, rng, cfg.g_gate) if cfg.arm == "G" and life_no > 1 else []
            rows["replayed"] += len(replay)
            for i, t in enumerate(order):
                b.step(t)
                for r in range(cfg.rehearsal):
                    if replay and r == 0 and i % 2 == 0:  # G: replay replaces half the background rehearsal
                        b.step(replay[(i // 2) % len(replay)])
                    else:
                        b.step(rng.choice(bg), teacher_text=rng.choice(bg))
        del day
        if life_no in checkpoints:
            cp = {"after_life": life_no, "seconds": round(time.time() - t0, 1), "memory_sha256": b.memory_hash()}
            for pl in probe_lives:
                if pl <= life_no:
                    cp[f"life{pl}"] = probe_facts(b, ps.lives[pl - 1])
            done = [c for c in ps.composition if c["life_b"] <= life_no]
            cp["composition"] = probe_composition(b, done[:40])
            neg = [n for n in ps.negatives if int(n["sister_fid"][1:4]) <= life_no][:20]
            cp["negatives_answered"] = sum(bool(first_answer(b.generate(n["prompt"])) ) for n in neg)
            if competence:
                cp["ppl_loss"] = {k: mean_loss(model, tok, v) for k, v in competence.items()}   # memory active
            rows["checkpoints"].append(cp)
            log(f"[{cfg.arm} seed {seed}] life {life_no}: " + ", ".join(
                f"life{pl} {sum(r['template'] for r in cp[f'life{pl}'])}/10" for pl in probe_lives if f"life{pl}" in cp)
                + f", comp {sum(c['ok'] for c in cp['composition'])}/{len(cp['composition'])}, {cp['seconds']}s")
    rows["backbone_end"] = backbone_hash(model)
    rows["final_state_sha256"] = b.memory_hash()
    b.close()
    return rows
