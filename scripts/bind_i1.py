"""BIND-I1: controlled three-condition interference intervention (Claude + Sol, 2026-10-26). DEV seeds 2-5 only.

Question: does teaching another entity B a value W for the SAME attribute as an earlier memory A->V damage A's preference
for V more than exposing the model to the same W under an unrelated attribute, or to unrelated material?

Per seed (one controlled synthetic stream, 5 lives x 10 facts, 4 training phrasings each, Gate v3 phrasings):
  life 1  : 4 TARGET facts (A_i, T_i, V_i), T_i distinct attributes, + 6 filler facts on non-target attributes.
  life 2  : 4 TREATMENT facts, one per target, entity B_i (new, distinct), + 6 filler facts. By condition:
              S  (B_i, T_i, W_i)   same attribute as the target, competing value W_i
              X  (B_i, X_i, W_i)   the SAME value W_i under an unrelated attribute X_i
              U  (B_i, X_i, U_i)   an unrelated value U_i (token count matched to W_i) under X_i
  lives 3-5: 10 filler facts each. Fillers never use target attributes; fillers and all other text are identical
            across conditions. A_i and V_i never reappear after life 1 (no target rehearsal).
Values are invented neutral words (no attribute-shaped suffix) so the same W fits both T_i and X_i. Target and X
attributes are drawn per seed from the nine non-relational attributes.

Training is the Gate v3 life loop (same step, rehearsal, night passes, KL teacher), copied verbatim from
being/v3/run.py live_stream so it can branch: life 1 is trained ONCE; then LoRA weights, AdamW state and the Python RNG
are snapshotted and restored for each condition, so S, X and U differ only in the treatment facts.

Probes at checkpoints 1 (shared, pre-branch), 2, 3, 5 per condition, and on the wiped backbone:
  target i (template of T_i for A_i): forced choice over a FIXED 50-candidate set [V_i, W_i, U_i, 47 never-taught
    neutral words], all scores saved; template recall; l(V|A), l(W|A), l(U|A).
  paired: l(V|B_i), l(W|B_i), l(U|B_i) under the T_i template (B's query for the target attribute).
  treatment acquisition: the treatment fact's own forced choice (its value vs 49 never-taught neutrals) and log-probs.
  competence monitors (eval, modern) and memory hash.

  python scripts/bind_i1.py --seed 2 --out results/dev3/bind_i1/seed2.json
"""
import argparse
import copy
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = {2, 3, 4, 5}
CONDITIONS = ("S", "X", "U")
CHECKPOINTS = (1, 2, 3, 5)
N_TARGETS, FACTS_PER_LIFE, N_LIVES, N_DISTRACT = 4, 10, 5, 47
CFG = {"lr": 1e-4, "kl_weight": 1.0, "night_passes": 4}        # L kl1.0 lr1e-4 (the D1 arm)


def build_stream(seed, lifegen, ntok=None):
    """Deterministic controlled stream for one seed. ntok(word) -> token count (tokenizer); None skips matching (tests).
    Returns a dict with per-condition lives (lists of Fact) and the design."""
    rng = random.Random(f"BIND-I1-{seed}")
    attrs = [a for a in lifegen.ATTRS if a not in lifegen.RELATIONAL]
    rng.shuffle(attrs)
    T, X, extra = attrs[:N_TARGETS], attrs[N_TARGETS:2 * N_TARGETS], attrs[2 * N_TARGETS:]
    filler_attrs = X + extra                                     # fillers never touch target attributes
    used = set()

    def word():
        while True:
            w = lifegen._word(rng, rng.choice((2, 3))).capitalize()
            if w.lower() not in used and len(w) >= 4:
                used.add(w.lower())
                return w

    def name():
        while True:
            n = f"{lifegen._word(rng, 2).capitalize()} {lifegen._word(rng, rng.choice((1, 2))).capitalize()}"
            if n.lower() not in used and n.split()[0].lower() != "david":
                used.add(n.lower())
                return n

    def fact(fid, life, e, a, v):
        return lifegen.Fact(fid, life, e, a, v, [t.format(e=e, v=v) for t in lifegen.TRAIN[a]])

    A = [name() for _ in range(N_TARGETS)]
    B = [name() for _ in range(N_TARGETS)]
    V = [word() for _ in range(N_TARGETS)]
    W = [word() for _ in range(N_TARGETS)]
    U = []
    for w in W:                                                  # U_i token-count matched to W_i when a tokenizer is given
        for _ in range(500):
            u = word()
            if ntok is None or ntok(u) == ntok(w):
                break
        else:
            raise RuntimeError(f"no token-matched U for {w}")
        U.append(u)
    distract = {i: [word() for _ in range(N_DISTRACT)] for i in range(N_TARGETS)}
    treat_distract = {i: [word() for _ in range(N_DISTRACT + 2)] for i in range(N_TARGETS)}

    def fillers(life, n):
        out = []
        while len(out) < n:
            e = name()
            for a in rng.sample(filler_attrs, min(len(filler_attrs), rng.choice((2, 3)))):
                if len(out) < n:
                    out.append(fact(f"I{seed}F{life:02d}{len(out):02d}", life, e, a, word()))
        return out

    targets = [fact(f"I{seed}L01T{i}", 1, A[i], T[i], V[i]) for i in range(N_TARGETS)]
    life1 = targets + fillers(1, FACTS_PER_LIFE - N_TARGETS)
    fill2 = fillers(2, FACTS_PER_LIFE - N_TARGETS)
    later = {L: fillers(L, FACTS_PER_LIFE) for L in range(3, N_LIVES + 1)}
    treatment = {
        "S": [fact(f"I{seed}L02B{i}", 2, B[i], T[i], W[i]) for i in range(N_TARGETS)],
        "X": [fact(f"I{seed}L02B{i}", 2, B[i], X[i], W[i]) for i in range(N_TARGETS)],
        "U": [fact(f"I{seed}L02B{i}", 2, B[i], X[i], U[i]) for i in range(N_TARGETS)],
    }
    lives = {c: [life1, treatment[c] + fill2] + [later[L] for L in range(3, N_LIVES + 1)] for c in CONDITIONS}
    design = {"seed": seed, "T": T, "X": X, "extra": extra, "A": A, "B": B, "V": V, "W": W, "U": U,
              "target_candidates": [[V[i], W[i], U[i]] + distract[i] for i in range(N_TARGETS)],
              "treatment_distractors": [treat_distract[i] for i in range(N_TARGETS)],
              "ntok": ({"W": [ntok(w) for w in W], "U": [ntok(u) for u in U], "V": [ntok(v) for v in V]}
                       if ntok else None)}
    return {"lives": lives, "targets": targets, "treatment": treatment, "design": design}


def check_stream(st, lifegen):
    """Structural guarantees the analysis relies on (also used by the tests)."""
    d = st["design"]
    for c in CONDITIONS:
        L = st["lives"][c]
        assert len(L) == N_LIVES and all(len(life) == FACTS_PER_LIFE for life in L), "facts per life"
        for life in L[1:]:
            for f in life:
                assert f.entity not in d["A"] and f.value not in d["V"], "target rehearsal after life 1"
        for life in L:
            for f in life:
                if f.fid.startswith(f"I{d['seed']}F"):
                    assert f.attr not in d["T"], "filler on a target attribute"
    for L in range(N_LIVES):
        if L == 1:
            continue
        assert [f.fid for f in st["lives"]["S"][L]] == [f.fid for f in st["lives"]["X"][L]] == \
               [f.fid for f in st["lives"]["U"][L]], "non-treatment lives must be identical"
    for c in CONDITIONS:
        assert [f.train_texts for f in st["lives"][c][1][N_TARGETS:]] == \
               [f.train_texts for f in st["lives"]["S"][1][N_TARGETS:]], "life-2 fillers identical"
    for i in range(N_TARGETS):
        s, x, u = (st["treatment"][c][i] for c in CONDITIONS)
        assert s.attr == d["T"][i] and x.attr == u.attr == d["X"][i] and s.value == x.value == d["W"][i]
        assert u.value == d["U"][i] and s.entity == x.entity == u.entity == d["B"][i]
        assert len(set(d["target_candidates"][i])) == 50 and d["target_candidates"][i][:3] == [d["V"][i], d["W"][i], d["U"][i]]
    taught = {f.value.lower() for c in CONDITIONS for life in st["lives"][c] for f in life}
    for i in range(N_TARGETS):
        assert not taught & {v.lower() for v in d["target_candidates"][i][3:]}, "distractor was taught"
        assert not taught & {v.lower() for v in d["treatment_distractors"][i]}, "treatment distractor was taught"
    assert len(set(d["T"]) | set(d["X"])) == 2 * N_TARGETS and not set(d["T"]) & set(lifegen.RELATIONAL)
    return True


def main():
    import torch
    from being.competence import load_sentences, mean_loss
    from being.v3 import lifegen
    from being.v3.core import Being3, V3Config, backbone_hash, exact, load_qwen
    from being.v3.lifegen import probes_for
    from being.v3.run import background_pool
    spec = __import__("importlib.util").util.spec_from_file_location("run_v3_dev", ROOT / "scripts/run_v3_dev.py")
    dev = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(dev)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seed not in SEEDS:
        sys.exit("BIND-I1 uses dev seeds 2-5 only")
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    model, tok = load_qwen()
    ntok = lambda w: len(tok(" " + w).input_ids)
    st = build_stream(a.seed, lifegen, ntok)
    check_stream(st, lifegen)
    cfg = V3Config(arm="L", **CFG)
    comp = {k: load_sentences(k) for k in ("eval", "modern")}
    base = {k: mean_loss(model, tok, v) for k, v in comp.items()}
    d = st["design"]
    res = {"which": "BIND-I1", "commit": commit, "config": asdict(cfg), "seed": a.seed,
           "backbone_sha256": backbone_hash(model), "base_ppl_loss": base,
           "design": {k: v for k, v in d.items()},
           "streams": {c: [[asdict(f) for f in life] for life in st["lives"][c]] for c in CONDITIONS},
           "status": "in_progress"}
    dev.write_atomic(out, res)

    @torch.no_grad()
    def probe(b, L, cond):
        rows = []
        for i, f in enumerate(st["targets"]):
            p = probes_for(f)["template"]
            cands = d["target_candidates"][i]
            lps = b.candidates_logprob(p, cands)
            g = b.generate(p)
            bf = st["treatment"]["S"][i]                         # B_i under the TARGET attribute T_i
            pb = probes_for(bf)["template"]
            lb = b.candidates_logprob(pb, [d["V"][i], d["W"][i], d["U"][i]])
            row = {"i": i, "fid": f.fid, "attr": f.attr, "template": exact(g, f.value),
                   "cands": [{"v": v, "lp": lp} for v, lp in zip(cands, lps)],
                   "fc": sum(x > lps[0] for x in lps[1:]) == 0, "fc_ties": sum(x == lps[0] for x in lps[1:]),
                   "lV_A": lps[0], "lW_A": lps[1], "lU_A": lps[2], "margin_best": lps[0] - max(lps[1:]),
                   "lV_B": lb[0], "lW_B": lb[1], "lU_B": lb[2]}
            if cond is not None and L >= 2:                     # treatment acquisition in this condition
                tf = st["treatment"][cond][i]
                tp = probes_for(tf)["template"]
                tl = b.candidates_logprob(tp, [tf.value] + d["treatment_distractors"][i])
                row["treat"] = {"fid": tf.fid, "attr": tf.attr, "value": tf.value,
                                "fc": sum(x > tl[0] for x in tl[1:]) == 0, "lp": tl[0], "margin_best": tl[0] - max(tl[1:])}
            rows.append(row)
        return {"after_life": L, "targets": rows,
                "ppl_loss": {k: mean_loss(model, tok, v) for k, v in comp.items()} if L > 0 else None}

    def run_life(b, rng, bg, life):                              # verbatim from being/v3/run.py live_stream (L arm)
        n0 = b_steps[0]
        day = [t for f in life for t in f.train_texts]
        rng.shuffle(day)
        for t in day:
            b.step(t, teacher_text=rng.choice(bg))
            for _ in range(cfg.rehearsal):
                b.step(rng.choice(bg), teacher_text=rng.choice(bg))
        for _ in range(cfg.night_passes):
            order = list(day)
            rng.shuffle(order)
            for i, t in enumerate(order):
                b.step(t, teacher_text=rng.choice(bg))
                for r in range(cfg.rehearsal):
                    b.step(rng.choice(bg), teacher_text=rng.choice(bg))
        return b_steps[0] - n0

    t0 = time.time()
    res["wiped"] = probe(dev._Wiped(model, tok, cfg), 0, None)
    bg = background_pool()
    rng = random.Random(a.seed * 7 + 3)
    b = Being3(model, tok, cfg, birth_seed=a.seed)
    b.birth(bg[:300])
    b_steps = [0]
    _step = b.step

    def counted(*x, **k):                                        # counts optimiser steps; behaviour unchanged
        b_steps[0] += 1
        return _step(*x, **k)
    b.step = counted
    h0 = backbone_hash(model)
    res["steps_life1"] = run_life(b, rng, bg, st["lives"]["S"][0])   # life 1 is identical in all conditions
    cp1 = probe(b, 1, None)
    cp1["memory_sha256"] = b.memory_hash()
    res["checkpoint1"] = cp1
    snap = {"lora": b.state(), "opt": copy.deepcopy(b.opt.state_dict()), "rng": rng.getstate(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    res["conditions"] = {}
    for cond in CONDITIONS:
        b.load_state(snap["lora"])
        b.opt.load_state_dict(copy.deepcopy(snap["opt"]))
        rng.setstate(snap["rng"])
        torch.set_rng_state(snap["torch"])
        if snap["cuda"] is not None:
            torch.cuda.set_rng_state_all(snap["cuda"])
        assert b.memory_hash() == cp1["memory_sha256"], "branch did not restore the life-1 state"
        cps, steps = [], []
        for L in range(2, N_LIVES + 1):
            steps.append(run_life(b, rng, bg, st["lives"][cond][L - 1]))
            if L in CHECKPOINTS:
                cp = probe(b, L, cond)
                cp["memory_sha256"] = b.memory_hash()
                cps.append(cp)
        res["conditions"][cond] = cps
        res.setdefault("steps", {})[cond] = steps
        res.setdefault("rng_after", {})[cond] = __import__("hashlib").sha256(repr(rng.getstate()).encode()).hexdigest()
        dev.write_atomic(out, res)
        print(time.strftime("%H:%M:%S"), f"seed {a.seed} condition {cond} done", flush=True)
    assert len({tuple(v) for v in res["steps"].values()}) == 1, "step counts differ between conditions"
    assert len(set(res["rng_after"].values())) == 1, "RNG streams diverged between conditions"
    res["backbone_start"], res["backbone_end"] = h0, backbone_hash(model)
    assert h0 == res["backbone_end"] == res["backbone_sha256"], "backbone changed"
    b.close()
    res["seconds"] = round(time.time() - t0, 1)
    res["status"] = "complete"
    dev.write_atomic(out, res)
    print("BIND-I1 SEED DONE", a.seed, res["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
