"""BIND-I1: controlled three-condition interference intervention (Claude + Sol, 2026-10-26). DEV seeds 2-5 only.

Question: does teaching another entity B a value W for the SAME attribute as an earlier, entity-specific memory A->V
damage A's preference for V more than teaching the same W under an unrelated attribute, or unrelated material?

Per seed (one controlled synthetic stream, 5 lives x 10 facts, 4 training phrasings each, Gate v3 phrasings):
  life 1  : 4 TARGET facts (A_i, T_i, V_i) + 4 COUNTER facts (C_i, T_i, Z_i) + 2 fillers. Each target attribute has two
            entities with different values, so life-1 acquisition is binding-sensitive (Sol, review of b7c87a8).
  life 2  : 4 TREATMENT facts, entity B_i (new), + 6 fillers. By condition:
              S  (B_i, T_i, W_i)   same attribute as the target, new competing value W_i
              X  (B_i, X_i, W_i)   the SAME value W_i under an unrelated attribute X_i
              U  (B_i, X_i, U_i)   an unrelated value U_i (token count matched to W_i) under X_i
  lives 3-5: 10 filler facts each. Fillers never use target attributes and are identical across conditions.
            A_i, C_i, V_i, Z_i never reappear after life 1 (no target rehearsal).
Values are invented neutral words (no attribute-shaped suffix) so the same W fits both T_i and X_i.

Training is the Gate v3 life loop (same step, rehearsal, night passes, KL teacher), copied verbatim from
being/v3/run.py live_stream so it can branch: life 1 is trained ONCE; then LoRA weights, AdamW state, the Python RNG and
the torch RNGs are snapshotted and restored for each condition, so S, X and U differ only in the treatment facts.
Branch integrity is recorded and asserted: optimiser step counts per life, training-token counts, and a hash of the
exact sequence of background and teacher texts (life-text positions marked, not hashed) must match across conditions.

Probes at checkpoints 1 (shared, pre-branch), 2, 3, 5 per condition, and on the wiped backbone. One fixed 50-candidate
pool per target i: [V_i, Z_i, W_i, U_i, 46 never-taught neutral words]; all scores saved.
  A_i under the T_i template: forced choice over the pool, template recall, l(V|A), l(Z|A), l(W|A), l(U|A).
  C_i under the T_i template: same pool (correct = Z_i), l(Z|C), l(V|C)  -> bidirectional life-1 binding.
  B_i under the T_i template (transfer control in X/U, the taught query in S): l(V|B), l(Z|B), l(W|B), l(U|B).
  treatment acquisition, all three conditions' treatment facts at every checkpoint, each under the attribute it is
    taught with: forced choice over the same pool (correct = W_i, or U_i in U), log-prob and margin_best.
  competence monitors (eval, modern) and memory hash.

  python scripts/bind_i1.py --seed 2 --out results/dev3/bind_i1/seed2.json
"""
import argparse
import copy
import hashlib
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
N_TARGETS, FACTS_PER_LIFE, N_LIVES, N_POOL = 4, 10, 5, 50
NAMED = ("V", "Z", "W", "U")                                   # pool positions 0..3
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
    C = [name() for _ in range(N_TARGETS)]
    B = [name() for _ in range(N_TARGETS)]
    V = [word() for _ in range(N_TARGETS)]
    Z = [word() for _ in range(N_TARGETS)]
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
    neutrals = [[word() for _ in range(N_POOL - len(NAMED))] for _ in range(N_TARGETS)]

    def fillers(life, n):
        out = []
        while len(out) < n:
            e = name()
            for a in rng.sample(filler_attrs, min(len(filler_attrs), rng.choice((2, 3)))):
                if len(out) < n:
                    out.append(fact(f"I{seed}F{life:02d}{len(out):02d}", life, e, a, word()))
        return out

    targets = [fact(f"I{seed}L01T{i}", 1, A[i], T[i], V[i]) for i in range(N_TARGETS)]
    counters = [fact(f"I{seed}L01C{i}", 1, C[i], T[i], Z[i]) for i in range(N_TARGETS)]
    life1 = targets + counters + fillers(1, FACTS_PER_LIFE - 2 * N_TARGETS)
    fill2 = fillers(2, FACTS_PER_LIFE - N_TARGETS)
    later = {L: fillers(L, FACTS_PER_LIFE) for L in range(3, N_LIVES + 1)}
    treatment = {
        "S": [fact(f"I{seed}L02B{i}", 2, B[i], T[i], W[i]) for i in range(N_TARGETS)],
        "X": [fact(f"I{seed}L02B{i}", 2, B[i], X[i], W[i]) for i in range(N_TARGETS)],
        "U": [fact(f"I{seed}L02B{i}", 2, B[i], X[i], U[i]) for i in range(N_TARGETS)],
    }
    lives = {c: [life1, treatment[c] + fill2] + [later[L] for L in range(3, N_LIVES + 1)] for c in CONDITIONS}
    design = {"seed": seed, "T": T, "X": X, "extra": extra, "A": A, "C": C, "B": B, "V": V, "Z": Z, "W": W, "U": U,
              "pool": [[V[i], Z[i], W[i], U[i]] + neutrals[i] for i in range(N_TARGETS)],
              "ntok": ({k: [ntok(w) for w in vs] for k, vs in (("V", V), ("Z", Z), ("W", W), ("U", U))}
                       if ntok else None)}
    return {"lives": lives, "targets": targets, "counters": counters, "treatment": treatment, "design": design}


def check_stream(st, lifegen):
    """Structural guarantees the analysis relies on (also used by the tests)."""
    d = st["design"]
    life1_only = set(d["A"]) | set(d["C"])
    for c in CONDITIONS:
        L = st["lives"][c]
        assert len(L) == N_LIVES and all(len(life) == FACTS_PER_LIFE for life in L), "facts per life"
        for life in L[1:]:
            for f in life:
                assert f.entity not in life1_only and f.value not in d["V"] + d["Z"], "target rehearsal after life 1"
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
        a, cf = st["targets"][i], st["counters"][i]
        assert a.attr == cf.attr == d["T"][i] and a.entity != cf.entity and a.value != cf.value
        s, x, u = (st["treatment"][c][i] for c in CONDITIONS)
        assert s.attr == d["T"][i] and x.attr == u.attr == d["X"][i] and s.value == x.value == d["W"][i]
        assert u.value == d["U"][i] and s.entity == x.entity == u.entity == d["B"][i]
        pool = d["pool"][i]
        assert len(set(v.lower() for v in pool)) == N_POOL and pool[:4] == [d[k][i] for k in NAMED]
    taught = {f.value.lower() for c in CONDITIONS for life in st["lives"][c] for f in life}
    for i in range(N_TARGETS):
        assert not taught & {v.lower() for v in d["pool"][i][len(NAMED):]}, "neutral was taught"
    assert len(set(d["T"]) | set(d["X"])) == 2 * N_TARGETS and not set(d["T"]) & set(lifegen.RELATIONAL)
    return True


def treat_index(cond):
    return NAMED.index("U" if cond == "U" else "W")


def main():
    import torch
    from being.competence import load_sentences, mean_loss
    from being.v3 import lifegen
    from being.v3.core import Being3, V3Config, backbone_hash, exact, load_qwen
    from being.v3.lifegen import probes_for
    from being.v3.run import background_pool
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_v3_dev", ROOT / "scripts/run_v3_dev.py")
    dev = importlib.util.module_from_spec(spec)
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
           "backbone_sha256": backbone_hash(model), "base_ppl_loss": base, "design": d,
           "streams": {c: [[asdict(f) for f in life] for life in st["lives"][c]] for c in CONDITIONS},
           "train_tokens": {c: [sum(len(tok(t).input_ids) for f in life for t in f.train_texts)
                                for life in st["lives"][c]] for c in CONDITIONS},
           "status": "in_progress"}
    dev.write_atomic(out, res)

    def fc_row(lps, k):
        rest = lps[:k] + lps[k + 1:]
        return {"fc": sum(x > lps[k] for x in rest) == 0, "ties": sum(x == lps[k] for x in rest),
                "lp": lps[k], "margin_best": lps[k] - max(rest)}

    @torch.no_grad()
    def probe(b, L):
        rows = []
        for i, (f, cf) in enumerate(zip(st["targets"], st["counters"])):
            pool = d["pool"][i]
            p = probes_for(f)["template"]
            lps = b.candidates_logprob(p, pool)
            g = b.generate(p)
            lc = b.candidates_logprob(probes_for(cf)["template"], pool)
            lb = b.candidates_logprob(probes_for(st["treatment"]["S"][i])["template"], pool[:4])  # B_i under T_i
            row = {"i": i, "fid": f.fid, "attr": f.attr, "template": exact(g, f.value),
                   "cands": [{"v": v, "lp": lp} for v, lp in zip(pool, lps)], **fc_row(lps, 0),
                   "lV_A": lps[0], "lZ_A": lps[1], "lW_A": lps[2], "lU_A": lps[3],
                   "counter": {"fid": cf.fid, **fc_row(lc, 1), "lZ_C": lc[1], "lV_C": lc[0],
                               "cands": [{"v": v, "lp": lp} for v, lp in zip(pool, lc)]},
                   "lV_B": lb[0], "lZ_B": lb[1], "lW_B": lb[2], "lU_B": lb[3], "treat": {}}
            for c in CONDITIONS:                                 # every condition's treatment fact, its taught attribute
                tf = st["treatment"][c][i]
                tl = b.candidates_logprob(probes_for(tf)["template"], pool)
                row["treat"][c] = {"fid": tf.fid, "attr": tf.attr, "value": tf.value, **fc_row(tl, treat_index(c)),
                                   "lps_named": tl[:4]}
            rows.append(row)
        return {"after_life": L, "targets": rows,
                "ppl_loss": {k: mean_loss(model, tok, v) for k, v in comp.items()} if L > 0 else None}

    trace = {"h": None, "day": set(), "n": 0}

    def run_life(b, rng, bg, life):                              # verbatim from being/v3/run.py live_stream (L arm)
        trace["h"], trace["n"] = hashlib.sha256(), 0
        day = [t for f in life for t in f.train_texts]
        trace["day"] = set(day)
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
        return {"steps": trace["n"], "bg_teacher_sha256": trace["h"].hexdigest()}

    t0 = time.time()
    res["wiped"] = probe(dev._Wiped(model, tok, cfg), 0)
    bg = background_pool()
    rng = random.Random(a.seed * 7 + 3)
    b = Being3(model, tok, cfg, birth_seed=a.seed)
    b.birth(bg[:300])
    _step = b.step

    def traced(text, teacher_text=None):                         # records the step sequence; behaviour unchanged
        trace["n"] += 1
        life_text = text in trace["day"]
        trace["h"].update(repr((trace["n"], "LIFE" if life_text else text, teacher_text)).encode())
        return _step(text, teacher_text=teacher_text)
    b.step = traced
    h0 = backbone_hash(model)
    res["life1_trace"] = run_life(b, rng, bg, st["lives"]["S"][0])    # life 1 is identical in all conditions
    cp1 = probe(b, 1)
    cp1["memory_sha256"] = b.memory_hash()
    res["checkpoint1"] = cp1
    snap = {"lora": b.state(), "opt": copy.deepcopy(b.opt.state_dict()), "rng": rng.getstate(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    res["conditions"], res["traces"], res["rng_after"] = {}, {}, {}
    for cond in CONDITIONS:
        b.load_state(snap["lora"])
        b.opt.load_state_dict(copy.deepcopy(snap["opt"]))
        rng.setstate(snap["rng"])
        torch.set_rng_state(snap["torch"])
        if snap["cuda"] is not None:
            torch.cuda.set_rng_state_all(snap["cuda"])
        assert b.memory_hash() == cp1["memory_sha256"], "branch did not restore the life-1 state"
        cps, traces = [], []
        for L in range(2, N_LIVES + 1):
            traces.append(run_life(b, rng, bg, st["lives"][cond][L - 1]))
            if L in CHECKPOINTS:
                cp = probe(b, L)
                cp["memory_sha256"] = b.memory_hash()
                cps.append(cp)
        res["conditions"][cond], res["traces"][cond] = cps, traces
        res["rng_after"][cond] = hashlib.sha256(repr(rng.getstate()).encode()).hexdigest()
        dev.write_atomic(out, res)
        print(time.strftime("%H:%M:%S"), f"seed {a.seed} condition {cond} done", flush=True)
    assert len({json.dumps(v, sort_keys=True) for v in res["traces"].values()}) == 1, \
        "step counts or background/teacher sequences differ between conditions"
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
