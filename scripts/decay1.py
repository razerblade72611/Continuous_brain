"""DECAY-1: why do life-1 memories become inaccessible under continued LoRA training? (Claude + Sol, 2026-10-27).
DEV seeds 2-5; the same controlled streams as BIND-I1 (scripts/bind_i1.py), mechanistic follow-up, not confirmation.

Life 1 is trained exactly as in BIND-I1 (same stream, same RNG, same loop), so checkpoint 1 must reproduce the archived
BIND-I1 checkpoint 1. Then the life-1 state (LoRA weights, AdamW state, Python/torch RNGs) is snapshotted and ONE
continuation life is run in five arms:
  A  U-type new facts (BIND-I1's life 2 in condition U)   KL to the frozen backbone, weight 1.0   (= BIND-I1 U, life 2)
  B  the same new facts                                    KL off (weight 0)
  C  background only (40 background sentences in place of the 40 new-fact texts)   KL to the frozen backbone, 1.0
  D  background only                                       KL off
  E  the same new facts                                    KL 1.0 to the END-OF-LIFE-1 model (previous self)
Every arm inherits the AdamW state. The schedule is identical: same step count and positions, same background-rehearsal
and teacher-text draws (KL-off arms draw and ignore them); asserted per arm by a hash of (step, kind, background text,
teacher text). C/D's 40 sentences are chosen by a separate per-seed RNG, so the training RNG is untouched.
E's teacher is the student with its adapters swapped for detached clones of the checkpoint-1 adapters during a no-grad
forward; teacher logits on a fixed sentence are hashed before and after the life and must match.

Measured: checkpoint 1 and checkpoint 2 per arm with the BIND-I1 probe (pool [V, Z, W, U, 46 neutrals]; A, C and B
queries; all three treatment facts), competence, memory hashes. Gradient diagnostic at checkpoint 1 (no optimiser step,
memory hash checked): new-fact CE, background CE and frozen-KL gradients vs the gradients of mean l(V|A) and of mean
[l(V|A) - l(Z|A)], plus the AdamW update direction. End-of-life-2 branch states are saved on the volume (not bundled) for
a separately approved extension.

  python scripts/decay1.py --seed 2 --out results/dev3/decay1/seed2.json --reference results/dev3/bind_i1/seed2.json
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("bind_i1_for_decay1", ROOT / "scripts/bind_i1.py")
i1 = importlib.util.module_from_spec(spec)
sys.modules["bind_i1_for_decay1"] = i1
spec.loader.exec_module(i1)

SEEDS = {2, 3, 4, 5}
ARMS = ("A", "B", "C", "D", "E")
ARM_SPEC = {"A": ("facts", "frozen"), "B": ("facts", "off"), "C": ("background", "frozen"),
            "D": ("background", "off"), "E": ("facts", "previous")}
CFG = dict(i1.CFG)                                             # L, lr 1e-4, KL 1.0, night 4
N_DIAG = 40


def background_day(seed, bg, targets, ntok):
    """C/D day texts: one background sentence per new-fact text, length-matched (closest token count, ties broken by a
    SEPARATE per-seed RNG, never reused). The training RNG is never touched."""
    r = random.Random(f"DECAY1-bgday-{seed}")
    pool = list(bg)
    r.shuffle(pool)
    lens = {s: ntok(s) for s in pool}
    used, out = set(), []
    for t in targets:
        n = ntok(t)
        best = min((s for s in pool if s not in used), key=lambda s: abs(lens[s] - n))
        used.add(best)
        out.append(best)
    return out


def compare_rows(a_rows, b_rows):
    """Max |diff| over every score the BIND-I1 probe records for the targets (A, C, B queries and candidates)."""
    m = 0.0
    for x, y in zip(a_rows, b_rows):
        for k in ("lV_A", "lZ_A", "lW_A", "lU_A", "lV_B", "lZ_B", "lW_B", "lU_B"):
            m = max(m, abs(x[k] - y[k]))
        for k in ("lZ_C", "lV_C"):
            m = max(m, abs(x["counter"][k] - y["counter"][k]))
        for c1, c2 in zip(x["cands"], y["cands"]):
            assert c1["v"] == c2["v"]
            m = max(m, abs(c1["lp"] - c2["lp"]))
    return m


def main():
    import torch
    import torch.nn.functional as F
    from being.competence import load_sentences, mean_loss
    from being.v3 import lifegen
    from being.v3.core import Being3, V3Config, backbone_hash, exact, load_qwen
    from being.v3.lifegen import probes_for
    from being.v3.run import background_pool
    spec = importlib.util.spec_from_file_location("run_v3_dev", ROOT / "scripts/run_v3_dev.py")
    dev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out")
    ap.add_argument("--reference", help="archived BIND-I1 result for this seed (required for a run)")
    ap.add_argument("--states", help="where to save end-of-life-2 branch states (default: next to --out)")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()
    if a.seed not in SEEDS:
        sys.exit("DECAY-1 uses dev seeds 2-5 only")
    if a.preflight:                                            # same prompts/pools as BIND-I1: tokenizer-only check
        from transformers import AutoTokenizer
        from being.v3.core import QWEN
        tok = AutoTokenizer.from_pretrained(QWEN)
        st = i1.build_stream(a.seed, lifegen, lambda w: len(tok(" " + w).input_ids))
        i1.check_stream(st, lifegen)
        pc = i1.prefix_check(st, lifegen, tok)
        print(f"DECAY-1 PREFLIGHT seed {a.seed}: {pc['n']} pairs, prefix failures {pc['fails']}")
        sys.exit(1 if pc["fails"] else 0)
    if not (a.out and a.reference):
        sys.exit("--out and --reference are required")
    out, ref_path = ROOT / a.out, ROOT / a.reference
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    ref = json.loads(ref_path.read_text())
    if ref.get("status") != "complete" or ref.get("seed") != a.seed:
        sys.exit("reference is not a complete BIND-I1 result for this seed")
    states_path = Path(a.states) if a.states else out.with_name(out.stem + "_states.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    model, tok = load_qwen()
    ntok = lambda w: len(tok(" " + w).input_ids)
    st = i1.build_stream(a.seed, lifegen, ntok)
    i1.check_stream(st, lifegen)
    if json.loads(json.dumps(st["design"])) != ref["design"]:
        sys.exit("stream design differs from the BIND-I1 reference")
    pc = i1.prefix_check(st, lifegen, tok)
    cfg = V3Config(arm="L", **CFG)
    comp = {k: load_sentences(k) for k in ("eval", "modern")}
    base = {k: mean_loss(model, tok, v) for k, v in comp.items()}
    d = st["design"]
    bg = background_pool()
    facts_life2 = st["lives"]["U"][1]
    bg_day = background_day(a.seed, bg, [t for f in facts_life2 for t in f.train_texts], lambda t: len(tok(t).input_ids))
    res = {"which": "DECAY-1", "commit": commit, "config": asdict(cfg), "seed": a.seed, "arms": ARM_SPEC,
           "backbone_sha256": backbone_hash(model), "base_ppl_loss": base, "prefix_check": pc,
           "reference": str(a.reference), "reference_commit": ref.get("commit"),
           "life2_facts": [asdict(f) for f in facts_life2], "bg_day": bg_day,
           "train_tokens": {"facts": sum(len(tok(t).input_ids) for f in facts_life2 for t in f.train_texts),
                            "background": sum(len(tok(t).input_ids) for t in bg_day)},
           "status": "in_progress"}
    dev.write_atomic(out, res)

    def fc_row(lps, k):
        rest = lps[:k] + lps[k + 1:]
        return {"fc": sum(x > lps[k] for x in rest) == 0, "ties": sum(x == lps[k] for x in rest),
                "lp": lps[k], "margin_best": lps[k] - max(rest)}

    @torch.no_grad()
    def probe(b, L):                                           # identical calls, order and batching to BIND-I1's probe
        rows = []
        for i, (f, cf) in enumerate(zip(st["targets"], st["counters"])):
            pool = d["pool"][i]
            p = probes_for(f)["template"]
            lps = b.candidates_logprob(p, pool)
            g = b.generate(p)
            lc = b.candidates_logprob(probes_for(cf)["template"], pool)
            lb = b.candidates_logprob(probes_for(st["treatment"]["S"][i])["template"], pool[:4])
            row = {"i": i, "fid": f.fid, "attr": f.attr, "template": exact(g, f.value),
                   "cands": [{"v": v, "lp": lp} for v, lp in zip(pool, lps)], **fc_row(lps, 0),
                   "lV_A": lps[0], "lZ_A": lps[1], "lW_A": lps[2], "lU_A": lps[3],
                   "counter": {"fid": cf.fid, **fc_row(lc, 1), "lZ_C": lc[1], "lV_C": lc[0],
                               "cands": [{"v": v, "lp": lp} for v, lp in zip(pool, lc)]},
                   "lV_B": lb[0], "lZ_B": lb[1], "lW_B": lb[2], "lU_B": lb[3], "treat": {}}
            for c in i1.CONDITIONS:
                tf = st["treatment"][c][i]
                tl = b.candidates_logprob(probes_for(tf)["template"], pool)
                row["treat"][c] = {"fid": tf.fid, "attr": tf.attr, "value": tf.value,
                                   **fc_row(tl, i1.treat_index(c)), "lps_named": tl[:4]}
            rows.append(row)
        return {"after_life": L, "targets": rows,
                "ppl_loss": {k: mean_loss(model, tok, v) for k, v in comp.items()} if L > 0 else None}

    trace = {"h": None, "n": 0}

    def run_life(b, rng, day_texts):                          # the BIND-I1 / live_stream L loop, with step kinds traced
        trace["h"], trace["n"] = hashlib.sha256(), 0

        def step(kind, text, teacher):
            trace["n"] += 1
            trace["h"].update(repr((trace["n"], kind, text if kind == "bg" else "LIFE", teacher)).encode())
            b.step(text, teacher_text=teacher)
        day = list(day_texts)
        rng.shuffle(day)
        for t in day:
            step("life", t, rng.choice(bg))
            for _ in range(cfg.rehearsal):
                step("bg", rng.choice(bg), rng.choice(bg))
        for _ in range(cfg.night_passes):
            order = list(day)
            rng.shuffle(order)
            for i, t in enumerate(order):
                step("life", t, rng.choice(bg))
                for r in range(cfg.rehearsal):
                    step("bg", rng.choice(bg), rng.choice(bg))
        return {"steps": trace["n"], "schedule_sha256": trace["h"].hexdigest()}

    def lp_diff(b, prompt, cand):                             # differentiable summed log-prob (answer_logprob's convention)
        p = b._ids(prompt)
        full = b._ids(prompt + " " + cand)
        lp = F.log_softmax(model(full).logits[0, :-1].float(), -1)
        n = p.shape[1] - 1
        return lp[n:].gather(-1, full[0, 1:][n:, None]).sum()

    def flat_grad(b, loss_fn, items):
        acc = None
        for it in items:
            g = torch.autograd.grad(loss_fn(it), b.lora, allow_unused=True)
            g = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, b.lora)])
            acc = g if acc is None else acc + g
        return acc / len(items)

    def diagnostic(b, rng_snapshot):
        dr = random.Random(f"DECAY1-diag-{a.seed}")
        teach = dr.sample(bg, N_DIAG)
        newfact = [t for f in facts_life2 for t in f.train_texts][:N_DIAG]
        gV = flat_grad(b, lambda i: lp_diff(b, probes_for(st["targets"][i])["template"], d["V"][i]), range(4))
        gB = flat_grad(b, lambda i: lp_diff(b, probes_for(st["targets"][i])["template"], d["V"][i])
                       - lp_diff(b, probes_for(st["targets"][i])["template"], d["Z"][i]), range(4))
        ce = lambda t: model(b._ids(t), labels=b._ids(t)).loss
        losses = {"newfact_ce": flat_grad(b, ce, newfact), "background_ce": flat_grad(b, ce, bg_day[:N_DIAG]),
                  "frozen_kl": flat_grad(b, lambda t: b._kl_to_frozen(t), teach)}
        st_ = b.opt.state
        m = torch.cat([st_[p]["exp_avg"].reshape(-1) for p in b.lora])
        v = torch.cat([st_[p]["exp_avg_sq"].reshape(-1) for p in b.lora])
        stepn = float(st_[b.lora[0]]["step"])
        b1, b2 = b.opt.defaults["betas"]
        upd = (m / (1 - b1 ** stepn)) / ((v / (1 - b2 ** stepn)).sqrt() + b.opt.defaults["eps"])
        losses["adam_update_dir"] = upd                       # parameter change is -lr * this
        cos = lambda x, y: float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-30))
        out = {"norm_gV": float(gV.norm()), "norm_gB": float(gB.norm())}
        for k, g in losses.items():
            out[k] = {"norm": float(g.norm()), "cos_gV": cos(g, gV), "cos_gB": cos(g, gB),
                      "dot_gV": float(torch.dot(g, gV)), "dot_gB": float(torch.dot(g, gB))}
        out["note"] = ("descent on a loss with positive cos_gV lowers l(V|A) to first order; for adam_update_dir the "
                       "parameter change is -lr*dir, so positive cos_gV also means the next update lowers l(V|A)")
        return out

    t0 = time.time()
    res["wiped"] = probe(dev._Wiped(model, tok, cfg), 0)
    rng = random.Random(a.seed * 7 + 3)                        # identical to BIND-I1
    b = Being3(model, tok, cfg, birth_seed=a.seed)
    b.birth(bg[:300])
    h0 = backbone_hash(model)
    res["life1_trace"] = run_life(b, rng, [t for f in st["lives"]["S"][0] for t in f.train_texts])
    cp1 = probe(b, 1)
    cp1["memory_sha256"] = b.memory_hash()
    res["checkpoint1"] = cp1
    snap = {"lora": b.state(), "opt": copy.deepcopy(b.opt.state_dict()), "rng": rng.getstate(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    res["diagnostic"] = diagnostic(b, snap)
    for p in b.lora:
        p.grad = None
    assert b.memory_hash() == cp1["memory_sha256"], "diagnostic changed the memory"
    dev.write_atomic(out, res)

    frozen_kl = b._kl_to_frozen
    prev = [x.detach().clone() for x in snap["lora"]]
    lins = [getattr(layer.self_attn, n) for layer in model.model.layers for n in ("q_proj", "v_proj")]

    def kl_to_previous(text):                                  # teacher = student with checkpoint-1 adapters (no grad)
        ids = b._ids(text)
        live = [lin._lora for lin in lins]
        for j, lin in enumerate(lins):
            lin._lora = (prev[2 * j], prev[2 * j + 1])
        try:
            with torch.no_grad():
                t = F.log_softmax(model(ids).logits, -1)
        finally:
            for lin, l in zip(lins, live):
                lin._lora = l
        s = F.log_softmax(model(ids).logits, -1)
        return F.kl_div(s, t, log_target=True, reduction="batchmean") / ids.shape[1]

    probe_sentence = comp["eval"][0]

    @torch.no_grad()
    def teacher_hash():
        ids = b._ids(probe_sentence)
        live = [lin._lora for lin in lins]
        for j, lin in enumerate(lins):
            lin._lora = (prev[2 * j], prev[2 * j + 1])
        try:
            lg = model(ids).logits.float().cpu().numpy().tobytes()
        finally:
            for lin, l in zip(lins, live):
                lin._lora = l
        return hashlib.sha256(lg).hexdigest()

    res["arms_out"], res["traces"], res["rng_after"], states = {}, {}, {}, {}
    for arm in ARMS:
        data, teacher = ARM_SPEC[arm]
        b.load_state(snap["lora"])
        b.opt.load_state_dict(copy.deepcopy(snap["opt"]))
        rng.setstate(snap["rng"])
        torch.set_rng_state(snap["torch"])
        if snap["cuda"] is not None:
            torch.cuda.set_rng_state_all(snap["cuda"])
        assert b.memory_hash() == cp1["memory_sha256"], "branch did not restore the life-1 state"
        b.cfg.kl_weight = 0.0 if teacher == "off" else CFG["kl_weight"]
        b._kl_to_frozen = kl_to_previous if teacher == "previous" else frozen_kl
        extra = {}
        if teacher == "previous":
            extra["teacher_hash_start"] = teacher_hash()
        texts = [t for f in facts_life2 for t in f.train_texts] if data == "facts" else bg_day
        res["traces"][arm] = run_life(b, rng, texts)
        if teacher == "previous":
            extra["teacher_hash_end"] = teacher_hash()
        b._kl_to_frozen = frozen_kl
        b.cfg.kl_weight = CFG["kl_weight"]
        cp = probe(b, 2)
        cp["memory_sha256"] = b.memory_hash()
        cp.update(extra)
        res["arms_out"][arm] = cp
        res["rng_after"][arm] = hashlib.sha256(repr(rng.getstate()).encode()).hexdigest()
        states[arm] = {"lora": [x.detach().cpu() for x in b.state()], "opt": copy.deepcopy(b.opt.state_dict()),
                       "rng": rng.getstate()}
        dev.write_atomic(out, res)
        print(time.strftime("%H:%M:%S"), f"seed {a.seed} arm {arm} done", flush=True)
    assert len({json.dumps(v, sort_keys=True) for v in res["traces"].values()}) == 1, "schedules differ between arms"
    assert len(set(res["rng_after"].values())) == 1, "RNG streams diverged between arms"
    res["backbone_start"], res["backbone_end"] = h0, backbone_hash(model)
    assert h0 == res["backbone_end"] == res["backbone_sha256"], "backbone changed"
    # reproduction against the archived BIND-I1 run: checkpoint 1, and arm A vs condition U at life 2
    u2 = next(c for c in ref["conditions"]["U"] if c["after_life"] == 2)
    res["reproduction"] = {"cp1_max_abs_diff": compare_rows(cp1["targets"], ref["checkpoint1"]["targets"]),
                           "cp1_memory_hash_match": cp1["memory_sha256"] == ref["checkpoint1"]["memory_sha256"],
                           "A_vs_U2_max_abs_diff": compare_rows(res["arms_out"]["A"]["targets"], u2["targets"]),
                           "A_vs_U2_memory_hash_match": res["arms_out"]["A"]["memory_sha256"] == u2["memory_sha256"]}
    torch.save(states, states_path)
    res["states_file"] = {"path": str(states_path.relative_to(ROOT)) if states_path.is_relative_to(ROOT) else str(states_path),
                          "sha256": hashlib.sha256(states_path.read_bytes()).hexdigest()}
    b.close()
    res["seconds"] = round(time.time() - t0, 1)
    res["status"] = "complete"
    dev.write_atomic(out, res)
    print("DECAY-1 SEED DONE", a.seed, res["seconds"], "s", json.dumps(res["reproduction"]), flush=True)


if __name__ == "__main__":
    main()
