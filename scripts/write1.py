"""WRITE-1: can a new life be written WITHOUT changing the parameters that hold earlier memories? (Claude + Sol,
2026-10-27). DEV seeds 2-5, the BIND-I1 streams; mechanistic follow-up to DECAY-1, not confirmation.

Life 1 is trained exactly as in BIND-I1 / DECAY-1 (checkpoint 1 must reproduce BIND-I1 exactly). From the full
checkpoint-1 state (LoRA, AdamW, Python and torch RNGs) ONE continuation life on the U-type new facts runs in three arms,
all with the same KL-to-frozen objective, schedule, facts and probes:
  A  continue writing the life-1 adapter, inherited AdamW         (= DECAY-1 arm A = BIND-I1 condition U, life 2)
  R  continue writing the life-1 adapter, FRESH AdamW state        (optimizer-reset control)
  P  life-1 adapter FROZEN and active; a fresh rank-16 adapter (same init scheme, seeded) is the only trainable
     memory for life 2, with a fresh AdamW; the model's output is backbone + life-1 adapter + life-2 adapter.
Readouts after life 2: A and R as usual. P in three modes:
  both (PRIMARY: no routing, no life label), old-only (integrity gate: must equal checkpoint 1 exactly, since the
  life-1 adapter is frozen), new-only (descriptive).
Schedules are asserted identical across arms (step kinds, background and teacher texts, RNG end state).

Integrity (Sol): the new adapter starts with exactly zero output and P's combined readout before life-2 training must
equal checkpoint 1; only new-adapter parameters train; the frozen-KL teacher disables both adapters; A at life 2 must
reproduce BIND-I1 condition U and DECAY-1 arm A exactly; trainable-parameter counts and steps are recorded per arm.

  python scripts/write1.py --seed 2 --out results/dev3/write1/seed2.json --reference results/dev3/bind_i1/seed2.json \
      --decay-reference results/dev3/decay1/seed2.json
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


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


i1 = _load("bind_i1_for_write1", "scripts/bind_i1.py")
dc = _load("decay1_for_write1", "scripts/decay1.py")

SEEDS = {2, 3, 4, 5}
ARMS = ("A", "R", "P")
P_MODES = ("both", "old", "new")
CFG = dict(i1.CFG)


def main():
    import torch
    from being.competence import load_sentences, mean_loss
    from being.v3 import lifegen
    from being.v3.core import Being3, V3Config, backbone_hash, exact, load_qwen
    from being.v3.lifegen import probes_for
    from being.v3.run import background_pool
    dev = _load("run_v3_dev_w1", "scripts/run_v3_dev.py")

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out")
    ap.add_argument("--reference")
    ap.add_argument("--decay-reference", help="archived DECAY-1 result for this seed (arm A must reproduce it)")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()
    if a.seed not in SEEDS:
        sys.exit("WRITE-1 uses dev seeds 2-5 only")
    if a.preflight:
        from transformers import AutoTokenizer
        from being.v3.core import QWEN
        tok = AutoTokenizer.from_pretrained(QWEN)
        st = i1.build_stream(a.seed, lifegen, lambda w: len(tok(" " + w).input_ids))
        i1.check_stream(st, lifegen)
        pc = i1.prefix_check(st, lifegen, tok)
        print(f"WRITE-1 PREFLIGHT seed {a.seed}: {pc['n']} pairs, prefix failures {pc['fails']}")
        sys.exit(1 if pc["fails"] else 0)
    if not (a.out and a.reference and a.decay_reference):
        sys.exit("--out, --reference and --decay-reference are required")
    dref = json.loads((ROOT / a.decay_reference).read_text())
    if dref.get("status") != "complete" or dref.get("seed") != a.seed:
        sys.exit("decay reference is not a complete DECAY-1 result for this seed")
    out, ref = ROOT / a.out, json.loads((ROOT / a.reference).read_text())
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    if ref.get("status") != "complete" or ref.get("seed") != a.seed:
        sys.exit("reference is not a complete BIND-I1 result for this seed")
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    model, tok = load_qwen()
    st = i1.build_stream(a.seed, lifegen, lambda w: len(tok(" " + w).input_ids))
    i1.check_stream(st, lifegen)
    if json.loads(json.dumps(st["design"])) != ref["design"]:
        sys.exit("stream design differs from the BIND-I1 reference")
    pc = i1.prefix_check(st, lifegen, tok)
    cfg = V3Config(arm="L", **CFG)
    comp = {k: load_sentences(k) for k in ("eval", "modern")}
    base = {k: mean_loss(model, tok, v) for k, v in comp.items()}
    d = st["design"]
    bg = background_pool()
    life2 = [t for f in st["lives"]["U"][1] for t in f.train_texts]
    res = {"which": "WRITE-1", "commit": commit, "config": asdict(cfg), "seed": a.seed, "arms": list(ARMS),
           "backbone_sha256": backbone_hash(model), "base_ppl_loss": base, "prefix_check": pc,
           "reference": str(a.reference), "reference_commit": ref.get("commit"), "status": "in_progress"}
    dev.write_atomic(out, res)

    def fc_row(lps, k):
        rest = lps[:k] + lps[k + 1:]
        return {"fc": sum(x > lps[k] for x in rest) == 0, "ties": sum(x == lps[k] for x in rest),
                "lp": lps[k], "margin_best": lps[k] - max(rest)}

    @torch.no_grad()
    def probe(b, L):                                           # identical to the BIND-I1 / DECAY-1 probe
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

    def run_life(b, rng, day_texts):                          # the BIND-I1 / live_stream L loop, step kinds traced
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

    def tensors_hash(ts):
        h = hashlib.sha256()
        for p in ts:
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

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
    old_params, old_opt = list(b.lora), b.opt
    lins = [getattr(layer.self_attn, n) for layer in model.model.layers for n in ("q_proj", "v_proj")]
    live_forward = [lin.forward for lin in lins]               # Being3's LoRA forwards (life-1 adapter)
    res["reproduction"] = {"cp1": dc.reproduce(cp1, ref["checkpoint1"])}
    dev.write_atomic(out, res)

    mode = {"m": "both"}

    def attach_new_adapter():
        g = torch.Generator().manual_seed(1_000_003 * a.seed + 17)
        new = []
        for lin in lins:
            na = torch.nn.Parameter((torch.randn(cfg.lora_r, lin.in_features, generator=g) / lin.in_features ** 0.5)
                                    .to(b.dev))
            nb = torch.nn.Parameter(torch.zeros(lin.out_features, cfg.lora_r, device=b.dev))
            lin._lora_new = (na, nb)
            new += [na, nb]

            def fwd(x, l=lin):
                y = l._orig_forward(x)
                if mode["m"] in ("both", "old"):
                    y = y + (x @ l._lora[0].T) @ l._lora[1].T
                if mode["m"] in ("both", "new"):
                    y = y + (x @ l._lora_new[0].T) @ l._lora_new[1].T
                return y
            lin.forward = fwd
        for p in old_params:
            p.requires_grad_(False)
        b.lora = new
        b.opt = torch.optim.AdamW(new, lr=cfg.lr, weight_decay=0.0)
        return new

    def detach_new_adapter():
        for lin, fw in zip(lins, live_forward):
            lin.forward = fw
            del lin._lora_new
        for p in old_params:
            p.requires_grad_(True)
        b.lora, b.opt = old_params, old_opt

    res["arms_out"], res["traces"], res["rng_after"], states = {}, {}, {}, {}
    for arm in ARMS:
        b.load_state(snap["lora"])
        b.opt.load_state_dict(copy.deepcopy(snap["opt"]))
        rng.setstate(snap["rng"])
        torch.set_rng_state(snap["torch"])
        if snap["cuda"] is not None:
            torch.cuda.set_rng_state_all(snap["cuda"])
        assert b.memory_hash() == cp1["memory_sha256"], "branch did not restore the life-1 state"
        if arm == "R":
            b.opt = torch.optim.AdamW(old_params, lr=cfg.lr, weight_decay=0.0)
        if arm == "P":
            new = attach_new_adapter()
            old_h = tensors_hash(old_params)
            assert all(float(p.abs().max()) == 0.0 for p in new[1::2]), "new adapter must start with zero output"
            pre = probe(b, 1)                                  # combined readout BEFORE life-2 training = checkpoint 1
            res["P_zero_init_check"] = dc.reproduce({**pre, "memory_sha256": cp1["memory_sha256"]}, cp1)
        res.setdefault("trainable_params", {})[arm] = sum(p.numel() for p in b.lora if p.requires_grad)
        res["traces"][arm] = run_life(b, rng, life2)
        res["rng_after"][arm] = hashlib.sha256(repr(rng.getstate()).encode()).hexdigest()
        if arm == "P":
            assert tensors_hash(old_params) == old_h == cp1["memory_sha256"], "life-1 adapter changed during P"
            outp = {}
            for m in P_MODES:
                mode["m"] = m
                cp = probe(b, 2)
                cp["memory_sha256"] = cp1["memory_sha256"] if m == "old" else tensors_hash(
                    (old_params + new) if m == "both" else new)
                outp[m] = cp
            mode["m"] = "both"
            res["arms_out"]["P"] = outp["both"]
            res["P_readouts"] = {"old": outp["old"], "new": outp["new"]}
            res["isolation_integrity"] = dc.reproduce({**outp["old"], "after_life": 1}, cp1)   # everything but the label
            res["P_cost"] = {"params_per_adapter": sum(p.numel() for p in new),
                             "adapters_active_at_inference": 2}
            states[arm] = {"old_lora": [p.detach().cpu() for p in old_params], "new_lora": [p.detach().cpu() for p in new],
                           "opt": copy.deepcopy(b.opt.state_dict()), "rng": rng.getstate()}
            detach_new_adapter()
        else:
            cp = probe(b, 2)
            cp["memory_sha256"] = b.memory_hash()
            res["arms_out"][arm] = cp
            states[arm] = {"lora": [p.detach().cpu() for p in b.lora], "opt": copy.deepcopy(b.opt.state_dict()),
                           "rng": rng.getstate()}
            if arm == "R":
                b.opt = old_opt
        dev.write_atomic(out, res)
        print(time.strftime("%H:%M:%S"), f"seed {a.seed} arm {arm} done", flush=True)
    assert len({json.dumps(v, sort_keys=True) for v in res["traces"].values()}) == 1, "schedules differ between arms"
    assert len(set(res["rng_after"].values())) == 1, "RNG streams diverged between arms"
    res["backbone_start"], res["backbone_end"] = h0, backbone_hash(model)
    assert h0 == res["backbone_end"] == res["backbone_sha256"], "backbone changed"
    u2 = next(c for c in ref["conditions"]["U"] if c["after_life"] == 2)
    res["reproduction"]["A_vs_U2"] = dc.reproduce(res["arms_out"]["A"], u2)
    res["reproduction"]["A_vs_DECAY1_A"] = dc.reproduce(res["arms_out"]["A"], dref["arms_out"]["A"])
    states["_checkpoint1"] = {"lora": [x.detach().cpu() for x in snap["lora"]], "opt": snap["opt"], "rng": snap["rng"],
                              "torch_rng": snap["torch"], "cuda_rng": snap["cuda"], "memory_sha256": cp1["memory_sha256"]}
    sp = out.with_name(out.stem + "_states.pt")
    torch.save(states, sp)
    res["states_file"] = {"path": sp.name, "sha256": hashlib.sha256(sp.read_bytes()).hexdigest()}
    b.close()
    res["seconds"] = round(time.time() - t0, 1)
    res["status"] = "complete"
    dev.write_atomic(out, res)
    print("WRITE-1 SEED DONE", a.seed, res["seconds"], "s", json.dumps(res["reproduction"]),
          "integrity", json.dumps(res["isolation_integrity"]), flush=True)


if __name__ == "__main__":
    main()
