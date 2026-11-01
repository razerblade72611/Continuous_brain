"""MERGE-ENG (Sol GO 2026-10-31): read-only engineering investigation of merged-vs-unmerged equivalence on the 24 saved
CONS-0A run-2 cortex checkpoints. No training; it does NOT revalidate CONS-0A (lab/CONS0A_RUN2_REPORT.md).

Configurations (one checkpoint at a time; cortex = fp32 LoRA r16 a32 on q/k/v/o/gate/up/down):
  phase A - base model loaded in bf16 (as in CONS-0A):
    R      bf16 base matmul + fp32 LoRA correction cast to bf16       (CONS-0A in-process path)
    M16    W += (dW).to(bf16), then plain bf16 forward                (CONS-0A fresh path: two roundings)
    M16b   W  = (W.float() + dW).to(bf16), plain bf16 forward         (careful bf16 merge: one rounding)
    C32u   targeted linears in fp32: x.float() @ W.float().T + fp32 LoRA, output cast to bf16
    C32m   targeted linears in fp32: x.float() @ (W.float() + dW).T,    output cast to bf16   (same casting as C32u)
  phase B - base model loaded in fp32:
    F32u   fp32 base + fp32 LoRA
    F32m   fp32 (W + dW)
Pairs: R~M16 (reproduces run 2), R~M16b, C32u~C32m, F32u~F32m, R~C32u (effect of the base matmul's precision).
Per pair: raw answer agreement (all probes / positives), grade agreement, summary-metric diffs, competence ppl diff,
first-position KL and top-1 agreement, and at the first diverging generated token the top-2 log-prob margin (both
configs). Harness validation: R and M16 answers are compared with run 2's in-process and fresh rows.
    python scripts/merge_eng.py --run2 results/cons0a_run2 --ckpts CKPT_DIR --model M --out results/merge_eng/merge_eng.json
"""
import argparse, importlib.util, json, math, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cons0a_for_merge_eng", ROOT / "scripts/cons0a.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)

CKPTS = [(f"cortex_{n}.pt", n, K) for n, K in (("QA_shared_s1", 1), ("I-QA_s5", 5), ("S-QA_s5", 5),
                                               ("V_shared_s1", 1), ("I-V_s5", 5), ("S-V_s5", 5))]
PHASE_A, PHASE_B = ("R", "M16", "M16b", "C32u", "C32m"), ("F32u", "F32m")
PAIRS = (("R", "M16"), ("R", "M16b"), ("C32u", "C32m"), ("F32u", "F32m"), ("R", "C32u"))


class Switch:
    """Routes every targeted Linear through the configuration selected by self.mode."""

    def __init__(self, model):
        import torch
        self.t, self.mode, self.lora, self.mods = torch, None, {}, []
        for name, mod in model.named_modules():
            if name.split(".")[-1] in C.TARGETS and isinstance(mod, torch.nn.Linear):
                assert mod.bias is None, name
                mod._me_W = mod.weight.data            # the unmodified base weight (never written)
                mod._me_name = name
                mod.forward = (lambda m: (lambda x: self._fwd(m, x)))(mod)
                self.mods.append(mod)
        assert len(self.mods) == model.config.num_hidden_layers * len(C.TARGETS), len(self.mods)

    def load(self, state, scale):
        t = self.t
        dev = self.mods[0]._me_W.device
        self.lora = {m._me_name: (state[m._me_name][0].to(dev, t.float32), state[m._me_name][1].to(dev, t.float32))
                     for m in self.mods}
        self.scale = scale

    def dW(self, m):
        A, B = self.lora[m._me_name]
        return (B @ A) * self.scale

    def set(self, mode):
        """Merged bf16 modes install a merged weight copy; every other mode uses the untouched base weight."""
        t = self.t
        with t.no_grad():
            for m in self.mods:
                if mode == "M16":
                    m.weight.data = m._me_W + self.dW(m).to(m._me_W.dtype)          # production: two roundings
                elif mode == "M16b":
                    m.weight.data = (m._me_W.float() + self.dW(m)).to(m._me_W.dtype)  # one rounding
                else:
                    m.weight.data = m._me_W
        self.mode = mode

    def _fwd(self, m, x):
        t, mode = self.t, self.mode
        if mode in ("M16", "M16b"):
            return t.nn.functional.linear(x, m.weight)
        A, B = self.lora[m._me_name]
        if mode in ("R", "F32u"):                     # base matmul in the model dtype + fp32 LoRA, cast back
            return t.nn.functional.linear(x, m._me_W) + ((x.to(t.float32) @ A.T @ B.T) * self.scale).to(x.dtype)
        if mode == "F32m":
            return t.nn.functional.linear(x, m._me_W + self.dW(m))
        xf = x.to(t.float32)
        if mode == "C32u":
            return (t.nn.functional.linear(xf, m._me_W.float()) + (xf @ A.T @ B.T) * self.scale).to(x.dtype)
        if mode == "C32m":
            return t.nn.functional.linear(xf, m._me_W.float() + self.dW(m)).to(x.dtype)
        raise ValueError(mode)


def gen_ids(model, tok, prompts, dev, bs=16):
    import torch
    out = []
    tok.padding_side = "left"
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = model.generate(**b, max_new_tokens=C.MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            out += [r.tolist() for r in g[:, b.input_ids.shape[1]:]]
    return out


def first_logprobs(model, tok, prompts, dev, bs=16):
    """Log-softmax of the next-token distribution at the end of each prompt (fp32, on CPU)."""
    import torch
    out = []
    tok.padding_side = "left"
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            b = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            out.append(torch.log_softmax(model(**b).logits[:, -1].float(), -1).cpu())
    return torch.cat(out)


def margin_at(model, tok, prompt, prefix_ids, toks, dev):
    """Log-prob margin top1 - top2 at the position after prompt+prefix, and logp of the two competing tokens."""
    import torch
    ids = tok(prompt, add_special_tokens=False).input_ids + prefix_ids
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids], device=dev)).logits[0, -1].float(), -1)
    top = torch.topk(lp, 2).values
    return {"top2_margin": float(top[0] - top[1]), "lp_a": float(lp[toks[0]]), "lp_b": float(lp[toks[1]])}


def evaluate(model, tok, P, dev, comp_sets, base_loss):
    from being.competence import mean_loss
    prompts = [C.frame(p["item"]) for p in P]
    ids = gen_ids(model, tok, prompts, dev)
    rows = []
    for p, g in zip(P, ids):
        ans = C.first_line(tok.decode(g, skip_special_tokens=True))
        r = {"kind": p["kind"], "life": p.get("life"), "answer": ans, "ids": g}
        if p["kind"] in ("template", "paraphrase", "temporal"):
            r |= C.grade(ans, p["value"])
        else:
            r |= {"correct": False, "unknown": C.is_unknown(ans), "empty": not C.norm(ans)}
        rows.append(r)
    ppl = {k: math.exp(mean_loss(model, tok, s)) / math.exp(base_loss[k]) for k, s in comp_sets.items()}
    return {"rows": rows, "ppl_ratio": ppl, "first_lp": first_logprobs(model, tok, prompts, dev)}


def compare(a, b, P, prompts, model, tok, dev, sw, L):
    import torch
    pos = [i for i, p in enumerate(P) if p["kind"] in ("template", "paraphrase")]
    ra, rb = a["rows"], b["rows"]
    agree = sum(x["answer"] == y["answer"] for x, y in zip(ra, rb)) / len(ra)
    agree_pos = sum(ra[i]["answer"] == rb[i]["answer"] for i in pos) / max(1, len(pos))
    grade = sum(x["correct"] == y["correct"] for x, y in zip(ra, rb)) / len(ra)
    sa, sb = C.summarise(ra, L), C.summarise(rb, L)
    kl = (a["first_lp"].exp() * (a["first_lp"] - b["first_lp"])).sum(-1)
    top1 = (a["first_lp"].argmax(-1) == b["first_lp"].argmax(-1)).float().mean()
    div = []
    for i, (x, y) in enumerate(zip(ra, rb)):
        if x["ids"] == y["ids"]:
            continue
        j = next((k for k, (u, v) in enumerate(zip(x["ids"], y["ids"])) if u != v), min(len(x["ids"]), len(y["ids"])))
        if j >= min(len(x["ids"]), len(y["ids"])):
            continue                                          # one is a prefix of the other (length only)
        div.append((i, j, x["ids"][j], y["ids"][j]))
    marg = {}
    for cfg in (a["cfg"], b["cfg"]):
        if cfg in (PHASE_A if sw.phase == "A" else PHASE_B):
            sw.set(cfg)
            marg[cfg] = [margin_at(model, tok, prompts[i], ra[i]["ids"][:j], (u, v), dev) for i, j, u, v in div]
    return {"agree": agree, "agree_pos": agree_pos, "grade_agree": grade,
            "summary_diff": {k: abs((sa[k] or 0) - (sb[k] or 0)) for k in sa},
            "ppl_diff": {k: abs(a["ppl_ratio"][k] - b["ppl_ratio"][k]) for k in a["ppl_ratio"]},
            "kl_first_mean": float(kl.mean()), "kl_first_max": float(kl.max()), "top1_first_agree": float(top1),
            "n_diverging": len(div), "n_diverging_answers": sum(x["answer"] != y["answer"] for x, y in zip(ra, rb)),
            "first_divergence_margins": {c: {"median_top2": statistics.median([m["top2_margin"] for m in v]) if v else None,
                                             "max_top2": max([m["top2_margin"] for m in v]) if v else None,
                                             "median_gap_ab": statistics.median([abs(m["lp_a"] - m["lp_b"]) for m in v]) if v else None}
                                         for c, v in marg.items()},
            "would_pass_cons0a_per_seed": agree >= C.MERGE_AGREE and max(
                abs(a["ppl_ratio"][k] - b["ppl_ratio"][k]) for k in a["ppl_ratio"]) <= C.PPL_TOL}


def main():
    import torch
    from being.v3 import lifegen
    ap = argparse.ArgumentParser()
    ap.add_argument("--run2", required=True); ap.add_argument("--ckpts", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stream-tokenizer", default=C.STREAM_TOK); ap.add_argument("--seeds", default="2,3,4,5")
    ap.add_argument("--phases", default="A,B"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--test-mode", action="store_true"); ap.add_argument("--no-reproduce", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {"which": "MERGE-ENG", "results": {}}
    dev = a.device
    assert dev != "cuda" or torch.cuda.is_available(), "CUDA not available"
    seeds = [int(s) for s in a.seeds.split(",")]
    streams = {s: C._stream(s, a.stream_tokenizer, a.test_mode)[:2] for s in seeds}
    for phase in a.phases.split(","):
        dtype = torch.bfloat16 if phase == "A" else torch.float32
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="eager").to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        comp_sets, base_loss = C._comp(tok, model, False)
        sw = Switch(model); sw.phase = phase
        cfgs = PHASE_A if phase == "A" else PHASE_B
        for s in seeds:
            eps, st = streams[s]
            r2 = None if a.no_reproduce else json.loads((Path(a.run2) / f"seed{s}/run_seed{s}.json").read_text())
            for fn, name, K in CKPTS:
                key = f"{s}:{name}"
                if res["results"].get(key, {}).get(f"phase_{phase}_done"):
                    continue
                t0 = time.time()
                P = C.probe_set(eps, st, K, s, lifegen)
                prompts = [C.frame(p["item"]) for p in P]
                sw.load(torch.load(Path(a.ckpts) / f"seed{s}" / fn), C.ALPHA / C.RANK)
                ev = {}
                for cfg in cfgs:
                    sw.set(cfg)
                    ev[cfg] = evaluate(model, tok, P, dev, comp_sets, base_loss) | {"cfg": cfg}
                entry = res["results"].setdefault(key, {"seed": s, "ckpt": name, "sleep": K})
                entry.setdefault("answers", {})
                for cfg in cfgs:
                    entry["answers"][cfg] = [x["answer"] for x in ev[cfg]["rows"]]
                    entry.setdefault("summary", {})[cfg] = C.summarise(ev[cfg]["rows"], K)
                    entry.setdefault("ppl_ratio", {})[cfg] = ev[cfg]["ppl_ratio"]
                for x, y in PAIRS:
                    if x in ev and y in ev:
                        entry.setdefault("pairs", {})[f"{x}~{y}"] = compare(ev[x], ev[y], P, prompts, model, tok, dev, sw, K)
                if phase == "A" and r2 is not None:          # harness validation against run 2's own evaluations
                    arm = {"QA_shared_s1": "S-QA", "V_shared_s1": "S-V"}.get(name, name.split("_")[0])
                    inproc = [x["answer"] for x in r2["arms"][arm]["eval"][str(K)]["rows"]]
                    fresh = [x["answer"] for x in json.loads((Path(a.run2) / f"seed{s}/fresh_seed{s}_{arm}_s{K}.json")
                                                            .read_text())["rows"]]
                    entry["reproduce"] = {
                        "R_vs_run2_inprocess": sum(u == v for u, v in zip(entry["answers"]["R"], inproc)) / len(inproc),
                        "M16_vs_run2_fresh": sum(u == v for u, v in zip(entry["answers"]["M16"], fresh)) / len(fresh),
                        "n": len(inproc), "same_len": len(inproc) == len(P)}
                entry[f"phase_{phase}_done"] = True
                entry[f"phase_{phase}_seconds"] = round(time.time() - t0, 1)
                out.write_text(json.dumps(res, indent=1))
                pr = entry.get("pairs", {})
                print(time.strftime("%H:%M:%S"), phase, key, f"{entry[f'phase_{phase}_seconds']}s",
                      {k: (round(v["agree"], 3), round(max(v["ppl_diff"].values()), 4)) for k, v in pr.items()
                       if k.split("~")[0] in cfgs}, entry.get("reproduce", ""), flush=True)
        del model, sw
        torch.cuda.empty_cache()
    res["environment"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev}
    out.write_text(json.dumps(res, indent=1))
    print("MERGE-ENG DONE", flush=True)


if __name__ == "__main__":
    main()
