"""CLS-0 gate: single-model resource benchmark of the episodic memory (being/cls/episodic.py) BEFORE any CLS training
run (Claude + Sol, 2026-10-27). Not a scientific result: it measures cost and checks that the mechanism works end to end.

Writes three short episodes into an EpisodicMemory on the given backbone and records, per episode: mask size and
attempts, optimiser epochs to exact reconstruction, wall-clock, peak GPU memory, adapter parameters; then routing
similarities for three cues (one per stored fact, plus an unrelated cue), reconstruction of the routed episode, process
RSS, model load time and VRAM, and the on-disk size of the stored adapters. Optionally also loads an external embedding
model (published K1 keys) and records its memory cost.

  python scripts/cls_bench.py --model /workspace/hf/Qwen3-8B --dtype bf16 --out results/cls/bench_qwen3_8b.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Episodes are fixed text (no model involvement). Two observed facts about different people with the SAME attribute
# (a routing pair) and one short narrative of about 80 tokens (the published episode length).
EPISODES = [
    ("e1_fact", "Griarnshial Yeithmaev works as a Pemyiasnind. By trade, Griarnshial Yeithmaev is a Pemyiasnind. "
                "Griarnshial Yeithmaev's occupation is Pemyiasnind. A Pemyiasnind by profession, that is "
                "Griarnshial Yeithmaev."),
    ("e2_fact", "Shosposk Fuskyyl works as a Drivtaefusk. By trade, Shosposk Fuskyyl is a Drivtaefusk. "
                "Shosposk Fuskyyl's occupation is Drivtaefusk. A Drivtaefusk by profession, that is Shosposk Fuskyyl."),
    ("e3_story", "On the third morning of the thaw, Velka Moradine carried her grandfather's brass compass down to the "
                 "river market and traded it for a crate of winter pears and a promise. The pear seller, a tall man "
                 "named Oskel, swore he would return the compass when the ferry ran again. By evening the ice had "
                 "cracked, the ferry had not come, and Velka sat on the crate counting lanterns across the water."),
]
CUES = [("Griarnshial Yeithmaev works as a", "e1_fact"), ("Shosposk Fuskyyl works as a", "e2_fact"),
        ("Who traded a brass compass at the river market?", "e3_story"),
        ("What is the boiling point of water at sea level?", None)]


def rss():
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS", "VmHWM")):
            k, v = line.split(":")
            out[k] = int(v.split()[0]) // 1024
    return out                                                # MiB


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from being.cls.episodic import EpisodicMemory

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--external", help="optional external embedding model path (Qwen3-Embedding-0.6B)")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--r", type=int, default=2, help="adapter rank (published: 2)")
    ap.add_argument("--init-mask", type=int, default=45, help="initial mask size (published: 45)")
    ap.add_argument("--key-layer", type=int, default=24)
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    res = {"which": "CLS-0 resource benchmark", "model": a.model, "dtype": a.dtype, "device": dev,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "gpu": torch.cuda.get_device_name(0) if cuda else None, "episodes": {}, "status": "in_progress"}
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if a.dtype == "bf16" else torch.float32,
                                                 attn_implementation="eager").to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    res["load_seconds"] = round(time.time() - t0, 1)
    res["n_layers"] = len(model.model.layers)
    res["vram_after_load_mib"] = torch.cuda.memory_allocated() // 2**20 if cuda else None
    res["rss_after_load_mib"] = rss()
    mem = EpisodicMemory(model, tok, layer=a.layer, key_layer=a.key_layer, max_epochs=a.max_epochs,
                         r=a.r, init_mask=a.init_mask)
    res["layer"], res["key_layer"], res["r"], res["init_mask"] = a.layer, mem.key_layer, a.r, a.init_mask
    for eid, text in EPISODES:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            ep = mem.write(eid, text, time_=1, source="bench:fixed-text")
            ok = True
        except RuntimeError as e:
            ep, ok = None, False
            res["episodes"][eid] = {"ok": False, "error": str(e), "attempts": getattr(e, "attempts", None),
                                    "seconds": round(time.time() - t, 1)}
        if ep is not None:
            res["episodes"][eid] = {"ok": ok, "mask": ep.mask, "attempts": ep.attempts,
                                    "seconds": round(time.time() - t, 1)}
        res["episodes"][eid]["peak_vram_mib"] = torch.cuda.max_memory_allocated() // 2**20 if cuda else None
        print(time.strftime("%H:%M:%S"), eid, json.dumps(res["episodes"][eid])[:300], flush=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    res["routing"] = []
    for cue, expect in CUES:
        top = mem.route(cue, k=len(mem.episodes))
        res["routing"].append({"cue": cue, "expect": expect, "ranking": top,
                               "correct": (top[0][0] == expect) if (top and expect) else None})
    stored = [e for e, _ in EPISODES if e in mem.episodes]
    res["reconstruction"] = {e: mem.reconstruct(e) for e in stored}
    res["reconstruction_exact"] = {e: res["reconstruction"][e].strip() == dict(EPISODES)[e].strip() for e in stored}
    sp = out.with_suffix(".adapters.pt")
    torch.save({k: {n: t.detach().cpu() for n, t in v.state_dict().items()} | {"in_idx": v.in_idx.cpu(),
                                                                                "out_idx": v.out_idx.cpu()}
                for k, v in mem.site.adapters.items()}, sp)
    res["adapters_file_bytes"] = sp.stat().st_size
    res["adapter_params_total"] = sum(p.numel() for v in mem.site.adapters.values() for p in v.parameters())
    if a.external:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        from transformers import AutoModel
        t = time.time()
        etok = AutoTokenizer.from_pretrained(a.external)
        emod = AutoModel.from_pretrained(a.external, dtype=torch.bfloat16 if cuda else torch.float32).to(dev).eval()
        with torch.no_grad():
            b = etok([t_ for _, t_ in EPISODES], padding=True, return_tensors="pt").to(dev)
            h = emod(**b).last_hidden_state
        res["external_embedding"] = {"load_and_embed_seconds": round(time.time() - t, 1),
                                     "peak_vram_mib": torch.cuda.max_memory_allocated() // 2**20 if cuda else None}
    res["rss_end_mib"] = rss()
    res["seconds_total"] = round(time.time() - t0, 1)
    res["status"] = "complete"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("CLS BENCH DONE", a.model, json.dumps({k: res[k] for k in ("vram_after_load_mib", "load_seconds",
                                                                       "seconds_total", "reconstruction_exact")}),
          flush=True)


if __name__ == "__main__":
    main()
