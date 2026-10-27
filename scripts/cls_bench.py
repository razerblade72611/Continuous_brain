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
    from being.cls.episodic import BudgetExceeded, EpisodicMemory

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--external", help="optional external embedding model path (Qwen3-Embedding-0.6B): K1 keys")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--r", type=int, default=2, help="adapter rank (published: 2)")
    ap.add_argument("--init-mask", type=int, default=45, help="initial mask size (published: 45)")
    ap.add_argument("--key-layer", type=int, default=24)
    ap.add_argument("--budget-seconds", type=float, default=2400, help="wall-clock budget for all episode writes")
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    res = {"which": "CLS-0 resource benchmark", "model": a.model, "dtype": a.dtype, "device": dev,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "gpu": torch.cuda.get_device_name(0) if cuda else None, "episodes": {}, "status": "in_progress",
           "budget_seconds": a.budget_seconds}

    def save():
        with open(out, "w") as f:
            json.dump(res, f, indent=1)

    def gpu_peaks():
        return ({"peak_alloc_mib": torch.cuda.max_memory_allocated() // 2**20,
                 "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20} if cuda else {})

    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16 if a.dtype == "bf16" else torch.float32, attn_implementation="eager").to(dev)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        res["load_seconds"] = round(time.time() - t0, 1)
        res["n_layers"] = len(model.model.layers)
        res["vram_after_load"] = gpu_peaks()
        res["rss_after_load_mib"] = rss()
        mem = EpisodicMemory(model, tok, layer=a.layer, key_layer=a.key_layer, max_epochs=a.max_epochs,
                             r=a.r, init_mask=a.init_mask)
        res["layer"], res["key_layer"], res["r"], res["init_mask"] = a.layer, mem.key_layer, a.r, a.init_mask
        t_writes = time.time()
        deadline = t_writes + a.budget_seconds
        slowest = 0.0
        # the published-length narrative first: if the budget only allows one episode, it is the representative one
        for eid, text in sorted(EPISODES, key=lambda x: x[0] != "e3_story"):
            left = deadline - time.time()
            if slowest and left < 1.5 * slowest:
                res["episodes"][eid] = {"ok": False, "skipped_budget": True, "seconds_left": round(left, 1)}
                save()
                continue
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            try:
                ep = mem.write(eid, text, time_=1, source="bench:fixed-text", deadline=deadline)
                res["episodes"][eid] = {"ok": True, "mask": ep.mask, "attempts": ep.attempts}
            except BudgetExceeded as e:
                res["episodes"][eid] = {"ok": False, "budget_censored": True, "attempts": e.attempts}
            except torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as e:
                res["episodes"][eid] = {"ok": False, "error": str(e), "attempts": getattr(e, "attempts", None)}
            res["episodes"][eid]["seconds"] = round(time.time() - t, 1)
            res["episodes"][eid].update(gpu_peaks())
            slowest = max(slowest, time.time() - t)
            print(time.strftime("%H:%M:%S"), eid, json.dumps(res["episodes"][eid])[:300], flush=True)
            save()
        stored = [e for e, _ in EPISODES if e in mem.episodes]
        res["routing_internal"] = []
        for cue, expect in CUES:
            top = mem.route(cue, k=len(mem.episodes)) if stored else []
            res["routing_internal"].append({"cue": cue, "expect": expect, "ranking": top,
                                            "correct": (top[0][0] == expect) if (top and expect in stored) else None})
        res["reconstruction"] = {e: mem.reconstruct(e) for e in stored}
        res["reconstruction_exact"] = {e: res["reconstruction"][e].strip() == dict(EPISODES)[e].strip() for e in stored}
        sp = out.with_suffix(".adapters.pt")
        torch.save({k: {n: t_.detach().cpu() for n, t_ in v.state_dict().items()} |
                    {"in_idx": v.in_idx.cpu(), "out_idx": v.out_idx.cpu()} for k, v in mem.site.adapters.items()}, sp)
        res["adapters_file_bytes"] = sp.stat().st_size
        res["adapter_params_total"] = sum(p.numel() for v in mem.site.adapters.values() for p in v.parameters())
        if a.external:
            from being.cls.keys import ExternalKeys
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            ek = ExternalKeys(a.external, dev, torch.bfloat16 if cuda else torch.float32)
            texts = dict(EPISODES)
            keys = ek.embed([texts[e] for e in stored]) if stored else None
            rows = []
            for cue, expect in CUES:
                if keys is None:
                    rows.append({"cue": cue, "expect": expect, "ranking": [], "correct": None})
                    continue
                sims = (keys @ ek.embed([cue], query=True)[0]).tolist()
                rank = sorted(zip(stored, sims), key=lambda x: -x[1])
                rows.append({"cue": cue, "expect": expect, "ranking": rank,
                             "correct": (rank[0][0] == expect) if expect in stored else None})
            res["routing_external"] = rows
            res["external_embedding"] = {"seconds": round(time.time() - t, 1), **gpu_peaks()}
        n_ok = sum(1 for v in res["episodes"].values() if v.get("ok"))
        res["status"] = ("BENCH_COMPLETE" if n_ok == len(EPISODES) and all(res["reconstruction_exact"].values())
                         else "BENCH_PARTIAL" if n_ok else "BENCH_FAILED")
    except torch.cuda.OutOfMemoryError as e:
        res["status"], res["error"] = "BENCH_FAILED", f"CUDA OOM: {str(e)[:300]}"
        res["oom_peaks"] = gpu_peaks()
    except Exception as e:                                   # any other failure is recorded, never reported as success
        res["status"], res["error"] = "BENCH_FAILED", f"{type(e).__name__}: {str(e)[:300]}"
    res["rss_end_mib"] = rss()
    res["seconds_total"] = round(time.time() - t0, 1)
    save()
    print("CLS BENCH", res["status"], a.model, json.dumps({k: res.get(k) for k in (
        "vram_after_load", "load_seconds", "seconds_total", "reconstruction_exact", "error")}), flush=True)
    sys.exit({"BENCH_COMPLETE": 0, "BENCH_PARTIAL": 2}.get(res["status"], 1))


if __name__ == "__main__":
    main()
