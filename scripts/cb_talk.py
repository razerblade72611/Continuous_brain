"""Talk to a Continuous Brain cortex interactively (a demo tool; NOT part of any experiment's frozen code).

Loads Qwen3-8B in fp32 on CPU (or GPU if available), merges one cortex checkpoint into it exactly as the experiments'
fp32 evaluation does, and lets you ask questions in the experiments' own answer frame.

  python3 scripts/cb_talk.py --model ~/hf/Qwen3-8B --ckpt <cortex_ARM_sL.pt> --prep <prep_seedS.json> --sleep L

Commands at the prompt:
  <question>          asked in the training construction ("Question: ...") - e.g. What does Pixgriask Yaelkaend do for a living?
  s: <stem>           a Statement probe, e.g.  s: Pixgriask Yaelkaend works as a
  raw: <text>         free continuation of your text with no frame (mostly plain Qwen3 behaviour)
  who                 people this cortex has experienced by the chosen sleep, with what it was taught (and some gaps)
  quit
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cons0a_talk", ROOT / "scripts/cons0a.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
ATTRS9 = ["city", "occupation", "favorite object", "pet species", "childhood fear", "workplace", "hobby", "birth month",
          "allergy"]


def load(model_dir, ckpt):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"loading {model_dir} in fp32 on {dev} (first load takes a few minutes) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        scale, n = C.ALPHA / C.RANK, 0
        with torch.no_grad():
            for name, mod in model.named_modules():
                if name in state:
                    A, B = (x.to(mod.weight.device, torch.float32) for x in state[name])
                    mod.weight += (B @ A) * scale; n += 1
        print(f"merged cortex {Path(ckpt).name}: {n} projections", flush=True)
    else:
        print("no --ckpt: talking to the untrained base model", flush=True)
    return model, tok, dev


def who(prep, sleep):
    st = prep["stream"]
    facts = [f for f in st["facts"] if f["life"] <= sleep]
    known = {}
    for f in sorted(facts, key=lambda f: f["life"]):
        known.setdefault(f["entity"], {})[f["attr"]] = f["value"]
    print(f"\n-- experienced by sleep {sleep} (what it was taught) --")
    for e in sorted(known):
        group = "A" if e in st.get("A", []) else ("H" if e in st.get("H", []) else "transient")
        print(f"  {e} [{group}]: " + "; ".join(f"{a} = {v}" for a, v in known[e].items()))
        gaps = [a for a in ATTRS9 if a not in known[e]]
        if gaps:
            print(f"      never told: {', '.join(gaps[:4])}{' ...' if len(gaps) > 4 else ''}")
    if "split" in prep:
        print("  (A people received 'unknown' training on some gaps; H people never did)")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--ckpt"); ap.add_argument("--prep")
    ap.add_argument("--sleep", type=int, default=5); ap.add_argument("--max-new", type=int, default=8)
    a = ap.parse_args()
    prep = json.loads(Path(a.prep).read_text()) if a.prep else None
    model, tok, dev = load(a.model, a.ckpt)
    if prep:
        who(prep, a.sleep)
    print("Ask a question (or: s: <stem> | raw: <text> | who | quit)")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        if line == "who":
            who(prep, a.sleep) if prep else print("(start with --prep to see who this cortex knows)"); continue
        t = time.time()
        if line.startswith("raw:"):
            out = C.generate(model, tok, [line[4:].strip()], dev, max_new=max(a.max_new, 40))[0]
            print(out.strip())
        else:
            item = C.statement(line[2:].strip().rstrip(".").rstrip("…").rstrip()) if line.startswith("s:") else C.question(line)
            print("  " + C.first_line(C.generate(model, tok, [C.frame(item)], dev, max_new=a.max_new)[0]))
        print(f"  ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
