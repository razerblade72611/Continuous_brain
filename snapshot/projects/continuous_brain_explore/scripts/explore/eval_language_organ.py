"""Measure a language-organ backend on a labelled corpus of W1 caretaker messages and write a JSON report.

    # here (CPU, no weights): the development stand-in
    python scripts/explore/eval_language_organ.py --backend dev --out evidence/explore/language/dev_parser_eval.json

    # owner's machine: a real model fetched by fetch_models.py
    python scripts/explore/eval_language_organ.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct \
        --local-dir ~/cb_models/Qwen--Qwen2.5-0.5B-Instruct --device cpu --max-messages 300

Corpus (ground truth from W1's obligations layer, see continuous_brain_organs/language_eval.py):

* ``main``   -- the default W1Config: seeds 72001.. (engineering family ``w1-language-eng``), 3000-step episodes,
  W1's privileged scripted baseline as actor with noisy answers (so every feedback frame occurs);
* ``stress`` -- the same actor under a high-change caretaker (arrivals 1/15 per step, changes 1/40 per step per
  obligation, difficulty 1.0 from the start) on seeds 72101.., 900-step episodes, for amendment/cancellation
  coverage.

Metrics: exact match of ``interpret`` overall, per event type (new/amend/cancel/question/feedback/info) and per
family; per-field accuracy; deadline-step accuracy; answer-grammar validity and correctness (answers composed
from an *oracle* memory, graded by W1's own parser and grader); attempts; latency.  ``--corpus-out`` saves the
labelled corpus so a slow backend can be run later on exactly the same items with ``--corpus-in``; the corpus
behind the committed dev-parser evidence is ``evidence/explore/language/w1_language_corpus.json.gz`` (its
``items_sha256`` is in the report), so another machine with a different numpy/Crafter build scores the
very same messages.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from continuous_brain_organs import language_eval as E  # noqa: E402
from continuous_brain_organs.language import FEW_SHOT, LanguageOrgan  # noqa: E402
from continuous_brain_organs.model_store import model_dir_name  # noqa: E402

STRESS = {"arrival_rate0": 1 / 15, "arrival_rate_max": 1 / 10, "change_rate": 1 / 40, "difficulty0": 1.0}
DEV_LABEL = ("DevelopmentParser: a development stand-in that knows W1's caretaker templates, NOT a language model. "
             "These numbers measure the stand-in (an upper reference on W1's closed template language), not any "
             "language model and not the individual.")


def git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def read_corpora(path: str) -> dict:
    data = Path(path).expanduser().read_bytes()
    return json.loads(gzip.decompress(data) if path.endswith(".gz") else data)


def write_corpora(path: str, corpora: dict) -> None:
    data = json.dumps(corpora, sort_keys=True, separators=(",", ":")).encode()
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(data, mtime=0) if path.endswith(".gz") else data)


def corpora_sha256(corpora: dict) -> str:
    """SHA-256 of the labelled items (canonical JSON), independent of how the file was compressed."""
    items = {k: v["items"] for k, v in sorted(corpora.items())}
    return hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_corpora(args) -> dict:
    if args.corpus_in:
        loaded = read_corpora(args.corpus_in)
        keep = ("main", "stress") if args.corpus == "both" else (args.corpus,)
        out = {k: v for k, v in loaded.items() if k in keep}
        if args.max_messages:
            for v in out.values():
                v["items"] = v["items"][:args.max_messages]     # a prefix: per-seed order is preserved
        return out
    out = {}
    if args.corpus in ("main", "both"):
        seeds = [E.SEED_BASE + 1 + i for i in range(args.seeds)]
        print(f"building main corpus: seeds {seeds[0]}..{seeds[-1]}, {args.episode_length} steps", flush=True)
        out["main"] = E.build_corpus(seeds, episode_length=args.episode_length, max_messages=args.max_messages)
    if args.corpus in ("stress", "both"):
        seeds = [E.SEED_BASE + 101 + i for i in range(args.stress_seeds)]
        print(f"building stress corpus: seeds {seeds[0]}..{seeds[-1]}, 900 steps", flush=True)
        out["stress"] = E.build_corpus(seeds, episode_length=900, config=STRESS, max_messages=args.max_messages)
    return out


def make_organ(args) -> LanguageOrgan:
    if args.backend == "dev":
        return LanguageOrgan.from_config({"backend": "dev"})
    if not args.model:
        raise SystemExit("--backend hf needs --model")
    local = args.local_dir or str(Path(args.models_dir).expanduser() / model_dir_name(args.model))
    cfg = {"backend": "hf", "model_id": args.model, "revision": args.revision, "local_dir": local,
           "device": args.device, "dtype": args.dtype, "max_attempts": args.max_attempts,
           "hash_weights": args.hash_weights, "audit_path": args.audit_out, "threads": args.threads}
    if args.n_shots is not None:
        cfg["n_shots"] = args.n_shots
    if args.max_new_tokens:
        cfg["max_new_tokens"] = args.max_new_tokens
    return LanguageOrgan.from_config(cfg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["dev", "hf"], default="dev")
    ap.add_argument("--model", help="Hugging Face model id (hf backend)")
    ap.add_argument("--revision", default=None, help="expected commit sha (checked against models.lock.json)")
    ap.add_argument("--local-dir", default=None, help="model directory (default: <models-dir>/<org>--<name>)")
    ap.add_argument("--models-dir", default="~/cb_models")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--dtype", default="auto", help="auto | float32 | bfloat16 | float16")
    ap.add_argument("--hash-weights", choices=["full", "lock"], default="full")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--n-shots", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--corpus", choices=["main", "stress", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=20, help="main corpus episodes")
    ap.add_argument("--stress-seeds", type=int, default=10)
    ap.add_argument("--episode-length", type=int, default=3000)
    ap.add_argument("--max-messages", type=int, default=None,
                    help="cap per corpus (for slow backends); with --corpus-in, the first N items of each corpus")
    ap.add_argument("--corpus-in", default=None,
                    help="load corpora written by --corpus-out (.json or .json.gz), e.g. the committed "
                         "evidence/explore/language/w1_language_corpus.json.gz")
    ap.add_argument("--corpus-out", default=None, help="save the labelled corpora (.json or .json.gz)")
    ap.add_argument("--audit-out", default=None, help="hf backend: JSONL of every prompt/response/latency")
    ap.add_argument("--progress", type=int, default=0, help="print progress every N messages")
    ap.add_argument("--out", default=None, help="report path (default runs/explore/language/eval_<backend>.json)")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    corpora = build_corpora(args)
    if args.corpus_out:
        write_corpora(args.corpus_out, corpora)
    organ = make_organ(args)
    few = {m for m, _ in FEW_SHOT}
    results = {}
    for name, corpus in corpora.items():
        print(f"evaluating {name}: {len(corpus['items'])} messages", flush=True)
        r = E.evaluate(organ, corpus, progress=args.progress or None)
        results[name] = {"corpus": E.corpus_summary(corpus),
                         "items_identical_to_a_few_shot_example": sum(it["message"] in few for it in corpus["items"]),
                         **r}
        print(f"  exact match {r['exact_match']['correct']}/{r['exact_match']['total']}, deadline "
              f"{r['deadline_step_accuracy']['correct']}/{r['deadline_step_accuracy']['total']}, answers valid "
              f"{r['answers']['grammar_valid']['correct']}/{r['answers']['grammar_valid']['total']}", flush=True)
    life = [ev for rec in organ.records if rec["method"] == "interpret" for ev in rec["output"]][:15]
    t = time.perf_counter()
    story = organ.narrate(life)
    narrate_s = time.perf_counter() - t
    prov = organ.provenance()
    report = {
        "format": "cb-language-eval/1",
        "label": DEV_LABEL if args.backend == "dev" else
        f"{args.model} through HFBackend (transformers causal LM, greedy, few-shot, JSON + schema + bounded retry)",
        "measures": "interpret() on W1 caretaker messages vs. ground truth from W1's obligations layer; answer() "
                    "composed from an oracle memory and graded by W1's parser/grader; latency on this machine",
        "does_not_measure": "open-ended language (W1's caretaker is template-based), the individual's own memory, "
                            "or narrative quality (one sample is included, unscored)",
        "provenance": prov,
        "machine": {"python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor() or platform.machine()},
        "git": {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
        "command": " ".join(["scripts/explore/eval_language_organ.py"] + (argv if argv is not None else sys.argv[1:])),
        "corpus": {"source": args.corpus_in or "built by this run", "items_sha256": corpora_sha256(corpora)},
        "results": results,
        "narrate_sample": {"events": len(life), "text": story, "latency_s": narrate_s},
        "wall_seconds": time.perf_counter() - t0,
    }
    out = Path(args.out) if args.out else REPO / "runs" / "explore" / "language" / f"eval_{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")
    print(f"wrote {out} ({report['wall_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
