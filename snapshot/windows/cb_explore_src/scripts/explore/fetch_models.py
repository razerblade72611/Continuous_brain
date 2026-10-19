"""Download the language/vision-language organ's open-weight models and pin them (trust on first use).

Run this on a machine with internet access to the Hugging Face Hub (the owner's WSL, or a rented GPU box):

    python scripts/explore/fetch_models.py --set small                  # Qwen2.5-0.5B + SmolLM2-360M
    python scripts/explore/fetch_models.py --model Qwen/Qwen2.5-1.5B-Instruct
    python scripts/explore/fetch_models.py --set small --verify-only    # re-hash everything, no network
    python scripts/explore/fetch_models.py --set small --split-mb 300   # also write <=300 MiB chunks

What it does, per model:

1. First download: resolves the requested revision (default ``main``) to its exact commit SHA on the Hub,
   downloads that commit with ``huggingface_hub.snapshot_download`` into ``<models-dir>/<org>--<name>/``
   (ONNX exports and Meta's ``original/`` checkpoint are skipped), SHA-256-hashes every file, cross-checks the
   hashes of large (LFS) files against the Hub's own LFS SHA-256, and records revision + every file's size and
   SHA-256 in ``<models-dir>/models.lock.json``.
2. Later runs: the lock's revision is used (a different ``--revision`` is refused unless ``--repin``); after
   the (idempotent) download every file is re-hashed and must equal the lock -- a changed, missing or extra
   file makes the script refuse (exit 2) and leave the lock untouched.
3. ``--split-mb N`` additionally writes each file as N-MiB chunks plus ``split_manifest.json`` under
   ``<models-dir>/_split/<org>--<name>/``; move that folder through any channel with a per-file size limit and
   rebuild + verify it on the other side with ``scripts/explore/reassemble_model.py``.

Gated models (meta-llama/Llama-3.1-8B-Instruct): accept the licence on the model's Hub page first, then export
``HF_TOKEN=<a read token>`` (or pass ``--token-env NAME``).  No network access happens at import time.

Exit codes: 0 ok; 2 verification failed / refused; 3 bad arguments or revision conflict; 4 download error.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from continuous_brain_organs import model_store as ms  # noqa: E402

CARDS = REPO / "continuous_brain_organs" / "model_cards.json"
DEFAULT_MODELS_DIR = "~/cb_models"


def load_cards(path: Path = CARDS) -> dict:
    cards = json.loads(Path(path).read_text())
    return cards


# --- network operations (module-level so tests can replace them; huggingface_hub imported lazily) ---------
def resolve_revision(model_id: str, revision: str, token: str | None) -> tuple[str, dict]:
    """Return (commit sha, {path: hub lfs sha256}) for ``revision`` of ``model_id``."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True, token=token)
    lfs = {}
    for s in info.siblings or []:
        meta = getattr(s, "lfs", None)
        sha = meta.get("sha256") if isinstance(meta, dict) else getattr(meta, "sha256", None)
        if sha:
            lfs[s.rfilename] = sha
    return info.sha, lfs


def download(model_id: str, revision: str, dest: Path, ignore_patterns: list, token: str | None) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_id, revision=revision, local_dir=str(dest), ignore_patterns=ignore_patterns,
                      token=token)


# --- main ----------------------------------------------------------------------------------------------
def fetch_one(model_id: str, models_dir: Path, lock: dict, args, ignore: list, token: str | None) -> int:
    name = ms.model_dir_name(model_id)
    dest = models_dir / name
    entry = lock["models"].get(model_id)
    if args.verify_only:
        if entry is None:
            print(f"[{model_id}] not in the lock; nothing to verify", file=sys.stderr)
            return 2
        problems, _ = ms.verify_entry(models_dir, entry, full=True)
        if problems:
            print(f"[{model_id}] VERIFICATION FAILED against {models_dir / ms.LOCK_NAME}:", file=sys.stderr)
            for p in problems:
                print(f"    {p}", file=sys.stderr)
            return 2
        print(f"[{model_id}] ok: {len(entry['files'])} files match the lock (revision {entry['revision']})")
        return 0
    requested = (args.revisions or {}).get(model_id)
    if entry is not None and requested and requested != entry["revision"] and not args.repin:
        print(f"[{model_id}] the lock pins revision {entry['revision']}; refusing --revision {requested} "
              "(pass --repin to replace the pin deliberately)", file=sys.stderr)
        return 3
    repin = entry is not None and args.repin
    try:
        if entry is not None and not repin:
            sha, hub_lfs = entry["revision"], None
        else:
            sha, hub_lfs = resolve_revision(model_id, requested or "main", token)
        print(f"[{model_id}] revision {sha}: downloading into {dest}", flush=True)
        download(model_id, sha, dest, ignore, token)
    except Exception as e:  # network, auth (gated), disk
        print(f"[{model_id}] download failed: {type(e).__name__}: {e}", file=sys.stderr)
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            print("    gated model: accept its licence on huggingface.co and export HF_TOKEN", file=sys.stderr)
        return 4
    files = {rel: meta for rel, meta in ms.hash_tree(dest).items() if not ms.is_ignored(rel, ignore)}
    if entry is not None and not repin:
        problems = ms.compare_files(entry["files"], files)
        if problems:
            print(f"[{model_id}] REFUSED: files differ from the lock (trust on first use broken):", file=sys.stderr)
            for p in problems:
                print(f"    {p}", file=sys.stderr)
            return 2
        print(f"[{model_id}] ok: {len(files)} files match the lock")
    else:
        bad = [rel for rel, h in (hub_lfs or {}).items() if rel in files and files[rel]["sha256"] != h]
        if bad:
            print(f"[{model_id}] REFUSED: downloaded files disagree with the Hub's LFS sha256: {bad}", file=sys.stderr)
            return 2
        checked = sum(1 for rel in (hub_lfs or {}) if rel in files)
        lock["models"][model_id] = {
            "dir": name, "revision": sha, "files": files, "hub_lfs_sha256_checked": checked,
            "first_fetched_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "huggingface_hub": _hub_version(), "ignore_patterns": ignore,
            **({"repinned_from": entry["revision"]} if repin else {})}
        ms.save_lock(models_dir / ms.LOCK_NAME, lock)
        print(f"[{model_id}] pinned (trust on first use): revision {sha}, {len(files)} files, "
              f"{sum(f['size'] for f in files.values()) / 1e9:.2f} GB, {checked} LFS hashes cross-checked")
    if args.split_mb:
        path = ms.split_model(models_dir, model_id, lock["models"][model_id], models_dir / "_split", args.split_mb)
        print(f"[{model_id}] split into <= {args.split_mb} MiB chunks: {path.parent}")
    return 0


def _hub_version() -> str:
    try:
        import huggingface_hub
        return huggingface_hub.__version__
    except ImportError:
        return "unknown"


def parse_args(argv):
    cards = load_cards()
    ids = [m["id"] for m in cards["models"]]
    ap = argparse.ArgumentParser(
        description="Download and pin (trust on first use) the open-weight models of the Continuous Brain "
                    "language organ; verify them against models.lock.json on every later run.",
        epilog="Models: " + ", ".join(ids) + ".  Sets: " +
               "; ".join(f"{k} = {', '.join(v)}" for k, v in cards["sets"].items()),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], metavar="ID", help="a model id (repeatable)")
    ap.add_argument("--set", choices=sorted(cards["sets"]) + ["all"], help="a named set of models")
    ap.add_argument("--models-dir", default=os.environ.get("CB_MODELS_DIR", DEFAULT_MODELS_DIR),
                    help="where the weights and models.lock.json live (default: $CB_MODELS_DIR or ~/cb_models)")
    ap.add_argument("--revision", action="append", default=[], metavar="ID=REV",
                    help="revision (branch, tag or commit) for a first download, e.g. "
                         "Qwen/Qwen2.5-0.5B-Instruct=main; refused if it conflicts with the lock")
    ap.add_argument("--repin", action="store_true", help="deliberately replace an existing pin with a new revision")
    ap.add_argument("--verify-only", action="store_true", help="no network: re-hash local files against the lock")
    ap.add_argument("--split-mb", type=int, default=0, metavar="N",
                    help="also write every file as N-MiB chunks + split_manifest.json under <models-dir>/_split/")
    ap.add_argument("--token-env", default="HF_TOKEN", help="environment variable holding a Hub token (gated models)")
    ap.add_argument("--allow-unlisted", action="store_true", help="allow a model id that is not in model_cards.json")
    ap.add_argument("--list", action="store_true", help="list the model cards and the lock status, then exit")
    args = ap.parse_args(argv)
    revs = {}
    for r in args.revision:
        if "=" not in r:
            ap.error(f"--revision wants ID=REV, got {r!r}")
        k, v = r.split("=", 1)
        revs[k] = v
    args.revisions = revs
    chosen = list(args.model)
    if args.set:
        chosen += ids if args.set == "all" else cards["sets"][args.set]
    args.chosen = list(dict.fromkeys(chosen))
    if not args.chosen and not args.list:
        ap.error("choose models with --model ID and/or --set NAME (or --list)")
    unknown = [m for m in args.chosen if m not in ids]
    if unknown and not args.allow_unlisted:
        ap.error(f"not in model_cards.json: {unknown} (pass --allow-unlisted to fetch anyway)")
    if args.split_mb < 0:
        ap.error("--split-mb must be positive")
    return args, cards


def main(argv=None) -> int:
    args, cards = parse_args(argv)
    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    lock = ms.load_lock(models_dir / ms.LOCK_NAME)
    if args.list:
        for m in cards["models"]:
            e = lock["models"].get(m["id"])
            status = f"pinned {e['revision'][:12]}" if e else "not downloaded"
            lic = m["licence"]
            flags = (" GATED" if lic["gated"] else "") + (" NON-APACHE" if lic["non_apache"] else "")
            print(f"{m['id']:40s} {m['role']:16s} {m['parameters'] / 1e9:5.2f}B "
                  f"{m['expected_disk_bytes'] / 1e9:6.2f} GB  {lic['spdx_or_name']}{flags}  [{status}]")
        return 0
    ignore = list(cards["default_ignore_patterns"])
    token = os.environ.get(args.token_env) or None
    worst = 0
    for model_id in args.chosen:
        worst = max(worst, fetch_one(model_id, models_dir, lock, args, ignore, token))
    return worst


if __name__ == "__main__":
    sys.exit(main())
