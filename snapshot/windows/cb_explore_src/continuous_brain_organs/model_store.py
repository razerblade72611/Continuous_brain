"""Local model store: file hashing, the ``models.lock.json`` format, and trust-on-first-use verification.

Used by ``scripts/explore/fetch_models.py`` (which downloads and writes the lock on the owner's machine), by
``scripts/explore/reassemble_model.py`` and by ``language.HFBackend`` (which refuses weights that disagree with
the lock).  Nothing here touches the network.

Lock format (``models.lock.json``, next to the model directories)::

    {"format": "cb-models-lock/1",
     "models": {"Qwen/Qwen2.5-0.5B-Instruct": {
         "dir": "Qwen--Qwen2.5-0.5B-Instruct",            # relative to the lock's directory
         "revision": "<40-hex commit sha resolved at first download>",
         "files": {"model.safetensors": {"size": 988097824, "sha256": "<hex>"}, ...},
         "hub_lfs_sha256_checked": 1,                      # files whose Hub LFS sha256 agreed at download
         "first_fetched_utc": "...", "huggingface_hub": "1.32.0"}}}

Trust on first use: the first download records the revision and every file's SHA-256; every later run (and
every load by ``HFBackend``) checks the files against that record and refuses on any difference -- a changed,
missing or extra file, or a different revision.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path

LOCK_FORMAT = "cb-models-lock/1"
LOCK_NAME = "models.lock.json"
# huggingface_hub keeps download bookkeeping under <local_dir>/.cache; it is not part of the model.
IGNORED_DIRS = {".cache", "__pycache__"}
IGNORED_FILES = {LOCK_NAME, ".DS_Store"}


def model_dir_name(model_id: str) -> str:
    """Directory name of a model inside the models directory (``org/name`` -> ``org--name``)."""
    if not model_id or model_id.count("/") != 1 or ".." in model_id:
        raise ValueError(f"not a Hugging Face model id: {model_id!r}")
    return model_id.replace("/", "--")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_files(root: Path) -> list[str]:
    """Every regular file under ``root`` (relative POSIX paths, sorted), skipping download bookkeeping."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in filenames:
            if name in IGNORED_FILES:
                continue
            p = Path(dirpath) / name
            if p.is_file():
                out.append(p.relative_to(root).as_posix())
    return sorted(out)


def hash_tree(root: Path) -> dict:
    """``{relpath: {"size": bytes, "sha256": hex}}`` for every file under ``root``."""
    root = Path(root)
    return {rel: {"size": (root / rel).stat().st_size, "sha256": sha256_file(root / rel)} for rel in list_files(root)}


def files_digest(files: dict) -> str:
    """One SHA-256 over the sorted ``(path, sha256)`` pairs: a short name for a whole set of weights."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(files[rel]["sha256"].encode())
        h.update(b"\n")
    return h.hexdigest()


def load_lock(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"format": LOCK_FORMAT, "models": {}}
    lock = json.loads(path.read_text())
    if lock.get("format") != LOCK_FORMAT:
        raise ValueError(f"{path}: unknown lock format {lock.get('format')!r}")
    return lock


def save_lock(path: Path, lock: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def compare_files(expected: dict, actual: dict) -> list[str]:
    """Human-readable differences between two ``hash_tree`` results (empty list = identical)."""
    problems = []
    for rel in sorted(set(expected) - set(actual)):
        problems.append(f"missing file {rel}")
    for rel in sorted(set(actual) - set(expected)):
        problems.append(f"unexpected file {rel}")
    for rel in sorted(set(expected) & set(actual)):
        e, a = expected[rel], actual[rel]
        if e["size"] != a["size"]:
            problems.append(f"size of {rel}: lock {e['size']} != disk {a['size']}")
        elif e["sha256"] != a["sha256"]:
            problems.append(f"sha256 of {rel}: lock {e['sha256'][:16]}... != disk {a['sha256'][:16]}...")
    return problems


def is_ignored(rel: str, patterns) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in patterns or ())


def verify_entry(models_dir: Path, entry: dict, full: bool = True) -> tuple[list[str], dict]:
    """Check a model directory against its lock entry.

    ``full=True`` re-hashes every file; ``full=False`` checks names and sizes only and returns the lock's hashes
    (fast for multi-GB weights that were fully verified before).  Files matching the entry's
    ``ignore_patterns`` (the ones fetch_models never downloads, e.g. ``onnx/*``) are not part of the model.
    Returns ``(problems, files)``."""
    d = Path(models_dir) / entry["dir"]
    if not d.is_dir():
        return [f"model directory {d} does not exist"], {}
    ignore = entry.get("ignore_patterns", ())
    rels = [r for r in list_files(d) if not is_ignored(r, ignore)]
    if full:
        actual = {rel: {"size": (d / rel).stat().st_size, "sha256": sha256_file(d / rel)} for rel in rels}
    else:
        actual = {}
        for rel in rels:
            size = (d / rel).stat().st_size
            want = entry["files"].get(rel)
            actual[rel] = {"size": size, "sha256": want["sha256"] if want and want["size"] == size else "?"}
    return compare_files(entry["files"], actual), actual


def find_lock_entry(local_dir: Path, model_id: str, lock_path: Path | None = None) -> tuple[Path | None, dict | None]:
    """Locate the lock entry describing ``local_dir`` (default lock: ``<local_dir>/../models.lock.json``)."""
    local_dir = Path(local_dir).resolve()
    lock_path = Path(lock_path) if lock_path else local_dir.parent / LOCK_NAME
    if not lock_path.exists():
        return None, None
    lock = load_lock(lock_path)
    entry = lock["models"].get(model_id)
    if entry is None or (lock_path.parent / entry["dir"]).resolve() != local_dir:
        return lock_path, None
    return lock_path, entry


# --- moving weights through a channel with a per-file size limit -----------------------------------------
SPLIT_FORMAT = "cb-model-split/1"
SPLIT_MANIFEST = "split_manifest.json"
_BLOCK = 1 << 22


def split_model(models_dir: Path, model_id: str, entry: dict, out_dir: Path, chunk_mb: int) -> Path:
    """Write every file of a locked model as ``<relpath>.partNNNN`` chunks of at most ``chunk_mb`` MiB under
    ``out_dir/<entry dir>/``, plus ``split_manifest.json`` (per-chunk and per-file SHA-256, the revision).
    The source files are re-hashed while splitting and must equal the lock.  Returns the manifest path."""
    if chunk_mb < 1:
        raise ValueError("chunk_mb must be >= 1")
    chunk = int(chunk_mb) << 20
    src = Path(models_dir) / entry["dir"]
    dst = Path(out_dir) / entry["dir"]
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {"format": SPLIT_FORMAT, "model_id": model_id, "dir": entry["dir"], "revision": entry["revision"],
                "chunk_bytes": chunk, "files": {}}
    for rel in sorted(entry["files"]):
        want = entry["files"][rel]
        whole = hashlib.sha256()
        parts, size, i = [], 0, 0
        with open(src / rel, "rb") as f:
            while True:
                name = f"{rel}.part{i:04d}"
                (dst / name).parent.mkdir(parents=True, exist_ok=True)
                h, n = hashlib.sha256(), 0
                with open(dst / name, "wb") as out:
                    while n < chunk:
                        b = f.read(min(_BLOCK, chunk - n))
                        if not b:
                            break
                        out.write(b)
                        h.update(b)
                        whole.update(b)
                        n += len(b)
                if n == 0 and i > 0:
                    (dst / name).unlink()
                    break
                parts.append({"name": name, "size": n, "sha256": h.hexdigest()})
                size += n
                i += 1
                if n < chunk:
                    break
        if size != want["size"] or whole.hexdigest() != want["sha256"]:
            raise ValueError(f"{rel}: the file on disk no longer matches the lock; refusing to split it")
        manifest["files"][rel] = {"size": size, "sha256": want["sha256"], "parts": parts}
    path = dst / SPLIT_MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def reassemble_model(split_dir: Path, models_dir: Path, lock_path: Path | None = None) -> dict:
    """Rebuild a model split by ``split_model`` into ``models_dir/<dir>``, verifying every chunk and every
    file, then record (or check) it in ``models_dir/models.lock.json``.  Raises ``ValueError`` on any mismatch
    and leaves no partially written file behind."""
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / SPLIT_MANIFEST).read_text())
    if manifest.get("format") != SPLIT_FORMAT:
        raise ValueError(f"{split_dir}: not a {SPLIT_FORMAT} manifest")
    lock_path = Path(lock_path) if lock_path else Path(models_dir) / LOCK_NAME
    lock = load_lock(lock_path)
    old = lock["models"].get(manifest["model_id"])
    if old is not None and (old["revision"] != manifest["revision"] or
                            compare_files(old["files"], {k: {"size": v["size"], "sha256": v["sha256"]}
                                                         for k, v in manifest["files"].items()})):
        raise ValueError(f"{manifest['model_id']}: the lock at {lock_path} pins different files or revision")
    dst = Path(models_dir) / manifest["dir"]
    for rel, meta in sorted(manifest["files"].items()):
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".reassembling")
        whole = hashlib.sha256()
        try:
            with open(tmp, "wb") as out:
                for part in meta["parts"]:
                    h, n = hashlib.sha256(), 0
                    with open(split_dir / part["name"], "rb") as f:
                        while True:
                            b = f.read(_BLOCK)
                            if not b:
                                break
                            out.write(b)
                            h.update(b)
                            whole.update(b)
                            n += len(b)
                    if n != part["size"] or h.hexdigest() != part["sha256"]:
                        raise ValueError(f"chunk {part['name']} is corrupt (size or sha256 differs from the manifest)")
            if whole.hexdigest() != meta["sha256"] or tmp.stat().st_size != meta["size"]:
                raise ValueError(f"{rel}: reassembled file does not match the manifest sha256")
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()
    entry = {"dir": manifest["dir"], "revision": manifest["revision"],
             "files": {k: {"size": v["size"], "sha256": v["sha256"]} for k, v in manifest["files"].items()},
             "source": "reassembled from split chunks"}
    problems, _ = verify_entry(Path(models_dir), entry, full=True)
    if problems:
        raise ValueError(f"{dst}: after reassembly: " + "; ".join(problems[:5]))
    if old is None:
        lock["models"][manifest["model_id"]] = entry
        save_lock(lock_path, lock)
    return entry
