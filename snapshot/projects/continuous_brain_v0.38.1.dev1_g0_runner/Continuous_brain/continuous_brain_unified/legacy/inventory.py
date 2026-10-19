#!/usr/bin/env python3
"""Read-only inventory of a legacy continuous_brain tree (DAV-74 starting point).

Standard library only, so it can be copied onto any machine and run with the
system Python against the canonical WSL tree or a mirror.

For every candidate mechanism in the candidate spec it records, without
writing anything under the root:
  * module / evaluator / trainer / runner / test / protocol files: SHA-256,
    size, and whether the legacy tree's own manifests (MANIFEST_SHA256.json,
    SHA256SUMS*.txt, RELEASE_MANIFEST*.json) list the same hash
  * module API: classes (bases, public methods and their arguments), top-level
    functions, and intra-package imports, from the AST (nothing is imported)
  * result directories: every file's SHA-256; result.json's format, every
    top-level `*_passed` flag, split/scope/seed information, and every
    SHA-256 the result itself recorded, checked against the files present
  * result bundles: SHA-256 and member list
  * the protected base checkpoint

Every candidate is UNVERIFIED. Promotion to the manifest is a DAV-74 decision
made against the canonical tree, not something this tool can do.

Usage:
  python3 inventory.py --root <legacy tree> --spec legacy/candidate_spec.json \
      --root-label wsl-canonical --out legacy/candidates.json
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "cb-legacy-candidates/1"
SPEC_FORMAT = "cb-legacy-candidate-spec/1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GROUPS = ("module", "support", "evaluator", "trainer", "runner", "test", "protocol")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Tree:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__", ".git"))
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if p.is_symlink():
                    continue
                self.files.append(p.relative_to(self.root).as_posix())
        self._hash_cache: dict[str, str] = {}
        self.legacy_hashes, self.legacy_manifest_files = self._legacy_manifests()

    def sha(self, rel: str) -> str:
        if rel not in self._hash_cache:
            self._hash_cache[rel] = sha256_file(self.root / rel)
        return self._hash_cache[rel]

    def inside(self, rel: str) -> bool:
        """True if rel is a relative path that stays inside the root (nothing outside is ever read)."""
        if not rel or rel.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", rel) or ".." in Path(rel).parts:
            return False
        try:
            (self.root / rel).resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    def exists(self, rel: str) -> bool:
        if not self.inside(rel):
            return False
        p = (self.root / rel)
        return p.is_file() and not p.is_symlink()

    def glob(self, pattern: str) -> list[str]:
        return [f for f in self.files if fnmatch.fnmatchcase(f, pattern)]

    def _legacy_manifests(self) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
        """{relpath: {manifest name: sha}} from every manifest the legacy tree ships."""
        out: dict[str, dict[str, str]] = {}
        files: dict[str, str] = {}
        for rel in self.files:
            if "/" in rel:
                continue
            name = rel
            try:
                if re.match(r"^(MANIFEST_SHA256|RELEASE_MANIFEST.*|.*_SHA256)\.json$", name):
                    data = json.loads((self.root / rel).read_text(encoding="utf-8"))
                    table = data.get("files", data) if isinstance(data, dict) else {}
                    for k, v in table.items():
                        if isinstance(v, str) and HEX64.match(v):
                            out.setdefault(k, {})[name] = v
                    files[name] = self.sha(rel)
                elif re.match(r"^SHA256SUMS.*\.txt$", name):
                    for line in (self.root / rel).read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2 and HEX64.match(parts[0]):
                            out.setdefault(parts[1].lstrip("*").strip(), {})[name] = parts[0]
                    files[name] = self.sha(rel)
            except (ValueError, OSError):
                files[name] = "UNREADABLE"
        return out, files


def module_api(path: Path) -> dict:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as e:
        return {"parse_error": str(e)}

    def args(fn: ast.FunctionDef) -> list[str]:
        a = fn.args
        return [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]

    classes, functions, imports = [], [], set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [{"name": n.name, "args": args(n)} for n in node.body
                       if isinstance(n, ast.FunctionDef) and (not n.name.startswith("_") or n.name == "__init__")]
            classes.append({"name": node.name, "bases": [ast.unparse(b) for b in node.bases], "methods": methods})
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            functions.append({"name": node.name, "args": args(node)})
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            if node.level or (node.module or "").startswith("continuous_brain"):
                imports.add(mod)
    return {"classes": classes, "functions": functions, "package_imports": sorted(imports)}


def file_record(tree: Tree, rel: str, with_api: bool = False) -> dict:
    if not tree.inside(rel):
        return {"relpath": rel, "exists": False, "outside_root": True}
    rec = {"relpath": rel, "exists": tree.exists(rel)}
    if not rec["exists"]:
        return rec
    rec["sha256"] = tree.sha(rel)
    rec["bytes"] = (tree.root / rel).stat().st_size
    listed = tree.legacy_hashes.get(rel, {})
    rec["legacy_manifests"] = {m: ("match" if h == rec["sha256"] else "MISMATCH") for m, h in sorted(listed.items())}
    if with_api and rel.endswith(".py"):
        rec["api"] = module_api(tree.root / rel)
    return rec


def _walk_hashes(obj, path: str = "", key: str = "") -> list[tuple[str, str, str]]:
    """Every 64-hex string inside a JSON value as (JSON path, its key, digest)."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(_walk_hashes(v, f"{path}/{k}", str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:64]):
            found.extend(_walk_hashes(v, f"{path}[{i}]", key))
    elif isinstance(obj, str) and HEX64.match(obj):
        found.append((path, key, obj))
    return found


def result_summary(tree: Tree, rel: str) -> dict:
    out: dict = {"relpath": rel, "sha256": tree.sha(rel)}
    try:
        data = json.loads((tree.root / rel).read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        out["parse_error"] = str(e)
        return out
    if not isinstance(data, dict):
        return out
    out["format"] = data.get("format")
    out["passed_flags"] = {k: v for k, v in data.items() if k.endswith("_passed")}
    for key in ("split", "scope", "family", "historical_status", "fresh_causal_seed_ranges"):
        if isinstance(data.get(key), (dict, str)):
            out[key] = data[key]
    for key in ("training_performed", "protected_unchanged"):
        if key in data:
            out[key] = data[key]
    checks = {}
    for jpath, key, digest in _walk_hashes(data):
        entry = {"sha256": digest, "file_check": None}
        if "/" in key or key.endswith((".pt", ".json", ".py", ".zip", ".npz")):
            entry["file_check"] = ("match" if tree.sha(key) == digest else "MISMATCH") if tree.exists(key) else "absent"
        checks[jpath] = entry
    out["recorded_hashes"] = checks
    return out


def cross_reference(tree: Tree, record: dict) -> None:
    """Mark recorded hashes that equal a file this run already hashed (cheap reverse lookup)."""
    known = {}
    for rel, digest in tree._hash_cache.items():
        known.setdefault(digest, []).append(rel)
    for d in record["result_dirs"]:
        for r in d.get("results", []):
            for entry in r.get("recorded_hashes", {}).values():
                if entry["sha256"] in known:
                    entry["equals_file"] = sorted(known[entry["sha256"]])


def result_dir_record(tree: Tree, rel_dir: str) -> dict:
    if not tree.inside(rel_dir):
        return {"relpath": rel_dir, "exists": False, "outside_root": True, "files": [], "checkpoints_present": []}
    prefix = rel_dir.rstrip("/") + "/"
    members = [f for f in tree.files if f.startswith(prefix)]
    rec: dict = {"relpath": rel_dir, "exists": bool(members), "files": []}
    for f in members:
        entry = {"relpath": f, "sha256": tree.sha(f), "bytes": (tree.root / f).stat().st_size}
        if f.endswith(".zip"):
            try:
                with zipfile.ZipFile(tree.root / f) as z:
                    infos = z.infolist()
                    entry["members"] = [{"name": i.filename, "bytes": i.file_size} for i in infos[:200]]
                    entry["member_count"] = len(infos)
            except zipfile.BadZipFile as e:
                entry["zip_error"] = str(e)
        rec["files"].append(entry)
        if f.endswith("result.json"):
            rec.setdefault("results", []).append(result_summary(tree, f))
    rec["checkpoints_present"] = [f for f in members if f.endswith((".pt", ".pth", ".ckpt"))]
    return rec


def inventory(root: Path, spec: dict, root_label: str) -> dict:
    if spec.get("format") != SPEC_FORMAT:
        raise SystemExit(f"candidate spec must have format {SPEC_FORMAT!r}")
    tree = Tree(root)
    counts: dict[str, int] = {}
    for f in tree.files:
        ext = Path(f).suffix.lower() or "(none)"
        counts[ext] = counts.get(ext, 0) + 1
    candidates = []
    for c in spec["candidates"]:
        rec = {k: c[k] for k in ("mechanism_id", "slot_options", "lineage_version", "dossier_status", "admissible")}
        rec["notes"] = list(c.get("notes", []))
        for group in GROUPS:
            rels: list[str] = []
            for item in c.get(group, []):
                rels.extend(tree.glob(item) if any(ch in item for ch in "*?[") else [item])
            rec[group] = [file_record(tree, r, with_api=group in ("module", "support")) for r in sorted(set(rels))]
        rec["result_dirs"] = [result_dir_record(tree, d) for d in c.get("result_dirs", [])]
        candidates.append(rec)
    for rec, c in zip(candidates, spec["candidates"]):
        cross_reference(tree, rec)
        reasons = [f"hashes computed on {root_label!r}; DAV-74 must confirm them on the canonical tree"]
        if not any(r.get("exists") for r in rec["module"]):
            reasons.append("module not found in this tree")
        if not any(r.get("exists") for r in rec["evaluator"]):
            reasons.append("evaluator not found in this tree")
        if not any(d["checkpoints_present"] for d in rec["result_dirs"]):
            reasons.append("no checkpoint files present: learned state cannot be loaded from this tree")
        for d in rec["result_dirs"]:
            for r in d.get("results", []):
                failed = [k for k, v in r.get("passed_flags", {}).items() if v is False]
                if failed:
                    reasons.append(f"{r['relpath']} records {failed} = false")
        if not c.get("admissible", True):
            reasons.append("formal failure: regression check only, never a capability")
        rec["status"] = "UNVERIFIED"
        rec["unverified_reasons"] = reasons
    protected = spec.get("protected_checkpoint")
    prot_rec = None
    if protected:
        rel = protected["relpath"]
        prot_rec = {"relpath": rel, "expected_sha256": protected["sha256"], "exists": tree.exists(rel),
                    "outside_root": not tree.inside(rel)}
        if prot_rec["exists"]:
            prot_rec["sha256"] = tree.sha(rel)
            prot_rec["matches"] = prot_rec["sha256"] == protected["sha256"]
    return {"format": FORMAT, "root_label": root_label,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool_sha256": sha256_file(Path(__file__).resolve()),
            "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            "tree_summary": {"files": len(tree.files), "by_extension": dict(sorted(counts.items()))},
            "legacy_manifests": tree.legacy_manifest_files, "protected_checkpoint": prot_rec,
            "candidates": candidates}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="legacy tree (read only)")
    ap.add_argument("--spec", required=True, help="candidate spec JSON")
    ap.add_argument("--root-label", required=True, help="a name for the tree, recorded instead of its path")
    ap.add_argument("--out", required=True, help="output JSON (must be outside the root)")
    args = ap.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if root == out or root in out.parents:
        print("refusing to write inside the legacy root", file=sys.stderr)
        return 2
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    result = inventory(root, spec, args.root_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    n = len(result["candidates"])
    found = sum(any(m.get("exists") for m in c["module"]) for c in result["candidates"])
    print(f"{n} candidates, {found} with a module present; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
