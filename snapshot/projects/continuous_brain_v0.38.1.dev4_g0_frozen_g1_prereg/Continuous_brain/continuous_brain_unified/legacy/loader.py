"""Import a verified legacy module from the legacy root without shadowing anything.

The legacy code is the frozen `continuous_brain` package. Importing a module
executes more than its own file: the package `__init__` files and every
in-package module it imports. So before importing, the loader walks the
module's imports (AST only, nothing executed) to the transitive closure inside
the package, and requires every one of those files to be listed in the
manifest entry with a matching SHA-256 (role "support"). Then:
  * if some other `continuous_brain` is already imported, the import is refused
  * bytecode writing is disabled, so no `__pycache__` appears under the root
  * the module file is re-hashed after import
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType

from continuous_brain_protocols import sha256_file

from .manifest import LegacyHashMismatch, ManifestEntry

LEGACY_PACKAGE = "continuous_brain"


class LegacyImportConflict(RuntimeError):
    pass


def _module_name(relpath: str) -> str:
    p = PurePosixPath(relpath)
    if p.suffix != ".py" or not p.parts or p.parts[0] != LEGACY_PACKAGE:
        raise ValueError(f"legacy modules live under {LEGACY_PACKAGE}/: {relpath!r}")
    parts = p.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _module_file(root: Path, dotted: str) -> str | None:
    parts = dotted.split(".")
    for cand in (PurePosixPath(*parts).with_suffix(".py"), PurePosixPath(*parts, "__init__.py")):
        if (root / cand).is_file():
            return str(cand)
    return None


def _package_chain(relpath: str) -> list[str]:
    parts = PurePosixPath(relpath).parts[:-1]
    return [str(PurePosixPath(*parts[:i], "__init__.py")) for i in range(1, len(parts) + 1)]


def package_dependencies(root: Path, relpath: str) -> set[str]:
    """Relative paths of every in-package file importing `relpath` would execute (excluding itself)."""
    deps: set[str] = set()
    todo, seen = [relpath], set()
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for init in _package_chain(rel):
            if (root / init).is_file() and init not in seen:
                deps.add(init)
                todo.append(init)
        package = _module_name(rel) if rel.endswith("__init__.py") else _module_name(rel).rsplit(".", 1)[0]
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")
                    base = base[:len(base) - (node.level - 1)]
                    mod = ".".join(base + ([node.module] if node.module else []))
                else:
                    mod = node.module or ""
                if mod.split(".")[0] != LEGACY_PACKAGE:
                    continue
                targets.append(mod)
                targets.extend(f"{mod}.{a.name}" for a in node.names)      # "from . import x" may name a module
            elif isinstance(node, ast.Import):
                targets.extend(a.name for a in node.names if a.name.split(".")[0] == LEGACY_PACKAGE)
            for t in targets:
                f = _module_file(root, t)
                if f is not None and f != relpath and f not in seen:
                    deps.add(f)
                    todo.append(f)
    deps.discard(relpath)
    return deps


def import_verified_module(entry: ManifestEntry, root: Path) -> ModuleType:
    mod_file = entry.file("module")
    path = (root / mod_file.relpath).resolve()
    if sha256_file(path) != mod_file.sha256:
        raise LegacyHashMismatch(f"{mod_file.relpath} changed since verification")
    listed = {f.relpath: f.sha256 for f in entry.files}
    for dep in sorted(package_dependencies(root, mod_file.relpath)):
        if dep not in listed:
            raise LegacyImportConflict(f"importing {mod_file.relpath} would execute {dep}, which the manifest "
                                       f"entry does not hash (add it with role 'support')")
        if sha256_file(root / dep) != listed[dep]:
            raise LegacyHashMismatch(f"{dep} changed since verification")
    existing = sys.modules.get(LEGACY_PACKAGE)
    if existing is not None:
        loc = Path(getattr(existing, "__file__", "") or "").resolve()
        if root not in loc.parents:
            raise LegacyImportConflict(f"a different {LEGACY_PACKAGE!r} is already imported from {loc}")
    sys.path.insert(0, str(root))
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                   # nothing is written under the legacy root
    try:
        module = importlib.import_module(_module_name(mod_file.relpath))
    finally:
        sys.dont_write_bytecode = old_bytecode
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    loaded = Path(module.__file__).resolve()
    if loaded != path:
        raise LegacyImportConflict(f"imported {loaded}, expected {path}")
    if sha256_file(path) != mod_file.sha256:
        raise LegacyHashMismatch(f"{mod_file.relpath} changed during import")
    return module
