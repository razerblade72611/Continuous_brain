#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import site
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Run a script with AUTHOR site-packages appended after the current venv."
    )
    ap.add_argument("--author-site", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("script_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    author_site = str(Path(args.author_site).expanduser().resolve())
    script = str(Path(args.script).expanduser().resolve())

    if author_site not in sys.path:
        # Append, do not prepend: current Continuous Brain packages (torch,
        # numpy, etc.) remain authoritative. AUTHOR is only a fallback.
        sys.path.append(author_site)

    try:
        import torch
        import ai2thor
        from ai2thor.controller import Controller
    except Exception as exc:
        print("RUNTIME_BRIDGE_IMPORT_FAILURE:", repr(exc))
        print("current_python:", sys.executable)
        print("author_site:", author_site)
        raise

    print("RUNTIME_BRIDGE_IMPORT: PASS")
    print("current_python:", sys.executable)
    print("torch_version:", getattr(torch, "__version__", "unknown"))
    print("torch_file:", getattr(torch, "__file__", "unknown"))
    print("ai2thor_file:", getattr(ai2thor, "__file__", "unknown"))
    print("author_site_appended_last: true")

    # Preserve evaluator CLI exactly.
    forwarded = list(args.script_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    sys.argv = [script] + forwarded
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
