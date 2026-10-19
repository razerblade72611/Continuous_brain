#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.metadata as md, platform
from pathlib import Path

PKGS=("torch","torch-directml","numpy")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="constraints/directml.local.txt")
    args=ap.parse_args()
    rows=[f"# python={platform.python_version()} platform={platform.platform()}"]
    missing=[]
    for p in PKGS:
        try: rows.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError: missing.append(p)
    if missing:
        raise SystemExit("refusing to invent a DirectML lock; missing installed packages: "+", ".join(missing))
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(rows)+"\n")
    print(out); print(out.read_text(), end="")
if __name__=="__main__": main()
