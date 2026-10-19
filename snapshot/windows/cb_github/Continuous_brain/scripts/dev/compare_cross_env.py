#!/usr/bin/env python3
"""Compare an E1 report produced on another machine with the frozen cloud G0-route hashes.

Prints, per mechanism: whether the other machine's unmodified G0 route reproduced the cloud
result.json byte-for-byte, and whether its own E1 passed. ENGINEERING_ONLY.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: compare_cross_env.py runs/dev/<tag>_e1_report.json")
    rep = json.loads(Path(sys.argv[1]).read_text())
    ref = json.loads((REPO / "evidence/dev6/cloud_g0_route_hashes.json").read_text())["by_n"].get(str(rep["n"]))
    if ref is None:
        raise SystemExit(f"no cloud reference for n={rep['n']}")
    ok = True
    for row in rep["rows"]:
        exp = ref[row["mechanism"]]
        same = row["g0_sha"] == exp["raw_result_sha256"] and row["seeds"] == exp["seeds"]
        ok &= same and row["E1"] == "PASS"
        print(f"{row['mechanism']:45s} cross-env G0 identical: {same!s:5s}  local E1: {row['E1']}")
    print("CROSS_ENV_ALL", "PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
