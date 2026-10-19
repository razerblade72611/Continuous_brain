#!/usr/bin/env python3
"""
Cross-arm qualification equivalence check for recall comparison v2.

Because M2's normalized gate starts with multiplier 1.0 and cannot learn until
after update-3 backward, the linear and gated arms should have matching
forward/gradient behavior through the first three qualification forwards,
except for the gated arm's own newly nonzero gate gradients at update 3.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


FIELDS = (
    "update1_loss",
    "update1_candidate_grad_max",
    "update2_loss",
    "update2_memory_max",
    "update2_candidate_grad_max",
    "update2_write_gate_grad_max",
    "update2_recall_projection_grad_max",
    "update3_loss",
    "update3_memory_max",
    "update3_recall_max",
    "update3_candidate_grad_max",
    "update3_write_gate_grad_max",
    "update3_recall_projection_grad_max",
)

RTOL = 2e-5
ATOL = 1e-9


def parse(path):
    text = Path(path).read_text()
    out = {}
    for field in FIELDS:
        m = re.search(
            rf"^{re.escape(field)}:\s*([-+0-9.eE]+)\s*$",
            text,
            flags=re.MULTILINE,
        )
        if not m:
            raise RuntimeError(f"{path}: missing {field}")
        out[field] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", required=True)
    ap.add_argument("--gated", required=True)
    args = ap.parse_args()

    a = parse(args.linear)
    b = parse(args.gated)

    print("WORKING_MEMORY_RECALL_CROSS_ARM_EQUIVALENCE_V2")
    for field in FIELDS:
        av = a[field]
        bv = b[field]
        ok = math.isclose(av, bv, rel_tol=RTOL, abs_tol=ATOL)
        print(
            f"{field}: linear={av:.12g} gated={bv:.12g} "
            f"match={str(ok).lower()}"
        )
        if not ok:
            raise AssertionError(
                f"neutral gated arm not matched for {field}: "
                f"{av} vs {bv}"
            )

    print("NEUTRAL_M2_MATCHES_M15_THROUGH_UPDATE3_FORWARD: true")
    print("WORKING_MEMORY_RECALL_CROSS_ARM_EQUIVALENCE_OK")


if __name__ == "__main__":
    main()
