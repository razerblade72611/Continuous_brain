#!/usr/bin/env python3
"""Combine fixed-final TRAIN and representation-portability diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


RULES = ("invert_lr", "vertical", "vertical_reverse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--portability", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    train = json.loads(Path(args.train).read_text())
    dev = json.loads(Path(args.dev).read_text())
    port = json.loads(Path(args.portability).read_text())

    ts = train["summary"]
    ds = dev["summary"]
    pr = port["results"]

    train_rule_mean = sum(ts["normal"][r] for r in RULES) / len(RULES)
    dev_rule_mean = sum(ds["normal"][r] for r in RULES) / len(RULES)
    generalization_gap = train_rule_mean - dev_rule_mean

    raw_dev = pr["raw_memory"]["mean_dev_accuracy"]
    recall_dev = pr["canonical_recall"]["mean_dev_accuracy"]
    hidden_dev = pr["hidden"]["mean_dev_accuracy"]

    findings = []

    if train_rule_mean < 0.80 or ts["normal"]["six_way"] < 0.65:
        findings.append(
            "FINAL_TRAIN_UNDERLEARNED_OR_UNSTABLE"
        )
    elif generalization_gap > 0.15:
        findings.append(
            "LARGE_TRAIN_TO_DEV_EXECUTIVE_GENERALIZATION_GAP"
        )

    if raw_dev >= 0.95 and recall_dev < 0.90:
        findings.append(
            "RAW_MEMORY_PORTABLE_BUT_CANONICAL_RECALL_DEGRADES"
        )
    elif raw_dev >= 0.95 and recall_dev >= 0.95:
        findings.append(
            "BOTH_RAW_AND_CANONICAL_CONTENT_LINEarly_PORTABLE"
        )
    elif raw_dev < 0.90:
        findings.append(
            "WRITER_MEMORY_ITSELF_HAS_WEAK_V3_PORTABILITY"
        )

    if recall_dev >= 0.95 and dev_rule_mean < 0.75:
        findings.append(
            "CONTENT_PRESENT_BUT_EXECUTIVE_RULE_USE_GENERALIZES_POORLY"
        )

    payload = {
        "format": "canonical_failure_diagnosis_v1",
        "train_rule_pair_mean": train_rule_mean,
        "train_six_way": ts["normal"]["six_way"],
        "dev_rule_pair_mean": dev_rule_mean,
        "dev_six_way": ds["normal"]["six_way"],
        "train_to_dev_rule_gap": generalization_gap,
        "raw_memory_dev_cue_decode": raw_dev,
        "canonical_recall_dev_cue_decode": recall_dev,
        "hidden_dev_cue_decode": hidden_dev,
        "findings": findings,
        "fresh_test_touched": False,
    }

    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )

    print("CANONICAL_FAILURE_DIAGNOSIS_V1")
    print(
        f"TRAIN rule_pair_mean={train_rule_mean:.4f} "
        f"six_way={ts['normal']['six_way']:.4f}"
    )
    print(
        f"DEV rule_pair_mean={dev_rule_mean:.4f} "
        f"six_way={ds['normal']['six_way']:.4f}"
    )
    print(
        f"TRAIN_to_DEV_rule_gap={generalization_gap:+.4f}"
    )
    print(
        f"raw_memory_DEV_cue_decode={raw_dev:.4f}"
    )
    print(
        f"canonical_recall_DEV_cue_decode={recall_dev:.4f}"
    )
    print(
        f"hidden_DEV_cue_decode={hidden_dev:.4f}"
    )
    print("findings:")
    for item in findings:
        print(" -", item)
    print("FRESH_TEST_TOUCHED: false")
    print("CANONICAL_FAILURE_DIAGNOSIS_COMPLETE")


if __name__ == "__main__":
    main()
