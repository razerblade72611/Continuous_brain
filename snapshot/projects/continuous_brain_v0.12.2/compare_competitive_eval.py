#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def pct(x):
    return f"{100.0*x:.1f}%" if x is not None else "-"


def f4(x):
    return f"{x:.4f}" if x is not None else "-"


def f6(x):
    return f"{x:.6f}" if x is not None else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    rows = []
    for name in args.files:
        data = json.loads(Path(name).read_text())
        condition = data["condition"]
        for length, s in data["lengths"].items():
            rows.append({
                "condition": condition,
                "length": length,
                "success": s["closed_loop_success_rate"],
                "return": s["closed_loop_mean_return"],
                "steps": s["closed_loop_mean_steps"],
                "motor_loss": s["teacher_motor_loss"],
                "world_loss": s["teacher_world_loss"],
                "action_acc": s["teacher_action_accuracy"],
                "entropy": s["teacher_selector_entropy"],
            })

    rows.sort(key=lambda r: (float(r["length"]), r["condition"]))

    header = (
        f"{'length':>7}  {'condition':>11}  {'success':>8}  "
        f"{'return':>8}  {'steps':>7}  {'motor':>10}  "
        f"{'world':>10}  {'act_acc':>8}  {'entropy':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['length']:>7}  {r['condition']:>11}  "
            f"{pct(r['success']):>8}  {f4(r['return']):>8}  "
            f"{f4(r['steps']):>7}  {f6(r['motor_loss']):>10}  "
            f"{f6(r['world_loss']):>10}  {pct(r['action_acc']):>8}  "
            f"{f4(r['entropy']):>9}"
        )


if __name__ == "__main__":
    main()
