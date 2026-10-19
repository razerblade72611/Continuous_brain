#!/usr/bin/env python3
"""Reanalyze saved v0.20.3 actions; no torch, training, or checkpoint access.

Requires NumPy. Replays only recorded actions on the already-used seeds.
Equal-prefix comparisons are sensitivity analyses, not full-episode estimates.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
ENV_SHA256 = "5617f61210db2e29bbbb98713f9e2df08bb622677a5b66297c391cab6cc7daee"
EPS = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def quality(row):
    return int(row["completed"]), -row["collision_delta"], row["reward"]


def compare(left, right, epsilon=0.0):
    for a, b in zip(quality(left)[:2], quality(right)[:2]):
        if a != b:
            return "outside" if a > b else "top2"
    difference = left["reward"] - right["reward"]
    return "outside" if difference > epsilon else "top2" if difference < -epsilon else "tie"


def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    vx, vy = sum(x*x for x in dx), sum(y*y for y in dy)
    if vx <= EPS or vy <= EPS:
        return 0.0
    return sum(x*y for x, y in zip(dx, dy)) / math.sqrt(vx*vy)


def original_group(rows):
    return {"n": len(rows), "completion_rate": mean(r["completed"] for r in rows),
            "mean_collision_delta": mean(r["collision_delta"] for r in rows),
            "mean_reward": mean(r["reward"] for r in rows)}


def group(rows):
    steps = sum(r["steps"] for r in rows)
    collisions = sum(r["collision_delta"] for r in rows)
    return {**original_group(rows), "completions": sum(r["completed"] for r in rows),
            "collisions": collisions, "steps": steps,
            "mean_steps": steps/len(rows), "median_steps": median(r["steps"] for r in rows),
            "collisions_per_100_recorded_steps": 100*collisions/steps}


def original_summary(boundaries):
    rows = [r for b in boundaries for r in b["options"]]
    result = {
        "boundaries": len(boundaries), "option_counterfactuals": len(rows),
        "top2": original_group([r for r in rows if r["top2"]]),
        "outside_top2": original_group([r for r in rows if not r["top2"]]),
        "best_realized_option_outside_top2_rate": mean(not b["best_realized_in_top2"] for b in boundaries),
        "selector_top1_is_best_realized_rate": mean(b["actual_selector_choice"] == b["best_realized_option"] for b in boundaries),
    }
    for x, xkey in [("selector_logit", "selector_logit"), ("selector_rank", "selector_rank")]:
        for y, ykey in [("completion", "completed"), ("collision", "collision_delta"), ("reward", "reward")]:
            if x == "selector_rank" and y == "completion":
                continue
            result[f"{x}_vs_{y}_pearson"] = pearson([r[xkey] for r in rows], [r[ykey] for r in rows])
    return result


def same_summary(actual, expected, location="summary"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{location}: keys differ")
        for key in expected:
            same_summary(actual[key], expected[key], f"{location}.{key}")
    else:
        require(isinstance(actual, (int, float)) and math.isfinite(actual), f"{location}: invalid number")
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f"{location}: {actual} != {expected}")


def validate(data):
    require(data["format"] == "continuous_brain.option_support_counterfactual_v1", "unexpected format")
    boundaries = data["boundaries"]
    require(bool(boundaries), "empty evidence")
    seen = set()
    for b in boundaries:
        key = (b["seed"], b["step"])
        require(key not in seen, f"duplicate boundary {key}")
        seen.add(key)
        require(data["seed_window"]["start"] <= b["seed"] <= data["seed_window"]["end"], "seed outside declared window")
        require(type(b["step"]) is int and 1 <= b["step"] <= 120, "invalid step")
        rows = b["options"]
        require(len(rows) == 6 and {r["option"] for r in rows} == set(range(6)), "missing/duplicate options")
        require([r["selector_rank"] for r in rows] == list(range(1, 7)), "ranks not ordered 1..6")
        require(all(rows[i]["selector_logit"] >= rows[i+1]["selector_logit"] for i in range(5)), "logits disagree with ranks")
        for r in rows:
            require(type(r["completed"]) is bool and type(r["terminated"]) is bool, "invalid flags")
            require(type(r["top2"]) is bool and r["top2"] == (r["selector_rank"] <= 2), "incorrect top2 label")
            require(type(r["steps"]) is int and 1 <= r["steps"] <= 12, "invalid segment length")
            require(r["steps"] == len(r["actions"]), "action count mismatch")
            require(all(type(a) is int and 0 <= a <= 4 for a in r["actions"]), "invalid action")
            require(type(r["collision_delta"]) is int and 0 <= r["collision_delta"] <= r["steps"], "invalid collision count")
            require(all(math.isfinite(r[k]) for k in ["reward", "selector_logit"]), "nonfinite value")
        best = max(rows, key=quality)
        require(b["best_realized_option"] == best["option"], "winner mismatch")
        require(b["best_realized_in_top2"] == best["top2"], "winner group mismatch")
        require(b["actual_selector_choice"] == rows[0]["option"], "actual choice not rank 1")
    same_summary(data["summary"], original_summary(boundaries))


def load_environment():
    path = ROOT / "source/environment.py"
    require(sha(path.read_bytes()) == ENV_SHA256, "environment snapshot changed")
    spec = importlib.util.spec_from_file_location("v0203_audit_environment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DetourWorld


def analyze(data):
    validate(data)
    Environment = load_environment()
    boundaries = data["boundaries"]
    rows = [r for b in boundaries for r in b["options"]]
    counts, coverage, raw_comparison, tolerant_comparison = Counter(), Counter(), Counter(), Counter()
    prefix_raw, prefix_tolerant, three_raw, three_tolerant = Counter(), Counter(), Counter(), Counter()
    per_boundary, margins = [], []
    max_error, replayed, previous_seed, env = 0.0, 0, None, None
    closed_seeds = set()
    for b in boundaries:
        if b["seed"] != previous_seed:
            if previous_seed is not None:
                closed_seeds.add(previous_seed)
            require(b["seed"] not in closed_seeds, "noncontiguous episode rows")
            previous_seed = b["seed"]
            env = Environment(seed=previous_seed, max_episode_steps=120)
            env.reset()  # Original diagnostic also resets after constructor reset.
        require(env.steps == b["step"] - 1, "saved actual actions do not cover boundary history")
        traces, end_states = {}, {}
        start_distance = float(__import__("numpy").linalg.norm(env.target-env.agent))
        for row in b["options"]:
            branch, reward = copy.deepcopy(env), 0.0
            start_collisions, trace = branch.collisions, []
            for index, action in enumerate(row["actions"]):
                _, value, done, info = branch.step(action)
                reward += value
                trace.append({"completed": bool(info["reached"]), "collision_delta": branch.collisions-start_collisions,
                              "reward": reward, "steps": index+1})
                require(not done or index == len(row["actions"])-1, "actions continued beyond terminal")
            final = trace[-1]
            error = abs(final["reward"]-row["reward"])
            max_error = max(max_error, error)
            require(error <= EPS, "replayed reward mismatch")
            require(all(final[k] == row[k] for k in ["completed", "collision_delta", "steps"]), "replayed outcome mismatch")
            traces[row["option"]], end_states[row["option"]] = trace, branch
            replayed += 1
        top = max((r for r in b["options"] if r["top2"]), key=quality)
        outside = max((r for r in b["options"] if not r["top2"]), key=quality)
        outcome, robust = compare(outside, top), compare(outside, top, EPS)
        raw_comparison[outcome] += 1
        tolerant_comparison[robust] += 1
        any_top, any_out = top["completed"], outside["completed"]
        coverage["both" if any_top and any_out else "top2_only" if any_top else "outside_only" if any_out else "neither"] += 1
        horizon = min(top["steps"], outside["steps"])
        pt, po = traces[top["option"]][horizon-1], traces[outside["option"]][horizon-1]
        cp, cpt = compare(po, pt), compare(po, pt, EPS)
        # All branches share the minimum commitment of three steps, unless terminal.
        first3 = []
        for row in b["options"]:
            trace = traces[row["option"]]
            require(len(trace) >= 3 or trace[-1]["completed"] or end_states[row["option"]].steps == 120, "short nonterminal segment")
            first3.append({**row, **trace[min(2, len(trace)-1)]})
        t3 = max((r for r in first3 if r["top2"]), key=quality)
        o3 = max((r for r in first3 if not r["top2"]), key=quality)
        three_raw[compare(o3, t3)] += 1
        three_tolerant[compare(o3, t3, EPS)] += 1
        reason = "tie"
        if outcome == "outside":
            reason = "completion" if outside["completed"] != top["completed"] else "collision" if outside["collision_delta"] != top["collision_delta"] else "reward"
            counts[f"wins_on_{reason}"] += 1
            counts["winner_completed"] += outside["completed"]
            counts["winner_not_completed"] += not outside["completed"]
            counts["shorter" if outside["steps"] < top["steps"] else "longer" if outside["steps"] > top["steps"] else "same_length"] += 1
            counts["longer_identical_prefix_extension"] += top["steps"] < outside["steps"] and top["actions"] == outside["actions"][:top["steps"]]
            counts["shorter_identical_prefix"] += outside["steps"] < top["steps"] and outside["actions"] == top["actions"][:outside["steps"]]
            counts[f"winner_rank_{outside['selector_rank']}"] += 1
            prefix_raw[cp] += 1
            prefix_tolerant[cpt] += 1
            margins.append(outside["reward"]-top["reward"])
        per_boundary.append({
            "seed": b["seed"], "step": b["step"], "agent_x": float(env.agent[0]), "agent_y": float(env.agent[1]),
            "target_distance": start_distance, "top2_option": top["option"], "outside_option": outside["option"],
            "outside_rank": outside["selector_rank"], "top2_completed": top["completed"], "outside_completed": outside["completed"],
            "top2_collisions": top["collision_delta"], "outside_collisions": outside["collision_delta"],
            "top2_steps": top["steps"], "outside_steps": outside["steps"], "top2_reward": top["reward"], "outside_reward": outside["reward"],
            "reward_difference": outside["reward"]-top["reward"], "raw_comparison": outcome, "tolerant_comparison": robust,
            "outside_win_reason": reason if outcome == "outside" else "not_outside_win",
            "common_prefix_steps": horizon, "common_prefix_raw": cp, "common_prefix_tolerant": cpt,
            "common_prefix_reward_difference": po["reward"]-pt["reward"],
        })
        env = end_states[b["actual_selector_choice"]]
    for key in ["wins_on_completion", "wins_on_collision", "wins_on_reward"]:
        counts.setdefault(key, 0)
    for key in ["both", "top2_only", "outside_only", "neither"]:
        coverage.setdefault(key, 0)
    return {
        "scope": "Offline reanalysis and deterministic replay of recorded actions; no neural inference, training, new policy execution, or checkpoint access.",
        "generalization_established": False,
        "new_seed_window_consumed": False,
        "numerical_tie_absolute_tolerance": EPS,
        "validation": {"all_summary_fields_reproduced": True, "winner_labels_reproduced": True,
                       "recorded_action_segments_replayed": replayed, "max_absolute_reward_error": max_error,
                       "completion_collision_and_length_replay_matches": True, "all_boundary_histories_reconstructed": True},
        "observed_sample": {"boundaries": len(boundaries), "option_segments": len(rows),
                            "episode_seeds": sorted(set(b["seed"] for b in boundaries)),
                            "boundaries_per_seed": dict(Counter(b["seed"] for b in boundaries))},
        "groups": {"top2": group([r for r in rows if r["top2"]]), "outside_top2": group([r for r in rows if not r["top2"]])},
        "rank_profiles": {str(rank): group([r for r in rows if r["selector_rank"] == rank]) for rank in range(1, 7)},
        "completion_coverage_by_boundary": dict(coverage),
        "best_group_comparison_raw": dict(raw_comparison),
        "best_group_comparison_tolerant": dict(tolerant_comparison),
        "reported_outside_wins": dict(counts),
        "outside_win_reward_margins": {"mean": mean(margins), "median": median(margins), "min": min(margins), "max": max(margins),
                                       "at_most_numerical_tolerance": sum(m <= EPS for m in margins),
                                       "below_one_step_cost_0_004": sum(m < .004 for m in margins)},
        "original_outside_winning_pairs_at_common_prefix": {"raw": dict(prefix_raw), "tolerant": dict(prefix_tolerant),
            "interpretation": "Same two originally selected options, truncated at the shorter recorded length. Removes unequal exposure only; not equal full-episode continuation."},
        "all_boundaries_first_three_steps": {"raw": dict(three_raw), "tolerant": dict(three_tolerant),
            "interpretation": "Re-select best in each group from three-step prefixes (absorbing terminal outcomes). Short-horizon sensitivity only; four outside vs two inside candidates remain."},
        "boundaries": per_boundary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"input/continuous_brain_v0203_results.zip")
    parser.add_argument("--output", type=Path, default=ROOT/"analysis.json")
    parser.add_argument("--csv", type=Path, default=ROOT/"boundary_comparisons.csv")
    args = parser.parse_args()
    require(len({p.resolve() for p in [args.input, args.output, args.csv]}) == 3, "input/output paths must differ")
    original_bytes = args.input.read_bytes()
    with ZipFile(args.input) as archive:
        require(archive.namelist().count("result.json") == 1, "result.json missing or duplicated")
        result_bytes = archive.read("result.json")
    result = analyze(json.loads(result_bytes))
    result["provenance"] = {"input_archive_sha256": sha(original_bytes), "input_result_json_sha256": sha(result_bytes),
                            "environment_snapshot_sha256": ENV_SHA256, "analyzer_sha256": sha(Path(__file__).read_bytes())}
    require(args.input.read_bytes() == original_bytes, "input changed during analysis")
    for path in [args.output, args.csv]:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["boundaries"][0]))
        writer.writeheader()
        writer.writerows(result["boundaries"])
    print(json.dumps({k: result[k] for k in ["validation", "completion_coverage_by_boundary", "reported_outside_wins", "original_outside_winning_pairs_at_common_prefix"]}, indent=2))
    print("ANALYSIS_REPRODUCED: true")
    print("TRAINING_PERFORMED: false")


if __name__ == "__main__":
    main()
