"""Fail-closed TRAIN gate: derive scores from all 1,600 condition records."""
from __future__ import annotations

import math
import re

from .canonical_executive_rule_world_v3 import generate_canonical_rule_spec, target_action

RULES = ("invert_lr", "vertical", "vertical_reverse")
FACTORS = ("corridor_length", "goal_x_offset", "goal_y_offset", "cue_steps", "move_step")
MODES = ("normal", "donor", "zero", "off")


def summary(rows):
    if not rows:
        raise ValueError("empty evaluation")
    result = {}
    for mode in MODES:
        result[mode] = {
            rule: sum(all(row["members"][cue][rule][mode]["correct"] for cue in ("left", "right"))
                      for row in rows) / len(rows)
            for rule in RULES
        }
        result[mode]["six_way"] = sum(
            all(row["members"][cue][rule][mode]["correct"]
                for cue in ("left", "right") for rule in RULES)
            for row in rows) / len(rows)
    result["none_max_diff"] = max(row["members"][cue][rule]["none_diff"]
                                   for row in rows for cue in ("left", "right") for rule in RULES)
    return result


def factors(rows):
    result = {}
    for field in FACTORS:
        result[field] = {}
        for value in sorted({row["factors"][field] for row in rows}):
            group = [row for row in rows if row["factors"][field] == value]
            scores = summary(group)
            entry = {"n": len(group), "six_way_full": scores["normal"]["six_way"]}
            for rule in RULES:
                entry[rule + "_pair_full"] = scores["normal"][rule]
                entry[rule + "_donor"] = scores["donor"][rule]
            result[field][str(value)] = entry
    return result


def validate_report(report):
    if report.get("format") != "action_intent_executive_final_train_v1_1":
        raise ValueError("report must be regenerated with the repaired evaluator")
    if report.get("split") != "train_only":
        raise ValueError("this gate accepts only v3 TRAIN")
    if report.get("v3_dev_touched") is not False or report.get("v3_test_touched") is not False:
        raise ValueError("held-out split flags are missing or invalid")
    preservation = report.get("preservation", {})
    required = ("admitted_bit_exact", "base_bit_exact", "executive_bit_exact", "checkpoint_files_unchanged")
    if any(preservation.get(key) is not True for key in required):
        raise ValueError("missing measured preservation checks")
    provenance = report.get("provenance", {})
    update = provenance.get("global_update")
    if type(update) is not int or update not in (1600, 3200, 4800):
        raise ValueError("not a predeclared full-epoch checkpoint")
    for key in ("checkpoint_sha256", "admitted_checkpoint_sha256", "historical_base_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(key, ""))) is None:
            raise ValueError(f"missing checkpoint identity: {key}")
    if provenance["admitted_checkpoint_sha256"] != "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c":
        raise ValueError("wrong admitted checkpoint")
    if provenance["historical_base_sha256"] != "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f":
        raise ValueError("wrong historical base")
    sources = provenance.get("source_sha256", {})
    if len(sources) < 7 or any(re.fullmatch(r"[0-9a-f]{64}", str(v)) is None for v in sources.values()):
        raise ValueError("missing evaluated source identity")
    rows = report.get("per_condition", [])
    ids = [row.get("condition_id") for row in rows]
    if any(type(cid) is not int for cid in ids) or len(ids) != 1600 or set(ids) != set(range(1600)):
        raise ValueError("requires each TRAIN condition 0..1599 exactly once")
    for row in rows:
        spec = generate_canonical_rule_spec(row["condition_id"])
        if row.get("factors") != {field: getattr(spec, field) for field in FACTORS}:
            raise ValueError("condition factors disagree with the frozen world")
        for cue in ("left", "right"):
            donor = "right" if cue == "left" else "left"
            for rule in RULES:
                member = row["members"][cue][rule]
                difference = member["none_diff"]
                if type(difference) not in (int, float) or not math.isfinite(difference) or difference < 0:
                    raise ValueError("invalid NONE difference")
                for mode in MODES:
                    entry = member[mode]
                    prediction = entry["pred"]
                    if type(prediction) is not int or prediction not in range(5) or type(entry["correct"]) is not bool:
                        raise ValueError("invalid prediction record")
                    target = target_action(donor if mode == "donor" else cue, rule)
                    if entry["correct"] != (prediction == target):
                        raise ValueError("reported correctness disagrees with action")
    measured_summary, measured_factors = summary(rows), factors(rows)
    if report.get("summary") != measured_summary or report.get("factor_summary") != measured_factors:
        raise ValueError("reported summaries disagree with raw predictions")
    return measured_summary, measured_factors


def decide(report):
    scores, factor_scores = validate_report(report)
    checks = {}
    for rule in RULES:
        checks[rule + "_pair"] = scores["normal"][rule] >= .95
        checks[rule + "_donor"] = scores["donor"][rule] >= .95
        checks[rule + "_off"] = scores["off"][rule] <= .05
        checks[rule + "_zero"] = scores["zero"][rule] <= .05
    checks["six_way"] = scores["normal"]["six_way"] >= .90
    checks["none_exact"] = scores["none_max_diff"] == 0.0
    failures = []
    for field, levels in factor_scores.items():
        for level, score in levels.items():
            if not (score["six_way_full"] >= .75 and all(
                score[rule + "_pair_full"] >= .85 and score[rule + "_donor"] >= .85 for rule in RULES
            )):
                failures.append({"factor": field, "value": level, **score})
    checks["factors"] = not failures
    return {"format": "action_intent_train_decision_v1_1", "pass": all(checks.values()),
            "checks": checks, "factor_failures": failures, "summary": scores,
            "provenance": report["provenance"], "scope": "teacher_prefix_response_mapping_only",
            "ready_for_fresh_v4": False}
