"""Evaluate episodes while separating policy inputs from scoring labels.

Policy inputs: pixels, observed sensors, visible rule ID, previous executed
action. No target, cue label, teacher action, condition ID or phase flag.
"""
from __future__ import annotations

from .canonical_executive_rule_world_v3 import RULE_NONE


def run_rollout(env, policy, *, mode="autonomous", image_size=96):
    if mode not in ("autonomous", "teacher_until_instruction"):
        raise ValueError("unknown rollout mode")
    observation = env.reset()
    previous_action = None
    trace = []
    response_correct = False
    response_reached = False
    response_pred = None
    while not env.done:
        rid = env.instruction_id()
        prediction = policy(observation.copy(), env.render_rgb(image_size), rid, previous_action)
        if type(prediction) is not int or prediction not in range(env.action_dim):
            raise ValueError("policy must return a valid integer action")
        teacher_used = mode == "teacher_until_instruction" and rid == RULE_NONE
        executed = int(env.teacher_action()) if teacher_used else prediction
        response = bool(env.at_response_step())
        record = {"step": int(env.steps), "rule_id": rid, "prediction": prediction,
                  "executed_action": executed, "teacher_used": teacher_used,
                  "at_response": response}
        observation, _, _, info = env.step(executed)
        trace.append(record)
        previous_action = executed
        if response:
            response_reached = True
            response_pred = executed
            response_correct = bool(info["response_correct"])
    policy_steps = [step for step in trace if not step["teacher_used"]]
    return {
        "success": response_reached and response_correct,
        "response_reached": response_reached,
        "response_pred": response_pred,
        "timed_out": not response_reached,
        "steps": len(trace),
        "teacher_actions": sum(step["teacher_used"] for step in trace),
        "first_non_up_policy_step": next((step["step"] for step in policy_steps
                                         if not step["at_response"] and step["executed_action"] != 1), None),
        "trace": trace,
    }


def summarize_rollouts(rows):
    rules = ("invert_lr", "vertical", "vertical_reverse")
    if not rows:
        raise ValueError("no rollout records")
    scores = {}
    for mode in ("teacher_until_instruction", "autonomous"):
        scores[mode] = {}
        all_members = []
        for rule in rules:
            pairs = []
            for row in rows:
                pair = [row["members"][cue][rule][mode] for cue in ("left", "right")]
                pairs.append(all(member["success"] for member in pair))
                all_members.extend(pair)
            scores[mode][rule] = sum(pairs) / len(pairs)
        scores[mode]["six_way"] = sum(
            all(row["members"][cue][rule][mode]["success"]
                for cue in ("left", "right") for rule in rules)
            for row in rows) / len(rows)
        scores[mode]["episode_success"] = sum(member["success"] for member in all_members) / len(all_members)
        scores[mode]["response_reached"] = sum(member["response_reached"] for member in all_members) / len(all_members)
        scores[mode]["mean_steps"] = sum(member["steps"] for member in all_members) / len(all_members)
    return scores
