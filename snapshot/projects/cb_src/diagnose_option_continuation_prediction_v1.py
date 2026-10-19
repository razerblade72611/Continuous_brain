#!/usr/bin/env python3
"""Frozen, CPU, one-option interventions followed by direct closed-loop control."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

INPUT_SHA = "3e18c74f21f0a64a1c5c774d62445d00c7b084cdd0dcb30ffbe9b88152bc6a4e"
BASE_SHA = "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f"
PROTECTED = {
    "state/child_v113_dml_smoke/infant_training.pt": BASE_SHA,
    "state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt": "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c",
    "state/option_intent_executive_v1/latest.portable.pt": "041acf1bcdf46736a37c12c343cde2d9316b65cf4b4ee08604cc9c572585e5e2",
    "state/recurrent_option_progress_v1/latest.portable.pt": "237dd2029bfeb72acb953395f65966d7c3b1ecbfb0453fa7ff07fac8fb440018",
}
SOURCES = {
    "continuous_brain/infant.py": "592f5a29e3c7b6b374f9637f0fb4db47d3fd3f484d36b24352249b60876db3fe",
    "continuous_brain/planning_v100.py": "63c1cf1fb21270229615f736be491ff96e693d1083aa1d632c6caf7b1d8ad3ea",
    "continuous_brain/planning_v102.py": "e1574eb3d6f38038cbfa1de9127f5ab7d9351e1d956144428d6a4abff90b589c",
    "continuous_brain/planning_v103.py": "35541798a810ae50ba287f784819f29f142f06cf22429bd8b84f1afe48c0c074",
    "continuous_brain/environment.py": "5617f61210db2e29bbbb98713f9e2df08bb622677a5b66297c391cab6cc7daee",
}
ARM_NAMES = ("direct", "historical_top2", "historical_all6", "oracle_top2", "oracle_all6")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def fingerprints(root):
    names = set(PROTECTED) | set(SOURCES)
    names.update(p.relative_to(root).as_posix() for p in (root/"continuous_brain").rglob("*.py"))
    return {name: sha(root/name) for name in sorted(names)}


def validate_input(data):
    require(data.get("format") == "continuous_brain.option_support_counterfactual_v1", "wrong input format")
    rows = data.get("boundaries", [])
    require(len(rows) == 80, "exactly 80 saved boundaries required")
    require({b["seed"] for b in rows} == set(range(14001, 14016)), "unexpected seed set")
    require(len({(b["seed"], b["step"]) for b in rows}) == 80, "duplicate boundaries")
    for b in rows:
        require([r["selector_rank"] for r in b["options"]] == list(range(1, 7)), "incomplete ranks")
        require({r["option"] for r in b["options"]} == set(range(6)), "incomplete options")
        require(b["actual_selector_choice"] == b["options"][0]["option"], "saved choice is not selector top1")
        for r in b["options"]:
            require(r["steps"] == len(r["actions"]) and 1 <= r["steps"] <= 12, "invalid saved segment")
            require(r["top2"] == (r["selector_rank"] <= 2), "invalid support label")
    return rows


class Runtime:
    def __init__(self, project, checkpoint, threads=1):
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(project))
        import torch
        import numpy as np
        from continuous_brain.infant import InfantBrain, InfantConfig, one_hot_action
        from continuous_brain.environment import DetourWorld
        from continuous_brain.planning_v100 import _env_image_tensor, _env_context
        from continuous_brain.planning_v103 import (
            _active_option_should_terminate, _option_policy_action,
            _selector_candidates, _hierarchical_option_choice,
        )
        self.torch, self.np = torch, np
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        require(sha(checkpoint) == BASE_SHA, "canonical checkpoint SHA mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = InfantBrain(InfantConfig(**payload["config"]), vision_parent_dims=payload.get("vision_parent_dims", {}))
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.Env, self.image, self.context, self.one_hot = DetourWorld, _env_image_tensor, _env_context, one_hot_action
        self.should_stop, self.policy_action, self.candidates, self.hierarchical = (
            _active_option_should_terminate, _option_policy_action, _selector_candidates, _hierarchical_option_choice)
        self.cache = {}
        self.model_before = self.model_digest()

    def model_digest(self):
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            h.update(tensor.detach().cpu().contiguous().reshape(-1).view(self.torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def new_env(self, seed):
        env = self.Env(seed=int(seed), max_episode_steps=120)
        env.reset()  # Matches constructor reset followed by diagnostic reset.
        return env

    def root(self, env):
        key = env.agent.tobytes()+env.target.tobytes()+struct.pack("dd", env.energy, env.last_reward)
        if key not in self.cache:
            spatial = self.model.target_spatial_bus(self.image(env, self.model.cfg.image_size, "cpu"))
            self.cache[key] = self.model.planning.encode_state(spatial, self.context(env._obs(), "cpu")[1])
        return self.cache[key]

    def select(self, root):
        return int(self.model.options.selector_logits(root).argmax(dim=-1)[0].item())

    def stop(self, root, option, age):
        return self.should_stop(self.model, root, option, age, 3, .5)

    def action(self, root, option):
        return self.policy_action(self.model, root, option)

    def step_prediction(self, root, action):
        a = self.one_hot(self.torch.tensor([action]), self.model.cfg.action_dim)
        return self.model.planning.step(root, a)

    def prediction_fields(self, prediction):
        fields = {"reward": float(prediction["reward"][0]),
                  "success_probability": float(self.torch.sigmoid(prediction["success_logit"])[0]),
                  "collision_probability": float(self.torch.sigmoid(prediction["collision_logit"])[0]),
                  "context": prediction["context"][0].tolist()}
        require(all(math.isfinite(x) for x in [fields["reward"], fields["success_probability"], fields["collision_probability"], *fields["context"]]), "nonfinite prediction")
        return fields

    def choices_before_execution(self, root):
        choices = {}
        for name, topk in [("historical_top2", 2), ("historical_all6", 6)]:
            choices[name] = self.hierarchical(self.model, root, option_depth=2, selector_topk=topk,
                beam_width=8, max_option_steps=12, discount=.95, min_commit=3, terminate_threshold=.5)
        return choices

    def imagined_options(self, root):
        result = []
        for option, logit in self.candidates(self.model, root, 6):
            latent, trajectory = root, []
            for age in range(1, 13):
                action = self.action(latent, option)
                pred = self.step_prediction(latent, action)
                latent = pred["latent"]
                terminate = self.stop(latent, option, age)
                trajectory.append({"action": action, "terminate": terminate, **self.prediction_fields(pred)})
                if terminate:
                    break
            result.append({"option": option, "selector_logit": logit, "steps": trajectory})
        return result

    def prediction_audit(self, start, branch):
        env, records = copy.deepcopy(start), []
        imagined = self.root(env)
        # Same executed actions for both probes isolates state-prediction drift.
        for event in branch["events"][:min(12, branch["initial_option_steps"])]:
            root, action = self.root(env), event["action"]
            one = self.step_prediction(root, action)
            opened = self.step_prediction(imagined, action)
            imagined = opened["latent"]
            obs, reward, done, info = env.step(action)
            require(abs(reward-event["reward"]) <= 1e-12, "prediction audit replay mismatch")
            records.append({"depth": len(records)+1, "action": action,
                "actual": {"reward": reward, "success": bool(info["reached"]), "collision": bool(info["collision"]),
                           "context": self.context(obs, "cpu")[1][0].tolist()},
                "observed_state": self.prediction_fields(one), "rolled_state_same_actions": self.prediction_fields(opened)})
            if done:
                break
        return records


def continue_branch(runtime, start, forced_option):
    env = copy.deepcopy(start)
    start_step, start_collisions = env.steps, env.collisions
    option, age, initial, initial_steps, events = int(forced_option), 0, True, None, []
    require(start_step < 120, "terminal starting state")
    for _ in range(120-start_step):
        root = runtime.root(env)
        if runtime.stop(root, option, age):
            if initial:
                initial_steps, initial = age, False
            option, age = runtime.select(root), 0
        action = runtime.action(root, option)
        obs, reward, done, info = env.step(action)
        events.append({"step": env.steps, "option": option, "action": action, "reward": float(reward),
                       "collision": bool(info["collision"]), "completed": bool(info["reached"])})
        age += 1
        if done:
            break
    require(events and (events[-1]["completed"] or env.steps == 120), "unfinished branch")
    return {"option": int(forced_option), "completed": events[-1]["completed"],
        "collision_delta": int(env.collisions-start_collisions), "steps": len(events), "terminal_step": int(env.steps),
        "reward": sum(e["reward"] for e in events),
        "discounted_reward": sum(.95**i*e["reward"] for i, e in enumerate(events)),
        "initial_option_steps": age if initial else initial_steps,
        "initial_option_end": "episode_end" if initial else "learned_termination",
        "events": events}


def check_prefix(branch, saved):
    prefix = branch["events"][:saved["steps"]]
    require([e["action"] for e in prefix] == saved["actions"], "saved action prefix mismatch")
    require(sum(e["collision"] for e in prefix) == saved["collision_delta"], "saved collision prefix mismatch")
    require(any(e["completed"] for e in prefix) == saved["completed"], "saved completion prefix mismatch")
    require(abs(sum(e["reward"] for e in prefix)-saved["reward"]) <= 1e-10, "saved reward prefix mismatch")
    if saved["terminated"]:
        require(branch["initial_option_end"] == "learned_termination" and branch["initial_option_steps"] == saved["steps"], "learned termination mismatch")
    else:
        require(branch["initial_option_steps"] >= saved["steps"], "initial option stopped before saved segment")


def oracle_key(row):
    return int(row["completed"]), -row["collision_delta"], -row["steps"] if row["completed"] else 0, -row["selector_rank"]


def validate_records(records):
    fixture = Path(__file__).resolve().parent/"option_continuation_prediction_v1/input.json"
    require(sha(fixture) == INPUT_SHA, "saved boundary fixture changed")
    expected = validate_input(json.loads(fixture.read_text()))
    require(len(records) == len(expected), "incomplete report evidence")
    prefix_collisions, last_seed = 0, None
    for record, saved in zip(records, expected):
        require((record["seed"], record["step"]) == (saved["seed"], saved["step"]), "boundary identity/order mismatch")
        if record["seed"] != last_seed:
            prefix_collisions, last_seed = 0, record["seed"]
        require(record["start_collisions"] == prefix_collisions, "starting collisions mismatch")
        branches = record["branches"]
        require(len(branches) == 6 and {b["option"] for b in branches} == set(range(6)), "incomplete branches")
        saved_by_option = {r["option"]: r for r in saved["options"]}
        for branch in branches:
            s = saved_by_option[branch["option"]]
            require(branch["selector_rank"] == s["selector_rank"] and branch["top2"] == s["top2"], "branch support mismatch")
            events = branch["events"]
            require(events and len(events) == branch["steps"], "missing branch events")
            require([e["step"] for e in events] == list(range(record["step"], branch["terminal_step"]+1)), "branch time mismatch")
            require(branch["terminal_step"] <= 120 and (branch["completed"] or branch["terminal_step"] == 120), "branch budget/termination mismatch")
            require(all(not e["completed"] for e in events[:-1]) and events[-1]["completed"] == branch["completed"], "terminal event mismatch")
            require(all(type(e["action"]) is int and 0 <= e["action"] < 5 and type(e["option"]) is int and 0 <= e["option"] < 6 for e in events), "invalid action/option")
            require(sum(e["collision"] for e in events) == branch["collision_delta"], "branch collision sum mismatch")
            require(abs(sum(e["reward"] for e in events)-branch["reward"]) <= 1e-10, "branch reward sum mismatch")
            require(abs(sum(.95**i*e["reward"] for i,e in enumerate(events))-branch["discounted_reward"]) <= 1e-10, "discounted sum mismatch")
            require(1 <= branch["initial_option_steps"] <= branch["steps"], "initial option duration mismatch")
            require(all(e["option"] == branch["option"] for e in events[:branch["initial_option_steps"]]), "initial option did not own its actions")
            if branch["initial_option_end"] == "episode_end":
                require(branch["initial_option_steps"] == branch["steps"], "incorrect episode-end label")
            else:
                require(branch["initial_option_end"] == "learned_termination" and 3 <= branch["initial_option_steps"] < branch["steps"], "incorrect termination label")
            require(len(branch["prediction_audit"]) == min(12, branch["initial_option_steps"]), "missing prediction evidence")
            check_prefix(branch, s)
        by_option = {b["option"]: b for b in branches}
        choices = {"direct": saved["actual_selector_choice"],
            **{name: record["predictions_before_execution"][name]["chosen_option"] for name in ["historical_top2", "historical_all6"]},
            "oracle_top2": max((b for b in branches if b["top2"]), key=oracle_key)["option"],
            "oracle_all6": max(branches, key=oracle_key)["option"]}
        require(record["arms"].keys() == choices.keys(), "comparator set mismatch")
        for name, choice in choices.items():
            require(choice in by_option, "invalid chosen option")
            compact = {k:v for k,v in by_option[choice].items() if k not in {"events", "prediction_audit"}}
            require(record["arms"][name] == compact, "comparator outcome not supported by branch")
        prefix_collisions += saved["options"][0]["collision_delta"]


def summarize(records):
    validate_records(records)
    by_seed = defaultdict(list)
    for r in records:
        by_seed[r["seed"]].append(r)
    # The direct policy must reach the same episode outcome from each saved state.
    for seed, rows in by_seed.items():
        outcomes = {(r["arms"]["direct"]["completed"], r["arms"]["direct"]["terminal_step"],
                     r["start_collisions"]+r["arms"]["direct"]["collision_delta"]) for r in rows}
        require(len(outcomes) == 1, f"direct continuation is inconsistent across boundaries for seed {seed}")
    summary = {}
    for arm in ARM_NAMES:
        gains = sum(r["arms"][arm]["completed"] and not r["arms"]["direct"]["completed"] for r in records)
        losses = sum(r["arms"]["direct"]["completed"] and not r["arms"][arm]["completed"] for r in records)
        completion_diffs, collision_diffs = [], []
        for seed, rows in sorted(by_seed.items()):
            completion_diffs.append(statistics.mean(int(r["arms"][arm]["completed"])-int(r["arms"]["direct"]["completed"]) for r in rows))
            collision_diffs.append(statistics.mean(r["arms"][arm]["collision_delta"]-r["arms"]["direct"]["collision_delta"] for r in rows))
        ci = {}
        for name, values in [("completion_difference", completion_diffs), ("collision_difference", collision_diffs)]:
            rng = random.Random(20260919)
            boots = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
            ci[name] = {"mean_over_seeds": statistics.mean(values), "seed_bootstrap_percentile_95": [boots[49], boots[1949]]}
        summary[arm] = {"completion_gains_vs_direct_boundaries": gains, "completion_losses_vs_direct_boundaries": losses,
            "boundary_mean_completion": statistics.mean(r["arms"][arm]["completed"] for r in records),
            "boundary_mean_remaining_collisions": statistics.mean(r["arms"][arm]["collision_delta"] for r in records),
            "paired_equal_seed_summary": ci}
    accuracy = {}
    audits = [a for r in records for b in r["branches"] for a in b["prediction_audit"]]
    for mode in ["observed_state", "rolled_state_same_actions"]:
        accuracy[mode] = {"transitions": len(audits),
            "reward_mae": statistics.mean(abs(a[mode]["reward"]-a["actual"]["reward"]) for a in audits),
            "success_brier": statistics.mean((a[mode]["success_probability"]-a["actual"]["success"])**2 for a in audits),
            "collision_brier": statistics.mean((a[mode]["collision_probability"]-a["actual"]["collision"])**2 for a in audits),
            "context_mae": statistics.mean(statistics.mean(abs(x-y) for x,y in zip(a[mode]["context"],a["actual"]["context"])) for a in audits)}
        accuracy[mode]["collision_events"] = sum(a["actual"]["collision"] for a in audits)
        accuracy[mode]["observed_collision_rate"] = statistics.mean(a["actual"]["collision"] for a in audits)
        accuracy[mode]["mean_predicted_collision_probability"] = statistics.mean(a[mode]["collision_probability"] for a in audits)
        accuracy[mode]["collision_true_positives_at_0_5"] = sum(a["actual"]["collision"] and a[mode]["collision_probability"] >= .5 for a in audits)
        accuracy[mode]["always_zero_collision_brier"] = accuracy[mode]["observed_collision_rate"]
    duration = {}
    for arm in ["historical_top2", "historical_all6"]:
        loss_cases, capped, capped_and_long = [], 0, 0
        for record in records:
            selected, direct = record["arms"][arm], record["arms"]["direct"]
            imagined = next(p for p in record["imagined_options"] if p["option"] == selected["option"])
            cut_off = not imagined["steps"][-1]["terminate"]
            capped += cut_off
            capped_and_long += cut_off and selected["initial_option_steps"] > len(imagined["steps"])
            if direct["completed"] and not selected["completed"]:
                loss_cases.append({"seed": record["seed"], "step": record["step"], "option": selected["option"],
                    "selector_rank": selected["selector_rank"], "imagined_initial_steps": len(imagined["steps"]),
                    "imagined_terminated": not cut_off, "real_initial_steps": selected["initial_option_steps"],
                    "real_initial_end": selected["initial_option_end"], "remaining_collisions": selected["collision_delta"]})
        duration[arm] = {"selected_macros_capped_without_predicted_termination": capped,
            "capped_selected_macros_that_really_continue_longer": capped_and_long,
            "completion_loss_cases": loss_cases}
    return {"arms": summary, "prediction_accuracy_on_initial_option_prefixes": accuracy,
            "imagination_execution_duration_diagnostic": duration,
            "interpretation": "One initial intervention followed by direct control. Boundary rates are not autonomous planner episode success rates. Bootstrap is exploratory on 15 spent development seeds."}


def run_experiment(runtime, data, output):
    rows, records, env, previous_seed = validate_input(data), [], None, None
    raw_path = output.with_name("boundaries.jsonl")
    with raw_path.open("x") as stream:
        for index, saved_boundary in enumerate(rows):
            seed = saved_boundary["seed"]
            if seed != previous_seed:
                env, previous_seed = runtime.new_env(seed), seed
            require(env.steps == saved_boundary["step"]-1, "saved history gap")
            root = runtime.root(env)
            observed = runtime.candidates(runtime.model, root, 6)
            for (option, logit), saved in zip(observed, saved_boundary["options"]):
                require(option == saved["option"] and abs(logit-saved["selector_logit"]) <= 1e-4, "selector replay mismatch")
            require(runtime.select(root) == saved_boundary["actual_selector_choice"], "direct selector tie/replay mismatch")
            # These predictions and choices are captured before any real branch.
            predictions = runtime.choices_before_execution(root)
            imagined_options = runtime.imagined_options(root)
            branches = []
            for saved in saved_boundary["options"]:
                branch = continue_branch(runtime, env, saved["option"])
                check_prefix(branch, saved)
                branch.update({"selector_rank": saved["selector_rank"], "top2": saved["top2"],
                               "prediction_audit": runtime.prediction_audit(env, branch)})
                branches.append(branch)
            by_option = {b["option"]: b for b in branches}
            chosen = {"direct": saved_boundary["actual_selector_choice"],
                **{name: p["chosen_option"] for name,p in predictions.items()},
                "oracle_top2": max((b for b in branches if b["top2"]), key=oracle_key)["option"],
                "oracle_all6": max(branches, key=oracle_key)["option"]}
            arms = {name: {k:v for k,v in by_option[choice].items() if k not in {"events", "prediction_audit"}} for name,choice in chosen.items()}
            record = {"seed": seed, "step": saved_boundary["step"], "start_collisions": env.collisions,
                "predictions_before_execution": predictions, "imagined_options": imagined_options,
                "branches": branches, "arms": arms}
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False)+"\n")
            stream.flush()
            print(f"boundaries={index+1}/80 branches={(index+1)*6}/480 cached_states={len(runtime.cache)}", flush=True)
            # Advance only the saved actually-executed prefix to the next boundary.
            for action in saved_boundary["options"][0]["actions"]:
                env.step(action)
    return records, summarize(records)


def verify_report(path):
    report = json.loads(path.read_text())
    require(report.get("software_status") == "COMPLETE", "run is not complete")
    require(report.get("training_performed") is False and report.get("protected_unchanged") is True, "preservation flags invalid")
    raw = path.with_name("boundaries.jsonl")
    require(sha(raw) == report["raw_records_sha256"], "raw evidence hash mismatch")
    records = [json.loads(line) for line in raw.read_text().splitlines() if line]
    require(report["summary"] == summarize(records), "summary not reproduced from raw evidence")
    require(report["validation"]["saved_prefixes_matched"] == 480, "incomplete prefix checks")
    require(report["provenance"]["input_sha256"] == INPUT_SHA, "report input mismatch")
    for name, digest in {**PROTECTED, **SOURCES}.items():
        require(report["provenance"]["protected_and_core_sources"][name] == digest, "report provenance mismatch")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.verify_report:
        verify_report(args.verify_report)
        print("REPORT_RECOMPUTED: true")
        return
    require(args.output is not None, "--output is required")
    require(args.threads > 0, "--threads must be positive")
    root, output = args.project_root.resolve(), args.output.resolve()
    require(not output.exists() and not output.with_name("boundaries.jsonl").exists(), "output already exists; use a new run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(__file__).resolve().parent/"option_continuation_prediction_v1/input.json"
    require(sha(data_path) == INPUT_SHA, "saved boundary fixture changed")
    data = json.loads(data_path.read_text())
    validate_input(data)
    before = fingerprints(root)
    for name, expected in {**PROTECTED, **SOURCES}.items():
        require(before[name] == expected, f"provenance mismatch: {name}")
    started = time.monotonic()
    runtime = Runtime(root, root/next(iter(PROTECTED)), args.threads)
    print("OPTION_CONTINUATION_PREDICTION_V1\nCPU_ONLY: true\nTRAINING_PERFORMED: false", flush=True)
    print("SCOPE: historical planning_v103 comparators; not a reproduction of the unprovided v0.20.2 goal wrapper", flush=True)
    try:
        with runtime.torch.inference_mode():
            records, summary = run_experiment(runtime, data, output)
        require(runtime.model_digest() == runtime.model_before, "model tensors changed in memory")
        require(fingerprints(root) == before, "protected checkpoint/source bytes changed")
        report = {"format": "continuous_brain.option_continuation_prediction_v1", "software_status": "COMPLETE",
            "training_performed": False, "protected_unchanged": True, "generalization_established": False,
            "v019_remains_closed": True, "new_seed_window_consumed": False,
            "scope": "480 one-option interventions at 80 direct-policy states; natural termination; common direct continuation; absolute episode deadline 120.",
            "planner_scope": "Existing planning_v103 hierarchical search, depth 2, beam 8, imagined option horizon 12, discount .95, topk 2 or 6. No goal wrapper or memory input.",
            "provenance": {"protected_and_core_sources": before, "input_sha256": INPUT_SHA,
                "torch_version": runtime.torch.__version__, "numpy_version": runtime.np.__version__, "cpu_threads": args.threads,
                "script_sha256": sha(Path(__file__)), "model_state_digest": runtime.model_before},
            "validation": {"strict_checkpoint_load": True, "saved_prefixes_matched": len(records)*6,
                "saved_selector_rank_and_logit_replay": True, "direct_continuation_consistent_within_each_seed": True,
                "model_tensors_unchanged": True},
            "summary": summary, "raw_records_sha256": sha(output.with_name("boundaries.jsonl")),
            "elapsed_seconds": time.monotonic()-started}
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
        verify_report(output)
    except BaseException as error:
        error_report = {"software_status": "ERROR", "error": repr(error), "training_performed": False,
                        "protected_unchanged": fingerprints(root) == before}
        output.with_name("error.json").write_text(json.dumps(error_report, indent=2)+"\n")
        output.write_text(json.dumps(error_report, indent=2)+"\n")
        raise
    print(json.dumps(summary, indent=2), flush=True)
    print("PROTECTED_UNCHANGED: true\nMODEL_TENSORS_UNCHANGED: true\nSAVED_PREFIXES_MATCHED: 480\nOPTION_CONTINUATION_DIAGNOSTIC_COMPLETE", flush=True)
    print("report:", output, flush=True)


if __name__ == "__main__":
    main()
