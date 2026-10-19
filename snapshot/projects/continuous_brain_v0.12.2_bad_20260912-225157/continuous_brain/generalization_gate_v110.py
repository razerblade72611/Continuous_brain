"""Paired generalization diagnostics for the existing learned option planner.

Integrated from the consultation proposal; see CONSULTATION_VALIDATION.md for
executed checks and limits. Uses the actual project model and simulators.
A matching, valid fixed-layout P4b report is mandatory. Report validation checks
consistency and protocol, not authenticity of a human-editable JSON document.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .curriculum_v110 import _make_env
from .infant import InfantBrain
from .parented_train import load_checkpoint
from .planning_v100 import _env_context, _env_image_tensor
from .planning_v103 import _hierarchical_action
from .planning_v102 import _resolve_option
from .device_v111 import resolve_device, device_report
from .evidence_v114 import valid_p4b, P4B_PROTOCOL, write_json

MODULE_VERSION = "generalization_gate_v110.consolidated-r1"


def source_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_info(device):
    return {**device_report(device), "python": platform.python_version(), "numpy": np.__version__}


def validate_evaluation_settings(kind, episodes, steps, seed_start, evaluation_role,
                                 option_depth=2, selector_topk=2, beam_width=8,
                                 max_option_steps=12, min_commit=3, discount=.95,
                                 terminate_threshold=.5):
    if kind not in ("randomized_detour", "two_barrier"):
        raise ValueError("Unknown environment kind")
    if any(int(v) != v or v < 1 for v in (episodes, steps, option_depth, selector_topk,
                                          beam_width, max_option_steps, min_commit)):
        raise ValueError("Counts and planner limits must be positive integers")
    if int(seed_start) != seed_start or seed_start < 0:
        raise ValueError("Seed start must be a nonnegative integer")
    if not 0 <= discount <= 1 or not 0 <= terminate_threshold <= 1:
        raise ValueError("Discount and termination threshold must be in [0, 1]")
    end = seed_start + episodes - 1
    overlaps = lambda start, last: seed_start <= last and end >= start
    if overlaps(51001, 51100):
        raise ValueError("Seeds 51001-51100 remain reserved by the original project")
    if evaluation_role == "diagnostic":
        if overlaps(61001, 61100) or overlaps(71001, 71100):
            raise ValueError("Diagnostic runs must not consume confirmation seeds")
    elif evaluation_role == "confirmation":
        expected = 61001 if kind == "randomized_detour" else 71001
        if (seed_start, episodes, steps) != (expected, 100, 160):
            raise ValueError("Confirmation requires the reserved 100-episode, 160-step protocol")
        if (option_depth, selector_topk, beam_width, max_option_steps, min_commit,
                discount, terminate_threshold) != (2, 2, 8, 12, 3, .95, .5):
            raise ValueError("Confirmation planner settings must remain fixed")
    else:
        raise ValueError("Evaluation role must be diagnostic or confirmation")


def paired_comparison(candidate, reference):
    seeds = [r["seed"] for r in candidate]
    if not seeds or seeds != [r["seed"] for r in reference] or len(set(seeds)) != len(seeds):
        raise ValueError("Comparison requires matching, unique paired seeds")
    metrics = {"completion_rate": ("completed", 1), "collisions_reduced": ("collisions", -1),
               "reward": ("reward", 1)}
    result = {}
    for name, (field, sign) in metrics.items():
        diffs = np.asarray([sign * (float(a[field]) - float(b[field]))
                            for a, b in zip(candidate, reference)])
        lo, hi = _bootstrap_ci(diffs)
        result[name] = {"mean_difference": float(diffs.mean()), "lo": lo, "hi": hi,
                        "supported_advantage": lo > 0,
                        "status": "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive"}
    return result

# Fixed before any run. Inherited from planning_v103's fixed-geometry gate
# as a starting point, kept for comparability -- not a claim these exact
# numbers are the right bar for a harder, randomized task.
PASS_THRESHOLDS = {
    "absolute_completion": 0.20,
    "completion_advantage": 0.05,
    "max_collision_regression_per_episode": 2.0,
}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bootstrap_ci(diffs: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% CI on the mean of paired per-episode differences."""
    diffs = np.asarray(diffs, dtype=float)
    if diffs.ndim != 1 or not len(diffs) or not np.isfinite(diffs).all() or n_boot < 1:
        raise ValueError("Bootstrap needs nonempty finite paired differences")
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verify_replication_prerequisite(replication_report_path: str | Path, checkpoint_path: str | Path) -> dict:
    """
    Raises RuntimeError if the fixed-layout replication has not actually
    passed for THIS checkpoint. This is the enforcement asked for: not a
    convention, a required argument checked before any episode runs.
    """
    report = json.loads(Path(replication_report_path).read_text())
    actual_hash = _sha256_file(checkpoint_path)
    report_hash = report.get("checkpoint_sha256")
    if report_hash != actual_hash:
        raise RuntimeError(
            f"Replication report checkpoint hash does not match the checkpoint being evaluated.\n"
            f"  report checkpoint_sha256:    {report_hash}\n"
            f"  actual checkpoint sha256:    {actual_hash}\n"
            f"Run evaluate-hierarchical-options against THIS checkpoint first."
        )
    if report.get("p4b_passed") is not True:
        raise RuntimeError(
            f"Replication report shows p4b_passed={report.get('p4b_passed')!r} for this checkpoint. "
            f"The historical v0.10.3 pass is documented history about a different checkpoint, not a "
            f"property this one has been shown to have. Diagnose before running the generalization gate."
        )
    if (report.get("format") != "continuous_brain.p4b_hierarchical_option_planning_gate.v103"
            or not valid_p4b(report, actual_hash)):
        raise RuntimeError("Replication must match the original 100-episode P4b protocol and measured thresholds")
    return {"replication_report_sha256": _sha256_file(replication_report_path),
            "protocol": dict(P4B_PROTOCOL),
            "verification_scope": "JSON consistency, not signed execution attestation",
            "replication_report": str(Path(replication_report_path).resolve()),
            "replication_checkpoint_sha256": report_hash, "replication_p4b_passed": True}


@torch.no_grad()
def _run_generalized_episode(model: InfantBrain, kind: str, seed: int, *, mode: str,
                             steps: int, option_depth: int, selector_topk: int,
                             beam_width: int, max_option_steps: int, discount: float,
                             min_commit: int, terminate_threshold: float, device: str) -> dict:
    env = _make_env(kind, seed, steps)
    obs = env.reset()
    initial_observation = obs.tolist()
    layout = env.visual_snapshot()
    decision_trace = []
    hidden = model.nucleus.initial_state(1, device)
    memory = torch.zeros(1, model.cfg.memory_dim, dtype=torch.float32, device=device)
    previous_action = torch.zeros(1, dtype=torch.long, device=device)
    current_option, option_age = None, 0
    option_switches = 0
    total_reward = 0.0
    completed = False
    steps_used = int(steps)
    high_level_trace = []

    for step in range(1, int(steps) + 1):
        image = _env_image_tensor(env, model.cfg.image_size, device)
        context, context7 = _env_context(obs, device)
        semantic = model.vision(image)
        spatial_online = model.spatial(image)
        fused = model.fuse_vision(semantic, spatial_online)
        nuc = model.nucleus(fused, context, memory, previous_action, hidden)
        hidden = nuc["hidden"]

        decision = None
        if mode == "reactive":
            action = int(torch.argmax(model.motor(nuc["workspace"]), dim=-1)[0].item())
        elif mode == "option_direct":
            root = model.planning.encode_state(model.target_spatial_bus(image), context7)
            current_option, option_age, switched = _resolve_option(
                model, root, current_option, option_age, min_commit, terminate_threshold)
            option_t = torch.tensor([current_option], dtype=torch.long, device=device)
            action = int(model.options.action_logits(root, option_t).argmax(dim=-1)[0].item())
            option_switches += int(switched)
            decision = {"option": current_option, "option_age": option_age, "switched": switched}
            option_age += 1
        elif mode == "hierarchical":
            stable_spatial = model.target_spatial_bus(image)
            decision = _hierarchical_action(
                model, stable_spatial, context7, current_option, option_age,
                option_depth=option_depth, selector_topk=selector_topk, beam_width=beam_width,
                max_option_steps=max_option_steps, discount=discount, min_commit=min_commit,
                terminate_threshold=terminate_threshold,
            )
            action = int(decision["action"])
            current_option = int(decision["option"])
            option_switches += int(decision["switched"])
            option_age = int(decision["option_age"]) + 1
            if decision["high_level_decision"]:
                high_level_trace.append({
                    **decision,
                    "step": int(step), "chosen_option": current_option,
                    "switched": bool(decision["switched"]),
                })
        else:
            raise ValueError(mode)

        before_obs = obs.tolist()
        obs, reward, done, info = env.step(action)
        decision_trace.append(dict(step=step, observation=before_obs, action=action,
            next_observation=obs.tolist(), reward=float(reward), done=bool(done),
            collision=bool(info.get("collision", False)), option_decision=decision))
        total_reward += float(reward)
        previous_action = torch.tensor([action + 1], dtype=torch.long, device=device)
        if info.get("reached"):
            completed = True
            steps_used = step
        if done:
            break

    return dict(seed=seed, kind=kind, mode=mode, completed=completed, steps=steps_used,
                collisions=int(env.collisions), reward=total_reward, option_switches=option_switches,
                high_level_trace=high_level_trace, decision_trace=decision_trace,
                actual_steps=step, initial_observation=initial_observation, layout=layout)


def _summary(rows: list[dict], max_steps: int) -> dict:
    return {
        "episodes": len(rows),
        "completion_rate": _mean(float(r["completed"]) for r in rows),
        "mean_reward": _mean(r["reward"] for r in rows),
        "mean_steps_penalized": _mean(r["steps"] if r["completed"] else int(max_steps) + 1 for r in rows),
        "mean_collisions": _mean(r["collisions"] for r in rows),
        "mean_option_switches": _mean(r.get("option_switches", 0) for r in rows),
    }


def diagnose_gate_result(result: dict) -> dict:
    """
    Interpretable-evidence support, not a pass/fail restatement: identifies
    which specific threshold criterion failed (if any) and which specific
    seeds in the hierarchical arm failed to complete or collided most, so
    a failure can be inspected rather than only reported.
    """
    thresholds = result["pass_thresholds_fixed_before_evaluation"]
    h, r = result["hierarchical_option_planning"], result["reactive"]
    adv = result["advantages_vs_reactive"]
    failed_criteria = []
    if h["completion_rate"] < thresholds["absolute_completion"]:
        failed_criteria.append(f"absolute_completion: {h['completion_rate']:.3f} < {thresholds['absolute_completion']}")
    if not (adv["completion_percentage_points"] / 100.0 > thresholds["completion_advantage"]):
        failed_criteria.append(
            f"completion_advantage: {adv['completion_percentage_points']/100.0:.3f} <= {thresholds['completion_advantage']}"
        )
    if adv["collision_regression"] > thresholds["max_collision_regression_per_episode"]:
        failed_criteria.append(
            f"collision_regression: {adv['collision_regression']:.3f} > {thresholds['max_collision_regression_per_episode']}"
        )

    hierarchical_episodes = result["episodes"]["hierarchical_option_planning"]
    non_completed = sorted([e["seed"] for e in hierarchical_episodes if not e["completed"]])
    by_collisions = sorted(hierarchical_episodes, key=lambda e: -e["collisions"])[:10]

    return {
        "benchmark_threshold_passed": result["benchmark_threshold_passed"],
        "failed_criteria": failed_criteria if not result["benchmark_threshold_passed"] else [],
        "statistically_supported_advantage": result["statistically_supported_advantage"],
        "non_completed_seeds": non_completed,
        "highest_collision_seeds": [{"seed": e["seed"], "collisions": e["collisions"]} for e in by_collisions],
        "next_step": (
            "Inspect non_completed_seeds and highest_collision_seeds directly (each episode's "
            "high_level_trace is in the full result) before proposing any training change."
            if not result["benchmark_threshold_passed"] or not result["statistically_supported_advantage"]["completion_rate"]
            else "Completion advantage has paired statistical support on these seeds. Diagnose traces; confirmation requires the reserved protocol."
        ),
    }


def evaluate_generalization_gate(
    checkpoint: str | Path, *, replication_report: str | Path, kind: str = "randomized_detour",
    episodes: int = 100, steps: int = 160, seed_start: int = 41001, evaluation_role: str = "diagnostic", option_depth: int = 2,
    selector_topk: int = 2, beam_width: int = 8, max_option_steps: int = 12, discount: float = 0.95,
    min_commit: int = 3, terminate_threshold: float = 0.50, device: str = "cpu",
    output: str | Path | None = None,
) -> dict:
    started = time.perf_counter()
    checkpoint = Path(checkpoint).expanduser().resolve()
    validate_evaluation_settings(kind, episodes, steps, seed_start, evaluation_role,
        option_depth, selector_topk, beam_width, max_option_steps, min_commit,
        discount, terminate_threshold)
    if output and Path(output).expanduser().resolve() in {checkpoint, Path(replication_report).expanduser().resolve()}:
        raise ValueError("Output must not overwrite the checkpoint or replication report")
    before_hash = _sha256_file(checkpoint)
    replication = verify_replication_prerequisite(replication_report, checkpoint)  # raises if not satisfied

    device = resolve_device(device)
    model, saved = load_checkpoint(checkpoint, device)
    model.eval()
    metadata = dict(saved.get("metadata") or {})

    if metadata.get("option_cortex_version") != "latent-temporal-options-v1":
        raise RuntimeError("Requires an existing trained temporal-options checkpoint")
    reactive, hierarchical, direct = [], [], []
    for i in range(int(episodes)):
        seed = int(seed_start) + i
        reactive.append(_run_generalized_episode(
            model, kind, seed, mode="reactive", steps=steps, option_depth=option_depth,
            selector_topk=selector_topk, beam_width=beam_width, max_option_steps=max_option_steps,
            discount=discount, min_commit=min_commit, terminate_threshold=terminate_threshold, device=device,
        ))
        hierarchical.append(_run_generalized_episode(
            model, kind, seed, mode="hierarchical", steps=steps, option_depth=option_depth,
            selector_topk=selector_topk, beam_width=beam_width, max_option_steps=max_option_steps,
            discount=discount, min_commit=min_commit, terminate_threshold=terminate_threshold, device=device,
        ))
        direct.append(_run_generalized_episode(
            model, kind, seed, mode="option_direct", steps=steps, option_depth=option_depth,
            selector_topk=selector_topk, beam_width=beam_width, max_option_steps=max_option_steps,
            discount=discount, min_commit=min_commit, terminate_threshold=terminate_threshold, device=device,
        ))
        if not (reactive[-1]["initial_observation"] == hierarchical[-1]["initial_observation"] == direct[-1]["initial_observation"]
                and reactive[-1]["layout"] == hierarchical[-1]["layout"] == direct[-1]["layout"]):
            raise RuntimeError("Paired arms did not start in the same environment")
        print(f"generalization_gate[{kind}]={i + 1}/{episodes}")

    r, h = _summary(reactive, steps), _summary(hierarchical, steps)
    completion_advantage = h["completion_rate"] - r["completion_rate"]
    collision_regression = h["mean_collisions"] - r["mean_collisions"]

    completed_h = np.array([float(e["completed"]) for e in hierarchical])
    completed_r = np.array([float(e["completed"]) for e in reactive])
    collisions_h = np.array([float(e["collisions"]) for e in hierarchical])
    collisions_r = np.array([float(e["collisions"]) for e in reactive])
    reward_h = np.array([float(e["reward"]) for e in hierarchical])
    reward_r = np.array([float(e["reward"]) for e in reactive])

    ci_completion = _bootstrap_ci(completed_h - completed_r)
    ci_collisions = _bootstrap_ci(collisions_r - collisions_h)  # positive = hierarchical safer
    ci_reward = _bootstrap_ci(reward_h - reward_r)

    benchmark_threshold_passed = bool(
        h["completion_rate"] >= PASS_THRESHOLDS["absolute_completion"]
        and completion_advantage > PASS_THRESHOLDS["completion_advantage"]
        and collision_regression <= PASS_THRESHOLDS["max_collision_regression_per_episode"]
    )
    # An interval "supports an advantage" only if it excludes zero in the
    # favorable direction -- spanning zero is reported as NOT supported,
    # regardless of where the point estimate or the threshold check landed.
    statistically_supported_advantage = {
        "completion_rate": bool(ci_completion[0] > 0),
        "collisions_reduced": bool(ci_collisions[0] > 0),
        "reward": bool(ci_reward[0] > 0),
    }

    result = {
        "format": "continuous_brain.generalization_gate.v110",
        "module_version": MODULE_VERSION,
        "source_fingerprint": source_fingerprint(),
        "module_sha256": _sha256_file(__file__),
        "runtime": runtime_info(device),
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation_role": evaluation_role,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_metadata": metadata,
        "replication_prerequisite": replication,
        "kind": kind,
        "experiment": {
            "paired_episodes": int(episodes), "steps_per_episode": int(steps),
            "seed_start": int(seed_start), "seed_end": int(seed_start) + int(episodes) - 1,
            "option_depth": int(option_depth), "selector_topk": int(selector_topk),
            "beam_width": int(beam_width), "max_option_steps": int(max_option_steps),
            "discount": float(discount), "min_commit": int(min_commit),
            "terminate_threshold": float(terminate_threshold),
            "training_performed_during_evaluation": False, "memory_written": False,
        },
        "reactive": r,
        "hierarchical_option_planning": h,
        "option_direct": _summary(direct, steps),
        "paired_advantages_vs_direct": paired_comparison(hierarchical, direct),
        "advantages_vs_reactive": {
            "completion_percentage_points": 100.0 * completion_advantage,
            "collisions_reduced": r["mean_collisions"] - h["mean_collisions"],
            "collision_regression": collision_regression,
            "reward_absolute": h["mean_reward"] - r["mean_reward"],
        },
        "paired_95pct_confidence_intervals": {
            "completion_rate_difference": {"lo": ci_completion[0], "hi": ci_completion[1]},
            "collisions_reduced": {"lo": ci_collisions[0], "hi": ci_collisions[1]},
            "reward_difference": {"lo": ci_reward[0], "hi": ci_reward[1]},
        },
        "pass_thresholds_fixed_before_evaluation": dict(PASS_THRESHOLDS),
        "benchmark_threshold_passed": benchmark_threshold_passed,
        "statistically_supported_advantage": statistically_supported_advantage,
        "reporting_note": "benchmark_threshold_passed and statistically_supported_advantage are independent "
                          "and must be read together, never collapsed into one verdict: a threshold pass with "
                          "an interval spanning zero is a threshold pass without statistical support, not a "
                          "claim of demonstrated superiority.",
        "episodes": {"reactive": reactive, "hierarchical_option_planning": hierarchical, "option_direct": direct},
    }
    result["diagnosis"] = diagnose_gate_result(result)
    result["completion_evidence_status"] = (
        "positive" if ci_completion[0] > 0 else "negative" if ci_completion[1] < 0 else "inconclusive")
    result["confirmation_supported"] = bool(evaluation_role == "confirmation"
        and benchmark_threshold_passed and ci_completion[0] > 0)
    result["checkpoint_unchanged"] = _sha256_file(checkpoint) == before_hash
    if not result["checkpoint_unchanged"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    if output:
        write_json(output, result)
    return result
