#!/usr/bin/env python3
"""
Continuous Brain v0.19.2 — Current Planner Lineage Requalification v1

READ ONLY. NO TRAINING.

Why:
- v0.19 is closed.
- The integration audit found that the current canonical 14K child already
  contains the historical trained temporal options + recurrent planning stack.
- Before connecting admitted memory to that stack, re-run the existing
  hierarchical planner on the current canonical checkpoint.

This script is deliberately fail-closed:
- it calls the evaluator in-process and validates the returned object;
- it never decides from a pre-existing/stale result file;
- it hashes all protected artifacts before and after;
- any evaluator exception terminates the run immediately;
- it performs no training and does not save a model checkpoint.

Historical seeds are REUSED. This is lineage/capability requalification,
NOT a fresh generalization result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from continuous_brain.planning_v103 import evaluate_hierarchical_options

CURRENT_BASE = Path("state/child_v113_dml_smoke/infant_training.pt")
ADMITTED_MEMORY = Path(
    "state/working_memory_recall_capacity_control_phase1_v1/latest.portable.pt"
)
V018_INTENT = Path("state/option_intent_executive_v1/latest.portable.pt")
V019_CLOSED = Path("state/recurrent_option_progress_v1/latest.portable.pt")

HIST_V103 = Path(
    os.path.expanduser(
        "~/projects/continuous_brain_v0.10.3/continuous_brain_project/"
        "state/child_v103_hierarchical/infant_training.pt"
    )
)
HIST_FINAL_REPORT = Path(
    os.path.expanduser(
        "~/projects/continuous_brain_v0.10.3/continuous_brain_project/"
        "results/P4B_v103_FINAL_BLIND_12001_12100.json"
    )
)

EXPECTED = {
    "current_base":
        "086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f",
    "admitted_memory":
        "f9976894357b795525ce464c4ff5ef5a882f00073ed81444184c1bc79756282c",
    "v018_intent":
        "041acf1bcdf46736a37c12c343cde2d9316b65cf4b4ee08604cc9c572585e5e2",
    "v019_closed":
        "237dd2029bfeb72acb953395f65966d7c3b1ecbfb0453fa7ff07fac8fb440018",
    "hist_v103":
        "ba1792d340b761ae436585f77a6a80dd7188b4daf108e4ab4c0c7bf8091fae07",
    "hist_final_report":
        "cecd34bc9fb1a99c33efbc1c0a04c1493487e01fc0744dfe7b00ba868d0712ad",
    "planning_v103_source":
        "35541798a810ae50ba287f784819f29f142f06cf22429bd8b84f1afe48c0c074",
    "infant_source":
        "592f5a29e3c7b6b374f9637f0fb4db47d3fd3f484d36b24352249b60876db3fe",
}

REQUIRED_PLANNING_VERSION = (
    "recurrent-latent-v1+conservative-bellman-q-v1+temporal-options-v1"
)
REQUIRED_OPTION_VERSION = "latent-temporal-options-v1"

SMOKE = dict(
    episodes=10,
    steps=120,
    seed_start=9001,
    option_depth=2,
    selector_topk=2,
    beam_width=8,
    max_option_steps=12,
    discount=.95,
    min_commit=3,
    terminate_threshold=.50,
)

FINAL = dict(
    episodes=100,
    steps=120,
    seed_start=12001,
    option_depth=2,
    selector_topk=2,
    beam_width=8,
    max_option_steps=12,
    discount=.95,
    min_commit=3,
    terminate_threshold=.50,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def require_sha(label: str, path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing {path}")
    got = sha256(path)
    if got != expected:
        raise RuntimeError(
            f"{label}: SHA mismatch\nexpected={expected}\nactual={got}\npath={path}"
        )
    return got


def protected_paths():
    return {
        "current_base": CURRENT_BASE,
        "admitted_memory": ADMITTED_MEMORY,
        "v018_intent": V018_INTENT,
        "v019_closed": V019_CLOSED,
    }


def snapshot_hashes():
    return {
        name: sha256(path)
        for name, path in protected_paths().items()
    }


def require_protected_unchanged(before):
    after = snapshot_hashes()
    if before != after:
        raise RuntimeError(
            "protected artifact changed during read-only requalification:\n"
            + json.dumps({"before": before, "after": after}, indent=2)
        )
    return after


def load_payload(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def state_dict(payload):
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected checkpoint payload: {type(payload)}")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("checkpoint has no model state_dict")
    return model


def metadata(payload):
    m = payload.get("metadata")
    if not isinstance(m, dict):
        return {}
    return m


def compare_legacy_group(current_sd, hist_sd, prefix):
    """
    Compare every historical tensor under prefix with the same-named current
    tensor when shape/dtype match. Growth-only current tensors are not treated
    as historical tensors.
    """
    rows = []
    missing = []
    shape_mismatch = []
    exact = 0
    changed = []

    hist_keys = sorted(k for k in hist_sd if k.startswith(prefix))
    for k in hist_keys:
        hv = hist_sd[k]
        if k not in current_sd:
            missing.append(k)
            continue
        cv = current_sd[k]
        if tuple(cv.shape) != tuple(hv.shape) or cv.dtype != hv.dtype:
            shape_mismatch.append({
                "key": k,
                "historical_shape": list(hv.shape),
                "current_shape": list(cv.shape),
                "historical_dtype": str(hv.dtype),
                "current_dtype": str(cv.dtype),
            })
            continue
        same = torch.equal(cv.cpu(), hv.cpu())
        if same:
            exact += 1
        else:
            diff = float((cv.float().cpu() - hv.float().cpu()).abs().max())
            changed.append({"key": k, "max_abs_diff": diff})
        rows.append(k)

    return {
        "prefix": prefix,
        "historical_tensor_count": len(hist_keys),
        "comparable_tensor_count": len(rows),
        "exact_tensor_count": exact,
        "changed_tensor_count": len(changed),
        "missing_tensor_count": len(missing),
        "shape_mismatch_count": len(shape_mismatch),
        "all_comparable_exact": bool(rows and not changed),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "changed": changed[:100],
    }


def verify_current_metadata(current):
    meta = metadata(current)
    required = {
        "planning_cortex_version": REQUIRED_PLANNING_VERSION,
        "option_cortex_version": REQUIRED_OPTION_VERSION,
        "planning_updates": 4000,
        "planning_bellman_q_updates": 3000,
        "option_updates": 3000,
    }
    checks = {
        k: (meta.get(k) == v)
        for k, v in required.items()
    }

    growth = meta.get("growth_equivalence_audit") or {}
    by_interface = growth.get("by_interface") or {}
    exact_growth = bool(growth.get("exact_function_preservation"))
    growth_zero = {
        name: by_interface.get(name)
        for name in (
            "option_action",
            "option_q",
            "option_selector",
            "option_term",
            "planning_latent",
            "planning_step",
            "target_spatial",
        )
    }
    all_zero = all(
        value == 0.0
        for value in growth_zero.values()
    )

    return {
        "required_metadata": required,
        "checks": checks,
        "all_required_metadata_match": all(checks.values()),
        "growth_exact_function_preservation": exact_growth,
        "growth_interface_max_abs_difference": growth_zero,
        "growth_relevant_interfaces_all_zero": all_zero,
        "growth_source_sha256": meta.get("growth_source_sha256"),
        "growth_version": meta.get("growth_version"),
        "updates": current.get("updates"),
    }


def source_hashes():
    import continuous_brain.infant as infant
    import continuous_brain.planning_v103 as p103

    infant_path = Path(infant.__file__).resolve()
    planning_path = Path(p103.__file__).resolve()
    hashes = {
        "infant_path": str(infant_path),
        "infant_sha256": sha256(infant_path),
        "planning_v103_path": str(planning_path),
        "planning_v103_sha256": sha256(planning_path),
    }
    if hashes["infant_sha256"] != EXPECTED["infant_source"]:
        raise RuntimeError("current infant.py differs from audited source")
    if hashes["planning_v103_sha256"] != EXPECTED["planning_v103_source"]:
        raise RuntimeError("current planning_v103.py differs from audited source")
    return hashes


def validate_result(result, expected_args, expected_checkpoint_sha):
    if not isinstance(result, dict):
        raise RuntimeError("evaluator returned non-dict")
    if result.get("format") != "continuous_brain.p4b_hierarchical_option_planning_gate.v103":
        raise RuntimeError("unexpected evaluator result format")
    if result.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise RuntimeError("evaluator result checkpoint SHA mismatch")
    ex = result.get("experiment") or {}
    for k, v in expected_args.items():
        actual = ex.get({
            "episodes": "paired_episodes",
            "steps": "steps_per_episode",
        }.get(k, k))
        if actual != v:
            raise RuntimeError(
                f"evaluator protocol mismatch: {k} expected {v} got {actual}"
            )
    if ex.get("training_performed_during_evaluation") is not False:
        raise RuntimeError("evaluator does not attest no training")
    for key in (
        "reactive",
        "option_direct_diagnostic",
        "hierarchical_option_planning",
    ):
        if not isinstance(result.get(key), dict):
            raise RuntimeError(f"missing evaluation summary: {key}")
        if int(result[key].get("episodes", -1)) != int(expected_args["episodes"]):
            raise RuntimeError(f"episode count mismatch in {key}")
    if not isinstance(result.get("p4b_passed"), bool):
        raise RuntimeError("p4b_passed missing/not bool")
    return result


def summarize(result):
    return {
        "p4b_passed": result["p4b_passed"],
        "reactive": result["reactive"],
        "option_direct": result["option_direct_diagnostic"],
        "hierarchical": result["hierarchical_option_planning"],
    }


def historical_report():
    if not HIST_FINAL_REPORT.is_file():
        return {
            "available": False,
            "path": str(HIST_FINAL_REPORT),
        }
    require_sha(
        "historical_final_report",
        HIST_FINAL_REPORT,
        EXPECTED["hist_final_report"],
    )
    data = json.loads(HIST_FINAL_REPORT.read_text())
    return {
        "available": True,
        "path": str(HIST_FINAL_REPORT),
        "sha256": EXPECTED["hist_final_report"],
        "p4b_passed": data.get("p4b_passed"),
        "checkpoint_sha256": data.get("checkpoint_sha256"),
        "reactive": data.get("reactive"),
        "option_direct": data.get("option_direct_diagnostic"),
        "hierarchical": data.get("hierarchical_option_planning"),
    }


def metric_delta(current, historical):
    if not historical.get("available"):
        return None
    pairs = {
        "reactive": "reactive",
        "option_direct": "option_direct",
        "hierarchical": "hierarchical",
    }
    out = {}
    for ck, hk in pairs.items():
        out[ck] = {}
        for metric in (
            "completion_rate",
            "mean_collisions",
            "mean_reward",
            "mean_steps_penalized",
        ):
            cv = current[ck].get(metric)
            hv = historical[hk].get(metric)
            if cv is not None and hv is not None:
                out[ck][metric] = float(cv) - float(hv)
    return out


def resolve_device(name):
    if name == "cpu":
        return "cpu"
    if name == "auto":
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            return "cpu"
    return name


def run_eval(label, args, device, output_path, expected_sha, protected_before):
    # Stale output cannot influence decision.
    if output_path.exists():
        output_path.unlink()

    print(f"\n=== {label} ===")
    print("device:", device)
    print("protocol:", json.dumps(args, sort_keys=True))

    # Any exception here propagates and terminates the entire run.
    result = evaluate_hierarchical_options(
        CURRENT_BASE,
        device=device,
        output=None,
        **args,
    )
    validate_result(result, args, expected_sha)
    require_protected_unchanged(protected_before)
    atomic_json(output_path, result)

    # Decision is from in-memory returned result, not the file.
    print(label + "_P4B:", result["p4b_passed"])
    print(
        "reactive completion/collisions:",
        result["reactive"]["completion_rate"],
        result["reactive"]["mean_collisions"],
    )
    print(
        "direct completion/collisions:",
        result["option_direct_diagnostic"]["completion_rate"],
        result["option_direct_diagnostic"]["mean_collisions"],
    )
    print(
        "hierarchical completion/collisions:",
        result["hierarchical_option_planning"]["completion_rate"],
        result["hierarchical_option_planning"]["mean_collisions"],
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        default="state/planner_lineage_requalification_v1",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or a torch device string",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("CURRENT_PLANNER_LINEAGE_REQUALIFICATION_V1")
    print("READ_ONLY: true")
    print("TRAINING_PERFORMED: false")
    print("GENERALIZATION_CLAIMED: false")
    print("HISTORICAL_SEEDS_REUSED: true")

    # Hard provenance checks.
    require_sha("current_base", CURRENT_BASE, EXPECTED["current_base"])
    require_sha("admitted_memory", ADMITTED_MEMORY, EXPECTED["admitted_memory"])
    require_sha("v018_intent", V018_INTENT, EXPECTED["v018_intent"])
    require_sha("v019_closed", V019_CLOSED, EXPECTED["v019_closed"])
    require_sha("hist_v103", HIST_V103, EXPECTED["hist_v103"])
    sources = source_hashes()

    protected_before = snapshot_hashes()

    current = load_payload(CURRENT_BASE)
    historical = load_payload(HIST_V103)

    meta_check = verify_current_metadata(current)
    if not meta_check["all_required_metadata_match"]:
        raise RuntimeError("current checkpoint planning/option metadata mismatch")
    if not meta_check["growth_exact_function_preservation"]:
        raise RuntimeError("growth equivalence audit did not establish exact preservation")
    if not meta_check["growth_relevant_interfaces_all_zero"]:
        raise RuntimeError("relevant planning/option growth interface has non-zero preservation error")

    current_sd = state_dict(current)
    hist_sd = state_dict(historical)
    parity = {
        prefix: compare_legacy_group(current_sd, hist_sd, prefix)
        for prefix in (
            "planning.",
            "options.",
            "target_spatial.",
        )
    }

    hist_report = historical_report()

    device = resolve_device(args.device)
    current_sha = EXPECTED["current_base"]

    smoke = run_eval(
        "CURRENT_SMOKE_9001_9010",
        SMOKE,
        device,
        out / "current_smoke_9001_9010.json",
        current_sha,
        protected_before,
    )

    if not smoke["p4b_passed"]:
        final_result = None
        status = "SMOKE_FAIL"
        print("Final 100-episode historical protocol not run because smoke failed.")
    else:
        final_result = run_eval(
            "CURRENT_FINAL_REQUAL_12001_12100",
            FINAL,
            device,
            out / "current_final_12001_12100.json",
            current_sha,
            protected_before,
        )
        status = "PASS" if final_result["p4b_passed"] else "FINAL_FAIL"

    protected_after = require_protected_unchanged(protected_before)

    result = {
        "format": "continuous_brain.current_planner_lineage_requalification_v1",
        "status": status,
        "read_only": True,
        "training_performed": False,
        "historical_seeds_reused": True,
        "generalization_established": False,
        "current_checkpoint_sha256": current_sha,
        "historical_v103_checkpoint_sha256": EXPECTED["hist_v103"],
        "source_hashes": sources,
        "current_metadata": meta_check,
        "legacy_tensor_parity": parity,
        "historical_final_report": hist_report,
        "current_smoke": summarize(smoke),
        "current_final": summarize(final_result) if final_result else None,
        "current_minus_historical": (
            metric_delta(summarize(final_result), hist_report)
            if final_result else None
        ),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_unchanged": protected_before == protected_after,
        "v019_remains_closed": True,
        "admitted_memory_preserved": True,
        "v018_intent_preserved": True,
    }
    atomic_json(out / "result.json", result)

    print("\nLEGACY_TENSOR_PARITY")
    for prefix, r in parity.items():
        print(
            prefix,
            "historical=", r["historical_tensor_count"],
            "comparable=", r["comparable_tensor_count"],
            "exact=", r["exact_tensor_count"],
            "changed=", r["changed_tensor_count"],
            "missing=", r["missing_tensor_count"],
            "shape_mismatch=", r["shape_mismatch_count"],
        )

    print("\nPROTECTED_UNCHANGED:", result["protected_unchanged"])
    print("V019_REMAINS_CLOSED: true")
    print("ADMITTED_MEMORY_PRESERVED: true")
    print("V018_INTENT_PRESERVED: true")
    print("TRAINING_PERFORMED: false")
    print("GENERALIZATION_ESTABLISHED: false")
    print("CURRENT_PLANNER_P4B_REQUALIFICATION:", status)

    if status != "PASS":
        raise SystemExit(4)


if __name__ == "__main__":
    main()
