#!/usr/bin/env python3
"""Require measured, checkpoint-bound evidence; missing evidence never means pass."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_brain.evidence_v114 import (BUILD_ID, EXACT_SOURCE_SHA, STAGES, valid_p4b,
                                           valid_training, write_json)


def check(migration, trainings, p4bs, bootstrap, world, generalized):
    gates = {}
    gates["migration_exact"] = (migration.get("build_id") == BUILD_ID
        and migration.get("source_checkpoint_sha256") == EXACT_SOURCE_SHA
        and migration.get("exact_function_preservation") is True
        and migration.get("max_abs_difference") == 0.0
        and migration.get("source_tensor_mismatches") == 0
        and migration.get("learned_parameters_added") == 0)
    previous = migration.get("source_checkpoint_sha256")
    runtime_ids = []
    for stage in STAGES:
        tr, p4 = trainings[stage], p4bs[stage]
        gates[f"{stage}_training_integrity"] = bool(valid_training(tr, stage)
            and tr.get("source_checkpoint_sha256") == migration.get("source_checkpoint_sha256")
            and tr.get("source_legacy_sha256") == migration.get("source_legacy_sha256")
            and (tr.get("stage_admission") or {}).get("checkpoint_sha256") == previous)
        endpoint = tr.get("checkpoint_sha256")
        gates[f"{stage}_p4b_regression"] = valid_p4b(p4, endpoint)
        previous = endpoint
        runtime_ids.append((p4.get("_v114_evidence") or {}).get("runtime_id"))
    final = trainings["moderate"].get("checkpoint_sha256")
    for name, result in (("bootstrap", bootstrap), ("world", world), ("generalized", generalized)):
        evidence = result.get("_v114_evidence") or {}
        gates[f"{name}_checkpoint_binding"] = (len(final or "") == 64
            and evidence.get("checkpoint_sha256") == final
            and evidence.get("build_id") == BUILD_ID)
        runtime_ids.append(evidence.get("runtime_id"))
    gates["same_qualification_runtime"] = (all(isinstance(v, str) and len(v) == 64 for v in runtime_ids)
                                           and len(set(runtime_ids)) == 1)
    gates["bootstrap_retention"] = bootstrap.get("passed") is True
    gates["world_retention"] = (world.get("world_consolidation_passed") is True
        and all((world.get("domains") or {}).get(d, {}).get("world_model_passed") is True
                for d in ("foraging", "detour")))
    gates["generalized_randomized_detour_development"] = (
        generalized.get("kind") == "randomized_detour" and generalized.get("seed_start") == 41001
        and generalized.get("episodes") == 100
        and isinstance(generalized.get("completion_rate"), (int, float))
        and .20 <= generalized["completion_rate"] <= 1.0
        and generalized.get("training_performed_during_evaluation") is False)
    return dict(format="continuous_brain.v114.protected_growth_readiness", build_id=BUILD_ID,
                gates=gates, protected_growth_passed=all(gates.values()),
                reserved_blind_range="51001-51100", blind_evaluation_included=False,
                meaning="Development readiness only; capability is measured by the user's qualification run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--migration-result", required=True)
    for stage in STAGES:
        parser.add_argument(f"--{stage}-training-result", required=True)
        parser.add_argument(f"--{stage}-p4b-result", required=True)
    for name in ("bootstrap", "world-retention", "generalized"):
        parser.add_argument(f"--{name}-result", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    def load(path):
        return json.loads(Path(path).read_text())
    result = check(load(args.migration_result),
        {s: load(getattr(args, f"{s}_training_result")) for s in STAGES},
        {s: load(getattr(args, f"{s}_p4b_result")) for s in STAGES},
        load(args.bootstrap_result), load(args.world_retention_result), load(args.generalized_result))
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["protected_growth_passed"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
