#!/usr/bin/env python3
"""Negative controls for the I1 evaluator: every validity check must turn False when the fault it
guards against is injected into a copy of a finished run directory.

usage: python scripts/dev/i1_tamper_check.py RUN_DIR [OUT_JSON]

The run directory is copied (without historical_core/ and persist/) to a temporary directory per
fault; the original is never modified. Exit code 0 only if every fault is detected by its check and
the untampered copy is fully valid."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from continuous_brain_unified.integration.evaluate import evaluate  # noqa: E402


def _rows(d: Path, arm: str) -> list[dict]:
    return [json.loads(x) for x in (d / arm / "lives.jsonl").read_text().splitlines() if x.strip()]


def _first(rows: list[dict]) -> dict:
    return next(r for r in rows if "record" in r)


def _with_normal_second_half(src: Path):
    normal = {r["sid"]: r for r in _rows(src, "NORMAL")}

    def mutate(rows):
        for r in rows:
            r.pop("aborted", None)
            if "record" in r:
                for k in ("second_goal", "second_observation", "second_strategies", "second_completed",
                          "first_completed_store"):
                    r["record"][k] = normal[r["sid"]]["record"][k]
    return mutate


def faults(src: Path) -> list[tuple]:
    def set_goal_cycle(R):
        _first(R)["record"]["first_goal_store_status"]["completed_goals"][0]["completed_cycle"] = 99
    return [
        ("store goal completed_cycle altered", "FIXED_SCAFFOLD", set_goal_cycle, "fidelity_to_frozen_v036_driver"),
        ("capability after goal 1 altered", "FIXED_SCAFFOLD",
         lambda R: _first(R)["record"].__setitem__("capability_after_first", [0.5, 0.5]), "fidelity_to_frozen_v036_driver"),
        ("intervention mark tick differs from NORMAL", "DONOR",
         lambda R: _first(R)["record"]["marks"].__setitem__("imprint", -1), "replay_bit_exact_before_every_intervention"),
        ("pre-intervention digest differs from NORMAL", "ERASED",
         lambda R: _first(R)["record"]["digests"].__setitem__("imprint_pre", "0" * 64),
         "replay_bit_exact_before_every_intervention"),
        ("intervention applied at the wrong tick", "ROLLBACK",
         lambda R: _first(R)["runtime"]["interventions"][0].__setitem__("tick", 0), "interventions_applied_exactly_as_planned"),
        ("intervention in an unplanned arm", "NORMAL",
         lambda R: _first(R)["runtime"]["interventions"].append({"tick": 1, "target": "x", "kind": "erase"}),
         "interventions_applied_exactly_as_planned"),
        ("goal erase has no effect", "AUTHORITY_GOAL", _with_normal_second_half(src),
         "store_authority_controls_change_the_individual"),
        ("skills erase has no effect", "AUTHORITY_SKILLS", _with_normal_second_half(src),
         "store_authority_controls_change_the_individual"),
        ("curriculum erase has no effect", "AUTHORITY_CURRICULUM", _with_normal_second_half(src),
         "store_authority_controls_change_the_individual"),
        ("driver-side model forward", "PERSIST_SAVE",
         lambda R: _first(R)["runtime"]["model_forwards"].__setitem__("neutral_model:i1|driver", 1),
         "no_driver_side_model_forwards"),
        ("cross-slot nested call", "NORMAL",
         lambda R: _first(R)["runtime"]["nested_by_route"].__setitem__(
             "skills->goal_genesis:EndogenousGoalGenerator.generate", 1), "no_driver_side_mechanism_calls"),
        ("blocked driver-side call", "ERASED",
         lambda R: _first(R)["runtime"]["host_blocked"].__setitem__("skills", 1), "no_driver_side_mechanism_calls"),
        ("driver record disagrees with store lifecycle", "NORMAL",
         lambda R: _first(R)["record"].__setitem__("first_completed_store", not _first(R)["record"]["first_completed_store"]),
         "driver_record_matches_store_lifecycle"),
        ("host never read its state", "NORMAL",
         lambda R: _first(R)["runtime"]["state_reads_nonempty"].__setitem__("curriculum", 0), "host_state_read_from_store"),
        ("arm error", "PERSIST_RESUME", lambda R: _first(R).__setitem__("error", "injected"), "no_errors"),
        ("global randomness changed an individual", "RNG_PERTURBED",
         lambda R: _first(R)["record"].__setitem__("final_store_digest", "0" * 64), "no_hidden_mutable_state"),
        ("a frozen model changed during a life", "DONOR",
         lambda R: _first(R)["runtime"]["frozen_after"].__setitem__("neutral_model", "0" * 64), "no_hidden_mutable_state"),
        ("authority abort by an unrelated exception", "AUTHORITY_SKILLS",
         lambda R: [r.__setitem__("aborted", "ValueError: unrelated crash") for r in R if "aborted" in r],
         "store_authority_controls_change_the_individual"),
    ]


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        ignore = shutil.ignore_patterns("historical_core", "persist", "sealed")
        base = Path(tmp) / "untampered"
        shutil.copytree(src, base, ignore=ignore)
        v0 = evaluate(base)["validity"]
        out.append({"fault": None, "all_valid": all(v0.values())})
        for i, (name, arm, mutate, check) in enumerate(faults(src)):
            d = Path(tmp) / f"f{i}"
            shutil.copytree(src, d, ignore=ignore)
            rows = _rows(d, arm)
            mutate(rows)
            (d / arm / "lives.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            v = evaluate(d)["validity"]
            out.append({"fault": name, "arm": arm, "check": check, "detected": v[check] is False,
                        "all_valid": all(v.values())})
        d = Path(tmp) / "manifest"
        shutil.copytree(src, d, ignore=ignore)
        h = json.loads((d / "DONOR" / "header.json").read_text())
        h["manifest_sha256"] = "0" * 64
        (d / "DONOR" / "header.json").write_text(json.dumps(h))
        v = evaluate(d)["validity"]
        out.append({"fault": "one arm ran different code (manifest)", "arm": "DONOR",
                    "check": "manifest_consistent_across_arms", "detected": v["manifest_consistent_across_arms"] is False,
                    "all_valid": all(v.values())})
        d = Path(tmp) / "donor"
        shutil.copytree(src, d, ignore=ignore)
        plan = json.loads((d / "plan_DONOR.json").read_text())
        k0 = sorted(plan)[0]
        plan[k0][0]["donor_sid"] = int(k0) + 2                          # a donor chosen by something other than ids
        (d / "plan_DONOR.json").write_text(json.dumps(plan))
        v = evaluate(d)["validity"]
        out.append({"fault": "donor not the id-fixed partner", "arm": "DONOR", "check": "donor_pairing_fixed_by_ids",
                    "detected": v["donor_pairing_fixed_by_ids"] is False, "all_valid": all(v.values())})
        d = Path(tmp) / "sealed_view"
        shutil.copytree(src, d, ignore=ignore)
        visible = evaluate(d, sealed=True)
        allowed = {"arm_manifests", "expected_manifest_sha256", "driver_side_model_forwards",
                   "cross_slot_nested_calls", "host_blocked_calls", "unbound_calls"}
        value_free = (set(visible) == {"format", "status", "run", "n", "validity", "valid", "validity_diagnostics",
                                       "sealed_report_sha256", "note"}
                      and set(visible["validity_diagnostics"]) == allowed)
        out.append({"fault": "sealed (preflight) view is value-free (no outcome-bearing counts)", "arm": None,
                    "check": "sealed view is value-free", "detected": value_free, "all_valid": False})
    ok = out[0]["all_valid"] and all(r["detected"] and not r["all_valid"] for r in out[1:])
    rep = {"format": "cb-i1-tamper/1", "run": src.name, "faults": len(out) - 1, "all_detected": ok, "results": out}
    for r in out:
        print(("BASE " if r["fault"] is None else ("ok   " if r["detected"] else "MISS ")) +
              (r["fault"] or f"untampered copy valid={r['all_valid']}"))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(rep, indent=2) + "\n")
    print("ALL DETECTED" if ok else "SOME FAULT NOT DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
