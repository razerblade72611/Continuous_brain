"""I1 evaluator: integrated-individual checks from the arm outputs of one run directory.

Thresholds are read from the NORMAL arm header, where the arm process recorded them from the
frozen legacy evaluator modules (constant name, value, source SHA-256). Nothing is retyped here.

Verdict structure
  validity  (blocking)  fidelity to the frozen v0.36 driver (the store-held goal, all fields; the
                         environment capabilities; the autobiography bytes), bit-exact replay
                         (store digest and tick) before every intervention, interventions applied
                         exactly as planned (tick, target, kind) and nowhere else, one manifest
                         across all arms, store authority controls, host state read from the store,
                         no driver-side mechanism calls or model forwards, no errors, legacy tree
                         unchanged, pins verified, no hidden mutable state (an RNG-perturbed twin
                         of every NORMAL life is bit-identical; frozen models and fixed-parameter
                         objects are unchanged before/after every life), donor pairing fixed by ids
  primary   C1-C10       rates against frozen thresholds
  secondary S1-S4        preregistered predictions, the new integration outcome, and its mechanism at
                         matched goal-2 episodes (reported, never gating)
Status is ENGINEERING_ONLY unless the run was made on a family an APPROVED preregistration opened.
"""
from __future__ import annotations

import json, random
from pathlib import Path

AUTHORITY_ARMS = ["AUTHORITY_SKILLS", "AUTHORITY_GOAL", "AUTHORITY_CURRICULUM"]
PLANNED = ["ERASED", "DONOR", "ROLLBACK", *AUTHORITY_ARMS]
ARMS = ["LEGACY_REF", "NORMAL", "FIXED_SCAFFOLD", "ERASED", "DONOR", "ROLLBACK", "PERSIST_SAVE", "PERSIST_RESUME",
        "RNG_PERTURBED", *AUTHORITY_ARMS]
S4_K = (1, 2, 4)
# The authoritative I1 validity schema: every report carries exactly these checks, in this order, and the
# approval gate (runner.approve), the lifecycle suite and the tests import this tuple rather than a copy.
VALIDITY_KEYS = ("fidelity_to_frozen_v036_driver", "replay_bit_exact_before_every_intervention",
                 "interventions_applied_exactly_as_planned", "manifest_consistent_across_arms",
                 "store_authority_controls_change_the_individual", "host_state_read_from_store",
                 "driver_record_matches_store_lifecycle", "no_driver_side_model_forwards",
                 "no_driver_side_mechanism_calls", "no_hidden_mutable_state", "donor_pairing_fixed_by_ids",
                 "no_errors", "legacy_tree_unchanged", "pins_verified")            # preregistered matched goal-2 checkpoints for the S4 mechanism analysis
MARK_OF = {"ERASED": "imprint", "DONOR": "imprint", "ROLLBACK": "before_second",
           **{a: "before_second" for a in AUTHORITY_ARMS}}
# an AUTHORITY abort counts only if it is the erased mechanism's own error for missing state
EXPECTED_ABORTS = {"AUTHORITY_SKILLS": "RuntimeError: developmental phases are not initialized"}
SEAL_NOTE = ("primary and secondary values are withheld from display and committed by SHA-256; the sealed file "
             "is not encrypted and anyone with the run directory can read it. Open only after the scientific "
             "I1 result is frozen.")
STRATEGY_SCAFFOLD, STRATEGY_DIRECT = 1, 0


def load(run: Path, arm: str) -> dict[int, dict]:
    return {r["sid"]: r for r in (json.loads(x) for x in (run / arm / "lives.jsonl").read_text().splitlines() if x)}


def rate(xs):
    xs = [bool(x) for x in xs]
    return (sum(xs) / len(xs)) if xs else None, len(xs)


def goal_fields(g):
    return None if g is None else {k: v for k, v in g.items() if k != "__type__"}


def s4_mechanism(N: dict, F: dict, both: list) -> dict:
    """S4 (preregistered secondary, estimation only, non-gating): the S3 mechanism at matched goal-2
    checkpoints k in S4_K. For each k, the at-risk individuals are the first-goal completers of both arms
    whose goal-2 pursuit reached check k in BOTH arms (still pursuing after k episodes, by the store-held
    lifecycle's own completion check). Deltas are from check 0 of the same pursuit:
      reliability: the goal commitment's self-model reliability (what the goal is scored on)
      capability:  the environment's true capability on that commitment
    Reported: per-arm means, the paired difference controller - fixed with the paired bootstrap 95% CI
    (5000 resamples, seed 0, as S3). Descriptive only: successful work steps (throughput) and failed work
    steps (surprises) per goal-2 episode, from the frozen episode metrics."""
    out = {"k": {}, "definition": "at-risk = reached goal-2 check k in both arms; deltas from check 0"}
    for k in S4_K:
        risk = [s for s in both if len(N[s]["second_series"]) > k and len(F[s]["second_series"]) > k]
        row = {"n_at_risk": len(risk)}
        for q in ("reliability", "capability"):
            a = [N[s]["second_series"][k][q] - N[s]["second_series"][0][q] for s in risk]
            b = [F[s]["second_series"][k][q] - F[s]["second_series"][0][q] for s in risk]
            row[q] = {"controller_mean_gain": (sum(a) / len(a)) if a else None,
                      "fixed_scaffold_mean_gain": (sum(b) / len(b)) if b else None,
                      "paired_difference": ((sum(a) - sum(b)) / len(a)) if a else None,
                      "paired_bootstrap_95ci": bootstrap_diff(a, b)}
        out["k"][str(k)] = row
    for name, R in (("controller", N), ("fixed_scaffold", F)):
        eps = [e for s in both for e in R[s]["second_series"] if "work_successes" in e]
        out[f"descriptive_{name}"] = {
            "episodes": len(eps),
            "successful_work_steps_per_episode": (sum(e["work_successes"] for e in eps) / len(eps)) if eps else None,
            "failed_work_steps_per_episode": (sum(e["work_failures"] for e in eps) / len(eps)) if eps else None}
    return out


def bootstrap_diff(a: list, b: list, n_boot: int = 5000, seed: int = 0):
    """Paired bootstrap CI for mean(a) - mean(b) over individuals (same individuals in both)."""
    if not a:
        return None
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    return [diffs[int(.025 * n_boot)], diffs[int(.975 * n_boot) - 1]]


def _plan_entries(run: Path, arm: str) -> dict[str, list]:
    p = run / f"plan_{arm}.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text())
    return {sid: (e["interventions"] if isinstance(e, dict) else e) for sid, e in raw.items()}


def _store_goal(rec: dict):
    """The authoritative first goal: the one the store-held lifecycle holds (completed or active)."""
    st = rec.get("first_goal_store_status")
    if st is None:
        return None
    if st["completed_goals"]:
        return goal_fields(st["completed_goals"][0])
    return goal_fields(st["active"])


def _second_half(rec: dict) -> tuple:
    return (json.dumps(goal_fields(rec.get("second_goal")), sort_keys=True),
            json.dumps(rec.get("second_observation"), sort_keys=True),
            tuple(rec.get("second_strategies") or ()), rec.get("second_completed"), rec.get("first_completed_store"))


def evaluate(run: Path, *, sealed: bool = False, scientific: bool = False,
             expected_manifest: str | None = None) -> dict:
    """Write i1_report.json (or, sealed, sealed/i1_report.json) and return the visible report.

    sealed: the full report goes to run/sealed/ and only its SHA-256, the validity block and the
    counts are returned and written to run/i1_validity.json (engineering preflight). See SEAL_NOTE."""
    run = Path(run)
    A = {arm: load(run, arm) for arm in ARMS}
    headers = {arm: json.loads((run / arm / "header.json").read_text()) for arm in ARMS}
    TH = headers["NORMAL"]["frozen_constants"]
    sids = sorted(A["NORMAL"])
    N = {s: A["NORMAL"][s].get("record") for s in sids}
    errors = {arm: [s for s, r in A[arm].items() if "error" in r] for arm in ARMS}
    if errors["NORMAL"] or errors["LEGACY_REF"]:
        report = {"format": "cb-i1-report/2", "status": "SCIENTIFIC" if scientific else "ENGINEERING_ONLY",
                  "run": run.name, "verdict": "BLOCKED", "errors": {k: v for k, v in errors.items() if v}}
        (run / "i1_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    lives = [r for arm in ARMS if arm != "LEGACY_REF" for r in A[arm].values() if "record" in r]

    # ---------------------------------------------------------------- validity
    fid = []
    for s in sids:
        ref, fx = A["LEGACY_REF"][s], A["FIXED_SCAFFOLD"][s].get("record")
        if fx is None:
            fid.append(False); continue
        if ref["first_goal"] is None:
            fid.append(fx["first_goal"] is None); continue
        fid.append(_store_goal(fx) == goal_fields(ref["first_goal"])
                   and fx["first_completed_store"] == ref["first_completed"]
                   and fx["capability_after_first"] == ref["capability_after_first"]
                   and fx["digests"].get("before_second_autobiography") == ref["memory_sha256"])
    replay = []
    for arm in PLANNED:
        mark = MARK_OF[arm]
        for s, r in A[arm].items():
            if "record" in r:
                got, want = r["record"], N[s]
                replay.append(got["digests"].get(f"{mark}_pre") == want["digests"].get(f"{mark}_pre")
                              and got["marks"].get(mark) == want["marks"].get(mark))
    planned_ok, planned_counts = True, {}
    for arm in PLANNED:
        plan = _plan_entries(run, arm)
        planned_counts[arm] = len(plan)
        ran = {str(s) for s, r in A[arm].items() if "record" in r}
        if ran != set(plan):
            planned_ok = False
        for sid, ivs in plan.items():
            r = A[arm].get(int(sid), {})
            got = [(x["tick"], x["target"], x["kind"]) for x in r.get("runtime", {}).get("interventions", [])]
            if got != [(iv["start_tick"], iv["target"], iv["kind"]) for iv in ivs]:
                planned_ok = False
    for arm in ("NORMAL", "FIXED_SCAFFOLD", "PERSIST_SAVE", "PERSIST_RESUME", "RNG_PERTURBED"):
        if any(r.get("runtime", {}).get("interventions") for r in A[arm].values() if "record" in r):
            planned_ok = False
    want_manifest = expected_manifest
    if want_manifest is None and (run / "manifest.json").is_file():
        want_manifest = json.loads((run / "manifest.json").read_text()).get("manifest_sha256")
    arm_manifests = {arm: h.get("manifest_sha256") for arm, h in headers.items()}
    manifest_ok = want_manifest is not None and all(v == want_manifest for v in arm_manifests.values())
    authority = {}
    for arm in AUTHORITY_ARMS:
        res = []
        for s, r in A[arm].items():
            if "record" not in r:
                continue
            if "aborted" in r:
                res.append(r["aborted"] == EXPECTED_ABORTS.get(arm))
            else:
                res.append(_second_half(r["record"]) != _second_half(N[s]))
        authority[arm] = {"n": len(res), "changed_or_aborted": sum(res),
                          "aborted": sum(1 for r in A[arm].values() if "aborted" in r),
                          "unexpected_aborts": sorted({r["aborted"] for r in A[arm].values()
                                                       if "aborted" in r and r["aborted"] != EXPECTED_ABORTS.get(arm)})}
    authority_ok = all(v["n"] > 0 and v["changed_or_aborted"] == v["n"] for v in authority.values())
    store_state = all(all(v > 0 for v in r["runtime"]["state_reads_nonempty"].values())
                      for r in A["NORMAL"].values() if "record" in r and r["record"]["first_goal"] is not None)
    driver_forwards = sum(v for r in lives for k, v in r["runtime"].get("model_forwards", {}).items()
                          if k.endswith("|driver"))
    host_forwards = sum(v for r in lives for k, v in r["runtime"].get("model_forwards", {}).items()
                        if not k.endswith("|driver"))
    cross_nested = sum(v for r in lives for k, v in r["runtime"].get("nested_by_route", {}).items()
                       if k.split("->", 1)[0] != k.split("->", 1)[1].split(":", 1)[0])
    blocked = sum(sum(r["runtime"].get("host_blocked", {}).values()) for r in lives)
    unbound = sum(sum(r["runtime"].get("unbound_calls", {}).values()) for r in lives)
    driver_matches_store = all(r["first_completed"] == r["first_completed_store"]
                               for r in N.values() if r and r["first_goal"] is not None)
    # no hidden mutable state: global randomness does not matter, and no process-level object changes
    rng_twins = [s for s in sids if "record" in A["RNG_PERTURBED"].get(s, {})]
    rng_ok = len(rng_twins) == len(sids) and all(
        (A["RNG_PERTURBED"][s]["record"]["final_store_digest"], A["RNG_PERTURBED"][s]["record"]["final_tick"],
         A["RNG_PERTURBED"][s]["record"]["life_trace_sha256"]) ==
        (N[s]["final_store_digest"], N[s]["final_tick"], N[s]["life_trace_sha256"]) for s in rng_twins)
    frozen = [json.dumps(r["runtime"].get(k), sort_keys=True) for r in lives for k in ("frozen_before", "frozen_after")]
    frozen_ok = bool(frozen) and len(set(frozen)) == 1 and json.loads(frozen[0]) is not None
    # donor pairing fixed by ids: the donor of sid is sid XOR 1, its birth autobiography exactly
    dplan = json.loads((run / "plan_DONOR.json").read_text()) if (run / "plan_DONOR.json").is_file() else {}
    import base64, hashlib
    def _donor_entry_ok(sid: str, e: list) -> bool:
        partner = N.get(int(sid) ^ 1)
        got = A["DONOR"].get(int(sid), {}).get("record")
        return (len(e) == 1 and e[0].get("donor_sid") == int(sid) ^ 1 and partner is not None and got is not None
                and e[0].get("payload_b64") == partner["birth_state_b64"]
                # what the store held right after the intervention is exactly the partner's autobiography
                and got["digests"].get("imprint_autobiography") ==
                    hashlib.sha256(base64.b64decode(partner["birth_state_b64"])).hexdigest())
    donor_ok = set(dplan) == {str(s) for s in sids} and all(_donor_entry_ok(sid, e) for sid, e in dplan.items())
    safety = json.loads((run / "safety.json").read_text()) if (run / "safety.json").is_file() else {}
    validity = {"fidelity_to_frozen_v036_driver": all(fid) and len(fid) == len(sids),
                "replay_bit_exact_before_every_intervention": all(replay) and bool(replay),
                "interventions_applied_exactly_as_planned": planned_ok,
                "manifest_consistent_across_arms": manifest_ok,
                "store_authority_controls_change_the_individual": authority_ok,
                "host_state_read_from_store": store_state,
                "driver_record_matches_store_lifecycle": driver_matches_store,
                "no_driver_side_model_forwards": driver_forwards == 0 and host_forwards > 0,
                "no_driver_side_mechanism_calls": blocked == 0 and unbound == 0 and cross_nested == 0,
                "no_hidden_mutable_state": rng_ok and frozen_ok,
                "donor_pairing_fixed_by_ids": donor_ok,
                "no_errors": not any(errors.values()),
                "legacy_tree_unchanged": safety.get("legacy_tree_unchanged") is True,
                "pins_verified": safety.get("pins_verified") is True}
    if tuple(validity) != VALIDITY_KEYS:                  # the schema cannot drift silently
        raise AssertionError(f"validity keys {tuple(validity)} differ from VALIDITY_KEYS")
    valid = all(validity.values())

    # ---------------------------------------------------------------- primary
    authored = [s for s in sids if N[s] and N[s]["first_goal"] is not None]
    completers = [s for s in authored if N[s]["first_completed_store"]]       # the store's lifecycle decides

    def first_strategy_ok(r):
        init = {int(k): v for k, v in (r["initial_phase_strategies"] or {}).items()}
        fp, sp = r["evaluator_fragile_primary"], r["evaluator_stable_primary"]
        first = r["first_strategies"][0] if r["first_strategies"] else None
        return (init.get(fp) == STRATEGY_SCAFFOLD and init.get(sp) == STRATEGY_DIRECT
                and (first is None or first == STRATEGY_SCAFFOLD))

    rb = {s: A["ROLLBACK"][s] for s in completers if "record" in A["ROLLBACK"].get(s, {})}
    vals = {
        "first_goal_endogenous": rate([N[s]["first_goal"] is not None and
                                       N[s]["first_goal"]["primary"] == N[s]["evaluator_fragile_primary"]
                                       for s in sids]),
        "first_goal_completes": rate([bool(N[s] and N[s]["first_completed_store"]) for s in sids]),
        "history_guides_initial_strategy": rate([first_strategy_ok(N[s]) for s in authored]),
        "second_goal_authored": rate([N[s]["second_goal"] is not None for s in completers]),
        "curriculum_advances": rate([bool((N[s]["second_observation"] or {}).get("advanced")) for s in completers]),
        "erased_store_autobiography_no_goal": rate([A["ERASED"][s]["record"]["first_goal"] is None
                                                    for s in sids if "record" in A["ERASED"].get(s, {})]),
        "donor_store_autobiography_flips_goal": rate([
            A["DONOR"][s]["record"]["first_goal"] is not None and N[s]["first_goal"] is not None and
            A["DONOR"][s]["record"]["first_goal"]["primary"] != N[s]["first_goal"]["primary"]
            for s in sids if "record" in A["DONOR"].get(s, {})]),
        "rolled_back_autobiography_repeats_goal": rate([bool((r["record"]["second_observation"] or {})
                                                             .get("same_as_previous")) for r in rb.values()]),
        "updated_vs_rolled_back_diverges": rate([not r["updated_vs_rolled_back_same_semantics"] for r in rb.values()]),
        "persistence_bit_exact": rate([
            "record" in A["PERSIST_RESUME"].get(s, {}) and
            A["PERSIST_RESUME"][s]["record"]["final_store_digest"] == N[s]["final_store_digest"] and
            A["PERSIST_RESUME"][s]["record"]["final_tick"] == N[s]["final_tick"] and
            A["PERSIST_RESUME"][s]["record"]["life_trace_sha256"] == N[s]["life_trace_sha256"]
            for s in authored]),
    }
    primary = {}
    for name, (v, n) in vals.items():
        th = TH[name]["value"]
        primary[name] = {"value": v, "n": n, "threshold": th, "threshold_source": f"{TH[name]['module']}.{TH[name]['constant']}",
                         "meets": v is not None and n > 0 and v >= th}

    # ---------------------------------------------------------------- secondary
    F = {s: A["FIXED_SCAFFOLD"][s].get("record") for s in sids}
    s1 = rate([F[s] is not None and N[s]["digests"].get("before_second_autobiography") ==
               F[s]["digests"].get("before_second_autobiography") and N[s]["first_strategies"] == F[s]["first_strategies"]
               for s in authored])
    sec_strats = [x for s in completers for x in N[s]["second_strategies"]]
    both = [s for s in completers if F[s] is not None and F[s]["first_completed_store"]]
    nc, fc = [bool(N[s]["second_completed"]) for s in both], [bool(F[s]["second_completed"]) for s in both]
    secondary = {
        "S1_first_half_identical_controller_vs_fixed_scaffold": {"value": s1[0], "n": s1[1],
                                                                 "prediction": "1.0 (controller and goal share the 0.12 criterion)"},
        "S2_second_goal_episodes_pursued_DIRECT": {"value": (sum(1 for x in sec_strats if x == STRATEGY_DIRECT) /
                                                              len(sec_strats)) if sec_strats else None,
                                                    "episodes": len(sec_strats),
                                                    "prediction": "majority DIRECT (the controller graduates on the 0.12 gain)"},
        "S3_second_goal_completion": {"controller": rate(nc)[0], "fixed_scaffold": rate(fc)[0], "n": len(both),
                                      "difference_controller_minus_fixed": (rate(nc)[0] - rate(fc)[0]) if both else None,
                                      "paired_bootstrap_95ci": bootstrap_diff(nc, fc),
                                      "prediction": "estimation only: no directional prediction registered"},
        "S4_mechanism_matched_episodes": {**s4_mechanism(N, F, both),
                                          "status": "preregistered secondary; estimation only; engineering-motivated "
                                                    "(defined after the dev7 diagnostic); not an admission criterion"},
    }
    verdict = ("BLOCKED" if not valid else
               "PASS" if all(p["meets"] for p in primary.values()) else "FAIL")
    diagnostics = {"fidelity_failures": [s for s, ok in zip(sids, fid) if not ok][:20],
                   "replay_checks": len(replay), "planned_individuals": planned_counts,
                   "authority_controls": authority, "arm_manifests": arm_manifests,
                   "expected_manifest_sha256": want_manifest, "host_side_model_forwards": host_forwards,
                   "driver_side_model_forwards": driver_forwards, "cross_slot_nested_calls": cross_nested,
                   "host_blocked_calls": blocked, "unbound_calls": unbound,
                   "rng_perturbed_twins_identical": sum(1 for s in rng_twins if
                                                        A["RNG_PERTURBED"][s]["record"]["final_store_digest"] ==
                                                        N[s]["final_store_digest"]),
                   "frozen_object_digest_sets": len(set(frozen)), "donor_plan_entries": len(dplan)}
    report = {"format": "cb-i1-report/2", "status": "SCIENTIFIC" if scientific else "ENGINEERING_ONLY",
              "run": run.name, "n": len(sids),
              "individuals": [sids[0], sids[-1]], "authored_first_goal": len(authored), "first_goal_completers": len(completers),
              "validity": validity, "validity_diagnostics": diagnostics, "primary": primary, "secondary": secondary,
              "verdict": verdict if scientific else None,
              "proposed_verdict_if_this_were_a_scientific_family": None if scientific else verdict,
              "errors": {k: v for k, v in errors.items() if v}}
    data = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if sealed:
        (run / "sealed").mkdir(exist_ok=True)
        (run / "sealed" / "i1_report.json").write_text(data)
        import hashlib
        visible = {"format": "cb-i1-validity/2", "status": "ENGINEERING_ONLY", "run": run.name, "n": len(sids),
                   "validity": validity, "valid": valid,
                   # only value-free diagnostics: counts of planned individuals or replay checks would
                   # reveal the number of first-goal completers (C2), so they stay in the sealed file
                   "validity_diagnostics": {k: diagnostics[k] for k in ("arm_manifests", "expected_manifest_sha256",
                                                                       "driver_side_model_forwards",
                                                                       "cross_slot_nested_calls", "host_blocked_calls",
                                                                       "unbound_calls")},
                   "sealed_report_sha256": hashlib.sha256(data.encode()).hexdigest(), "note": SEAL_NOTE}
        (run / "i1_validity.json").write_text(json.dumps(visible, indent=2, sort_keys=True) + "\n")
        return visible
    (run / "i1_report.json").write_text(data)
    return report


def print_report(rep: dict) -> None:
    print(json.dumps({k: rep.get(k) for k in ("status", "validity", "verdict",
                                               "proposed_verdict_if_this_were_a_scientific_family",
                                               "sealed_report_sha256") if k in rep}, indent=1))
    for k, p in (rep.get("primary") or {}).items():
        v = "None" if p["value"] is None else str(round(p["value"], 4))
        print(f"  {k:42s} {v:>7s}  n={p['n']:<4d} >= {p['threshold']:<5} {'ok' if p['meets'] else 'NOT MET'}")
    if rep.get("secondary"):
        print(json.dumps(rep["secondary"], indent=1))
