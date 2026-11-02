"""Independent protocol fixtures and adversarial checks; no abst0 implementation imports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("sol_analyzer", Path(__file__).parents[1] / "scripts/abst0_sol_analyze.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def change(p, edit):
    d = json.loads(p.read_text())
    edit(d)
    write(p, d)


@pytest.fixture
def artifacts(tmp_path):
    # A small hand-defined stream: familiar A/H transition cells, an update,
    # and a new transient in each later life so FIRST has a denominator.
    schema = {a: {"template": f"Statement: {{e}} {a} ...", "para0": f"Statement: {{e}} alternate {a} ..."} for a in S.ATTRS}
    for seed in (12, 13, 14, 15):
        facts = []
        for L, e, attr, value in [(1, "Alice", "city", "Paris"), (1, "Henry", "city", "Rome"),
                                  (2, "Ada", "city", "York"), (2, "Hugh", "city", "Oslo"),
                                  (2, "Alice", "hobby", "Old"), (3, "Tess", "city", "Lima"),
                                  (4, "Alice", "allergy", "Pollen"), (4, "Henry", "allergy", "Seed"),
                                  (4, "Alice", "hobby", "New"), (4, "Theo", "city", "Cairo"),
                                  (5, "Tim", "city", "Quito")]:
            facts.append({"fid": f"f{len(facts)}", "life": L, "entity": e, "attr": attr, "value": value})
        stream = {"seed": seed, "facts": facts, "A": ["Alice", "Ada"], "H": ["Henry", "Hugh"],
                  "transient": ["Tess", "Theo", "Tim"], "T": ["Train"], "V": ["Vera"],
                  "fresh": [["Vera", "city"]], "transitions": [["Alice", "allergy"], ["Henry", "allergy"]],
                  "updates": [["Alice", "hobby"]],
                  "intro": {"Alice": 1, "Henry": 1, "Ada": 2, "Hugh": 2, "Tess": 3, "Theo": 4, "Tim": 5}}
        sub = tmp_path / f"seed{seed}"
        prov = {"code": S.FROZEN_CODE, "seed": seed, "model_manifest": "", "test_mode": True}
        prep = {"provenance": prov, "stream": stream,
                "stream_sha256": S.digest(json.dumps(stream, sort_keys=True).encode())}
        write(sub / f"prep_seed{seed}.json", prep)
        psha = S.digest((sub / f"prep_seed{seed}.json").read_bytes())
        exposure = {"cells": {"Alice|allergy": {"exposed": True}}}
        write(sub / f"exposure_seed{seed}.json", exposure)
        train = {"seed": seed, "provenance": {**prov, "prep_sha256": psha, "sleeps": 5, "arms": "P,U,G", "E": 4},
                 "sleeps": {str(L): {"budget": [10, 100, 20]} for L in range(1, 6)},
                 "arms": {a: {str(L): {"ckpt_sha256": S.digest(f"{seed}:{a}:{L}".encode()), "items": 10,
                                       "tokens": 100, "loss_tokens": 20} for L in range(1, 6)} for a in ("P", "U", "G")},
                 "exposure_sha256": S.digest((sub / f"exposure_seed{seed}.json").read_bytes())}
        write(sub / f"train_seed{seed}.json", train)
        for L in range(1, 6):
            inventory = S.semantic_inventory(stream, L)
            for arm in S.ARMS:
                rows = []
                for cat, cells in inventory.items():
                    if cat in ("HNM", "ANM"):
                        cells = dict(list(cells.items())[:40])
                    for key, truth in sorted(cells.items()):
                        if cat == "UPD" and truth["value"] is None:
                            continue
                        e, attr = key.split("|")
                        kinds = schema[attr] if cat in S.POSITIVE else {"template": schema[attr]["template"]}
                        value = truth["value"] if cat == "UPD" else truth
                        for kind, stem in kinds.items():
                            answer = value if value is not None and arm != "N" else "UNKNOWN"
                            if arm == "G" and cat == "HNM" and L == 5:
                                answer = "Confabulated"
                            row = {"cat": cat, "key": key, "kind": kind, "item": stem.replace("{e}", e),
                                   "answer": answer, "outcome": S.classify(answer, value)}
                            if cat == "UPD":
                                row.update(value=value, old=truth["old"], new=truth["new"])
                            else:
                                row["value"] = value
                            rows.append(row)
                for mode in (("base",) if arm == "N" else ("merged", "unmerged")):
                    name = f"eval_N_s{L}.json" if arm == "N" else f"eval_{arm}_s{L}_{mode}.json"
                    ev = {"sleep": L, "provenance": {**prov, "prep_sha256": psha, "sleep": L, "mode": mode,
                          "ckpt_sha256": None if arm == "N" else train["arms"][arm][str(L)]["ckpt_sha256"]},
                          "env": {"tf32": False}, "ppl_ratio": {"eval": 1.0, "modern": 1.0}, "rows": rows,
                          "measures": {"POS_correct": -999}}  # ignored, never used as an oracle
                    write(sub / name, ev)
    return tmp_path, schema


def run(artifacts):
    root, schema = artifacts
    return S.analyze(root, test_mode=True, model_manifest="", schema=schema)


def test_hand_counted_outcomes_and_unequal_seed_denominators():
    table = S.outcome_table([{"graded": k} for k in ("correct", "correct", "substantive", "unknown", "empty")])
    assert table["rates"]["correct"] == {"numerator": 2, "denominator": 5, "value": .4}
    assert table["answer_rate"]["value"] == .6
    assert table["selective_accuracy"]["value"] == 2 / 3
    pooled = S.pool_fractions({"12": S.fraction(1, 1), "13": S.fraction(0, 9)})
    assert pooled["value"] == .5
    assert pooled["descriptive_micro"]["value"] == .1
    assert S.pool_fractions({"12": S.fraction(1, 1), "13": S.fraction(0, 0)})["value"] is None


@pytest.mark.parametrize("article", ["a", "an", "the", "A", "AN", "THE"])
def test_inherited_leading_article_rule(article):
    assert S.classify(f"  {article} draembrernhound.  ", "draembrernhound") == "correct"
    assert S.classify(f"{article} UNKNOWN", None) == "unknown"
    assert S.classify(f"{article} UNKNOWN: draembrernhound", None) == "substantive"


def test_article_rule_is_word_bounded_not_substring_removal():
    for word in ("another", "anagram", "theme", "atheist"):
        assert S.normalized(word) == word
    assert S.classify("draembrernhound", "a draembrernhound") == "correct"
    assert S.normalized("one a thing") == "oneathing"


def test_complete_success_and_saved_measures_ignored(artifacts):
    r = run(artifacts)
    assert r["errors"] == []
    assert r["verdict"] == r["criteria"]["verdict"] == "ABST0_FUNCTIONAL_AND_GROUNDED"
    assert all(r["criteria"][g] for g in ("D1", "D2", "D3", "D4"))
    assert r["mechanism"]["contrasts"]["5"]["G_minus_U"] == 1
    assert r["per_seed"]["12"]["U"]["5"]["metrics"]["POS_correct"]["value"] == 1


def test_common_forged_gold_not_rescued_by_twin_agreement(artifacts):
    root, _ = artifacts
    for mode in ("merged", "unmerged"):
        def forge(d):
            row = next(r for r in d["rows"] if r["cat"] == "POS")
            row.update(value="Forged", answer="Forged", outcome="correct")
        change(root / f"seed12/eval_U_s4_{mode}.json", forge)
    r = run(artifacts)
    assert r["verdict"] == "ABST0_INVALID"
    assert r["integrity"]["12:U:4"]["ok"] is True
    assert r["integrity"]["12:U:4"]["valid"] is False
    assert any("value" in e for e in r["integrity"]["12:U:4"]["validation_errors"])


@pytest.mark.parametrize("arm,verdict", [("U", "ABST0_INVALID"), ("P", "ABST0_INVALID"),
                                         ("G", "ABST0_FUNCTIONAL_MECHANISM_NOT_ASSESSABLE")])
def test_missing_twin_has_arm_specific_verdict(artifacts, arm, verdict):
    root, _ = artifacts
    (root / f"seed12/eval_{arm}_s3_unmerged.json").unlink()
    r = run(artifacts)
    assert r["verdict"] == verdict
    assert arm in r["invalid_arms"]


@pytest.mark.parametrize("attack", ["answer", "outcome", "checkpoint", "nonfinite", "missing_probe", "wording"])
def test_integrity_and_inventory_attacks(artifacts, attack):
    root, _ = artifacts
    def edit(d):
        if attack == "answer": d["rows"][0]["answer"] = "Corrupt"
        if attack == "outcome": d["rows"][0]["outcome"] = "empty"
        if attack == "checkpoint": d["provenance"]["ckpt_sha256"] = "0" * 64
        if attack == "nonfinite": d["ppl_ratio"]["modern"] = float("nan")
        if attack == "missing_probe": d["rows"].pop()
        if attack == "wording": d["rows"][0]["item"] += " altered"
    change(root / "seed12/eval_U_s3_unmerged.json", edit)
    assert run(artifacts)["verdict"] == "ABST0_INVALID"


def test_shared_missing_probe_fails_even_when_twins_agree(artifacts):
    root, _ = artifacts
    for mode in ("merged", "unmerged"):
        change(root / f"seed12/eval_U_s2_{mode}.json", lambda d: d["rows"].pop())
    assert run(artifacts)["verdict"] == "ABST0_INVALID"


def test_empty_is_not_rejection_and_competence_limit(artifacts):
    root, _ = artifacts
    for mode in ("merged", "unmerged"):
        def edit(d):
            for row in d["rows"]:
                if row["cat"] == "FRESH": row.update(answer="", outcome="empty")
            d["ppl_ratio"]["modern"] = 1.051
        change(root / f"seed12/eval_U_s5_{mode}.json", edit)
    r = run(artifacts)
    assert not r["criteria"]["D2"] and not r["criteria"]["D4"]
    assert r["verdict"] == "ABST0_NO_ADVANCE"


def test_exposure_tamper_and_production_rejects_test_mode(artifacts):
    root, schema = artifacts
    assert S.analyze(root, schema=schema)["verdict"] == "ABST0_INVALID"
    change(root / "seed12/exposure_seed12.json", lambda d: d["cells"]["Alice|allergy"].update(exposed=False))
    r = run(artifacts)
    assert r["verdict"] == "ABST0_INVALID" and any("exposure" in e for e in r["errors"])


def test_missing_monitor_and_ppl_tolerance(artifacts):
    root, _ = artifacts
    change(root / "seed12/eval_U_s1_unmerged.json", lambda d: d["ppl_ratio"].update(eval=1.00011))
    assert run(artifacts)["verdict"] == "ABST0_INVALID"


def test_mechanism_requires_three_improving_seeds_not_just_pooled_margin(artifacts):
    root, _ = artifacts
    for seed in (14, 15):
        for mode in ("merged", "unmerged"):
            def edit(d):
                for row in d["rows"]:
                    if row["cat"] == "HNM": row.update(answer="UNKNOWN", outcome="unknown")
            change(root / f"seed{seed}/eval_G_s5_{mode}.json", edit)
    r = run(artifacts)
    assert r["mechanism"]["contrasts"]["5"]["G_minus_U"] == .5
    assert r["mechanism"]["contrasts"]["5"]["n_seeds_U_lower"] == 2
    assert r["verdict"] == "ABST0_FUNCTIONAL"


def test_transition_H_cannot_hide_behind_successful_A(artifacts):
    root, _ = artifacts
    for seed in (12, 13, 14, 15):
        for mode in ("merged", "unmerged"):
            def edit(d):
                for row in d["rows"]:
                    if row["cat"] == "TPOST" and row["key"].startswith("Henry|"):
                        row.update(answer="Wrong", outcome="substantive")
            change(root / f"seed{seed}/eval_U_s5_{mode}.json", edit)
    r = run(artifacts)
    assert r["pooled"]["U"]["5"]["metrics"]["TPOST_correct"]["value"] == .5
    assert not r["criteria"]["D3"] and r["verdict"] == "ABST0_NO_ADVANCE"


def test_competence_seed_mean_gate_even_when_each_seed_within_max(artifacts):
    root, _ = artifacts
    for seed in (12, 13, 14, 15):
        for mode in ("merged", "unmerged"):
            change(root / f"seed{seed}/eval_U_s3_{mode}.json", lambda d: d["ppl_ratio"].update(eval=1.03))
    r = run(artifacts)
    assert not r["invalid_arms"]
    assert not r["criteria"]["D4"] and r["verdict"] == "ABST0_NO_ADVANCE"


def test_missing_monitor_is_invalid_not_a_zero_ratio(artifacts):
    root, _ = artifacts
    change(root / "seed12/eval_U_s1_merged.json", lambda d: d["ppl_ratio"].pop("modern"))
    assert run(artifacts)["verdict"] == "ABST0_INVALID"


def test_strict_json_rejects_duplicate_keys(tmp_path):
    (tmp_path / "bad.json").write_text('{"answer":"Paris","answer":"Rome"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        S.Reader(tmp_path).read("bad.json")


def test_trained_checkpoint_must_have_a_real_hash(artifacts):
    root, _ = artifacts
    change(root / "seed12/train_seed12.json", lambda d: d["arms"]["U"]["1"].update(ckpt_sha256=None))
    assert run(artifacts)["verdict"] == "ABST0_INVALID"


def test_comparison_reports_disagreement_without_changing_independent_result(artifacts):
    result = run(artifacts)
    saved = copy.deepcopy(result)
    reference = {"verdict": result["verdict"], "invalid_arms": [],
                 "criteria": {k: result["criteria"][k] for k in ("D1", "D2", "D3", "D4", "functional")},
                 "pooled": {"U": {"5": {"POS_correct": 1.0}}}}
    reference["criteria"]["mechanism"] = True
    assert S.compare_report(result, reference)["match"]
    reference["pooled"]["U"]["5"]["POS_correct"] = .4
    diff = S.compare_report(result, reference)
    assert not diff["match"] and diff["differences"][0]["path"] == "pooled.U.5.POS_correct"
    assert result == saved


def test_no_experiment_imports():
    import ast
    tree = ast.parse(Path(S.__file__).read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(x and ("abst0" in x or x.startswith("being") or "cons0a" in x) for x in imports)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("eval", "exec", "__import__") for n in ast.walk(tree))
