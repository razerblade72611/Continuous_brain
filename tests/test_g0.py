import json
from pathlib import Path

from continuous_brain_unified.legacy.g0 import G0Disposition, G0Plan


def test_committed_g0_plan_preserves_failures_and_has_expected_classes():
    plan = G0Plan.load(Path(__file__).resolve().parents[1] / "legacy" / "g0_plan.json")
    assert plan.source_repo_commit.startswith("c011d40")
    assert plan.get("episodic_two_item_v023").disposition == G0Disposition.READY_FOR_G0
    assert plan.get("self_broadcast_v029").disposition == G0Disposition.CONTROL_ONLY
    assert plan.get("memory_guided_goal_options_v021").disposition == G0Disposition.SUPPORT_ONLY
    assert plan.get("working_memory_components_v013_v019").disposition == G0Disposition.NEEDS_ADJUDICATION
    assert plan.get("endogenous_goal_genesis_v0353").disposition == G0Disposition.NEEDS_REFRESHED_RESULT_PARSE
    assert not plan.get("self_broadcast_v029").can_prepare_adapter


def test_g0_verifier_is_root_label_aware_and_hash_strict(tmp_path):
    plan_path = tmp_path / "plan.json"
    f = tmp_path / "root" / "x.bin"
    f.parent.mkdir(); f.write_bytes(b"abc")
    import hashlib
    h = hashlib.sha256(b"abc").hexdigest()
    plan_path.write_text(json.dumps({
        "format":"cb-g0-plan/1", "source_bundle_sha256":"0"*64, "source_inventory_sha256":"1"*64,
        "source_repo_commit":"c011d40", "protected_checkpoint":{}, "counts":{}, "rules":[],
        "entries":[{"mechanism_id":"m","slot_options":["planner"],"lineage_version":"v",
                    "dossier_status":"PASS","disposition":"READY_FOR_G0","reason":"r",
                    "files":[{"root":"legacy-main","role":"module","relpath":"x.bin","sha256":h}]}]
    }))
    plan=G0Plan.load(plan_path)
    ok=plan.verify({"legacy-main":f.parent})[0]
    assert ok.ok and ok.files_checked == 1
    f.write_bytes(b"tampered")
    bad=plan.verify({"legacy-main":f.parent})[0]
    assert not bad.ok and bad.mismatches
