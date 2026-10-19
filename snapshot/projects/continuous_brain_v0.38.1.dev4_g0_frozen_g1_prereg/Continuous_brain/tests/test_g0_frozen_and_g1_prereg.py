import json
from pathlib import Path

import pytest

from continuous_brain_eval.seeds import SeedRegistry, SeedOverlapError
from continuous_brain_protocols import SeedFamily, SeedPurpose
from continuous_brain_unified.prereg import PreregistrationError, PreregistrationRegistry

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "consequence_replanning_v0252",
    "recurrent_self_trajectory_v031",
    "autobiographical_discrepancy_v0322",
    "event_consistent_self_development_v0341",
    "endogenous_goal_genesis_v0353",
    "recursive_autobiographical_curriculum_v0361",
}


def test_g0_evidence_registry_freezes_exactly_six_passes():
    d=json.loads((ROOT/'registry/g0_requalifications.json').read_text())
    assert d['prereg_id']=='G0-LEGACY-REQUAL-v1'
    assert d['verified_from_bundle']['sha256']=='a191bf3a06160a867c177f6377470879fe6b233a089f768b093250c7aaa277a5'
    assert d['verified_from_bundle']['manifest_entries_verified']==202
    rows={r['mechanism_id']:r for r in d['passes']}
    assert set(rows)==EXPECTED
    for r in rows.values():
        assert r['status']=='PASS'
        assert r['seed_family']['spent'] is True
        assert r['raw_legacy_pass_value'] is True
        assert r['frozen_check_count']==len(r['frozen_check_keys'])>0
        assert len(r['raw_result_sha256'])==64
        assert r['historical_test_reopened'] is False
        assert r['threshold_or_intervention_override_performed'] is False
        assert r['legacy_tree_metadata_unchanged'] is True
        assert r['pinned_evidence_still_matches'] is True
        assert r['dependency_closure_still_matches'] is True


def test_static_seed_registry_freezes_all_completed_g0_and_preflight_families():
    reg=SeedRegistry.load()
    for name in ('g0-v0252-dev','g0-v031-dev','g0-v0322-dev','g0-v0341-dev','g0-v0353-dev','g0-v0361-dev'):
        f=reg.get(name); assert f.spent
        with pytest.raises(SeedOverlapError, match='spent'): reg.check_usable(f)
    for name in ('g0-preflight-v023-eng','g0-preflight-v0252-eng','g0-preflight-v031-eng','g0-preflight-v0322-eng','g0-preflight-v0341-eng','g0-preflight-v0353-eng','g0-preflight-v0361-eng'):
        assert reg.get(name).spent


def test_g1_prereg_is_proposed_fail_closed_and_has_fresh_disjoint_ranges():
    d=json.loads((ROOT/'registry/g1_retention_v1.json').read_text())
    assert d['id']=='G1-COMPOSITION-RETENTION-v1' and d['status']=='PROPOSED'
    rows=d['mechanisms']; assert {r['mechanism_id'] for r in rows}==EXPECTED
    static=SeedRegistry.load()
    fams=[]
    for r in rows:
        assert r['required_check_count']==len(r['required_frozen_checks'])>0
        f=r['proposed_dev_family']
        fam=SeedFamily(f['name'],f['start'],f['stop'],SeedPurpose(f['purpose']))
        static.check_fresh(fam)
        fams.append(fam)
    for i,a in enumerate(fams):
        for b in fams[i+1:]: assert not a.overlaps(b)
    with pytest.raises(PreregistrationError):
        PreregistrationRegistry.load().require('G1-COMPOSITION-RETENTION-v1', item='G1')
