import json
from pathlib import Path
import pytest
from continuous_brain_unified.legacy.g0_runner import load_recipes, load_seed_plan, load_dependency_closure, _static_evaluator_preflight
from continuous_brain_unified.prereg import PreregistrationRegistry
from continuous_brain_eval.seeds import SeedRegistry
from continuous_brain_protocols import SeedPurpose

EXPECTED={
 'g0-v023-dev':288,'g0-v0252-dev':256,'g0-v031-dev':1024,
 'g0-v0322-dev':256,'g0-v0341-dev':128,'g0-v0353-dev':128,'g0-v0361-dev':128,
}

def test_first_wave_recipes_and_seed_counts():
    recipes,blocked=load_recipes(); seeds=load_seed_plan()
    assert set(recipes)=={'episodic_two_item_v023','consequence_replanning_v0252','recurrent_self_trajectory_v031','autobiographical_discrepancy_v0322','event_consistent_self_development_v0341','endogenous_goal_genesis_v0353','recursive_autobiographical_curriculum_v0361'}
    assert {n:s['stop']-s['start']+1 for n,s in seeds.items()}==EXPECTED
    assert 'persistent_self_v028' in blocked and 'episodic_goal_skill_v024' in blocked

def test_g0_prereg_is_approved_by_david_and_unlocks_g0():
    r=PreregistrationRegistry.load(); p=r.entries['G0-LEGACY-REQUAL-v1']
    assert p.status=='APPROVED' and p.approved_by=='David' and p.unlocks
    assert r.require('G0-LEGACY-REQUAL-v1',item='G0').id=='G0-LEGACY-REQUAL-v1'

def test_registered_g0_seed_ranges_are_exact_disjoint_and_v023_is_spent():
    seeds=load_seed_plan(); reg=SeedRegistry.load()
    fs=[]
    for name,s in seeds.items():
        f=reg.get(name); fs.append(f)
        assert f.purpose==SeedPurpose.DEV and not f.sealed
        assert f.spent is (name=='g0-v023-dev')
        assert (f.start,f.stop)==(s['start'],s['stop'])
    for i,a in enumerate(fs):
        for b in fs[i+1:]: assert not a.overlaps(b)

def test_static_preflight_requires_seed_main_and_state_dir(tmp_path:Path):
    good=tmp_path/'good.py'; good.write_text("TEST_IDS=range(1)\ndef main():\n    x='--state-dir'\n")
    assert _static_evaluator_preflight(good,'TEST_IDS')==[]
    imported=tmp_path/'imported.py'; imported.write_text("from other import TEST_IDS\ndef main():\n    x='--state-dir'\n")
    assert _static_evaluator_preflight(imported,'TEST_IDS')==[]
    bad=tmp_path/'bad.py'; bad.write_text('def nope(): pass\n')
    probs=_static_evaluator_preflight(bad,'TEST_IDS')
    assert len(probs)==3

def test_g0_seed_plan_status_records_partial_spend():
    raw=json.loads(Path('legacy/g0_seed_plan.json').read_text())
    assert raw['status']=='APPROVED_REGISTERED_PARTIALLY_SPENT'
    rows={x['name']:x for x in raw['families']}
    assert rows['g0-v023-dev']['spent'] is True
    assert all(rows[n]['spent'] is False for n in rows if n!='g0-v023-dev')

def test_dependency_closure_exists_for_every_first_wave_recipe():
    recipes,_=load_recipes(); closures=load_dependency_closure()
    assert set(recipes) <= set(closures)
    for mid in recipes:
        rels={x['relpath'] for x in closures[mid]}
        assert recipes[mid].evaluator in rels
        assert 'continuous_brain/__init__.py' in rels


def test_run_requires_confirmation_and_spends_before_launch(tmp_path, monkeypatch):
    import shutil
    import continuous_brain_unified.legacy.g0_runner as gr
    run=gr.run
    monkeypatch.setattr(gr,'_verify_dependency_closure',lambda legacy,mechanism_id:(True,[]))
    monkeypatch.setattr(gr,'verify_historical_core_archive',lambda archive:(True,[]))
    monkeypatch.setattr(gr,'_extract_historical_core',lambda archive,dest:{})
    monkeypatch.setattr(gr,'_require_runtime_preflight',lambda *a,**k:{})
    from continuous_brain_unified.prereg import PreregistrationError
    from continuous_brain_eval.seeds import SeedOverlapError
    from continuous_brain_protocols import sha256_file

    legacy=tmp_path/'legacy'; legacy.mkdir()
    evaluator=legacy/'evaluate_consequence_replanning_test_v1.py'
    evaluator.write_text("""TEST_ROUTE_IDS=range(1)

def main():
    import json, pathlib, sys
    flag='--state-dir'
    out=pathlib.Path(sys.argv[sys.argv.index(flag)+1]); out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps({'test_admission_passed': True}))
""")
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({
      'format':'cb-g0-plan/1','source_bundle_sha256':'x','source_inventory_sha256':'y','source_repo_commit':'z',
      'protected_checkpoint':{},'counts':{},'rules':[],
      'entries':[{'mechanism_id':'consequence_replanning_v0252','slot_options':['planner'],'lineage_version':'v0.25.2',
        'dossier_status':'ADMITTED/CLOSED PASS','disposition':'READY_FOR_G0','reason':'test',
        'files':[{'root':'wsl-canonical','role':'evaluator','relpath':evaluator.name,'sha256':sha256_file(evaluator)}]}]
    }))
    regdir=tmp_path/'registry'; regdir.mkdir()
    root=Path(__file__).resolve().parents[1]/'registry'
    shutil.copy(root/'preregistrations.json',regdir/'preregistrations.json')
    shutil.copy(root/'seed_ranges.json',regdir/'seed_ranges.json')

    with pytest.raises(PreregistrationError,match='confirm-open'):
        run('consequence_replanning_v0252',legacy,tmp_path/'out-no-confirm',registry_dir=regdir,plan_path=plan,historical_core_archive='dummy')
    assert not SeedRegistry.load(regdir).get('g0-v0252-dev').spent

    report=run('consequence_replanning_v0252',legacy,tmp_path/'out',registry_dir=regdir,plan_path=plan,
               confirm_open='G0-LEGACY-REQUAL-v1',historical_core_archive='dummy')
    assert report['status']=='PASS' and report['fresh_seed_family_spent_before_launch'] is True
    assert report['fresh_seed_family']['spent'] is True
    assert SeedRegistry.load(regdir).get('g0-v0252-dev').spent
    with pytest.raises(SeedOverlapError,match='spent'):
        run('consequence_replanning_v0252',legacy,tmp_path/'out-again',registry_dir=regdir,plan_path=plan,
            confirm_open='G0-LEGACY-REQUAL-v1',historical_core_archive='dummy')


def test_historical_core_manifest_pins_growth_depth_source():
    from continuous_brain_unified.legacy.g0_runner import load_historical_core
    spec=load_historical_core(); rows={x['relpath']:x['sha256'] for x in spec['files']}
    assert rows['continuous_brain/growth_v110.py']=='aa349757f5ed2b4eb998e9cc9c012f746f5b6f411e716710a491f642259465db'
    assert rows['continuous_brain/infant.py']=='592f5a29e3c7b6b374f9637f0fb4db47d3fd3f484d36b24352249b60876db3fe'


def test_preflight_engineering_ranges_are_disjoint_and_inside_reserve():
    from continuous_brain_unified.legacy.g0_runner import load_preflight_seed_plan
    rows=list(load_preflight_seed_plan().values()); reg=SeedRegistry.load(); reserve=reg.get('v0.38-ENGINEERING')
    for r in rows:
        f=reg.get(r['name'])
        assert f.purpose==SeedPurpose.ENGINEERING and not f.spent
        assert reserve.start <= f.start <= f.stop <= reserve.stop
    for i,a in enumerate(rows):
        for b in rows[i+1:]: assert a['stop'] < b['start'] or b['stop'] < a['start']
