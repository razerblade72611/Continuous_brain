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

def test_registered_g0_seed_ranges_are_exact_disjoint_and_unspent():
    seeds=load_seed_plan(); reg=SeedRegistry.load()
    fs=[]
    for name,s in seeds.items():
        f=reg.get(name); fs.append(f)
        assert f.purpose==SeedPurpose.DEV and not f.sealed and not f.spent
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

def test_g0_seed_plan_status_is_approved_registered_unspent():
    raw=json.loads(Path('legacy/g0_seed_plan.json').read_text())
    assert raw['status']=='APPROVED_REGISTERED_UNSPENT'

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
    from continuous_brain_unified.prereg import PreregistrationError
    from continuous_brain_eval.seeds import SeedOverlapError
    from continuous_brain_protocols import sha256_file

    legacy=tmp_path/'legacy'; legacy.mkdir()
    evaluator=legacy/'evaluate_multi_item_episodic_memory_test_v1.py'
    evaluator.write_text('''TEST_IDS=list(range(1))\n\ndef main():\n    import json, pathlib, sys\n    flag='--state-dir'\n    out=pathlib.Path(sys.argv[sys.argv.index(flag)+1]); out.mkdir(parents=True,exist_ok=True)\n    (out/'result.json').write_text(json.dumps({'test_admission_passed': True}))\n''')
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({
      'format':'cb-g0-plan/1','source_bundle_sha256':'x','source_inventory_sha256':'y','source_repo_commit':'z',
      'protected_checkpoint':{},'counts':{},'rules':[],
      'entries':[{'mechanism_id':'episodic_two_item_v023','slot_options':['episodic_memory'],'lineage_version':'v0.23.1',
        'dossier_status':'ADMITTED/CLOSED PASS','disposition':'READY_FOR_G0','reason':'test',
        'files':[{'root':'wsl-canonical','role':'evaluator','relpath':evaluator.name,'sha256':sha256_file(evaluator)}]}]
    }))
    regdir=tmp_path/'registry'; regdir.mkdir()
    root=Path(__file__).resolve().parents[1]/'registry'
    shutil.copy(root/'preregistrations.json',regdir/'preregistrations.json')
    shutil.copy(root/'seed_ranges.json',regdir/'seed_ranges.json')

    with pytest.raises(PreregistrationError,match='confirm-open'):
        run('episodic_two_item_v023',legacy,tmp_path/'out-no-confirm',registry_dir=regdir,plan_path=plan)
    assert not SeedRegistry.load(regdir).get('g0-v023-dev').spent

    report=run('episodic_two_item_v023',legacy,tmp_path/'out',registry_dir=regdir,plan_path=plan,
               confirm_open='G0-LEGACY-REQUAL-v1')
    assert report['status']=='PASS' and report['fresh_seed_family_spent_before_launch'] is True
    assert SeedRegistry.load(regdir).get('g0-v023-dev').spent
    with pytest.raises(SeedOverlapError,match='spent'):
        run('episodic_two_item_v023',legacy,tmp_path/'out-again',registry_dir=regdir,plan_path=plan,
            confirm_open='G0-LEGACY-REQUAL-v1')
