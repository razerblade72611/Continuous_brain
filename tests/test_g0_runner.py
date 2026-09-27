import pytest
from continuous_brain_unified.legacy.g0_runner import load_recipes, load_seed_plan
from continuous_brain_unified.prereg import PreregistrationRegistry, PreregistrationError
from continuous_brain_eval.seeds import SeedRegistry
from continuous_brain_protocols import SeedFamily, SeedPurpose

def test_first_wave_recipes_and_seed_counts():
    recipes,blocked=load_recipes(); seeds=load_seed_plan()
    assert set(recipes)=={'episodic_two_item_v023','consequence_replanning_v0252','recurrent_self_trajectory_v031','autobiographical_discrepancy_v0322','event_consistent_self_development_v0341','endogenous_goal_genesis_v0353','recursive_autobiographical_curriculum_v0361'}
    expected={'g0-v023-dev':288,'g0-v0252-dev':256,'g0-v031-dev':1024,'g0-v0322-dev':256,'g0-v0341-dev':128,'g0-v0353-dev':128,'g0-v0361-dev':128}
    assert {n:s['stop']-s['start']+1 for n,s in seeds.items()}==expected
    assert 'persistent_self_v028' in blocked and 'episodic_goal_skill_v024' in blocked

def test_g0_prereg_is_proposed_and_cannot_unlock():
    r=PreregistrationRegistry.load(); p=r.entries['G0-LEGACY-REQUAL-v1']; assert p.status=='PROPOSED' and not p.unlocks
    with pytest.raises(PreregistrationError): r.require('G0-LEGACY-REQUAL-v1',item='G0')

def test_proposed_g0_seed_ranges_are_pairwise_disjoint():
    fs=list(load_seed_plan().values())
    for i,a in enumerate(fs):
        for b in fs[i+1:]: assert a['stop']<b['start'] or b['stop']<a['start']


def test_proposed_g0_ranges_do_not_overlap_registered_history():
    reg=SeedRegistry.load()
    for f in load_seed_plan().values():
        fam=SeedFamily(f["name"],f["start"],f["stop"],SeedPurpose.DEV)
        reg.check_fresh(fam)
