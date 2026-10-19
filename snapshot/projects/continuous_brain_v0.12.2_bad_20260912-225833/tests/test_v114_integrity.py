import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import tarfile
import numpy as np
import pytest
import torch
from continuous_brain.evidence_v114 import (BUILD_ID, EXACT_SOURCE_SHA, STAGES, TARGETS, GATES,
    P4B_PROTOCOL, sha256, write_json, extract_outer_final_json, valid_p4b, valid_training, safe_extract)
from continuous_brain.infant import InfantBrain, InfantConfig
from continuous_brain.parented_train import save_checkpoint, load_checkpoint
from continuous_brain.whole_brain_v110 import expand_capacity
from continuous_brain.protected_growth_v114 import (protect_growth, train_protected_growth,
    _training_dataset, shadow_forward, _rank_anchor, _teacher_option_rows)
from test_v102 import _tiny_option_nursery


def p4(sha):
    return dict(checkpoint_sha256=sha, p4b_passed=True, experiment=dict(P4B_PROTOCOL),
                hierarchical_option_planning=dict(completion_rate=.38, mean_collisions=0),
                reactive=dict(completion_rate=0, mean_collisions=0))


def test_parser_keeps_outer_dictionary_and_skips_nested():
    assert extract_outer_final_json('progress {}\n{"passed":true,"metrics":{"x":1}}\nwarning') == {"passed":True,"metrics":{"x":1}}
    assert extract_outer_final_json('log [ {"nested":1} ]\n{"done":true}') == {"done":True}
    with pytest.raises(ValueError): extract_outer_final_json('nothing')


def test_p4b_rejects_wrong_hash_protocol_and_false_pass():
    report = p4('a'*64)
    assert valid_p4b(report, 'a'*64)
    assert not valid_p4b(report, 'b'*64)
    report['experiment']['seed_start'] = 51001
    assert not valid_p4b(report, 'a'*64)
    report = p4('a'*64);report['hierarchical_option_planning']['completion_rate'] = .13
    assert not valid_p4b(report, 'a'*64)


def test_float32_moderate_gate_and_exact_update_ceiling():
    tr = dict(build_id=BUILD_ID, stage='moderate', stage_updates_after=1000,
        legacy_tensors_changed=0, legacy_sha256='a'*64, source_legacy_sha256='a'*64,
        plasticity='growth-only-protected', checkpoint_sha256='b'*64,
        gate_report={'planning.growth':float(np.float32(.1)), 'options.growth':float(np.float32(.1)),
                     'target_vision.growth':0., 'target_spatial.growth':0.})
    assert valid_training(tr, 'moderate')
    tr['stage_updates_after']=1001
    assert not valid_training(tr, 'moderate')


def test_reject_archive_escape(tmp_path):
    archive=tmp_path/'bad.tar'
    with tarfile.open(archive,'w') as t:
        entry=tarfile.TarInfo('../outside.txt');entry.size=1;t.addfile(entry,io.BytesIO(b'x'))
    with pytest.raises(ValueError):safe_extract(archive,tmp_path/'out')
    assert not (tmp_path/'outside.txt').exists()


def test_training_refuses_sealed_or_evaluation_data(tmp_path):
    ds = _tiny_option_nursery(tmp_path/'data',episodes=3)
    ds.write_manifest({'training_prohibited':True})
    with pytest.raises(RuntimeError):_training_dataset(ds.directory)
    ds.write_manifest({'seed_start':51001})
    with pytest.raises(RuntimeError):_training_dataset(ds.directory)


def test_rank_anchor_penalizes_reversed_teacher_order():
    ref=torch.tensor([[.6,.3,.05]])
    assert _rank_anchor(ref,ref).item()==0
    assert _rank_anchor(ref.flip(-1),ref).item()>0


def test_readiness_fails_on_missing_or_stale_evidence():
    script=Path(__file__).parents[1]/'scripts/check_v114_protected_readiness.py'
    spec=importlib.util.spec_from_file_location('v114_check',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    migration=dict(build_id=BUILD_ID,source_checkpoint_sha256=EXACT_SOURCE_SHA,source_legacy_sha256='a'*64,
        exact_function_preservation=True,max_abs_difference=0.,source_tensor_mismatches=0,learned_parameters_added=0)
    ts,ps={},{};previous=EXACT_SOURCE_SHA
    for i,s in enumerate(STAGES):
        sha=str(i+1)*64
        ts[s]=dict(build_id=BUILD_ID,stage=s,stage_updates_after=TARGETS[s],legacy_tensors_changed=0,
            legacy_sha256='a'*64,source_legacy_sha256='a'*64,plasticity='growth-only-protected',checkpoint_sha256=sha,
            source_checkpoint_sha256=EXACT_SOURCE_SHA,stage_admission={'checkpoint_sha256':previous},
            gate_report={'planning.growth':float(np.float32(GATES[s])),'options.growth':float(np.float32(GATES[s])),
                         'target_vision.growth':0.,'target_spatial.growth':0.})
        ps[s]=p4(sha);ps[s]['_v114_evidence']={'runtime_id':'f'*64};previous=sha
    evidence=dict(checkpoint_sha256=previous,build_id=BUILD_ID,runtime_id='f'*64)
    bootstrap=dict(passed=True,_v114_evidence=evidence)
    world=dict(world_consolidation_passed=True,domains={d:{'world_model_passed':True} for d in ('foraging','detour')},_v114_evidence=evidence)
    gen=dict(kind='randomized_detour',seed_start=41001,episodes=100,completion_rate=.2,training_performed_during_evaluation=False,_v114_evidence=evidence)
    assert module.check(migration,ts,ps,bootstrap,world,gen)['protected_growth_passed']
    stale=copy.deepcopy(ps);stale['low']['checkpoint_sha256']='e'*64
    assert not module.check(migration,ts,stale,bootstrap,world,gen)['protected_growth_passed']
    del gen['seed_start']
    assert not module.check(migration,ts,ps,bootstrap,world,gen)['protected_growth_passed']


def test_full_route_training_resume_and_strict_state(tmp_path):
    torch.manual_seed(77)
    ds = _tiny_option_nursery(tmp_path/'training', episodes=4,steps=12)
    rows=ds.records()
    for row in rows:
        path=ds.parent_vision/(row['record_id']+'.npy')
        np.save(path,np.ones(8,dtype=np.float32))
        row['parent_vision']=str(path.relative_to(ds.directory));row['qas']=[{'question':'Here?','answer':'Here.'}]
    ds.rewrite(rows);ds.write_manifest({'vision_parent_id':'vision.test'})
    base=tmp_path/'base';base.mkdir()
    model=InfantBrain(InfantConfig.for_profile('proof'),vision_parent_dims={'vision.test':8})
    save_checkpoint(base/'infant_training.pt',model,updates=7,parent_id='vision.test',dataset_digest='test')
    del model
    exact=tmp_path/'exact';expand_capacity(base,exact)
    branch=tmp_path/'branch';migration=protect_growth(exact,branch)
    with pytest.raises(FileExistsError):protect_growth(exact,branch)
    uninterrupted=tmp_path/'uninterrupted';shutil.copytree(branch,uninterrupted)
    proof=write_json(tmp_path/'baseline.json',p4(migration['source_checkpoint_sha256']))
    kwargs=dict(rehearsal_datasets=[ds.directory], planning_dataset=ds.directory,
        planning_rehearsal_dataset=ds.directory,stage='shadow',batch_size=1,checkpoint_every=5,
        previous_p4b_result=proof,seed=77,device='cpu')
    with pytest.raises(RuntimeError):
        train_protected_growth(state_dir=branch,**{**kwargs,'previous_p4b_result':None},target_stage_updates=5)
    first=train_protected_growth(state_dir=branch,**kwargs,target_stage_updates=5,durable_state_dir=tmp_path/'durable')
    result=train_protected_growth(state_dir=branch,**kwargs,target_stage_updates=10,durable_state_dir=tmp_path/'durable')
    reference=train_protected_growth(state_dir=uninterrupted,**kwargs,target_stage_updates=10)
    a=torch.load(branch/'infant_training.pt',weights_only=False,map_location='cpu')
    b=torch.load(uninterrupted/'infant_training.pt',weights_only=False,map_location='cpu')
    assert all(torch.equal(t,b['model'][k]) for k,t in a['model'].items())
    assert result['legacy_tensors_changed']==0 and result['gate_report']['planning.growth']==0.
    assert result['route_counts']=={'perception':2,'language':2,'world':2,'planning':2,'options':1,'embodiment':1}
    digest=sha256(branch/'infant_training.pt')
    rerun=train_protected_growth(state_dir=branch,**kwargs,target_stage_updates=10)
    assert sha256(branch/'infant_training.pt')==digest==rerun['checkpoint_sha256']
    assert sha256(tmp_path/'durable/infant_training.pt')==digest
    with pytest.raises(ValueError):train_protected_growth(state_dir=branch,**kwargs,target_stage_updates=2001)
    with pytest.raises(RuntimeError):train_protected_growth(state_dir=branch,**{**kwargs,'stage':'micro'})
    # Normal inference remains identical at protected planner/option interfaces.
    before,_=load_checkpoint(exact/'infant_training.pt');after,_=load_checkpoint(branch/'infant_training.pt')
    sp=torch.randn(2,after.cfg.bus_dim);ctx=torch.randn(2,after.cfg.dynamics_context_dim)
    with torch.no_grad():
        z0=before.planning.encode_state(sp,ctx);z1=after.planning.encode_state(sp,ctx)
        assert torch.equal(z0,z1)
        assert torch.equal(before.options.selector_logits(z0),after.options.selector_logits(z1))
    # A missing protected head must fail, rather than be randomly initialized.
    del a['model']['planning.growth.gate'];torch.save(a,tmp_path/'broken.pt')
    with pytest.raises(RuntimeError):load_checkpoint(tmp_path/'broken.pt')


def test_colab_cache_binds_checkpoint_and_runtime_and_never_mutates_stage(tmp_path):
    spec=importlib.util.spec_from_file_location('colab_test',Path(__file__).parents[1]/'scripts/run_v114_colab.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    run=mod.ColabRun.__new__(mod.ColabRun)
    run.results=tmp_path/'results';run.durable_results=tmp_path/'drive';run.work=tmp_path/'work'
    run.state=tmp_path/'state';run.durable_state=tmp_path/'drive_state';run.runtime_id='a'*64
    for p in (run.results,run.durable_results,run.state,run.durable_state):p.mkdir()
    checkpoint=run.state/'infant_training.pt';checkpoint.write_bytes(b'current')
    calls=[]
    def cli(command,args,log_name):
        calls.append(command)
        out=Path(args[args.index('--output')+1]);write_json(out,{'passed':True})
        log=run.results/log_name;log.write_text('done')
        return 0,log
    run.cli=cli
    a=run.evaluate('eval.json','evaluate',[],run.state)
    assert run.evaluate('eval.json','evaluate',[],run.state)==a and len(calls)==1
    checkpoint.write_bytes(b'new checkpoint')
    b=run.evaluate('eval.json','evaluate',[],run.state)
    assert b['_v114_evidence']['checkpoint_sha256']!=a['_v114_evidence']['checkpoint_sha256'] and len(calls)==2
    run.runtime_id='b'*64;run.evaluate('eval.json','evaluate',[],run.state)
    assert len(calls)==3
    frozen=run.state/'infant_training.at_V114_SHADOW.pt';frozen.write_bytes(b'shadow')
    endpoint=run.endpoint_state('shadow')
    assert (endpoint/'infant_training.pt').read_bytes()==b'shadow'
    assert checkpoint.read_bytes()==b'new checkpoint'
    with pytest.raises(RuntimeError):run.require_stage_evidence()
