"""Full-inventory synthetic file controls for peer analyzer admission.
Historical prep bytes supply schema fixtures; synthesized answers are not results.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import prov1_analyze as peer
from prov1_sol_spec import *
from test_prov1_sol import make_perfect


@pytest.fixture
def files(tmp_path):
    with tarfile.open(ROOT/'results/prov1/phase0/preps.tar.gz') as archive:
        p=json.loads(archive.extractfile('prov1_prep_seed20.json').read())
    anchor=json.loads((ROOT/'results/support1/phase0/prep_seed20.json').read_text())
    for obj in (p,anchor):obj['test_mode']=True;obj['model_manifest']=''
    req=make_perfect(p,anchor);sd=tmp_path/'b/seed20';ref=tmp_path/'s1/seed20'
    sd.mkdir(parents=True);ref.mkdir(parents=True)
    save=lambda path,obj:path.write_bytes(canon(obj))
    save(sd/'prov1_prep_seed20.json',p);save(sd/'prov1_train_seed20.json',req['logs'][20]);save(ref/'prep_seed20.json',anchor)
    h='a'*64
    source_log={'schema':'SUPPORT1_SCHEMA_V1','experiment':'SUPPORT1','seed':20,
                'provenance':dict(code_hash=S1_HASH,prep_sha256=sha(canon(anchor)),model_manifest='',test_mode=True,
                                  gpu='cpu',arms=['P'],sleeps=5,fork=False),
                'arms':{'P':{str(L):dict(ckpt_sha256=h,full_state_hash=h) for L in range(1,6)}}}
    save(ref/'train_seed20.json',source_log)
    for L in range(1,6):
        rows=[]
        for q in anchor['probes']['observed'][str(L)]:
            ans=q['gold_current'] if q['gold_current']!='UNKNOWN' else 'Zorvane'
            rows.append(dict(probe_id=q['probe_id'],answer=ans,outcome=classify(ans,q['gold_current'])))
        ev=dict(schema='SUPPORT1_SCHEMA_V1',experiment='SUPPORT1',seed=20,arm='P',sleep=L,mode='merged',history='observed',
                provenance=dict(code_hash=S1_HASH,prep_sha256=sha(canon(anchor)),ckpt_sha256=h,model_manifest='',test_mode=True,qdiag=64),
                ppl_ratio=dict(eval=.9,modern=.9),base_denominator_sha256=h,rows=rows,env=dict(tf32=False,gpu='cpu'))
        save(ref/f'eval_P_s{L}_merged.json',ev)
    for e in req['evaluations']:
        e['provenance'].update({k:e[k] for k in ('arm','sleep','history','mode')})
        arm,L,mode,hist=e['arm'],e['sleep'],e['mode'],e['history']
        name=f'eval_N_s{L}.json' if arm=='N' else f'eval_{arm}_s{L}'+('_withheld' if hist=='withheld' else '')+f'_{mode}.json'
        save(sd/name,e)
    args=argparse.Namespace(phase='B',dir=str(sd.parent),support1_dir=str(ref.parent),seeds='20',test_mode=True,
                           admission=None,out=str(tmp_path/'analysis.json'))
    def run():
        peer.stage_analyze(args);return json.loads(Path(args.out).read_text())
    return sd,ref,p,req,args,run


def test_peer_valid_full_file_control(files):
    *_,run=files;assert run()['verdict']=='PROV1_ADVANCE'


@pytest.mark.parametrize('attack',['seed','model','test_mode'])
def test_peer_rejects_foreign_train_identity(files,attack):
    sd,_,_,_,_,run=files
    assert run()['verdict']=='PROV1_ADVANCE'
    path=sd/'prov1_train_seed20.json';tr=json.loads(path.read_text())
    if attack=='seed':tr['seed']=23
    elif attack=='model':tr['provenance']['model_manifest']='foreign-model'
    else:tr['provenance']['test_mode']=False
    path.write_text(json.dumps(tr))
    assert run()['verdict']=='PROV1_INVALID'


def test_peer_production_eval_rejects_wrong_gpu(files,tmp_path):
    _,_,p,req,_,_=files
    with tarfile.open(ROOT/'results/prov1/phase0/preps.tar.gz') as archive:
        raw=archive.extractfile('prov1_prep_seed20.json').read()
    p=json.loads(raw)
    ev=copy.deepcopy(next(e for e in req['evaluations'] if e['arm']=='RC' and e['sleep']==2 and e['mode']=='merged'))
    ps=sha(raw);ev['provenance'].update(test_mode=False,model_manifest=p['model_manifest'],prep_sha256=ps)
    path=tmp_path/'one.json';path.write_text(json.dumps(ev))
    call=lambda:peer.load_prov1_eval(path,p,ps,'RC',2,'merged','observed',B_CONDITIONS,True,'prov1',lambda x:x=='1'*64,False)
    ev['env']['gpu']='NVIDIA RTX 6000 Ada Generation';path.write_text(json.dumps(ev));assert not call()[1]
    ev['env']['gpu']='FOREIGN GPU';path.write_text(json.dumps(ev));assert call()[1]


def test_common_K5_requires_common_eligible_items():
    meta=[];rc={};ru={}
    for bid,construction,cat in [('s','statement','POS'),('q','question','POS-Q')]:
        for cond in ('FAITHFUL','DEL-Q','DEL-X'):
            meta.append(dict(base_probe_id=bid,condition=cond,cell=['a','city'],construction=construction,
                             cat=cat,item=bid,kind='template' if construction=='statement' else 'qt0',supported=True,n_entries=2))
            rc[cond,bid]='correct' if cond=='DEL-X' or cond=='FAITHFUL' and bid=='q' else 'unknown' if cond=='DEL-Q' else 'substantive'
            ru[cond,bid]='correct' if cond=='DEL-X' or cond=='FAITHFUL' and bid=='s' else 'substantive'
    st={20:{'A':['a'],'H':[]}};metas={20:{('observed',L):meta for L in (2,3,4,5)}}
    cands={20:{'RC':{L:rc for L in (2,3,4,5)},'RU':{L:ru for L in (2,3,4,5)}}}
    result=peer.common_k5([20],st,metas,cands,'RC','RU')
    assert result['A']['K5a_diff'] is None  # no faithful-correct construction is shared


def test_independent_directory_cli(files,tmp_path):
    import subprocess
    sd,ref,_,_,_,_=files;out=tmp_path/'independent.json'
    r=subprocess.run([sys.executable,str(ROOT/'scripts/prov1_sol_analyze.py'),'--phase','B','--dir',str(sd.parent),
                      '--support1-dir',str(ref.parent),'--seeds','20','--test-mode','--out',str(out)],capture_output=True,text=True)
    data=json.loads(out.read_text())
    assert r.returncode==0 and not data['errors'],(r.stdout,r.stderr,data)
    assert data['test_mode'] is True and data['production_ready'] is False


def test_independent_phase_A_directory_cli(files,tmp_path):
    import subprocess
    sd,ref,p,req,_,_=files
    h='1'*64
    source_log=json.loads((ref/'train_seed20.json').read_text())
    source_log['arms']={a:{str(L):dict(ckpt_sha256=h,full_state_hash=h) for L in range(1,6)} for a in ('P','U4')}
    source_log['fork']={'withheld':{'U4':dict(ckpt_sha256=h)}}
    (ref/'train_seed20.json').write_text(json.dumps(source_log))
    for ev in req['evaluations']:
        if ev['arm'] not in ('RP','RC'):continue
        ev=copy.deepcopy(ev);ev['arm']={'RP':'P','RC':'U4'}[ev['arm']]
        ev['provenance'].update(arm=ev['arm'],source='support1',conditions=list(A_CONDITIONS),monitors=False)
        ev['ppl_ratio']=None;ev['base_denominator_sha256']=None;ev['rows']=[]
        for q in p['probes'][ev['history']][str(ev['sleep'])]:
            if q['condition'] not in A_CONDITIONS:continue
            answer=q['gold_compliance'] if q['gold_compliance'] is not None else 'UNKNOWN'
            ev['rows'].append(dict(probe_id=q['probe_id'],answer=answer,outcome_truth=classify(answer,q['gold_truth']),
                                  outcome_compliance=None if q['gold_compliance'] is None else classify(answer,q['gold_compliance'])))
        name=f"eval_{ev['arm']}_s{ev['sleep']}"+('_withheld' if ev['history']=='withheld' else '')+f"_{ev['mode']}.json"
        (sd/name).write_text(json.dumps(ev))
    out=tmp_path/'phase_A.json'
    run=subprocess.run([sys.executable,str(ROOT/'scripts/prov1_sol_analyze.py'),'--phase','A','--dir',str(sd.parent),
                        '--support1-dir',str(ref.parent),'--seeds','20','--test-mode','--out',str(out)],capture_output=True,text=True)
    data=json.loads(out.read_text())
    assert run.returncode==0 and data['verdict']=='PROV1A_ZERO_SHOT_PASS',(run.stdout,run.stderr,data.get('errors'))


def test_independent_K6_joint_seed_gate():
    from prov1_sol_analyze import criterion_checks
    names=('POS','FIRST','POS-A','POS-H','RET','SEL-D','HOW','HNM','FRESH')
    metric={a:{str(L):{c:{out:{'value':1.} for out in ('correct','unknown')} for c in names} for L in range(1,6)} for a in ('RC','P-GATE')}
    per={str(s):{'RC':{str(L):dict(FAITHFUL={'HNM':{'unknown':{'value':1.}}},ppl_ratio={'eval':1.,'modern':1.}) for L in range(1,6)}} for s in range(20,24)}
    k5={'RC':{g:{k:{'value':1.} for k in ('K5a','K5b')} for g in ('A','H')}}
    # Statement passes 20-22; Question passes 21-23: only 21 and 22 pass jointly.
    forks={'RC':{'FORK-H':{'numerator':3},'FORK-H-Q':{'numerator':3},'joint':{'numerator':2}}}
    result=criterion_checks(metric,per,list(range(20,24)),'RC','B',k5,forks)
    assert result['gates']['K6'] is False and all(v for k,v in result['gates'].items() if k!='K6')


def test_independent_calibration_reports_malformed_train(tmp_path):
    from prov1_sol_io import verify_calibration
    (tmp_path/'cal').mkdir();(tmp_path/'cal/prov1_train_seed24.json').write_text('[]')
    result=verify_calibration(tmp_path,{},'a'*64,'b'*64,'c'*64,'cpu')
    assert result['pass_'] is False and result['errors']
