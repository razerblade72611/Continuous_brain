"""Independent reader tests using producer-generated fixtures only as file inputs.
Oracle-perfect synthetic answers verify analysis plumbing, not model capability.
"""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import prov1_sol_validate as validator
import prov1_sol_analyze as analyzer
from prov1_sol_spec import *


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    # Only fixture construction imports producers; independent modules never do.
    import tiny_hf_model
    import prov1
    import support1
    from transformers import AutoTokenizer
    model=tiny_hf_model.build(tmp_path_factory.mktemp('prov1_sol')/'m')
    tok=AutoTokenizer.from_pretrained(model)
    p=prov1.prep(20,tok,'',True);anchor=support1.prep(20,tok,'',True)
    return p,anchor,tok


def test_independent_prep_baseline(fixture):
    p,a,tok=fixture;r=validator.validate(p,a,tok)
    assert r['verdict']=='TEST_ONLY' and not r['errors']


@pytest.mark.parametrize('attack',['record','truth','DEL-X','condition_omission','ABSENT',
                                   'tokens','loss','history','cf_population','cf_log','budget'])
def test_prep_mutations(fixture,attack):
    original,a,tok=fixture;p=copy.deepcopy(original)
    rows=p['probes']['observed']['5']
    if attack=='record':rows[0]['record']='Memory record: fake - no experiences.'
    elif attack=='truth':rows[0]['gold_truth']='forged'
    elif attack=='DEL-X':next(r for r in rows if r['condition']=='DEL-X')['delx_attr']='fake'
    elif attack=='condition_omission':rows.pop()
    elif attack=='ABSENT':next(r for r in rows if r['condition']=='ABSENT')['item']='forged'
    elif attack=='tokens':p['schedule']['RC']['1'][0]['input_ids'][0]+=1
    elif attack=='loss':p['schedule']['RC']['1'][0]['n_loss']+=1
    elif attack=='history':p['fork_schedule']['RC'][0]['record']='wrong history'
    elif attack=='cf_population':next(r for r in p['schedule']['RC']['2'] if r['kind']=='cf')['cell'][0]=p['stream']['H'][0]
    elif attack=='cf_log':p['counterfactuals']['2'].pop()
    elif attack=='budget':p['budget']['RU']['1'][1]+=1
    assert validator.validate(p,a,tok)['verdict']=='FAIL'


def make_perfect(p,anchor):
    """One-seed test matrix: faithful values, DEL-Q UNKNOWN, DEL-X preserved."""
    seed=p['seed'];psha=sha(canon(p));h='1'*64
    from prov1_sol_analyze import exposure
    def delivered_rows(rows):
        counts=delivered(rows)
        return [dict(kind=k[0],cell=list(k[1]),src_cell=list(k[2]) or None,wording=k[3],label=k[4],count=n) for k,n in counts.items()]
    log=dict(schema=SCHEMA,experiment='PROV1',seed=seed,
             provenance=dict(code_hash=codehash(),prep_sha256=psha,model_manifest='',test_mode=True,gpu='cpu',
                             arms=list(ARMS),sleeps=5,fork=True),arms={},
             fork=dict(admitted=True,parent_full_state_hash={},withheld={}))
    for arm in ARMS:
        log['arms'][arm]={str(L):dict(ckpt_sha256=h,full_state_hash=h,budget=budget(p['schedule'][arm][str(L)]),
                                    delivered=delivered_rows(p['schedule'][arm][str(L)])) for L in range(1,6)}
    for arm in ('RU','RC'):
        log['fork']['parent_full_state_hash'][arm]=h
        log['fork']['withheld'][arm]=dict(ckpt_sha256=h,full_state_hash=h,delivered=delivered_rows(p['fork_schedule'][arm]))
    evals=[]
    for arm in ('N',*ARMS):
        for L in range(1,6):
            for hist in (('observed','withheld') if arm in ('RU','RC') and L==4 else ('observed',)):
                for mode in (('base',) if arm=='N' else ('merged','unmerged')):
                    rows=[]
                    for q in p['probes'][hist][str(L)]:
                        if q['condition'] not in B_CONDITIONS:continue
                        ans=q['gold_compliance'] if q['gold_compliance'] is not None else 'UNKNOWN'
                        rows.append(dict(probe_id=q['probe_id'],answer=ans,
                                         outcome_truth=classify(ans,q['gold_truth']),
                                         outcome_compliance=None if q['gold_compliance'] is None else classify(ans,q['gold_compliance'])))
                    evals.append(dict(schema=SCHEMA,experiment='PROV1',seed=seed,arm=arm,sleep=L,history=hist,mode=mode,
                        provenance=dict(code_hash=codehash(),prep_sha256=psha,ckpt_sha256=None if arm=='N' else h,
                                        model_manifest='',test_mode=True,conditions=list(B_CONDITIONS),monitors=True,source='prov1',
                                        arm=arm,sleep=L,history=hist,mode=mode),
                        ppl_ratio=dict(eval=1.,modern=1.),base_denominator_sha256=h,rows=rows,env=dict(gpu='cpu',tf32=False)))
    baselines={(seed,L):dict(rows=[dict(probe_id=q['probe_id'],answer=q['gold_current']) for q in anchor['probes']['observed'][str(L)]]) for L in range(1,6)}
    return dict(preps={seed:(p,psha)},logs={seed:log},evaluations=evals,anchors={seed:anchor},
                baselines=baselines,phase='B',test=True,manifest='',gpu='cpu')


def test_perfect_matrix(fixture):
    p,a,_=fixture;r=analyzer.analyze(**make_perfect(p,a))
    assert r['verdict']=='PROV1_GATE_ONLY',r.get('errors')
    assert r['production_ready'] is False and r['test_mode'] is True
    assert all(v for k,v in r['criteria']['RC']['gates'].items() if k!='K6')
    assert r['criteria']['RC']['gates']['K6'] is False  # one seed cannot satisfy >=3 seeds


def test_four_seed_perfect_matrix(fixture):
    import prov1,support1
    p,a,tok=fixture;request=make_perfect(p,a)
    for seed in (21,22,23):
        other=make_perfect(prov1.prep(seed,tok,'',True),support1.prep(seed,tok,'',True))
        for key in ('preps','logs','anchors','baselines'):request[key].update(other[key])
        request['evaluations']+=other['evaluations']
    r=analyzer.analyze(**request)
    assert r['verdict']=='PROV1_ADVANCE',r.get('errors')
    assert r['criteria']['RC']['gates']['K6'] is True
    # Three successes per construction can still be only two joint successes.
    for seed,cat in ((20,'FORK-H-Q'),(23,'FORK-H')):
        byid={q['probe_id']:q for q in request['preps'][seed][0]['probes']['withheld']['4']}
        for ev in request['evaluations']:
            if (ev['seed'],ev['arm'],ev['history'])!=(seed,'RC','withheld'):continue
            for row in ev['rows']:
                q=byid[row['probe_id']]
                if q['condition']=='FAITHFUL' and q['cat']==cat:
                    row.update(answer='SyntheticWrong',outcome_truth='substantive',outcome_compliance='substantive')
    r=analyzer.analyze(**request)
    assert not r['errors']
    assert r['fork']['RC']['FORK-H']['numerator']==3
    assert r['fork']['RC']['FORK-H-Q']['numerator']==3
    assert r['fork']['RC']['joint']['numerator']==2
    assert r['criteria']['RC']['gates']['K6'] is False


def test_phase_A_fixture(fixture):
    p,a,_=fixture;request=make_perfect(p,a);seed=p['seed'];h='1'*64
    request['phase']='A'
    request['logs']={seed:dict(arms={arm:{str(L):dict(ckpt_sha256=h) for L in range(1,6)} for arm in ('P','U4')},
                             fork=dict(withheld={'U4':dict(ckpt_sha256=h)}))}
    evaluations=[]
    for ev in request['evaluations']:
        if ev['arm'] not in ('RP','RC'):continue
        ev['arm']={'RP':'P','RC':'U4'}[ev['arm']]
        ev['provenance'].update(source='support1',conditions=list(A_CONDITIONS),monitors=False,arm=ev['arm'])
        ev['ppl_ratio']=None;ev['base_denominator_sha256']=None
        ev['rows']=[]
        for q in p['probes'][ev['history']][str(ev['sleep'])]:
            if q['condition'] not in A_CONDITIONS:continue
            ans=q['gold_compliance'] if q['gold_compliance'] is not None else 'UNKNOWN'
            ev['rows'].append(dict(probe_id=q['probe_id'],answer=ans,outcome_truth=classify(ans,q['gold_truth']),
                                  outcome_compliance=None if q['gold_compliance'] is None else classify(ans,q['gold_compliance'])))
        evaluations.append(ev)
    request['evaluations']=evaluations
    r=analyzer.analyze(**request)
    assert r['verdict']=='PROV1A_ZERO_SHOT_PASS',r.get('errors')
    assert r['criteria']['P']['gates']['K6'] is None


@pytest.mark.parametrize('attack',['forged_outcome','missing_row','mode','arm','history','checkpoint','denominator',
                                  'N_denominator','monitor','bool_monitor','TF32','missing_eval','duplicate_eval','train_exposure','fork_parent'])
def test_analysis_mutations(fixture,attack):
    p,a,_=fixture;request=make_perfect(p,a);ev=next(e for e in request['evaluations'] if e['arm']=='RC' and e['mode']=='merged' and e['sleep']==2)
    if attack=='forged_outcome':ev['rows'][0]['outcome_truth']='forged'
    elif attack=='missing_row':ev['rows'].pop()
    elif attack=='mode':ev['mode']='base'
    elif attack=='arm':ev['arm']='RP'
    elif attack=='history':ev['history']='withheld'
    elif attack=='checkpoint':ev['provenance']['ckpt_sha256']='2'*64
    elif attack=='denominator':ev['base_denominator_sha256']='2'*64
    elif attack=='N_denominator':next(e for e in request['evaluations'] if e['arm']=='N')['base_denominator_sha256']='2'*64
    elif attack=='monitor':ev['ppl_ratio']['eval']=float('nan')
    elif attack=='bool_monitor':ev['ppl_ratio']['eval']=True
    elif attack=='TF32':ev['env']['tf32']=True
    elif attack=='missing_eval':request['evaluations'].pop()
    elif attack=='duplicate_eval':request['evaluations'].append(copy.deepcopy(ev))
    elif attack=='train_exposure':request['logs'][p['seed']]['arms']['RC']['1']['delivered'][0]['count']+=1
    elif attack=='fork_parent':request['logs'][p['seed']]['fork']['parent_full_state_hash']['RC']='2'*64
    assert analyzer.analyze(**request)['verdict']=='PROV1_INVALID'


def test_K5_weights_cells_and_deduplicates_prompts():
    rows=[]
    for idx,(entity,works) in enumerate([('a',True),('b',False)]):
        for wording in range(3 if entity=='a' else 1):
            base=f'{entity}-{wording}'
            for cond,graded in [('FAITHFUL','correct'),('DEL-Q','unknown' if works else 'substantive'),('DEL-X','correct')]:
                rows.append(dict(probe_id=base+'-'+cond,base_probe_id=base,condition=cond,supported=True,n_entries=2,
                                 graded=graded,_sleep=2,cell=[entity,'city'],construction='question',cat='POS-Q',item=f'{entity} {wording}'))
    result,_=analyzer.reliance(rows,dict(A=['a','b'],H=[]))
    assert result['A']['K5a']['value']==.5
    assert result['A']['K5a']['denominator']==2
    assert result['H']['K5a']['value'] is None


def test_K5_update_probes_are_diagnostic_only():
    rows=[]
    for base,cat,attr in [('p','POS','city'),('u','UPD','occupation')]:
        for condition,graded in [('FAITHFUL','correct'),('DEL-Q','unknown' if cat=='POS' else 'substantive'),('DEL-X','correct')]:
            rows.append(dict(probe_id=base+'-'+condition,base_probe_id=base,condition=condition,cat=cat,
                             supported=True,n_entries=2,graded=graded,_sleep=2,cell=['a',attr],construction='statement',item=base))
    st={'A':['a'],'H':[]}
    assert analyzer.reliance(rows,st)[0]['A']['K5a']['value']==1.
    assert analyzer.reliance(rows,st,categories=None)[0]['A']['K5a']['value']==.5


def test_empty_required_H_does_not_pass_K5(fixture):
    original,a,_=fixture;p=copy.deepcopy(original)
    # Remove eligibility by an input-level fixture change; evaluation identity remains consistent.
    # This attacks aggregation, independent of the prep validator (which would also refuse altered prep bytes).
    for d in p['probes']['observed'].values():
        for r in d:
            if r['cell'][0] in p['stream']['H']:r['n_entries']=1
    request=make_perfect(p,a);r=analyzer.analyze(**request)
    assert r['criteria']['RC']['gates']['K5'] is False


def test_no_producer_imports():
    import ast
    banned={'prov1','support1','abst0','cons0a','cls0c_b','torch'}
    for name in ('prov1_sol_spec.py','prov1_sol_validate.py','prov1_sol_analyze.py','prov1_sol_io.py'):
        tree=ast.parse((ROOT/'scripts'/name).read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):assert not banned & {a.name.split('.')[0] for a in n.names}
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in banned
