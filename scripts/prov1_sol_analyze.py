#!/usr/bin/env python3
"""Independent PROV-1 analyzer draft, from protocol and literal JSON formats only.

No producer/peer-analyzer imports. The pure analyze() API supports both phases.
Production use is deliberately blocked until the pending calibration, checkpoint
admission and criterion-construction contracts are frozen. This is not a launch
gate yet. Draft choices are always reported, never silently treated as protocol.
"""
import argparse
import collections
import json
from pathlib import Path
import re
from prov1_sol_spec import *

VERSION='PROV1_SOL_ANALYZER_DRAFT_V1'
DRAFT_CHOICES={
    'K6':'both Statement and Question must each meet the 3/4-seed gate; ratification required',
    'K7':'equal weight to sleeps 2-5 within each seed, then equal weight to seeds; ratification required',
    'K5':'deduplicate identical base items, average eligible variants within construction, constructions within cell-sleep, then cell-sleeps within seed',
}


def hash_ok(s):
    return isinstance(s,str) and re.fullmatch('[0-9a-f]{64}',s) is not None


def expected_checkpoint(log,arm,L,h):
    return (log['arms'][arm][str(L)]['ckpt_sha256'] if h=='observed'
            else log['fork']['withheld'][arm]['ckpt_sha256'])


def validate_train(log,p,psha,code,manifest,test,gpu=None):
    errors=[];prov=log.get('provenance',{})
    expect={'code_hash':code,'prep_sha256':psha,'model_manifest':manifest,'test_mode':test,
            'arms':list(ARMS),'sleeps':5,'fork':True}
    for k,v in expect.items():
        if prov.get(k)!=v:errors.append('train provenance '+k)
    if log.get('seed')!=p['seed'] or log.get('schema')!=SCHEMA or log.get('experiment')!='PROV1':
        errors.append('train identity')
    if gpu is not None and prov.get('gpu')!=gpu:errors.append('train GPU')
    try:
        if set(log['arms'])!=set(ARMS):errors.append('train arm inventory')
        for arm in ARMS:
            if set(log['arms'][arm])!={str(L) for L in range(1,6)}:errors.append('train sleep inventory '+arm)
            for L in range(1,6):
                r=log['arms'][arm][str(L)]
                if not hash_ok(r['ckpt_sha256']) or not hash_ok(r['full_state_hash']):errors.append('train state identity')
                if r['budget']!=budget(p['schedule'][arm][str(L)]):errors.append('train budget')
                if exposure(r['delivered'])!=delivered(p['schedule'][arm][str(L)]):errors.append('train exposure')
        if log['fork'].get('admitted') is not True:errors.append('fork not admitted')
        for arm in ('RU','RC'):
            if log['fork']['parent_full_state_hash'][arm]!=log['arms'][arm]['3']['full_state_hash']:errors.append('fork parent')
            r=log['fork']['withheld'][arm]
            if not hash_ok(r['ckpt_sha256']) or not hash_ok(r['full_state_hash']):errors.append('fork checkpoint identity')
            if exposure(r['delivered'])!=delivered(p['fork_schedule'][arm]):errors.append('fork exposure')
    except (KeyError,TypeError,ValueError) as exc:errors.append('malformed train '+str(exc))
    return errors


def exposure(rows):
    out={}
    for r in rows:
        key=(r['kind'],tuple(r['cell']),tuple(r['src_cell'] or ()),r['wording'],r['label'])
        if key in out or type(r['count']) is not int or r['count']<=0:raise ValueError('invalid exposure counter')
        out[key]=r['count']
    return out


def validate_eval(ev,p,psha,ck,phase,test,code,manifest,gpu=None):
    errors=[];prov=ev.get('provenance',{});arm=ev.get('arm');h=ev.get('history');L=ev.get('sleep')
    conditions=list(A_CONDITIONS if phase=='A' else B_CONDITIONS)
    expected={'code_hash':code,'prep_sha256':psha,'ckpt_sha256':ck,'model_manifest':manifest,
              'test_mode':test,'conditions':conditions,'monitors':phase=='B',
              'source':'support1' if phase=='A' else 'prov1'}
    for k,v in expected.items():
        if prov.get(k)!=v or (k in ('test_mode','monitors') and type(prov.get(k)) is not bool):errors.append('eval provenance '+k)
    if ev.get('schema')!=SCHEMA or ev.get('experiment')!='PROV1' or ev.get('seed')!=p['seed']:
        errors.append('eval identity')
    if ev.get('mode') not in (('base',) if arm=='N' else ('merged','unmerged')):errors.append('eval mode')
    env=ev.get('env',{})
    if env.get('tf32') is not False:errors.append('TF32')
    if gpu is not None and env.get('gpu')!=gpu:errors.append('eval GPU')
    joined=[]
    try:
        expected_rows=[r for r in p['probes'][h][str(L)] if r['condition'] in conditions]
        actual=ev['rows']
        if [r['probe_id'] for r in actual]!=[r['probe_id'] for r in expected_rows]:errors.append('eval exact probe inventory/order')
        byid={r['probe_id']:r for r in expected_rows}
        for r in actual:
            q=byid[r['probe_id']]
            if not isinstance(r['answer'],str):raise ValueError('non-string answer')
            truth=classify(r['answer'],q['gold_truth'])
            compliance=None if q['gold_compliance'] is None else classify(r['answer'],q['gold_compliance'])
            if r.get('outcome_truth')!=truth or r.get('outcome_compliance')!=compliance:errors.append('stored grading inconsistent')
            joined.append(dict(q,answer=r['answer'],graded=truth,graded_compliance=compliance))
        ppl=ev.get('ppl_ratio')
        if phase=='B':
            if not isinstance(ppl,dict) or set(ppl)!={'eval','modern'} or any(not finite(x) or x<=0 for x in ppl.values()):
                errors.append('competence monitors')
            if not hash_ok(ev.get('base_denominator_sha256')):errors.append('base denominator identity')
        elif ppl is not None or ev.get('base_denominator_sha256') is not None:errors.append('unexpected phase-A monitors')
    except (KeyError,TypeError,ValueError) as exc:errors.append('malformed eval '+str(exc))
    return errors,joined


def twin(m,u):
    return (m['rows']==u['rows'] and m.get('base_denominator_sha256')==u.get('base_denominator_sha256')
            and (m.get('ppl_ratio') is None and u.get('ppl_ratio') is None or
                 isinstance(m.get('ppl_ratio'),dict) and isinstance(u.get('ppl_ratio'),dict)
                 and set(m['ppl_ratio'])==set(u['ppl_ratio'])=={'eval','modern'}
                 and all(finite(m['ppl_ratio'][k]) and finite(u['ppl_ratio'][k])
                         and abs(m['ppl_ratio'][k]-u['ppl_ratio'][k])<=1e-4 for k in ('eval','modern'))))


def select(rows,cohort,st):
    if cohort in ('POS-A','POS-H'):
        return [r for r in rows if r['cat']=='POS' and r['kind'].startswith('para') and r['cell'][0] in st[cohort[-1]]]
    if cohort in ('POS','FIRST','RET'):
        return [r for r in rows if r['cat']==cohort and r['kind'].startswith('para')]
    return [r for r in rows if r['cat']==cohort]


def rates(rows):
    return {out:fraction(sum(r['graded']==out for r in rows),len(rows)) for out in ('correct','unknown','empty','substantive')}


def reliance(rows,st,common=None):
    """Conditional paired rates; every unique eligible cell x sleep gets one weight.
    Duplicate POS/FIRST/RET prompts are not additional constructions.
    rows contains one seed's sleeps 2-5; caller adds _sleep.
    """
    groups={};counts={}
    by={r['probe_id']:r for r in rows}
    candidates={}
    for r in rows:
        if r['condition']!='FAITHFUL' or not r['supported'] or r['n_entries']<2 or r['graded']!='correct':continue
        key=(r['_sleep'],tuple(r['cell']),r['construction'],r['item'])
        q=by.get(r['base_probe_id']+'-DEL-Q');x=by.get(r['base_probe_id']+'-DEL-X')
        if q is None or x is None:continue
        if common is not None and key not in common:continue
        candidates.setdefault(key,(r,q,x))
    for group in ('A','H'):
        selected=[(k,v) for k,v in candidates.items() if k[1][0] in st[group]]
        cells=collections.defaultdict(lambda:collections.defaultdict(list))
        per_sleep=collections.Counter()
        for (L,c,construction,_),(f,q,x) in selected:
            cells[L,c][construction].append((int(q['graded']=='unknown'),int(x['graded']=='correct')))
        a=b=0.;components=[]
        for (L,c),cc in sorted(cells.items()):
            per_sleep[L]+=1
            aq=sum(sum(v[0] for v in variants)/len(variants) for variants in cc.values())/len(cc)
            bx=sum(sum(v[1] for v in variants)/len(variants) for variants in cc.values())/len(cc)
            a+=aq;b+=bx;components.append(dict(sleep=L,cell=list(c),DEL_Q_unknown=aq,DEL_X_correct=bx))
        groups[group]=dict(K5a=fraction(a,len(cells)),K5b=fraction(b,len(cells)),
                           eligible_unique_probes=len(selected),eligible_cells_by_sleep=dict(per_sleep),components=components)
    return groups,set(candidates)


def criterion_checks(metric,per_seed,seeds,arm,phase,k5,forks):
    checks={f'K{i}':[] for i in range(1,8)}
    def ck(k,name,x,op,y):
        checks[k].append(dict(requirement=name,actual=x,operator=op,threshold=y,
                              pass_=bool(ge(x,y) if op=='>=' else le(x,y))))
    def val(arm,L,cohort,out):return metric[arm][str(L)][cohort][out]['value']
    for L in range(1,6):
        x=val(arm,L,'POS','correct');p=val('P-GATE',L,'POS','correct')
        ck('K1',f'{L} POS floor',x,'>=',.5)
        ck('K1',f'{L} POS vs P',x,'>=',p-.1 if finite(p) else None)
        for g in ('POS-A','POS-H'):
            p=val('P-GATE',L,g,'correct')
            ck('K1',f'{L} {g}',val(arm,L,g,'correct'),'>=',p-.1 if finite(p) else None)
        if L>=2:
            ck('K1',f'{L} FIRST',val(arm,L,'FIRST','correct'),'>=',.5)
            for g in ('HNM','FRESH'):
                ck('K2',f'{L} {g}',val(arm,L,g,'unknown'),'>=',.9)
            for s in seeds:
                ck('K2',f'{s}/{L} HNM',per_seed[str(s)][arm][str(L)]['FAITHFUL']['HNM']['unknown']['value'],'>=',.8)
            for g in ('SEL-D','HOW'):ck('K3',f'{L} {g}',val(arm,L,g,'unknown'),'>=',.9)
    p=val('P-GATE',5,'RET','correct')
    ck('K4','5 RET',val(arm,5,'RET','correct'),'>=',p-.1 if finite(p) else None)
    if phase=='B':
        for L in range(1,6):
            for m in ('eval','modern'):
                fr={str(s):fraction(per_seed[str(s)][arm][str(L)]['ppl_ratio'][m],1) for s in seeds}
                ck('K4',f'{L} {m} pooled',pool(fr)['value'],'<=',1.02)
                for s in seeds:ck('K4',f'{s}/{L} {m}',fr[str(s)]['value'],'<=',1.05)
    for g in ('A','H'):
        for k,t in [('K5a',.8),('K5b',.9)]:ck('K5',g+' '+k,k5[arm][g][k]['value'],'>=',t)
    if phase=='B' and arm in forks:
        for construction in ('FORK-H','FORK-H-Q'):
            ck('K6',construction,forks[arm][construction]['numerator'],'>=',3)
    for g in ('POS-A','POS-H','RET','SEL-D','HOW','HNM','FRESH'):
        out='correct' if g in ('POS-A','POS-H','RET') else 'unknown'
        us=[val(arm,L,g,out) for L in range(2,6)];ps=[val('P-GATE',L,g,out) for L in range(2,6)]
        u=sum(us)/4 if all(finite(x) for x in us) else None
        p=sum(ps)/4 if all(finite(x) for x in ps) else None
        ck('K7',g,u,'>=',p-.05 if finite(p) else None)
    gates={k:all(x['pass_'] for x in v) for k,v in checks.items()}
    if phase=='A' or arm not in forks:gates['K6']=None
    return dict(gates=gates,pass_=all(v for k,v in gates.items() if phase=='B' or k!='K6'),checks=checks)


def analyze(preps,logs,evaluations,anchors,baselines,phase='B',test=False,code=None,manifest=None,gpu=None):
    """Inputs: seed -> (prep, raw-file-sha); seed -> train; list of eval JSONs;
    historical SUPPORT-1 preps; baseline P evals keyed (seed,L). This API never
    regards TEST_ONLY output as a production approval.
    """
    code=code or codehash();seeds=sorted(preps);errors=[];invalid=set();joined={};evmap={};per_seed={};forks={}
    arms=ARMS if phase=='B' else ('P','U4')
    try:
        for s in seeds:
            p,psha=preps[s]
            if p['test_mode'] is not test or p['code_hash']!=code or p['model_manifest']!=manifest:errors.append(f'{s}: prep binding')
            if phase=='B':
                ee=validate_train(logs[s],p,psha,code,manifest,test,gpu)
                errors += [f'{s}: {e}' for e in ee]
        for ev in evaluations:
            s,arm,L,h,mode=ev['seed'],ev['arm'],ev['sleep'],ev['history'],ev['mode']
            if s not in preps:raise ValueError('foreign eval seed')
            key=(s,arm,L,h,mode)
            if key in evmap:raise ValueError('duplicate eval identity')
            evmap[key]=ev
        for s in seeds:
            p,psha=preps[s];per_seed[str(s)]={}
            nrefs=set()
            if phase=='B':
                for L in range(1,6):
                    ev=evmap[s,'N',L,'observed','base']
                    ee,j=validate_eval(ev,p,psha,None,phase,test,code,manifest,gpu)
                    if not isinstance(ev.get('ppl_ratio'),dict) or any(not ge(ev['ppl_ratio'].get(m),1.) or
                            not le(ev['ppl_ratio'].get(m),1.) for m in ('eval','modern')):
                        ee.append('N reference must have unit ratios')
                    errors += [f'{s}/N/{L}: {e}' for e in ee]
                    nrefs.add(ev.get('base_denominator_sha256'))
                if len(nrefs)!=1 or not all(hash_ok(x) for x in nrefs):errors.append(f'{s}: inconsistent N denominator')
            for arm in arms:
                per_seed[str(s)][arm]={}
                for L in range(1,6):
                    pair=[]
                    for mode in ('merged','unmerged'):
                        ev=evmap[s,arm,L,'observed',mode]
                        ck=expected_checkpoint(logs[s],arm,L,'observed')
                        ee,j=validate_eval(ev,p,psha,ck,phase,test,code,manifest,gpu)
                        if phase=='B' and ev.get('base_denominator_sha256') not in nrefs:ee.append('foreign base denominator')
                        if ee:invalid.add(arm)
                        errors += [f'{s}/{arm}/{L}/{mode}: {e}' for e in ee]
                        pair.append(ev)
                        if mode=='merged':joined[s,arm,L,'observed']=[dict(r,_sleep=L) for r in j]
                    if not twin(*pair):invalid.add(arm);errors.append(f'{s}/{arm}/{L}: twins differ')
                    j=joined[s,arm,L,'observed'];conditions=A_CONDITIONS if phase=='A' else B_CONDITIONS
                    per_seed[str(s)][arm][str(L)]={cond:{cohort:rates(select([r for r in j if r['condition']==cond],cohort,p['stream']))
                                for cohort in ('POS','FIRST','POS-A','POS-H','RET','SEL-D','HOW','HNM','FRESH','SEL-Q','HOW-Q','REV-Q','GTQ')}
                                for cond in conditions}
                    per_seed[str(s)][arm][str(L)]['ppl_ratio']=pair[0].get('ppl_ratio')
            # Gate baseline: exact historical P answers, independently regraded on historical gold.
            per_seed[str(s)]['P-GATE']={}
            for L in range(1,6):
                base=baselines[s,L];aa=anchors[s];probes=aa['probes']['observed'][str(L)]
                if [r['probe_id'] for r in base['rows']]!=[r['probe_id'] for r in probes]:raise ValueError('historical P inventory')
                gg=[]
                for q,r in zip(probes,base['rows']):
                    gold=oracle(history(aa,'observed'),L).get(tuple(q['cell']),'UNKNOWN')
                    answer=r['answer'] if gold!='UNKNOWN' else 'UNKNOWN'
                    gg.append(dict(q,graded=classify(answer,gold)))
                per_seed[str(s)]['P-GATE'][str(L)]={g:rates(select(gg,g,p['stream'])) for g in ('POS','FIRST','POS-A','POS-H','RET','SEL-D','HOW','HNM','FRESH')}
            for arm in (('RU','RC') if phase=='B' else ('U4',)):
                pair=[]
                for mode in ('merged','unmerged'):
                    ev=evmap[s,arm,4,'withheld',mode];ck=expected_checkpoint(logs[s],arm,4,'withheld')
                    ee,j=validate_eval(ev,p,psha,ck,phase,test,code,manifest,gpu)
                    if phase=='B' and ev.get('base_denominator_sha256') not in nrefs:ee.append('fork denominator')
                    if ee:invalid.add(arm)
                    errors += [f'{s}/{arm}/fork: {e}' for e in ee];pair.append(ev)
                    if mode=='merged':joined[s,arm,4,'withheld']=j
                if not twin(*pair):invalid.add(arm);errors.append(f'{s}/{arm}/fork twins differ')
        expected=len(seeds)*(39 if phase=='B' else 22)
        if len(evmap)!=expected:errors.append('unexpected evaluation inventory')
        metrics={}
        for arm in (*arms,'P-GATE'):
            metrics[arm]={}
            for L in range(1,6):
                metrics[arm][str(L)]={}
                for cohort in ('POS','FIRST','POS-A','POS-H','RET','SEL-D','HOW','HNM','FRESH'):
                    metrics[arm][str(L)][cohort]={}
                    for out in ('correct','unknown','empty','substantive'):
                        fr={str(s):(per_seed[str(s)][arm][str(L)][cohort][out] if arm=='P-GATE'
                                    else per_seed[str(s)][arm][str(L)]['FAITHFUL'][cohort][out]) for s in seeds}
                        metrics[arm][str(L)][cohort][out]=pool(fr)
        k5={};k5per={};common={}
        for arm in arms:
            k5per[arm]={}
            for s in seeds:
                rr=[r for L in range(2,6) for r in joined[s,arm,L,'observed']]
                k5per[arm][str(s)],_=reliance(rr,preps[s][0]['stream'])
            k5[arm]={g:{k:pool({str(s):k5per[arm][str(s)][g][k] for s in seeds}) for k in ('K5a','K5b')} for g in ('A','H')}
        if phase=='B':
            for s in seeds:
                rr={a:[r for L in range(2,6) for r in joined[s,a,L,'observed']] for a in ('RU','RC')}
                keys=[reliance(rr[a],preps[s][0]['stream'])[1] for a in ('RU','RC')]
                common[str(s)]={a:reliance(rr[a],preps[s][0]['stream'],keys[0]&keys[1])[0] for a in ('RU','RC')}
        for arm in (('RU','RC') if phase=='B' else ('U4',)):
            forks[arm]={}
            for cat in ('FORK-H','FORK-H-Q'):
                fs={}
                for s in seeds:
                    obs=[r for r in joined[s,arm,4,'observed'] if r['condition']=='FAITHFUL' and r['cat']==cat]
                    wh=[r for r in joined[s,arm,4,'withheld'] if r['condition']=='FAITHFUL' and r['cat']==cat]
                    o= rates(obs)['correct'];w=rates(wh)['unknown']
                    fs[str(s)]=dict(observed_correct=o,withheld_unknown=w,history_sensitive=ge(o['value'],.5) and ge(w['value'],.8))
                forks[arm][cat]=dict(numerator=sum(x['history_sensitive'] for x in fs.values()),denominator=len(seeds),per_seed=fs)
        criteria={a:criterion_checks(metrics,per_seed,seeds,a,phase,k5,forks) for a in arms}
        # RP has no fork: its K6 is a reading that is not assessable, never a candidate failure/crash.
        candidate='RC' if phase=='B' else None
        if errors or invalid: verdict='PROV1_INVALID'
        elif phase=='A':verdict='PROV1A_ZERO_SHOT_PASS' if any(c['pass_'] for c in criteria.values()) else 'PROV1A_ZERO_SHOT_FAIL'
        elif criteria[candidate]['pass_']:verdict='PROV1_ADVANCE'
        else:verdict='PROV1_GATE_ONLY'  # P baseline's K1/K4 must still be checked below.
        baseline_ok=all(ge(metrics['P-GATE'][str(L)]['POS']['correct']['value'],.5) and
                        (L==1 or ge(metrics['P-GATE'][str(L)]['FIRST']['correct']['value'],.5)) for L in range(1,6))
        # Gate negatives are exact by construction. Competence is unchanged frozen P;
        # the bundle loader validates its recorded monitors before production use.
        if verdict=='PROV1_GATE_ONLY' and not baseline_ok:verdict='PROV1_NO_ADVANCE'
        return dict(analyzer='Sol',version=VERSION,phase=phase,test_mode=test,verdict=verdict,
                    production_ready=False,draft_choices=DRAFT_CHOICES,errors=errors,invalid_arms=sorted(invalid),
                    metrics=metrics,per_seed=per_seed,criteria=criteria,K5=k5,K5_per_seed=k5per,
                    K5_common_RU_RC=common,fork=forks,
                    limitations=['No tensor/RNG or model forward-pass execution during analysis.',
                                 'Calibration and relaxed checkpoint-admission schemas are pending; draft is not a launch gate.'])
    except (KeyError,ValueError,TypeError,IndexError,ZeroDivisionError) as exc:
        return dict(analyzer='Sol',version=VERSION,phase=phase,test_mode=test,verdict='PROV1_INVALID',
                    production_ready=False,draft_choices=DRAFT_CHOICES,errors=errors+[f'incomplete/malformed inputs: {exc}'],
                    invalid_arms=sorted(invalid))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request',required=True,help='JSON request with explicit prep, train, eval, anchor and baseline paths')
    ap.add_argument('--out',required=True)
    a=ap.parse_args();r=json.loads(Path(a.request).read_text())
    read=lambda p:json.loads(Path(p).read_text())
    seeds=[int(s) for s in r['seeds']]
    preps={s:(read(r['preps'][str(s)]),sha(Path(r['preps'][str(s)]).read_bytes())) for s in seeds}
    result=analyze(preps,{s:read(r['train'][str(s)]) for s in seeds},[read(p) for p in r['evals']],
                   {s:read(r['anchors'][str(s)]) for s in seeds},
                   {(s,L):read(r['baselines'][f'{s}/{L}']) for s in seeds for L in range(1,6)},
                   phase=r['phase'],test=r.get('test_mode',False),code=r['code_hash'],manifest=r['model_manifest'],gpu=r.get('gpu'))
    result['input_sha256']={p:sha(Path(p).read_bytes()) for p in [*r['evals'],*r['preps'].values(),*r['train'].values(),*r['anchors'].values(),*r['baselines'].values()]}
    result['source_sha256']=source_fingerprint(('prov1_sol_spec.py','prov1_sol_analyze.py'))
    if not result['test_mode'] and result['verdict']!='PROV1_INVALID':
        result['provisional_verdict']=result['verdict']
        result['verdict']='PROV1_ANALYZER_NOT_FROZEN'
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('verdict','production_ready','errors')}))
    # Deliberately nonzero for production: pending frozen contracts must not look launch-ready.
    return 0 if result['test_mode'] and not result['errors'] else 2


if __name__=='__main__':raise SystemExit(main())
