#!/usr/bin/env python3
"""Independent CPU prep validator, PROV-1 b88ca93 §11. No producer imports.

Production PASS requires the pinned tokenizer, model identity and hash-bound
SUPPORT-1 phase-0 preps. It attests prep correctness, not launch approval.
"""
import argparse
import json
from pathlib import Path
import random
import re
from prov1_sol_spec import *

VERSION = 'PROV1_SOL_VALIDATOR_V1'


def semantic(kind, slot, c, wording, label, mode='faithful', item=None, source=None):
    return dict(kind=kind, slot=slot, cell=list(c), wording=wording, label=label,
                rec_mode=mode, item_id=item, source_item_id=source)


def reconstruct(anchor, tok):
    """Reconstruct all training rows from the historical schedule and literal §11 RNG contract."""
    seed = anchor['seed']; st = anchor['stream']
    excluded = set(map(tuple, anchor['strata']['rev'])) | set(map(tuple, st['updates']))
    cells = ordered({cell(f) for f in st['facts'] if f['entity'] in st['A']} - excluded)
    population = [(c,k) for c in cells for k in (0,1)]
    random.Random(f'PROV1-cfcyc-{seed}').shuffle(population)
    pointer = 0; specs = {}; cf_log = {}
    for L in range(1,6):
        positives = anchor['positives'][str(L)]; n = len(positives)
        new = [semantic('pos','new',p['cell'],p['wording'],p['label'],item=p['item_id']) for p in positives] * 4
        replay = [semantic('replay','replay',r['cell'],r['wording'],r['label'],source=r['source_item_id'])
                  for r in anchor['schedule']['P'][str(L)] if r['slot']=='replay'] if L>1 else None
        blocks = anchor['base_blocks'][str(L)]
        negative = [semantic(kind,'third',c,k,'UNKNOWN') for part,kind in [('A','neg_a'),('T','neg_t')]
                    for c,k in blocks[part]] * 3
        cf = []; supported = oracle(st['facts'],L)
        for _ in range(2*n):
            for attempt in range(len(population)):
                c,k = population[pointer % len(population)]; pointer += 1
                if c in supported: break
            else: raise ValueError('counterfactual population has no eligible cell')
            cf.append(semantic('cf','third',c,k,'UNKNOWN','delq'))
        specs[L] = (new,replay,negative,cf)
        cf_log[str(L)] = [[r['cell'],r['wording']] for r in cf]
    bg_ids = tok.encode(' '.join((ROOT/'data/background.txt').read_text().splitlines()),add_special_tokens=False)

    def row(sp, facts, L):
        e,a = sp['cell']; rs,_ = entries(facts,e,L)
        if sp['rec_mode']=='delq': rs.pop(a,None)
        rec = record(e,rs)
        prefix = tok.encode(FRAME.format(item=rec+'\n'+question((e,a),sp['wording'])),add_special_tokens=False)
        tail = tok.encode(' '+sp['label'],add_special_tokens=False)+tok.encode('\n',add_special_tokens=False)
        return dict(sp,src_cell=None,input_ids=prefix+tail,loss_start=len(prefix),n_loss=len(tail),record=rec)

    def background(like, idx, slot):
        out=[]
        for j,r in enumerate(like):
            n=len(r['input_ids']); start=random.Random(f'CONS0A-bg-PROV1-{seed}-{idx+j}').randrange(max(1,len(bg_ids)-n))
            out.append(dict(kind='bg',slot=slot,cell=None,src_cell=None,wording=None,label=None,item_id=None,
                            source_item_id=None,input_ids=bg_ids[start:start+n],n_loss=r['n_loss'],loss_start=n-r['n_loss']))
        return out

    def rendered(L,h):
        fs=history(anchor,h); new,replay,neg,cf=specs[L]
        if h=='withheld':
            replacements={tuple(f['target']):f for f in anchor['fork']['facts']}
            new=[semantic('pos','new',cell(replacements[tuple(r['cell'])]),r['wording'],replacements[tuple(r['cell'])]['value'],
                          item=f"{replacements[tuple(r['cell'])]['fid']}-q{r['wording']}")
                 if tuple(r['cell']) in replacements else r for r in new]
        nr=[row(r,fs,L) for r in new]
        rr=background(nr,1_000_000,'replay') if replay is None else [row(r,fs,L) for r in replay]
        nn=[row(r,fs,L) for r in neg]; cc=[row(r,fs,L) for r in cf]
        thirds={'RP':background(nn+cc,2_000_000+L*10_000,'third'),
                'RU':nn+background(cc,3_000_000+L*10_000,'third'),'RC':nn+cc}
        order=list(range(len(nr)+len(rr)+len(nn)+len(cc)))
        random.Random(f'PROV1-perm-{seed}-{L}').shuffle(order)
        return {arm:[dict((nr+rr+thirds[arm])[j],pos=i) for i,j in enumerate(order)] for arm in ARMS}
    schedules={a:{} for a in ARMS}
    for L in range(1,6):
        r=rendered(L,'observed')
        for a in ARMS: schedules[a][str(L)]=r[a]
    wh=rendered(4,'withheld')
    return schedules,{'L':4,**{a:wh[a] for a in ('RU','RC')}},cf_log


def validate(p, anchor, tok=None, expected_code=None, manifest=None):
    errors=[]; checks=0
    def ck(ok,msg):
        nonlocal checks
        checks+=1
        if not ok: errors.append(msg)
    try:
        ck(p['schema']==SCHEMA and p['experiment']=='PROV1','schema/experiment')
        ck(type(p['test_mode']) is bool and p['test_mode']==anchor['test_mode'],'test-mode binding')
        ck(p['seed']==anchor['seed'] and p['seed'] in range(20,25),'seed binding')
        ck(p['support1_code_hash']==S1_HASH and anchor['code_hash']==S1_HASH,'frozen SUPPORT-1 code')
        ck(sha(b''.join(sha((ROOT/f).read_bytes()).encode() for f in S1_CODE_FILES))==S1_HASH,'local frozen SUPPORT-1 files')
        ck(p['code_hash']==(expected_code or codehash()),'PROV-1 code hash')
        ck(p['model_manifest']==anchor['model_manifest'] and (manifest is None or p['model_manifest']==manifest),'model identity')
        for key in ('stream','fork','strata','split'):
            ck(p[key]==anchor[key],f'historical {key}')
        ck(p['stream_sha256']==chash(p['stream'])==anchor['stream_sha256'],'stream digest')
        ck(p['support1_probes_sha256']==chash(anchor['probes']),'historical probe digest')
        for h,dd in anchor['probes'].items():
            ck(set(p['probes'][h])==set(dd),f'{h} sleep inventory')
            for L in dd:
                actual=p['probes'][h][L]; expected=expand(anchor,h,int(L))
                ck(len(actual)==len(expected),f'{h}/{L} probe inventory length')
                for i,(x,y) in enumerate(zip(actual,expected)):
                    ck(x==y,f'{h}/{L}/{i} probe/record/truth/compliance differs')
                    if tok:
                        ck(p['test_mode'] or len(tok.encode(FRAME.format(item=x['item']),add_special_tokens=False))<=256,
                           f'{h}/{L}/{i} probe too long')
        ck(set(p['probes'])==set(anchor['probes']),'history inventory')
        if tok:
            schedules,fork,cf=reconstruct(anchor,tok)
            ck(set(p['schedule'])==set(ARMS),'arm inventory')
            for a in ARMS:
                ck(set(p['schedule'][a])=={str(i) for i in range(1,6)},a+' sleep inventory')
                for L in range(1,6):
                    rows=p['schedule'][a][str(L)]; expected=schedules[a][str(L)]
                    ck(len(rows)==len(expected),f'{a}/{L} schedule length')
                    for i,(x,y) in enumerate(zip(rows,expected)):
                        ck(x==y,f'{a}/{L}/{i} schedule/source/record/tokens differs')
                        ck(p['test_mode'] or len(x['input_ids'])<=256,f'{a}/{L}/{i} too long')
                    ck(budget(rows)==p['budget'][a][str(L)],f'{a}/{L} stored budget')
                    ck(budget(rows)==budget(p['schedule']['RC'][str(L)]),f'{a}/{L} arm budget')
            ck(p['counterfactuals']==cf,'counterfactual cycler log')
            ck(set(p['fork_schedule'])=={'L','RU','RC'} and p['fork_schedule']['L']==4,'fork inventory')
            for a in ('RU','RC'):
                ck(p['fork_schedule'][a]==fork[a],a+' full fork schedule')
                fb=budget(p['fork_schedule'][a]); ob=budget(p['schedule'][a]['4'])
                ck(fb==p['fork_budget'][a],a+' stored fork budget')
                ck(p['test_mode'] or (fb[0]==ob[0] and fb[2]==ob[2] and abs(fb[1]-ob[1])<=.01*ob[1]),a+' fork tolerance')
            ck(p['fork_budget']['RU']==p['fork_budget']['RC'],'fork arm budgets')
        else:
            ck(p['test_mode'],'production requires tokenizer and schedule reconstruction')
        ck(p['test_mode'] or (bool(re.fullmatch('[0-9a-f]{64}',manifest or '')) and tok is not None),'external production bindings')
    except (KeyError,ValueError,TypeError,IndexError,ZeroDivisionError) as exc:
        errors.append(f'malformed prep: {type(exc).__name__}: {exc}')
    return dict(seed=p.get('seed'),test_mode=p.get('test_mode'),checks=checks,errors=errors,
                verdict='FAIL' if errors else 'TEST_ONLY' if p.get('test_mode') else 'PASS')


def load_anchor(directory,seed,test=False):
    directory=Path(directory); path=directory/f'prep_seed{seed}.json'; raw=path.read_bytes()
    if not test:
        lines=(directory/'PHASE0.sha256').read_text().splitlines()
        expected=[s.split()[0] for s in lines if len(s.split())==2 and s.split()[1].lstrip('*')==path.name]
        if expected != [sha(raw)]: raise ValueError('SUPPORT-1 anchor is not hash-bound by PHASE0.sha256')
        # External historical identity, independent of mutable manifest contents.
        hashes={20:'a07d6e4c3df4d365ffe693bc6ab7292840330ff17cffa2cc1435e749c3bd7de4',
                21:'d8afb085eecb7faf4c6a27e08fca6effe900be928e4186eea83b65ecfbcb44fe',
                22:'cfc15d72e85046e3a3de33c6704106f9d0eb0d5990dd89af062f919e4818c235',
                23:'3622a21cc1bdecf24590bdcea30f0199b514511934794af8dfa7050c55def804',
                24:'c8e51580f1b9b1693530dffaced14878351194f05bc2bebba91997bf6f707481'}
        if sha(raw)!=hashes[seed]: raise ValueError('SUPPORT-1 prep is not the approved frozen bytes')
    return json.loads(raw)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prep',nargs='+',required=True)
    ap.add_argument('--support1-phase0',required=True);ap.add_argument('--tokenizer')
    ap.add_argument('--model-manifest-sha');ap.add_argument('--out',required=True)
    a=ap.parse_args();tok=None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    results=[];hashes=[]
    for path in map(Path,a.prep):
        raw=path.read_bytes();p=json.loads(raw);hashes.append(sha(raw))
        anchor=load_anchor(a.support1_phase0,p['seed'],p.get('test_mode') is True)
        results.append(validate(p,anchor,tok,codehash(),a.model_manifest_sha))
    verdict='PASS' if results and all(r['verdict']=='PASS' for r in results) else 'TEST_ONLY' if results and all(r['verdict']=='TEST_ONLY' for r in results) else 'FAIL'
    report=dict(auditor='Sol',validator=VERSION,verdict=verdict,code_hash=codehash(),model_manifest=a.model_manifest_sha,
                prep_sha256=hashes,results=results,checks=sum(r['checks'] for r in results),
                errors=[e for r in results for e in r['errors']],
                source_sha256=source_fingerprint(('prov1_sol_spec.py','prov1_sol_validate.py')))
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('verdict','checks','errors')}))
    return 0 if verdict in ('PASS','TEST_ONLY') else 2


if __name__=='__main__':
    raise SystemExit(main())
