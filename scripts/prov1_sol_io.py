"""Independent directory/admission loader for the PROV-1 launcher layout.
No producer or peer-analyzer imports. Production failures produce a report,
including before a phase-1/admission contract is complete.
"""
import json
import re
from pathlib import Path
import tarfile
from prov1_sol_spec import *

GPU='NVIDIA RTX 6000 Ada Generation'
S1_BUNDLE_SHA='fa971aa2da915156013c1a0a95b17d2ea7ae38dfde381709e2df7da2204cc71c'


def read(path):return json.loads(Path(path).read_text())


def manifest(path):
    files={};meta={}
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        key,value=line.split(maxsplit=1)
        if len(key)==64 and all(c in '0123456789abcdef' for c in key):
            name=value.lstrip('*')
            if name in files or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe/duplicate manifest path')
            files[name]=key
        else:
            if key in meta:raise ValueError('duplicate manifest metadata')
            meta[key]=value
    return files,meta


def verify_calibration(p1,prep,prep_sha,code,model,gpu):
    """Independent regrade of the complete RP/RC/N calibration, not trust in pass=true."""
    from prov1_sol_analyze import validate_eval,twin,hash_ok
    p1=Path(p1);errors=[];inputs={};pairs={}
    def load(name):
        path=p1/name;inputs[name]=sha(path.read_bytes());return read(path)
    try:
        train=load('cal/prov1_train_seed24.json');pv=train.get('provenance',{})
        expected=dict(code_hash=code,prep_sha256=prep_sha,model_manifest=model,test_mode=False,
                      arms=['RP','RC'],sleeps=1,fork=False,gpu=gpu)
        if any(pv.get(k)!=v for k,v in expected.items()):errors.append('calibration train provenance')
        if train.get('schema')!=SCHEMA or train.get('experiment')!='PROV1' or train.get('seed')!=24:errors.append('calibration train identity')
        if prep.get('test_mode') is not False or prep.get('seed')!=24:errors.append('calibration prep identity')
        for arm in ('RP','RC'):
            state=train['arms'][arm]['1'];ck=state.get('ckpt_sha256')
            if not hash_ok(ck) or not hash_ok(state.get('full_state_hash')):errors.append('calibration checkpoint identity')
            pairs[arm]=[]
            for mode in ('merged','unmerged'):
                ev=load(f'cal/eval_{arm}_s1_{mode}.json')
                ee,joined=validate_eval(ev,prep,prep_sha,ck,'B',False,code,model,gpu)
                errors += [arm+'/'+mode+': '+x for x in ee];pairs[arm].append((ev,joined))
            if not twin(pairs[arm][0][0],pairs[arm][1][0]):errors.append(arm+' calibration twins')
        base=load('cal/eval_N_s1.json');ee,bj=validate_eval(base,prep,prep_sha,None,'B',False,code,model,gpu)
        errors += ['N: '+x for x in ee]
        evs=[base]+[e for arm in pairs for e,_ in pairs[arm]]
        denoms={e.get('base_denominator_sha256') for e in evs}
        if len(denoms)!=1 or not all(hash_ok(x) for x in denoms):errors.append('calibration shared base identity')
        if any(not ge(base['ppl_ratio'].get(k),1.) or not le(base['ppl_ratio'].get(k),1.) for k in ('eval','modern')):
            errors.append('calibration N unit ratios')
        rp=[r for r in pairs['RP'][0][1] if r['condition']=='FAITHFUL' and r['cat']=='POS' and r['kind']=='template']
        nn=[r for r in bj if r['condition']=='ABSENT' and r['cat']=='FRESH' and r['kind']=='template']
        acq=fraction(sum(r['graded']=='correct' for r in rp),len(rp));fresh=fraction(sum(r['graded']=='unknown' for r in nn),len(nn))
        if not ge(acq['value'],.8):errors.append('RP acquisition < .80')
        if not ge(fresh['value'],.9):errors.append('N FRESH < .90')
        if any(not le(e['ppl_ratio'][k],1.02) for arm in pairs for e,_ in pairs[arm] for k in ('eval','modern')):
            errors.append('calibration competence >1.02')
        report=load('cal/calcheck.json')
        if (report.get('pass') is not True or report.get('errors') or report.get('seed')!=24
                or report.get('test_mode') is not False or report.get('code_hash')!=code
                or report.get('model_manifest')!=model or report.get('prep_sha256')!=prep_sha):errors.append('calibration report binding')
        return dict(pass_=not errors,errors=errors,input_sha256=inputs,RP_acquisition=acq,N_FRESH=fresh,
                    base_denominator=next(iter(denoms)) if len(denoms)==1 else None)
    except (OSError,KeyError,ValueError,TypeError,IndexError,AttributeError) as exc:
        return dict(pass_=False,errors=errors+[f'incomplete/malformed calibration: {exc}'],input_sha256=inputs)


def directory_request(directory,support1,phase,seeds,test=False,phase0=None,phase1=None,gpu=None):
    """Construct the existing pure API request and independently check external bindings."""
    d=Path(directory);ref=Path(support1);p0=Path(phase0 or ROOT/'results/prov1/phase0')
    p1=Path(phase1 or ROOT/'results/prov1/phase1');gpu=gpu or (None if test else GPU)
    errors=[];binding={};code=codehash()
    request=dict(seeds=seeds,phase=phase,test_mode=test,code_hash=code,model_manifest=None,gpu=gpu,
                 preps={},train={},anchors={},baselines={},evals=[])
    try:
        for s in seeds:
            sd=d/f'seed{s}';p=sd/f'prov1_prep_seed{s}.json';pr=read(p)
            request['preps'][str(s)]=str(p);request['anchors'][str(s)]=str(ref/f'seed{s}/prep_seed{s}.json')
            if request['model_manifest'] is None:request['model_manifest']=pr['model_manifest']
            if pr['model_manifest']!=request['model_manifest']:errors.append('mixed prep model manifests')
            request['train'][str(s)]=str(sd/f'prov1_train_seed{s}.json') if phase=='B' else str(ref/f'seed{s}/train_seed{s}.json')
            for L in range(1,6):
                request['baselines'][f'{s}/{L}']=str(ref/f'seed{s}/eval_P_s{L}_merged.json')
                if phase=='B':request['evals'].append(str(sd/f'eval_N_s{L}.json'))
                for arm in (ARMS if phase=='B' else ('P','U4')):
                    for mode in ('merged','unmerged'):request['evals'].append(str(sd/f'eval_{arm}_s{L}_{mode}.json'))
            for arm in (('RU','RC') if phase=='B' else ('U4',)):
                for mode in ('merged','unmerged'):request['evals'].append(str(sd/f'eval_{arm}_s4_withheld_{mode}.json'))
        if not test:
            if seeds!=[20,21,22,23]:errors.append('production requires seeds 20-23')
            files,meta=manifest(p0/'PHASE0.sha256')
            if meta!=dict(CODE_HASH=code,MODEL_MANIFEST=request['model_manifest'],SUPPORT1_CODE_HASH=S1_HASH):errors.append('phase-0 code/model metadata')
            raw_preps={}
            with tarfile.open(p0/'preps.tar.gz') as archive:
                names={f'prov1_prep_seed{s}.json' for s in range(20,25)}
                members=archive.getmembers()
                if len(members)!=5 or {m.name for m in members}!=names or not all(m.isfile() for m in members):
                    raise ValueError('phase-0 archive inventory')
                for name in names:
                    raw_preps[name]=archive.extractfile(name).read()
                    if files.get(name)!=sha(raw_preps[name]):errors.append('phase-0 prep archive digest '+name)
            for s in seeds:
                p=Path(request['preps'][str(s)])
                if p.read_bytes()!=raw_preps[p.name]:errors.append('phase-2/phase-A prep differs from frozen phase 0')
            validator=read(d/'p0/validation.json')
            expected_hashes={sha(x) for x in raw_preps.values()}
            if (validator.get('verdict')!='PASS' or validator.get('errors') or validator.get('code_hash')!=code
                    or validator.get('model_manifest')!=request['model_manifest']
                    or set(validator.get('prep_sha256',[]))!=expected_hashes):errors.append('validator binding')
            binding['validator_sha256']=sha((d/'p0/validation.json').read_bytes())
            # The historical source has a fixed, previously independently verified archive hash.
            bundle=ROOT/'results/support1/phase2/support1_phase2.tar.gz'
            if sha(bundle.read_bytes())!=S1_BUNDLE_SHA:raise ValueError('historical SUPPORT-1 bundle identity')
            with tarfile.open(bundle) as archive:
                for s in seeds:
                    names=[f'prep_seed{s}.json',f'train_seed{s}.json']
                    names += [f'eval_P_s{L}_merged.json' for L in range(1,6)]
                    if phase=='A':names += [f'eval_U4_s{L}_merged.json' for L in range(1,6)]
                    for name in names:
                        raw=archive.extractfile(f'support1/seed{s}/{name}').read()
                        if (ref/f'seed{s}'/name).read_bytes()!=raw:errors.append('historical reference differs from frozen bundle '+name)
            f1,m1=manifest(p1/'PHASE1.sha256')
            required={'cal/calcheck.json','timings.json','projection.json','repro/repro.json','p0/validation.json'}
            if not required<=set(f1):errors.append('phase-1 manifest is missing required summary/admission artifacts')
            for name,digest in f1.items():
                if sha((p1/name).read_bytes())!=digest:errors.append('phase-1 digest '+name)
            if m1.get('CODE_HASH')!=code or m1.get('MODEL_MANIFEST')!=request['model_manifest'] or m1.get('GPU')!=gpu:
                errors.append('phase-1 metadata')
            if not re.fullmatch('[0-9a-f]{40}',m1.get('PIN','')):errors.append('phase-1 PIN identity')
            projection=read(p1/'projection.json')
            if projection.get('code_hash')!=code or projection.get('phase_A_affordable') is not True:
                errors.append('phase-1 projection binding/phase-A affordability')
            calprep=raw_preps['prov1_prep_seed24.json']
            cal=verify_calibration(p1,json.loads(calprep),sha(calprep),code,request['model_manifest'],gpu)
            binding['calibration']=cal;errors += cal['errors']
            # Raw calibration evidence must be bound, not only the summary's pass flag.
            for name,digest in cal['input_sha256'].items():
                if f1.get(name)!=digest:errors.append('phase-1 manifest does not bind '+name)
            binding['phase0_manifest_sha256']=sha((p0/'PHASE0.sha256').read_bytes())
            binding['phase1_manifest_sha256']=sha((p1/'PHASE1.sha256').read_bytes())
            binding['support1_bundle_sha256']=S1_BUNDLE_SHA
            if phase=='A':
                repro=read(p1/'repro/repro.json')
                pilot=read(ROOT/'results/support1/phase1/cal/train_seed24.json')['arms']['P']['1']['ckpt_sha256']
                if (repro.get('file_equal') is not True or repro.get('committed')!=pilot
                        or repro.get('regenerated')!=pilot):errors.append('SUPPORT-1 pilot did not match the frozen checkpoint')
                for s in seeds:
                    original=read(request['train'][str(s)])
                    regenerated=read(d/f'regen1/seed{s}/train_seed{s}.json')
                    adm=read(d/f'seed{s}/admission.json')
                    if adm.get('seed')!=s or adm.get('all_admitted') is not True:errors.append('admission status')
                    wanted={f'{arm}|{L}' for arm in ('P','U4') for L in range(1,6)}|{'U4|4w'}
                    if set(adm.get('records',{}))!=wanted:errors.append('admission inventory')
                    if regenerated.get('provenance')!=original.get('provenance'):errors.append('regeneration provenance')
                    for key in wanted:
                        arm,L=key.split('|');rec=adm['records'][key]
                        old=(original['fork']['withheld'][arm] if L=='4w' else original['arms'][arm][L])['ckpt_sha256']
                        new=(regenerated['fork']['withheld'][arm] if L=='4w' else regenerated['arms'][arm][L])['ckpt_sha256']
                        if rec.get('admitted') is not True or rec.get('logged_sha256')!=old or rec.get('regenerated_sha256')!=old or new!=old:
                            errors.append('exact checkpoint admission '+key)
                    binding[f'admission_seed{s}_sha256']=sha((d/f'seed{s}/admission.json').read_bytes())
        return request,errors,binding
    except (OSError,KeyError,ValueError,TypeError,AttributeError,tarfile.TarError) as exc:
        return request,errors+[f'incomplete/malformed directory bindings: {exc}'],binding
