"""Sol's full-inventory calibration counterexamples for producer 3a4d604/24984b1.
No forward passes, pods, SSH, credentials or external machines are used.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import tarfile
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prov1 as V


@pytest.fixture
def calibration(tmp_path):
    with tarfile.open(ROOT/'results/prov1/phase0/preps.tar.gz') as archive:
        raw=archive.extractfile('prov1_prep_seed24.json').read()
    p=json.loads(raw);prep=tmp_path/'prep.json';prep.write_bytes(raw);psha=V.sha(raw)
    hx=lambda s:V.sha(s)
    tr=dict(schema=V.SCHEMA,experiment=V.EXPERIMENT,seed=24,
            provenance=dict(code_hash=V.code_hash(),prep_sha256=psha,model_manifest=p['model_manifest'],
                            test_mode=False,gpu='NVIDIA RTX 6000 Ada Generation',arms=['RP','RC'],sleeps=1,fork=False),
            arms={arm:{'1':dict(ckpt_sha256=hx(arm),full_state_hash=hx(arm+'state'))} for arm in ('RP','RC')})
    docs={};paths={}
    for arm in ('RP','RC','N'):
        for mode in (('base',) if arm=='N' else ('merged','unmerged')):
            rows=[]
            for q in p['probes']['observed']['1']:
                if q['condition'] not in V.PHASE_B_CONDITIONS:continue
                answer='UNKNOWN' if arm=='N' else q['gold_compliance'] or 'UNKNOWN'
                rows.append(dict(probe_id=q['probe_id'],answer=answer,outcome_truth=V.outcome(answer,q['gold_truth']),
                                 outcome_compliance=V.outcome(answer,q['gold_compliance'])))
            pv=dict(code_hash=V.code_hash(),prep_sha256=psha,ckpt_sha256=None if arm=='N' else hx(arm),
                    model_manifest=p['model_manifest'],test_mode=False,conditions=list(V.PHASE_B_CONDITIONS),
                    monitors=True,source='prov1',arm=arm,sleep=1,history='observed',mode=mode)
            docs[arm,mode]=dict(schema=V.SCHEMA,experiment=V.EXPERIMENT,seed=24,arm=arm,sleep=1,history='observed',
                                mode=mode,provenance=pv,ppl_ratio=dict(eval=1. if arm=='N' else .9,modern=1. if arm=='N' else .9),
                                base_denominator_sha256=hx('denominator'),rows=rows,
                                env=dict(tf32=False,gpu='NVIDIA RTX 6000 Ada Generation'))
            paths[arm,mode]=tmp_path/f'{arm}_{mode}.json'
    train=tmp_path/'train.json';out=tmp_path/'calcheck.json'
    a=argparse.Namespace(prep=str(prep),train=str(train),out=str(out),base=str(paths['N','base']),
                        **{f'{arm.lower()}_{mode}':str(paths[arm,mode]) for arm in ('RP','RC') for mode in ('merged','unmerged')})
    def run():
        train.write_text(json.dumps(tr))
        for key,value in docs.items():paths[key].write_text(json.dumps(value))
        try:V.stage_calcheck(a);rc=0
        except SystemExit as exc:rc=exc.code
        return rc,json.loads(out.read_text())
    return tr,docs,run


def test_full_production_calibration_baseline(calibration):
    _,_,run=calibration;rc,result=run()
    assert rc==0 and result['pass'] is True and not result['errors']


def test_calcheck_requires_nonnull_checkpoint_hash(calibration):
    tr,docs,run=calibration
    for arm in ('RP','RC'):
        tr['arms'][arm]['1'].pop('ckpt_sha256')
        for mode in ('merged','unmerged'):docs[arm,mode]['provenance']['ckpt_sha256']=None
    rc,r=run();assert rc==4 and r['pass'] is False


@pytest.mark.parametrize('bad',[None,'not-a-sha256'])
def test_calcheck_requires_denominator_hash(calibration,bad):
    _,docs,run=calibration
    for ev in docs.values():ev['base_denominator_sha256']=bad
    rc,r=run();assert rc==4 and r['pass'] is False


def test_calcheck_rejects_boolean_monitors(calibration):
    _,docs,run=calibration
    for ev in docs.values():ev['ppl_ratio']={'eval':True,'modern':True}
    rc,r=run();assert rc==4 and r['pass'] is False


@pytest.mark.parametrize('attack',['model','test_mode','seed'])
def test_calcheck_rejects_foreign_train(calibration,attack):
    tr,_,run=calibration
    if attack=='model':tr['provenance']['model_manifest']='f'*64
    elif attack=='test_mode':tr['provenance']['test_mode']=True
    else:tr['seed']=20
    rc,r=run();assert rc==4 and r['pass'] is False


def test_launcher_refuses_unmounted_drive_directory(tmp_path):
    source=(ROOT/'scripts/cb_prov1.sh').read_text()
    begin=source.index('[[ $PIN =~');end=source.index('USD_PER_HR=',begin)
    drive=tmp_path/'drive_e';drive.mkdir()
    bindir=tmp_path/'bin';bindir.mkdir();df=bindir/'df'
    df.write_text('#!/bin/sh\nprintf "Avail\\n100G\\n"\n');df.chmod(0o755)
    # Only the local output preflight is extracted. No launcher, SSH or pod code is run.
    script=f'PIN={"0"*40}\nOUT={drive}/cb_results\nPATH={bindir}:$PATH\n'+source[begin:end].replace('/mnt/e',str(drive))
    run=subprocess.run(['bash','-c',script],capture_output=True,text=True)
    assert run.returncode!=0, 'ordinary directory was accepted as E: without verifying its mounted filesystem'


def test_mismatched_pilot_checkpoint_is_retained(tmp_path):
    source=(ROOT/'scripts/cb_prov1.sh').read_text()
    start=source.index('  # ---- SUPPORT-1 reproducibility pilot:')
    end=source.index('  # ---- PROV-1 calibration:',start)
    r=tmp_path/'run';old=tmp_path/'source';(r/'repro').mkdir(parents=True);(old/'cal').mkdir(parents=True)
    log=lambda h:{'arms':{'P':{'1':{'ckpt_sha256':h,'full_state_hash':h}}}}
    (r/'repro/train_seed24.json').write_text(json.dumps(log('a'*64)))
    (old/'cal/train_seed24.json').write_text(json.dumps(log('b'*64)))
    ck=r/'repro/cortex_P_s1.pt';ck.write_bytes(b'nonmatching pilot evidence')
    script=f'R={r}\nS1={old}\nS0={old}\nX1=true\nPY={sys.executable}\nM=unused\nMS=unused\nfin() {{ exit "$1"; }}\n'+source[start:end]
    ran=subprocess.run(['bash','-c',script],capture_output=True,text=True)
    assert ran.returncode==0,(ran.stdout,ran.stderr)
    assert json.loads((r/'repro/repro.json').read_text())['file_equal'] is False
    assert ck.exists(),'the nonmatching checkpoint was deleted before preservation/investigation'
