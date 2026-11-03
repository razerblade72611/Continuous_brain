"""Independent external-binding checks with complete, synthetic production-shaped calibration files."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
from test_prov1_phase1_review import calibration
from prov1_sol_io import verify_calibration,manifest,GPU
from prov1_sol_spec import sha,codehash


def check(tmp_path,run):
    run();d=tmp_path/'phase1';(d/'cal').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(tmp_path/'train.json',d/'cal/prov1_train_seed24.json')
    shutil.copyfile(tmp_path/'calcheck.json',d/'cal/calcheck.json')
    for arm in ('RP','RC'):
        for mode in ('merged','unmerged'):
            shutil.copyfile(tmp_path/f'{arm}_{mode}.json',d/f'cal/eval_{arm}_s1_{mode}.json')
    shutil.copyfile(tmp_path/'N_base.json',d/'cal/eval_N_s1.json')
    raw=(tmp_path/'prep.json').read_bytes();p=json.loads(raw)
    return verify_calibration(d,p,sha(raw),codehash(),p['model_manifest'],GPU)


def test_complete_calibration_regraded_independently(calibration,tmp_path):
    _,_,run=calibration;r=check(tmp_path,run)
    assert r['pass_'] is True,r['errors']
    assert len(r['input_sha256'])==7 and r['RP_acquisition']['value']==1.


@pytest.mark.parametrize('delta,accepted',[(5e-10,True),(1.5e-9,False)])
def test_calibration_unit_ratio_tolerance(calibration,tmp_path,delta,accepted):
    _,docs,run=calibration
    docs['N','base']['ppl_ratio']={'eval':1.+delta,'modern':1.-delta}
    r=check(tmp_path,run)
    assert r['pass_'] is accepted,r['errors']


def test_calibration_report_input_hashes_are_checked(calibration,tmp_path):
    _,_,run=calibration;assert check(tmp_path,run)['pass_'] is True
    p=tmp_path/'phase1/cal/calcheck.json';report=json.loads(p.read_text())
    report['inputs_sha256']['RP_merged']='0'*64;p.write_text(json.dumps(report))
    raw=(tmp_path/'prep.json').read_bytes();prep=json.loads(raw)
    r=verify_calibration(tmp_path/'phase1',prep,sha(raw),codehash(),prep['model_manifest'],GPU)
    assert r['pass_'] is False and 'calibration input digest binding' in r['errors']


@pytest.mark.parametrize('attack',['checkpoint','base','boolean','train_model'])
def test_independent_calibration_rejects_producer_bypasses(calibration,tmp_path,attack):
    tr,docs,run=calibration
    if attack=='checkpoint':
        for arm in ('RP','RC'):
            tr['arms'][arm]['1'].pop('ckpt_sha256')
            for mode in ('merged','unmerged'):docs[arm,mode]['provenance']['ckpt_sha256']=None
    elif attack=='base':
        for d in docs.values():d['base_denominator_sha256']=None
    elif attack=='boolean':
        for d in docs.values():d['ppl_ratio']={'eval':True,'modern':True}
    else:tr['provenance']['model_manifest']='wrong'
    r=check(tmp_path,run)
    assert r['pass_'] is False and r['errors']


@pytest.mark.parametrize('body',[f'{"a"*64}  ../outside\n',f'{"a"*64}  x\n{"b"*64}  x\n'])
def test_manifest_rejects_unsafe_or_duplicate_paths(tmp_path,body):
    p=tmp_path/'manifest';p.write_text(body)
    with pytest.raises(ValueError):manifest(p)
