"""Non-blocking-for-phase-1 followups for the explicitly later A/B admission review.
These do not exercise the phase-1 job or any external infrastructure.
"""
import argparse
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
from test_prov1_peer_admission import files
import prov1_analyze as peer
from prov1_sol_spec import codehash


@pytest.mark.xfail(strict=True,reason='later A/B review: withheld full_state_hash is not checked by the peer loader')
def test_withheld_state_identity_is_required(files):
    sd,_,_,_,_,run=files
    assert run()['verdict']=='PROV1_ADVANCE'
    path=sd/'prov1_train_seed20.json';tr=json.loads(path.read_text())
    tr['fork']['withheld']['RC'].pop('full_state_hash')
    path.write_text(json.dumps(tr))
    assert run()['verdict']=='PROV1_INVALID'


@pytest.mark.xfail(strict=True,reason='later A/B review: bind_frozen trusts a bare pass flag without the phase-1 manifest/raw calibration')
def test_production_binding_needs_calibration_evidence(tmp_path):
    d=tmp_path/'data';p1=tmp_path/'phase1';(p1/'cal').mkdir(parents=True)
    frozen=ROOT/'results/prov1/phase0'
    with tarfile.open(frozen/'preps.tar.gz') as archive:
        for s in range(20,24):
            sd=d/f'seed{s}';sd.mkdir(parents=True)
            (sd/f'prov1_prep_seed{s}.json').write_bytes(archive.extractfile(f'prov1_prep_seed{s}.json').read())
    hashes={line.split()[1]:line.split()[0] for line in (frozen/'PHASE0.sha256').read_text().splitlines()
            if len(line.split())==2 and line.split()[1].endswith('.json')}
    report={'pass':True,'errors':[],'code_hash':codehash(),'test_mode':False,
            'prep_sha256':hashes['prov1_prep_seed24.json'],'model_manifest':'FOREIGN MODEL'}
    (p1/'cal/calcheck.json').write_text(json.dumps(report))
    args=argparse.Namespace(phase0_dir=str(frozen),phase1_dir=str(p1))
    assert peer.bind_frozen(args,[20,21,22,23],d)
