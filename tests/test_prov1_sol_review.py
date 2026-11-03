"""Sol's counterexamples for b88ca93. Expected failures document producer defects;
remove strict-xfail markers only once the counterexamples are actually rejected.
No model, GPU, network, launcher or external machine is used here.
"""
import argparse
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prov1 as producer


class Recompute(Exception): pass


@pytest.mark.xfail(strict=True,reason='b88ca93 eval cache omits requested identity fields')
@pytest.mark.parametrize('field,new',[('arm','RU'),('sleep',4),('history','withheld'),('mode','unmerged')])
def test_eval_cache_refuses_foreign_request(tmp_path,monkeypatch,field,new):
    p=tmp_path/'prep.json';p.write_text(json.dumps({'test_mode':True}))
    ck=tmp_path/'checkpoint.pt';ck.write_bytes(b'no model is loaded')
    out=tmp_path/'eval.json'
    a=argparse.Namespace(prep=str(p),model='UNUSED',out=str(out),test_mode=True,
                        model_manifest_sha='',conditions='FAITHFUL',monitors=False,source='prov1',
                        ckpt=str(ck),arm='RC',sleep=1,history='observed',mode='merged')
    prov=dict(code_hash=producer.code_hash(),prep_sha256=producer.sha(p.read_bytes()),
              ckpt_sha256=producer.sha(ck.read_bytes()),model_manifest='',test_mode=True,
              conditions=['FAITHFUL'],monitors=False,source='prov1')
    old=dict(schema=producer.SCHEMA,experiment='PROV1',seed=20,arm='RC',sleep=1,
             history='observed',mode='merged',provenance=prov,rows=[])
    out.write_text(json.dumps(old))
    setattr(a,field,new)
    def recompute(*_):raise Recompute()
    monkeypatch.setattr(producer.A0,'load_fp32',recompute)
    # Recompute (or explicit refusal) is required; a successful stale-cache return is the defect.
    with pytest.raises((Recompute,SystemExit,ValueError)):
        producer.stage_evalck(a)


@pytest.mark.xfail(strict=True,reason='b88ca93 train cache verifies only main checkpoints, not required fork')
@pytest.mark.parametrize('damage',['missing','corrupt'])
def test_train_cache_checks_withheld_checkpoint(tmp_path,monkeypatch,damage):
    p=tmp_path/'prep.json';p.write_text(json.dumps({'test_mode':True,'seed':24}))
    a=argparse.Namespace(prep=str(p),out=str(tmp_path),test_mode=True,model_manifest_sha='',
                        arms='RC',sleeps=4,fork=True,model='UNUSED')
    main={}
    for L in range(1,5):
        ck=tmp_path/f'cortex_RC_s{L}.pt';ck.write_bytes(b'main')
        main[str(L)]={'ckpt_sha256':producer.sha(ck.read_bytes())}
    fork=tmp_path/'cortex_RC_s4_withheld.pt';fork.write_bytes(b'fork')
    fh=producer.sha(fork.read_bytes())
    if damage=='missing':fork.unlink()
    else:fork.write_bytes(b'changed')
    prov=dict(code_hash=producer.code_hash(),prep_sha256=producer.sha(p.read_bytes()),model_manifest='',
              test_mode=True,gpu='cpu',arms=['RC'],sleeps=4,fork=True)
    log=dict(provenance=prov,arms={'RC':main},fork={'admitted':True,'withheld':{'RC':{'ckpt_sha256':fh}}})
    (tmp_path/'prov1_train_seed24.json').write_text(json.dumps(log))
    def recompute(*_):raise Recompute()
    monkeypatch.setattr(producer.C,'load_model',recompute)
    with pytest.raises((Recompute,SystemExit,ValueError)):
        producer.stage_train(a)
