"""Raw phase-1 verification and production-shaped phase-2 plumbing fixtures.

Synthetic answers/checkpoint identities are only software tests, never results.
No model or producer analysis code is imported.
"""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
import support1_sol_analyze as A
import support1_sol_spec as S
from test_support1_sol import fake_eval, fake_train

P0 = ROOT/'results/support1/phase0'
P1 = ROOT/'results/support1/phase1'
CODE = '38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae'
MODEL = '69d069331c87531e7aa054e70085cef25752b3e6e3614b03e851a2e997a4b85b'
GPU = 'NVIDIA RTX 6000 Ada Generation'


def verify(p1=P1):
    return A.verify_phase1(p1, P0, CODE, MODEL, GPU, expected_pin='512ca58f728fdb76351ebf709b7aec0833cb3ae9')


def test_committed_phase1_independent_regrade():
    r = verify()
    assert r['verdict'] == 'SUPPORT1_PHASE1_GO' and r['errors'] == []
    assert r['calibration']['acquisition'] == S.fraction(8, 10)
    assert r['calibration']['N_fresh_unknown'] == S.fraction(20, 20)
    assert r['calibration']['POS_paraphrase']['rates']['correct'] == S.fraction(8, 12)
    assert r['restoration']['admitted'] and r['calibration']['twin']['pass_']
    assert r['projection']['chosen'] == dict(fork=True, qdiag=64, hours=10.06, usd=8.65)
    assert r['projection']['cumulative_projection'] == pytest.approx(9.84)
    assert r['archive']['checksum_recomputed'] is False
    assert r['archive']['claimed_sha256'] == '3e589fb85c05267185825ac01cb66607fea0996bd86cf2247e46519941d5712b'


def rebind(p1):
    cal = A.read(p1/'cal/calcheck.json')
    cal['eval_sha256'] = {m: S.sha((p1/'cal'/name).read_bytes()) for m, name in
        [('merged','eval_P_s1_merged.json'),('unmerged','eval_P_s1_unmerged.json'),('base','eval_N_s1.json')]}
    (p1/'cal/calcheck.json').write_bytes(S.canon(cal))
    lines = []
    for line in (p1/'PHASE1.sha256').read_text().splitlines():
        first, rest = line.split(' ', 1); rest=rest.strip()
        lines.append(S.sha((p1/rest).read_bytes())+'  '+rest if A.hash_ok(first) else line)
    (p1/'PHASE1.sha256').write_text('\n'.join(lines)+'\n')


@pytest.mark.parametrize('attack, needle', [('rate','counts/rates'),('bytes','eval-byte'),('raw_acquisition','raw calibration'),
    ('denominator','base denominator'),('plan','recorded phase-2 plan'),('gpu','GPU')])
def test_bound_phase1_tampering(tmp_path, attack, needle):
    p1 = tmp_path/'phase1'; shutil.copytree(P1, p1)
    if attack == 'gpu':
        r=A.verify_phase1(p1,P0,CODE,MODEL,'NVIDIA A40')
    else:
        if attack in ('rate','bytes'):
            cal=A.read(p1/'cal/calcheck.json')
            if attack=='rate': cal['pos_template_acquisition']=.9
            else: cal['eval_sha256']['merged']='0'*64
            (p1/'cal/calcheck.json').write_bytes(S.canon(cal))
        elif attack=='raw_acquisition':
            prep=A.read(P0/'prep_seed24.json'); by={p['probe_id']:p for p in prep['probes']['observed']['1']}
            for mode in ('merged','unmerged'):
                path=p1/f'cal/eval_P_s1_{mode}.json';e=A.read(path)
                row=next(r for r in e['rows'] if r['outcome']=='correct' and by[r['probe_id']]['cat']=='POS' and by[r['probe_id']]['kind']=='template')
                row.update(answer='UNKNOWN',outcome='unknown');path.write_bytes(S.canon(e))
            cal=A.read(p1/'cal/calcheck.json');cal['pos_template_acquisition']=.7
            (p1/'cal/calcheck.json').write_bytes(S.canon(cal))
        elif attack=='denominator':
            for mode in ('merged','unmerged'):
                path=p1/f'cal/eval_P_s1_{mode}.json';e=A.read(path);e['base_denominator_sha256']='0'*64;path.write_bytes(S.canon(e))
        elif attack=='plan':
            path=p1/'projection.json';e=A.read(path);e['choice']=e['plans'][1];path.write_bytes(S.canon(e))
        if attack!='bytes': rebind(p1)
        else:
            # Rehash the calcheck file in the outer manifest without repairing its false input reference.
            lines=(p1/'PHASE1.sha256').read_text().splitlines()
            lines[0]=S.sha((p1/'cal/calcheck.json').read_bytes())+'  cal/calcheck.json'
            (p1/'PHASE1.sha256').write_text('\n'.join(lines)+'\n')
        r=verify(p1)
    assert r['verdict']=='SUPPORT1_PHASE1_HOLD'
    assert any(needle in e for e in r['errors']),r['errors']


@pytest.fixture(scope='module')
def production_shaped_matrix(tmp_path_factory):
    """Mock complete production layout; oracle-perfect fake answers are not science."""
    d=tmp_path_factory.mktemp('support1_mock_prod_matrix')
    base=verify()['base_denominator_sha256']
    for seed in (20,21,22,23):
        p=A.read(P0/f'prep_seed{seed}.json');ph=S.sha((P0/f'prep_seed{seed}.json').read_bytes())
        t=fake_train(p);t['provenance'].update(code_hash=CODE,model_manifest=MODEL,test_mode=False,gpu=GPU,prep_sha256=ph)
        folder=d/f'seed{seed}';folder.mkdir();(folder/f'train_seed{seed}.json').write_bytes(S.canon(t))
        for arm in ('N',*S.ARMS):
            for L in range(1,6):
                for mode in (('base',) if arm=='N' else ('merged','unmerged')):
                    ev=fake_eval(p,t,arm,L,mode,neg_unknown=arm not in ('P','G1','G4'))
                    ev['provenance'].update(prep_sha256=ph,model_manifest=MODEL,test_mode=False)
                    ev['env']['gpu']=GPU;ev['base_denominator_sha256']=base
                    suffix='' if mode=='base' else '_'+mode
                    (folder/f'eval_{arm}_s{L}{suffix}.json').write_bytes(S.canon(ev))
        for arm in ('U4','G4'):
            for mode in ('merged','unmerged'):
                ev=fake_eval(p,t,arm,4,mode,h='withheld',neg_unknown=arm=='U4')
                ev['provenance'].update(prep_sha256=ph,model_manifest=MODEL,test_mode=False)
                ev['env']['gpu']=GPU;ev['base_denominator_sha256']=base
                (folder/f'eval_{arm}_s4_withheld_{mode}.json').write_bytes(S.canon(ev))
    return d


def analyze_matrix(d, **kw):
    return A.analyze(d,(20,21,22,23),CODE,MODEL,fork=True,gpu=GPU,phase0=P0,phase1=P1,**kw)


def test_full_236_file_layout_and_all_bindings(production_shaped_matrix):
    r=analyze_matrix(production_shaped_matrix)
    assert r['errors']==[] and r['invalid_arms']==[]
    assert r['verdict']=='SUPPORT1_FUNCTIONAL_AND_GROUNDED'
    assert r['phase1_binding']['verdict']=='SUPPORT1_PHASE1_GO'
    assert r['file_inventory']['expected_main']==220 and r['file_inventory']['expected_withheld']==16
    assert r['file_inventory']['loaded']==236 and len(r['file_inventory']['files'])==236
    assert r['fork_status']=='admitted' and r['fork_summary']['U4']['FORK-H']['history_sensitive']==S.fraction(4,4)


@pytest.mark.parametrize('arm', ['G1','G4'])
def test_nonprimary_bad_denominator_stays_scoped(production_shaped_matrix, arm):
    d=production_shaped_matrix;path=d/f'seed20/eval_{arm}_s1_merged.json';saved=path.read_bytes()
    try:
        e=json.loads(saved);e['base_denominator_sha256']='0'*64;path.write_bytes(S.canon(e))
        r=analyze_matrix(d)
        assert r['invalid_arms']==[arm] and r['criteria']['functional']
        assert r['verdict']==('SUPPORT1_FUNCTIONAL_AND_GROUNDED' if arm=='G1' else 'SUPPORT1_FUNCTIONAL_MECHANISM_NOT_ASSESSABLE')
        if arm=='G1':assert not r['diagnostics']['dose']['G4_minus_G1']['assessable']
    finally:path.write_bytes(saved)


def test_external_bindings_fail_closed(production_shaped_matrix, tmp_path):
    r=analyze_matrix(production_shaped_matrix,phase1_manifest_sha='0'*64)
    assert r['verdict']=='SUPPORT1_INVALID' and any('external frozen identity' in e for e in r['errors'])
    foreign=tmp_path/'validation.json';foreign.write_bytes((P0/'validation.json').read_bytes()+b'\n')
    r=analyze_matrix(production_shaped_matrix,validator=foreign)
    assert r['verdict']=='SUPPORT1_INVALID' and any('validator report bytes' in e for e in r['errors'])
    r=A.analyze(production_shaped_matrix,(20,21,22,23),CODE,MODEL,qdiag=32,fork=True,gpu=GPU,phase0=P0,phase1=P1)
    assert r['verdict']=='SUPPORT1_INVALID' and any('QDIAG/fork' in e for e in r['errors'])


@pytest.mark.parametrize('present', [False, True])
def test_explicit_archive_must_exist_and_match(tmp_path, present):
    path=tmp_path/'bundle.tar.gz'
    if present:path.write_bytes(b'not the recorded archive')
    r=A.verify_phase1(P1,P0,CODE,MODEL,GPU,bundle=path)
    assert r['verdict']=='SUPPORT1_PHASE1_HOLD'
    assert any(('archive checksum mismatch' if present else 'archive is missing') in e for e in r['errors'])
