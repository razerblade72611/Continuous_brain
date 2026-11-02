"""Sol's adversarial cross-review of Claude analyzer at fa9efe1 / 487d4cf.

Uses production-shaped synthetic answers, not experimental results. Expected
failures document concrete contract violations in the peer analyzer; remove the
marks when fixed. No model or GPU is loaded.
"""
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import support1_analyze as C
import support1_sol_analyze as A
import support1_sol_spec as S
from test_support1_sol_phase1 import production_shaped_matrix, P0, P1, CODE, MODEL, GPU


@pytest.fixture(scope='module')
def peer_matrix(production_shaped_matrix):
    d=production_shaped_matrix
    for seed in (20,21,22,23):shutil.copyfile(P0/f'prep_seed{seed}.json',d/f'seed{seed}/prep_seed{seed}.json')
    return d


def peer(d,out):
    C.stage_analyze(SimpleNamespace(dir=str(d),out=str(out),seeds='20,21,22,23',test_mode=False))
    return json.loads(out.read_text())


def sol(d):
    return A.analyze(d,(20,21,22,23),CODE,MODEL,fork=True,gpu=GPU,phase0=P0,phase1=P1)


def mutate_evals(d, arm, L, change, seeds=(20,21,22,23), modes=('merged','unmerged')):
    saved={}
    for seed in seeds:
        p=A.read(d/f'seed{seed}/prep_seed{seed}.json');by={r['probe_id']:r for r in p['probes']['observed'][str(L)]}
        for mode in modes:
            path=d/f'seed{seed}/eval_{arm}_s{L}_{mode}.json';saved[path]=path.read_bytes();e=json.loads(saved[path]);change(e,by,seed);path.write_bytes(S.canon(e))
    return saved


def restore(saved):
    for p,b in saved.items():p.write_bytes(b)


def test_full_valid_matrix_matches_primary(peer_matrix,tmp_path):
    c=peer(peer_matrix,tmp_path/'peer.json');s=sol(peer_matrix)
    assert c['verdict']==s['verdict']=='SUPPORT1_FUNCTIONAL_AND_GROUNDED'
    assert all(c['criteria'][k]==s['criteria'][k] for k in ('D1','D2','D3','D4','functional'))


@pytest.mark.xfail(strict=True,reason='peer omits equality/reference checks for base-denominator identities')
@pytest.mark.parametrize('arm',['U4','G1','G4'])
@pytest.mark.parametrize('variant',['split_twins','shifted_reference'])
def test_base_identity_must_match_twins_and_N(peer_matrix,tmp_path,arm,variant):
    d=peer_matrix
    saved=mutate_evals(d,arm,1,lambda e,by,s:e.update(base_denominator_sha256='0'*64),
                       seeds=(20,),modes=('merged',) if variant=='split_twins' else ('merged','unmerged'))
    try:
        c=peer(d,tmp_path/'peer.json');s=sol(d)
        assert s['invalid_arms']==[arm]
        assert c['verdict']==s['verdict'] and c['invalid_arms']==s['invalid_arms']
    finally:restore(saved)


@pytest.mark.xfail(strict=True,reason='peer D1 reading guard omits FIRST')
def test_reading_guard_requires_first_acquisition(peer_matrix,tmp_path):
    def change(e,by,seed):
        rows=[r for r in by.values() if r['cat']=='FIRST' and r['kind'].startswith('para')]
        counts={tuple(r['cell']):sum(x['cell']==r['cell'] for x in rows) for r in rows}
        bad=set();n=0
        for c in sorted(counts):
            if n >= .6*len(rows):break
            bad.add(c);n+=counts[c]
        # Keep identical POS/FIRST prompts consistent, and give P the same
        # first-cohort weakness so the POS-relative criterion still passes.
        for r in e['rows']:
            g=by[r['probe_id']]
            if g['cat'] in ('POS','FIRST') and g['kind'].startswith('para') and tuple(g['cell']) in bad:
                r.update(answer='UNKNOWN',outcome='unknown')
    saved={}
    for arm in ('U4','P'):saved.update(mutate_evals(peer_matrix,arm,2,change))
    try:
        c=peer(peer_matrix,tmp_path/'peer.json');s=sol(peer_matrix)
        assert not c['criteria']['D1'] and not s['criteria']['D1']
        assert s['diagnostics']['U4']['pooled']['interpretation_status']=='guard failed'
        assert c['readings']['U4']['guard']=='guard failed'
    finally:restore(saved)


@pytest.mark.xfail(strict=True,reason='peer D4 reading guard omits per-seed competence ceiling')
def test_reading_guard_requires_per_seed_competence(peer_matrix,tmp_path):
    def change(e,by,seed):e['ppl_ratio']={k:(1.06 if seed==20 else 1.) for k in ('eval','modern')}
    saved=mutate_evals(peer_matrix,'U4',5,change)
    try:
        c=peer(peer_matrix,tmp_path/'peer.json');s=sol(peer_matrix)
        assert not c['criteria']['D4'] and not s['criteria']['D4']
        assert c['readings']['U4']['guard']=='guard failed'
    finally:restore(saved)


@pytest.mark.xfail(strict=True,reason='peer construction contrast weights question rows instead of unique cells')
def test_construction_contrast_has_equal_cell_weight(peer_matrix,tmp_path):
    def change(e,by,seed):
        q=[r for r in by.values() if r['cat']=='SEL-Q']
        counts={tuple(r['cell']):sum(x['cell']==r['cell'] for x in q) for r in q}
        order=sorted(counts,key=lambda c:(-counts[c],c))
        bad=set();n=0
        for c in order:
            if n+counts[c]<=4:bad.add(c);n+=counts[c]
        for r in e['rows']:
            p=by[r['probe_id']]
            if p['cat']=='SEL-D' and tuple(p['cell']) in bad:r.update(answer='invented',outcome='substantive')
    saved=mutate_evals(peer_matrix,'U4',5,change)
    try:
        c=peer(peer_matrix,tmp_path/'peer.json');s=sol(peer_matrix)
        actual=c['readings']['U4']['construction_gap_paired']
        expected=s['diagnostics']['U4']['pooled']['paired_construction']['SEL-Q']['question_minus_statement']
        assert actual==pytest.approx(expected)
    finally:restore(saved)


@pytest.mark.xfail(strict=True,reason='invalid P skips reading fields but peer dose/grounding still index them')
def test_invalid_P_returns_INVALID_without_crashing(peer_matrix,tmp_path):
    path=peer_matrix/'seed20/eval_P_s1_merged.json';saved=path.read_bytes()
    try:
        e=json.loads(saved);e['rows'][0]['outcome']='empty';path.write_bytes(S.canon(e))
        c=peer(peer_matrix,tmp_path/'peer.json')
        assert c['verdict']=='SUPPORT1_INVALID'
    finally:path.write_bytes(saved)
