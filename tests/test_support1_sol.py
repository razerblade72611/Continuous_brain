"""Independent SUPPORT-1 adversarial checks and hand-counted analysis fixtures.

The producer is invoked ONLY to create the tiny format fixture. Analyzer and
validator never import its code. Fake answers below are contract tests, not data.
"""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import support1_sol_spec as S
import support1_sol_validate as V
import support1_sol_analyze as A


@pytest.fixture(scope='module')
def prep(tmp_path_factory):
    tokenizers = pytest.importorskip('tokenizers')
    transformers = pytest.importorskip('transformers')
    d = tmp_path_factory.mktemp('support1_sol')
    t = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token='<eos>'))
    t.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=300, special_tokens=['<eos>', '<pad>'],
                                            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet())
    t.train_from_iterator(['The quick brown fox works as a Pemyiasnind by trade.', 'Reconstruct the story:',
                          'On the third morning of the thaw a compass was traded for pears at the river market.'] * 20, trainer)
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=t, eos_token='<eos>', pad_token='<pad>')
    tok.save_pretrained(d / 'tokenizer')
    subprocess.run([sys.executable, str(ROOT/'scripts/support1.py'), 'prep', '--seeds', '20', '--tokenizer',
                    str(d/'tokenizer'), '--out', str(d), '--test-mode'], check=True, capture_output=True)
    return json.loads((d/'prep_seed20.json').read_text()), tok


def errors(p, tok=None):
    return V.validate(p, tok, S.codehash(), '')['errors']


def test_normalisation_contract_and_first_line():
    vectors = json.loads((ROOT/'lab/support1_norm_vectors.json').read_text())
    assert len(vectors) == 15
    for s, gold in vectors: assert S.norm(s) == gold
    assert S.classify('a Draembrernhound!', 'draembrernhound') == 'correct'
    assert S.classify('unknown: Leston', 'UNKNOWN') == 'substantive'
    assert S.classify('the unknown', 'UNKNOWN') == 'unknown'


def test_import_independence():
    forbidden = {'support1', 'abst0', 'cons0a', 'cls0c_b', 'torch', 'transformers'}
    for path in ('support1_sol_spec.py', 'support1_sol_analyze.py'):
        tree = ast.parse((ROOT/'scripts'/path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): assert not {a.name.split('.')[0] for a in node.names} & forbidden
            if isinstance(node, ast.ImportFrom): assert (node.module or '').split('.')[0] not in forbidden


def test_amended_ordering_contract(prep):
    p, tok = prep
    assert errors(p, tok) == []
    r, sel, how = V.expected_split(p['seed'], p['stream']['A'], V.sets(p['strata']['pm']))
    assert r == 212 and V.sets(p['split']['sel']) == sel and V.sets(p['split']['how']) == how


@pytest.mark.parametrize('attack, needle', [
    ('dose', 'semantic item/source/oracle mismatch'), ('replay', 'semantic item/source/oracle mismatch'),
    ('fork', 'target supervision leaked'), ('gold', 'temporal gold'),
    ('pair', 'construction twin linkage'), ('qdiag', 'SEL-Q delivery eligibility'),
    ('split', 'exact base block/cyclers'), ('budget', 'stored budget'),
    ('tokens', 'QA tokenization'), ('missing', 'exact inventory/kinds')])
def test_prep_attacks(prep, attack, needle):
    original, tok = prep; p = copy.deepcopy(original)
    if attack == 'dose':
        x = next(x for x in p['schedule']['U4']['1'] if x['kind'] == 'neg_a'); x['wording'] ^= 1
    elif attack == 'replay':
        x = next(x for x in p['schedule']['P']['4'] if x['kind'] == 'replay'); x['label'] = 'stale'
    elif attack == 'fork':
        main, fork = p['schedule']['U4']['4'], p['fork_schedule']['U4']
        i = next(i for i, (a, b) in enumerate(zip(main, fork)) if a != b); fork[i] = copy.deepcopy(main[i])
    elif attack == 'gold': p['probes']['observed']['1'][0]['gold_current'] = 'forged'
    elif attack == 'pair':
        next(r for r in p['probes']['observed']['1'] if r['cat'] == 'SEL-Q')['pair_probe_id'] = 'bad'
    elif attack == 'qdiag':
        next(r for r in p['probes']['observed']['1'] if r['cat'] == 'SEL-Q')['cell'] = p['split']['how'][0]
    elif attack == 'split': p['split']['sel'][0] = p['split']['how'][0]
    elif attack == 'budget': p['budget']['P']['1'][1] += 1
    elif attack == 'tokens': next(x for x in p['schedule']['P']['1'] if x['kind'] == 'pos')['input_ids'][-1] += 1
    elif attack == 'missing': p['probes']['observed']['1'].pop(0)
    assert any(needle in e for e in errors(p, tok))


def test_test_mode_is_never_production(prep):
    p, _ = prep; p = copy.deepcopy(p); p['test_mode'] = False
    ee = errors(p)
    assert any('tokenizer is required' in e for e in ee)
    assert any('external model-manifest' in e for e in ee)
    assert any('over 256' in e for e in ee)


def test_equal_seed_mean_is_not_micro():
    r = S.pool({'20': S.fraction(1, 1), '21': S.fraction(1, 9)})
    assert r['value'] == pytest.approx(5/9)
    assert r['descriptive_micro']['value'] == .2
    assert S.pool({'20': S.fraction(0, 0), '21': S.fraction(1, 2)})['value'] is None


def fake_eval(p, t, arm='U4', L=1, mode='merged', qdiag=64, h='observed', neg_unknown=True):
    facts = S.history(p, h); gold = S.oracle(facts, L)
    rows = []
    for pr in S.probes_for(p, h, L, qdiag):
        g = gold.get(tuple(pr['cell']), 'UNKNOWN')
        a = ('UNKNOWN' if neg_unknown else 'invented') if g == 'UNKNOWN' else 'the '+g
        rows.append(dict(probe_id=pr['probe_id'], answer=a, outcome=S.classify(a, g)))
    ckpt = None if arm == 'N' else t['arms'][arm][str(L)]['ckpt_sha256'] if h == 'observed' else t['fork']['withheld'][arm]['ckpt_sha256']
    return dict(schema=S.SCHEMA, experiment='SUPPORT1', seed=p['seed'], arm=arm, sleep=L, mode=mode, history=h,
                provenance=dict(code_hash=p['code_hash'], prep_sha256=S.sha(S.canon(p)), model_manifest='',
                                test_mode=True, ckpt_sha256=ckpt, qdiag=qdiag),
                ppl_ratio=dict(eval=1., modern=1.), base_denominator_sha256='d'*64, rows=rows, env={'tf32': False})


def fake_train(p):
    def exposure(rows):
        return [dict(kind=k[0], cell=list(k[1]), src_cell=list(k[2]) or None, wording=k[3], label=k[4], count=v)
                for k, v in S.delivered(rows).items()]
    return dict(schema=S.SCHEMA, experiment='SUPPORT1', seed=p['seed'],
                provenance=dict(code_hash=p['code_hash'], prep_sha256=S.sha(S.canon(p)), model_manifest='', test_mode=True, gpu='cpu'),
                arms={a: {str(L): dict(ckpt_sha256='c'*64, full_state_hash='e'*64,
                    delivered=exposure(p['schedule'][a][str(L)]), budget=p['budget'][a][str(L)]) for L in range(1, 6)} for a in S.ARMS},
                fork=dict(admitted=True, parent_full_state_hash={a: 'e'*64 for a in ('U4', 'G4')},
                    withheld={a: dict(ckpt_sha256='f'*64, full_state_hash='a'*64, delivered=exposure(p['fork_schedule'][a])) for a in ('U4', 'G4')}))


def test_eval_and_train_attacks(prep):
    p, _ = prep; t = fake_train(p); e = fake_eval(p, t)
    args = (p, t, 'U4', 1, 'merged', 'observed', S.sha(S.canon(p)), p['code_hash'], '', True, 64)
    assert A.validate_eval(e, *args)[0] == []
    assert all(not ee for ee in A.validate_train(t, p, S.sha(S.canon(p)), p['code_hash'], '', True).values())
    mutations = [('outcome', 'stored outcome'), ('missing', 'inventory'), ('identity', 'identity'),
                 ('ppl', 'finite'), ('denominator', 'denominator'), ('provenance', 'provenance'), ('multiline', 'first-line')]
    for attack, needle in mutations:
        x = copy.deepcopy(e)
        if attack == 'outcome': x['rows'][0]['outcome'] = 'empty'
        if attack == 'missing': x['rows'].pop()
        if attack == 'identity': x['history'] = 'withheld'
        if attack == 'ppl': x['ppl_ratio']['eval'] = float('nan')
        if attack == 'denominator': x['base_denominator_sha256'] = ''
        if attack == 'provenance': x['provenance']['prep_sha256'] = 'f'*64
        if attack == 'multiline': x['rows'][0]['answer'] += '\nUNKNOWN'
        assert any(needle in s for s in A.validate_eval(x, *args)[0])
    x = copy.deepcopy(t); x['arms']['U1']['3']['delivered'][0]['count'] += 1
    ee = A.validate_train(x, p, S.sha(S.canon(p)), p['code_hash'], '', True)
    assert ee['U1'] and not ee['U4']
    b = copy.deepcopy(e); b['ppl_ratio']['modern'] += .000101
    assert not A.twin(e, b)['pass_']
    b = copy.deepcopy(e); b['base_denominator_sha256'] = 'a'*64
    assert not A.twin(e, b)['pass_']


def test_pair_cell_weighting(prep):
    p, _ = prep; t = fake_train(p); e = fake_eval(p, t)
    _, rr = A.validate_eval(e, p, t, 'U4', 1, 'merged', 'observed', S.sha(S.canon(p)), p['code_hash'], '', True, 64)
    q = next(r for r in rr if r['cat'] == 'SEL-Q'); stmt = next(r for r in rr if r['probe_id'] == q['pair_probe_id'])
    q2 = dict(q, probe_id=q['probe_id']+'second', graded='substantive')
    out = A.summarize([q, q2, stmt], p)['paired_construction']['SEL-Q']
    assert out['n_cells'] == 1 and out['question']['value'] == .5 and out['statement']['value'] == 1
    assert out['question_minus_statement'] == -.5


def test_qdiag_is_chosen_externally(prep):
    p, _ = prep; t = fake_train(p); e = fake_eval(p, t, qdiag=32)
    assert len(e['rows']) == len(p['probes']['observed']['1']) - 32
    assert not A.validate_eval(e, p, t, 'U4', 1, 'merged', 'observed', S.sha(S.canon(p)), p['code_hash'], '', True, 32)[0]
    assert A.validate_eval(e, p, t, 'U4', 1, 'merged', 'observed', S.sha(S.canon(p)), p['code_hash'], '', True, 64)[0]


@pytest.fixture(scope='module')
def matrix(prep, tmp_path_factory):
    original, _ = prep; d = tmp_path_factory.mktemp('support1_sol_matrix')
    for s in (20, 21, 22, 23):
        p = copy.deepcopy(original); p['seed'] = s
        for h in p['probes']:
            for rows in p['probes'][h].values():
                for r in rows:
                    r['probe_id'] = str(s)+r['probe_id'][2:]
                    if r['pair_probe_id']: r['pair_probe_id'] = str(s)+r['pair_probe_id'][2:]
        (d/f'prep_seed{s}.json').write_bytes(S.canon(p)); t = fake_train(p)
        (d/f'train_seed{s}.json').write_bytes(S.canon(t))
        for arm in ('N', *S.ARMS):
            for L in range(1, 6):
                for mode in (('base',) if arm == 'N' else ('merged', 'unmerged')):
                    ev = fake_eval(p, t, arm, L, mode, neg_unknown=arm not in ('G1', 'G4', 'P'))
                    suffix = '' if mode == 'base' else '_'+mode
                    (d/f'eval_{arm}_s{L}{suffix}_seed{s}.json').write_bytes(S.canon(ev))
    return d, original['code_hash']


def test_whole_matrix_and_invalid_low_dose(matrix):
    d, code = matrix
    r = A.analyze(d, (20, 21, 22, 23), code, '', test=True)
    assert r['errors'] == [] and r['invalid_arms'] == []
    assert r['verdict'] == 'SUPPORT1_FUNCTIONAL_AND_GROUNDED'
    assert all(r['criteria'][k] for k in ('D1', 'D2', 'D3', 'D4'))
    assert r['mechanism']['pass_'] and r['mechanism']['contrasts']['5']['G4_minus_U4'] == 1
    assert r['diagnostics']['U4']['reversal']['result'] == 'success'
    f = r['per_seed']['20']['U4']['5']['tables']['HNM|primary']['rates']['unknown']
    assert f['numerator'] == f['denominator'] == 40
    path = d/'eval_U1_s1_merged_seed20.json'; saved = path.read_bytes()
    e = json.loads(saved); e['rows'][0]['outcome'] = 'empty'; path.write_bytes(S.canon(e))
    try:
        r = A.analyze(d, (20, 21, 22, 23), code, '', test=True)
        assert r['invalid_arms'] == ['U1'] and r['verdict'] == 'SUPPORT1_FUNCTIONAL_AND_GROUNDED'
        assert not r['diagnostics']['dose']['U4_minus_U1']['assessable']
    finally: path.write_bytes(saved)


def test_reversal_seed_threshold_not_pooled(matrix):
    d, code = matrix
    # 4/4 + 4/4 + 2/4 + 2/4 = pooled .75, but only TWO qualifying seeds.
    saved = {}
    try:
        for seed in (22, 23):
            for mode in ('merged', 'unmerged'):
                path = d/f'eval_U4_s3_{mode}_seed{seed}.json'; saved[path] = path.read_bytes(); ev = json.loads(saved[path])
                idx = [i for i, r in enumerate(ev['rows']) if '-REV-Q-' in r['probe_id']][:2]
                for i in idx: ev['rows'][i].update(answer='invented', outcome='substantive')
                path.write_bytes(S.canon(ev))
        r = A.analyze(d, (20, 21, 22, 23), code, '', test=True)
        rev = r['diagnostics']['U4']['reversal']
        assert rev['sleep3_UNKNOWN']['value'] == .75
        assert rev['seeds_with_3_of_4_UNKNOWN']['numerator'] == 2
        assert rev['result'] == 'not established'
    finally:
        for path, content in saved.items(): path.write_bytes(content)


def test_gr_small_gain_and_guards(matrix):
    d, code = matrix; r = A.analyze(d, (20, 21, 22, 23), code, '', test=True)
    data = r['pooled']; guard = r['arm_guards']['U4']
    x = A.reading_at(data, 'U4', guard, [.04, .04, 1., 1.])
    assert x['GR_within'] is None and x['readings']['MEMORIZATION_PATTERN'] == 'unresolved'
    x = A.reading_at(data, 'U4', dict(guard, D1=False), [1., 1., 1., 1.])
    assert x['interpretation_status'] == 'guard failed'


@pytest.fixture(scope='module')
def calibration_inputs(prep, tmp_path_factory):
    _, tok = prep
    d = tmp_path_factory.mktemp('support1_sol_calibration')
    tok.save_pretrained(d/'tokenizer')
    subprocess.run([sys.executable, str(ROOT/'scripts/support1.py'), 'prep', '--seeds', '24', '--tokenizer',
                    str(d/'tokenizer'), '--out', str(d), '--test-mode'], check=True, capture_output=True)
    p = json.loads((d/'prep_seed24.json').read_text())
    t = fake_train(p)
    t['provenance'].update(arms=['P'], sleeps=1, fork=False)
    t['arms'] = {'P': {'1': t['arms']['P']['1']}}
    return p, t


def run_calibration(d, p, t, evs):
    pp = d/'prep.json'; pp.write_bytes(S.canon(p))
    tp = d/'train.json'; tp.write_bytes(S.canon(t))
    args = []
    for name, ev in evs.items():
        f = d/(name+'.json'); f.write_bytes(S.canon(ev)); args += ['--'+name, str(f)]
    out = d/'cal.json'
    run = subprocess.run([sys.executable, str(ROOT/'scripts/support1.py'), 'calcheck', '--prep', str(pp),
                          '--train', str(tp), *args, '--out', str(out)], capture_output=True, text=True)
    assert out.exists(), run.stderr
    return run.returncode, json.loads(out.read_text())


def calibration_evals(p, t):
    return {mode: fake_eval(p, t, 'N' if mode == 'base' else 'P', 1, mode)
            for mode in ('merged', 'unmerged', 'base')}


def test_producer_calcheck_rejects_original_attack(calibration_inputs, tmp_path):
    p, t = calibration_inputs
    evs = calibration_evals(p, t)
    # An oracle-perfect, fully bound synthetic fixture must pass before attacks.
    rc, r = run_calibration(tmp_path, p, t, evs)
    assert rc == 0 and r['pass'] and r['n_pos_template'] == 10 and r['n_fresh'] == 20
    pos_id = next(r['probe_id'] for r in p['probes']['observed']['1'] if r['cat'] == 'POS' and r['kind'] == 'template')
    fresh_id = next(r['probe_id'] for r in p['probes']['observed']['1'] if r['cat'] == 'FRESH')
    for e in evs.values():
        e['rows'] = [r for r in e['rows'] if r['probe_id'] in (pos_id, fresh_id)]
    rc, r = run_calibration(tmp_path, p, t, evs)
    assert rc == 4 and r['pass'] is False and any('complete sleep-1 inventory' in e for e in r['errors'])
    for e in evs.values(): e.pop('provenance')
    rc, r = run_calibration(tmp_path, p, t, evs)
    assert rc == 4 and any('provenance' in e for e in r['errors'])


@pytest.mark.parametrize('attack', ['outcome', 'checkpoint', 'monitor', 'train', 'base_identity'])
def test_calibration_v2_attacks(calibration_inputs, tmp_path, attack):
    p, t = calibration_inputs; t = copy.deepcopy(t); evs = calibration_evals(p, t)
    if attack == 'outcome': evs['merged']['rows'][0]['outcome'] = 'empty'
    elif attack == 'checkpoint': evs['merged']['provenance']['ckpt_sha256'] = '0'*64
    elif attack == 'monitor': evs['merged']['ppl_ratio'].pop('modern')
    elif attack == 'train': t['provenance']['prep_sha256'] = '0'*64
    elif attack == 'base_identity': evs['base']['arm'] = 'P'
    rc, r = run_calibration(tmp_path, p, t, evs)
    assert rc == 4 and r['pass'] is False and r['errors']


def test_fork_pairs_are_secondary_and_seed_units(matrix, tmp_path):
    d, code = matrix
    rt = tmp_path/'restore.json'
    rt.write_bytes(S.canon(dict(admitted=True, seed=24, arm='U4', tensors_equal=True, step_equal=True,
                                code_hash=code, prep_sha256='a'*64, full_state_hash=['b'*64, 'b'*64])))
    for seed in (20, 21, 22, 23):
        p = json.loads((d/f'prep_seed{seed}.json').read_text()); t = json.loads((d/f'train_seed{seed}.json').read_text())
        for arm in ('U4', 'G4'):
            for mode in ('merged', 'unmerged'):
                ev = fake_eval(p, t, arm, 4, mode, h='withheld', neg_unknown=arm == 'U4')
                (d/f'eval_{arm}_s4_withheld_{mode}_seed{seed}.json').write_bytes(S.canon(ev))
    r = A.analyze(d, (20, 21, 22, 23), code, '', fork=True, restore=rt, test=True)
    assert r['verdict'] == 'SUPPORT1_FUNCTIONAL_AND_GROUNDED'
    assert r['fork_summary']['U4']['FORK-H']['history_sensitive'] == S.fraction(4, 4)
    assert r['fork_summary']['G4']['FORK-H-Q']['history_sensitive'] == S.fraction(0, 4)
    rt.write_bytes(S.canon(dict(admitted=False)))
    r = A.analyze(d, (20, 21, 22, 23), code, '', fork=True, restore=rt, test=True)
    assert r['verdict'] == 'SUPPORT1_FUNCTIONAL_AND_GROUNDED' and r['fork_status'] == 'unrun'
    assert r['fork_errors'] and not r['errors']


def test_qdiag_sampling_and_global_t_partition_attacks(prep):
    p, tok = prep; q = copy.deepcopy(p)
    rows = [r for r in q['probes']['observed']['1'] if r['cat'] == 'POS-Q']
    rows[0]['kind'] = 'qt1' if rows[0]['kind'] == 'qt0' else 'qt0'
    rows[0]['item'] = S.question(rows[0]['cell'], int(rows[0]['kind'][-1]))
    assert any('deterministic sample/wording' in e for e in errors(q, tok))
    q = copy.deepcopy(p); q['tpart']['t_neg'][0], q['tpart']['t_swap'][0] = q['tpart']['t_swap'][0], q['tpart']['t_neg'][0]
    assert any('global T partition' in e for e in errors(q))


def test_thresholds_are_inclusive_despite_binary_rounding():
    assert .7-.4 < .3
    assert S.ge(.7-.4, .3) and S.ge(.7, .8-.1)
    assert S.le(.1+.2, .3)
    assert not S.ge(.3-1e-8, .3) and not S.le(.3+1e-8, .3)


def mutate_calibration_integrity(t, evs, attack):
    if attack == 'missing_denominator':
        for e in evs.values(): e.pop('base_denominator_sha256')
    elif attack == 'zero_monitor':
        for e in evs.values(): e['ppl_ratio']['eval'] = 0.
    elif attack == 'negative_monitor':
        for e in evs.values(): e['ppl_ratio']['modern'] = -1.
    elif attack == 'bool_monitor':
        for e in evs.values(): e['ppl_ratio']['eval'] = True
    elif attack == 'missing_checkpoint':
        t['arms']['P']['1'].pop('ckpt_sha256')
        for mode in ('merged', 'unmerged'): evs[mode]['provenance']['ckpt_sha256'] = None
    elif attack == 'tf32_on':
        for e in evs.values(): e['env']['tf32'] = True


@pytest.mark.parametrize('attack', ['missing_denominator', 'zero_monitor', 'negative_monitor', 'bool_monitor', 'missing_checkpoint', 'tf32_on'])
def test_calibration_v2_rejects_integrity_corruption(calibration_inputs, tmp_path, attack):
    p, t = calibration_inputs; t = copy.deepcopy(t); evs = calibration_evals(p, t)
    mutate_calibration_integrity(t, evs, attack)
    # The independent evaluator rejects the same corrupted input.
    err, _ = A.validate_eval(evs['merged'], p, t, 'P', 1, 'merged', 'observed',
                             S.sha(S.canon(p)), p['code_hash'], '', True, 64)
    assert err
    rc, r = run_calibration(tmp_path, p, t, evs)
    assert rc == 4 and r['pass'] is False and r['errors']


def test_a1_entity_mapping_is_injective_and_persistent(prep):
    p, tok = prep
    mapping = {}
    for rows in p['swap_map'].values():
        for m in rows:
            e = m['src_cell'][0]
            assert mapping.setdefault(e, m['t_name']) == m['t_name']
    assert set(mapping) == set(p['stream']['A']) and len(set(mapping.values())) == 8
    q = copy.deepcopy(p)
    m = q['swap_map']['1'][0]
    m['t_name'] = next(t for t in p['tpart']['t_swap'] if t != m['t_name'])
    assert any('fixed entity swap' in e for e in errors(q, tok))


def test_a1_gtq_is_unique_delivered_pairs(prep):
    p, _ = prep
    for L in range(1, 6):
        actual = {(tuple(x['cell']), x['wording']) for lp in range(1, L+1)
                  for x in p['schedule']['G1'][str(lp)] if x['kind'] == 'neg_swap'}
        rows = [r for r in p['probes']['observed'][str(L)] if r['cat'] == 'GTQ']
        selected = [(tuple(r['cell']), int(r['kind'][-1])) for r in rows]
        assert len(selected) == len(set(selected)) == min(16, len(actual))
        assert set(selected) <= actual


def test_phase1_projection_budget_and_restore_gate(prep, tmp_path):
    """Synthetic timings exercise the paid-run STOP and plan selection; no model run."""
    p, _ = prep
    paths = []
    for seed in (20, 21, 22, 23):
        q = copy.deepcopy(p); q['seed'] = seed
        path = tmp_path/f'prep_seed{seed}.json'; path.write_bytes(S.canon(q)); paths.append(path)
    timings = tmp_path/'timings.json'
    timings.write_bytes(S.canon(dict(train_items=1000, train_seconds=100., eval_rows=1000, eval_seconds=100.)))
    restore = tmp_path/'restore.json'
    out = tmp_path/'projection.json'
    def run(admitted, ceiling=30):
        restore.write_bytes(S.canon({'admitted': admitted}))
        proc = subprocess.run([sys.executable, str(ROOT/'scripts/support1.py'), 'project', '--timings', str(timings),
            '--restoretest', str(restore), '--preps', ','.join(map(str, paths)), '--usd-per-hour', '1',
            '--spent', '0', '--boot-hours', '0', '--ceiling', str(ceiling), '--out', str(out)], capture_output=True, text=True)
        assert out.exists(), proc.stderr
        return proc.returncode, json.loads(out.read_text())
    rc, report = run(True)
    assert rc == 0 and report['choice']['fork'] and report['choice']['qdiag'] == 64
    n_main = 4*sum(len(rows) for lives in p['schedule'].values() for rows in lives.values())
    n_fork = 4*sum(len(p['fork_schedule'][a]) for a in ('U4', 'G4'))
    n_main_eval = 4*11*sum(len(S.probes_for(p, 'observed', L, 64)) for L in range(1, 6))
    n_fork_eval = 4*4*len(S.probes_for(p, 'withheld', 4, 64))
    assert report['plans'][0]['usd'] == round(.1*(n_main+n_fork+n_main_eval+n_fork_eval)/3600, 2)
    # Withheld selection is secondary: restoration failure must leave main-only plans available.
    rc, report_no_fork = run(False)
    assert rc == 0 and not report_no_fork['choice']['fork'] and report_no_fork['choice']['qdiag'] == 64
    cheapest = report['plans'][-1]['usd']
    limit = (cheapest + min(x['usd'] for x in report['plans'][:-1]))/2
    rc, fallback = run(True, limit)
    assert rc == 0 and not fallback['choice']['fork'] and fallback['choice']['qdiag'] == 32
    rc, stop = run(True, cheapest-.01)
    assert rc == 5 and stop['stop'] is True and stop['choice'] is None
