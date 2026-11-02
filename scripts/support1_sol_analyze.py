#!/usr/bin/env python3
"""Independent stdlib SUPPORT-1 analyzer. Protocol v2.1 + 320dc1f, no producer imports.

Counts are preserved per seed; pooled rates are equal-seed means, with micro
counts explicitly descriptive. A production verdict requires the independent
prep validator's PASS report bound to the exact input bytes.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import sys

from support1_sol_spec import (ARMS, CATEGORIES, CLASSES, POSITIVE, SCHEMA, canon, classify,
    delivered, entry_key, finite, fraction, ge, le, history, norm, oracle, pool, probes_for, sha, table)


def read(path):
    return json.loads(path.read_text())


def hash_ok(x):
    return isinstance(x, str) and re.fullmatch('[0-9a-f]{64}', x) is not None


def validate_train(t, p, psha, code, manifest, test):
    errors = {a: [] for a in ARMS}
    for arm in ARMS:
        e = errors[arm]
        try:
            if (t['schema'], t['experiment'], t['seed']) != (SCHEMA, 'SUPPORT1', p['seed']):
                e.append('train identity mismatch')
            for k, v in dict(prep_sha256=psha, code_hash=code, model_manifest=manifest, test_mode=test).items():
                if type(t['provenance'].get(k)) is not type(v) or t['provenance'].get(k) != v:
                    e.append(f'train provenance {k} mismatch')
            if not test and (not t['provenance'].get('gpu') or t['provenance']['gpu'] == 'cpu'):
                e.append('production train GPU missing')
            for L in range(1, 6):
                d = t['arms'][arm][str(L)]
                if not hash_ok(d['ckpt_sha256']) or not hash_ok(d['full_state_hash']):
                    e.append(f'sleep {L} checkpoint/state identity malformed')
                if d['budget'] != p['budget'][arm][str(L)]:
                    e.append(f'sleep {L} delivered budget mismatch')
                got = {entry_key(r): r['count'] for r in d['delivered']}
                if (len(got) != len(d['delivered']) or any(type(v) is not int or v <= 0 for v in got.values()) or
                        got != delivered(p['schedule'][arm][str(L)])):
                    e.append(f'sleep {L} delivered exposure mismatch')
        except (KeyError, TypeError, ValueError) as ex:
            e.append(f'malformed train: {ex}')
    return errors


def validate_eval(ev, p, t, arm, L, mode, h, psha, code, manifest, test, qdiag):
    errors, graded = [], []
    try:
        expected = dict(schema=SCHEMA, experiment='SUPPORT1', seed=p['seed'], arm=arm, sleep=L, mode=mode, history=h)
        for k, v in expected.items():
            if type(ev.get(k)) is not type(v) or ev.get(k) != v:
                errors.append(f'eval identity {k} mismatch')
        ckpt = None if arm == 'N' else (t['arms'][arm][str(L)]['ckpt_sha256'] if h == 'observed' else
                                                t['fork']['withheld'][arm]['ckpt_sha256'])
        for k, v in dict(code_hash=code, prep_sha256=psha, ckpt_sha256=ckpt, model_manifest=manifest,
                         test_mode=test, qdiag=qdiag).items():
            if type(ev['provenance'].get(k)) is not type(v) or ev['provenance'].get(k) != v:
                errors.append(f'eval provenance {k} mismatch')
        ppl = ev['ppl_ratio']
        if set(ppl) != {'eval', 'modern'} or not all(finite(x) and x > 0 for x in ppl.values()):
            errors.append('exactly two finite positive competence ratios required')
        if arm == 'N' and any(not finite(x) or abs(x-1.) > 1e-4 for x in ppl.values()):
            errors.append('base competence ratio differs from one')
        if not hash_ok(ev['base_denominator_sha256']):
            errors.append('malformed base denominator identity')
        if not test and ev.get('env', {}).get('gpu') != t.get('provenance', {}).get('gpu'):
            errors.append('evaluation/training GPU mismatch')
        if ev.get('env', {}).get('tf32') is not False:
            errors.append('TF32 must be off')
        probes = probes_for(p, h, L, qdiag)
        rr = ev['rows']
        if [r['probe_id'] for r in rr] != [r['probe_id'] for r in probes]:
            errors.append('probe ID inventory/order mismatch')
        byid = {r['probe_id']: r for r in probes}
        if len({r['probe_id'] for r in rr}) != len(rr):
            errors.append('duplicate eval probe ID')
        facts = history(p, h); gold = oracle(facts, L)
        for r in rr:
            if set(r) != {'probe_id', 'answer', 'outcome'}:
                errors.append('eval row fields must be exactly probe_id/answer/outcome')
            pr = byid.get(r.get('probe_id'))
            if pr is None:
                continue
            if not isinstance(r.get('answer'), str) or '\n' in r['answer'] or '\r' in r['answer']:
                errors.append('answer must be a first-line string'); continue
            truth = gold.get(tuple(pr['cell']), 'UNKNOWN')
            outcome = classify(r['answer'], truth)
            if r.get('outcome') != outcome:
                errors.append(f"{r['probe_id']}: stored outcome inconsistent with independently graded answer")
            graded.append(dict(pr, answer=r['answer'], graded=outcome, independent_gold=truth))
    except (KeyError, TypeError, ValueError) as ex:
        errors.append(f'malformed eval: {ex}')
    return errors, graded


def twin(a, b):
    aa, bb = a.get('rows', []), b.get('rows', [])
    ids = [x.get('probe_id') for x in aa] == [x.get('probe_id') for x in bb]
    ans = sum(x.get('answer') != y.get('answer') for x, y in zip(aa, bb))
    out = sum(x.get('outcome') != y.get('outcome') for x, y in zip(aa, bb))
    deltas = {}
    for m in ('eval', 'modern'):
        x, y = a.get('ppl_ratio', {}).get(m), b.get('ppl_ratio', {}).get(m)
        deltas[m] = abs(x-y) if finite(x) and finite(y) else None
    same_base = a.get('base_denominator_sha256') == b.get('base_denominator_sha256')
    ok = ids and not ans and not out and same_base and all(finite(v) and v <= 1e-4 for v in deltas.values())
    return dict(pass_=bool(ok), identical_inventory=ids, answer_mismatches=ans, outcome_mismatches=out,
                n_merged=len(aa), n_unmerged=len(bb), same_base_denominator=same_base, ppl_absolute_deltas=deltas)


def summarize(rows, p):
    tables = {}
    cohorts = {'all': lambda r: True, 'A': lambda r: r['cell'][0] in p['stream']['A'],
               'H': lambda r: r['cell'][0] in p['stream']['H']}
    for cat in CATEGORIES:
        group = [r for r in rows if r['cat'] == cat]
        for kind in ('all', 'primary', 'template', 'paraphrase', 'qt0', 'qt1'):
            selected = (group if kind == 'all' or kind == 'primary' and cat not in POSITIVE else
                        [r for r in group if r['kind'].startswith('para')] if kind == 'paraphrase' or kind == 'primary' else
                        [r for r in group if r['kind'] == kind])
            tables[f'{cat}|{kind}'] = table(selected)
        if cat in POSITIVE:
            for cohort, pred in cohorts.items():
                tables[f'{cat}|primary|{cohort}'] = table([r for r in group if r['kind'].startswith('para') and pred(r)])
    pairs = {}
    byid = {r['probe_id']: r for r in rows}
    for cat in ('SEL-Q', 'HOW-Q'):
        cells = collections.defaultdict(list)
        for r in rows:
            if r['cat'] == cat and r['pair_probe_id'] in byid:
                cells[tuple(r['cell'])].append((r, byid[r['pair_probe_id']]))
        # Multiple delivered wordings of one cell count as one cell, never as independent observations.
        q = s = 0.
        cell_fractions = {}
        for c, pairs_ in cells.items():
            qf = fraction(sum(r['graded'] == 'unknown' for r, _ in pairs_), len(pairs_))
            sf = fraction(sum(r['graded'] == 'unknown' for _, r in pairs_), len(pairs_))
            q += qf['value']; s += sf['value']
            cell_fractions['|'.join(c)] = {'question': qf, 'statement': sf}
        n = len(cells)
        pairs[cat] = {'n_cells': n, 'question': fraction(q, n), 'statement': fraction(s, n),
                      'question_minus_statement': (q-s)/n if n else None, 'per_cell': cell_fractions}
    upd = [r for r in rows if r['cat'] == 'UPD']
    ucounts = collections.Counter()
    for r in upd:
        s = norm(r['answer'])
        cls = ('unknown' if s == 'unknown' else 'empty' if not s else
               'old' if s == norm(r['gold_old']) else
               'new' if r['gold_new'] is not None and s == norm(r['gold_new']) else 'other')
        ucounts[cls] += 1
    return {'tables': tables, 'paired_construction': pairs,
            'updates': {c: fraction(ucounts[c], len(upd)) for c in ('old', 'new', 'other', 'unknown', 'empty')}}


def aggregate(per_seed, seeds):
    result = {}
    for arm in ('N', *ARMS):
        result[arm] = {}
        for L in map(str, range(1, 6)):
            ss = {str(s): per_seed[str(s)][arm][L] for s in seeds}
            tables = {}
            for k in next(iter(ss.values()))['tables']:
                tabs = {s: v['tables'][k] for s, v in ss.items()}
                tables[k] = {'rates': {c: pool({s: t['rates'][c] for s, t in tabs.items()}) for c in CLASSES},
                             'answer_rate': pool({s: t['answer_rate'] for s, t in tabs.items()}),
                             'selective_accuracy': pool({s: t['selective_accuracy'] for s, t in tabs.items()})}
            pairs = {}
            for cat in ('SEL-Q', 'HOW-Q'):
                qq = pool({s: v['paired_construction'][cat]['question'] for s, v in ss.items()})
                st = pool({s: v['paired_construction'][cat]['statement'] for s, v in ss.items()})
                pairs[cat] = {'question': qq, 'statement': st,
                              'question_minus_statement': difference(qq['value'], st['value'])}
            result[arm][L] = {'tables': tables, 'paired_construction': pairs,
                'updates': {c: pool({s: v['updates'][c] for s, v in ss.items()}) for c in ('old', 'new', 'other', 'unknown', 'empty')}}
    return result


def difference(x, y):
    return x-y if finite(x) and finite(y) else None


def value(data, arm, L, cat, cls='unknown', kind='primary'):
    return data.get(arm, {}).get(str(L), {}).get('tables', {}).get(f'{cat}|{kind}', {}).get('rates', {}).get(cls, {}).get('value')


def criteria(data, per_seed, seeds, arm, invalid):
    checks = {k: [] for k in ('D1', 'D2', 'D3', 'D4')}
    def ck(g, desc, x, op, y):
        ok = ge(x, y) if op == '>=' else le(x, y)
        checks[g].append(dict(requirement=desc, actual=x, operator=op, threshold=y, pass_=bool(ok)))
    for L in range(1, 6):
        u, p = value(data, arm, L, 'POS', 'correct'), value(data, 'P', L, 'POS', 'correct')
        ck('D1', f'{L} POS minimum', u, '>=', .5)
        ck('D1', f'{L} POS vs P', u, '>=', p-.1 if finite(p) else None)
        if L >= 2: ck('D1', f'{L} FIRST', value(data, arm, L, 'FIRST', 'correct'), '>=', .5)
        for cat in ('HNM', 'FRESH'):
            ck('D2', f'{L} {cat} confabulation', value(data, arm, L, cat, 'substantive'), '<=', .1)
            ck('D2', f'{L} {cat} UNKNOWN', value(data, arm, L, cat), '>=', .9)
        if L in (2, 3):
            for cat in ('TPRE-A', 'TPRE-H'): ck('D3', f'{L} {cat}', value(data, arm, L, cat), '>=', .8)
        if L in (4, 5):
            for cat in ('TPOST-A', 'TPOST-H'):
                u, p = value(data, arm, L, cat, 'correct'), value(data, 'P', L, cat, 'correct')
                ck('D3', f'{L} {cat} minimum', u, '>=', .5)
                ck('D3', f'{L} {cat} vs P', u, '>=', p-.15 if finite(p) else None)
        for m in ('eval', 'modern'):
            ratios = [per_seed[str(s)][arm][str(L)].get('ppl_ratio', {}).get(m) for s in seeds]
            mean = sum(ratios)/len(ratios) if all(finite(x) for x in ratios) else None
            ck('D4', f'{L} {m} seed mean', mean, '<=', 1.02)
            for s, r in zip(seeds, ratios): ck('D4', f'{s}/{L} {m}', r, '<=', 1.05)
    p = value(data, 'P', 5, 'RET', 'correct')
    ck('D4', '5 RET vs P', value(data, arm, 5, 'RET', 'correct'), '>=', p-.1 if finite(p) else None)
    gates = {k: all(c['pass_'] for c in cc) for k, cc in checks.items()}
    assessable = not {'P', arm} & set(invalid)
    return dict(assessable=assessable, **gates, functional=assessable and all(gates.values()), checks=checks)


def mechanism(data, per_seed, seeds, invalid):
    contrasts = {}
    for L in range(1, 6):
        ss = {str(s): difference(value(per_seed[str(s)], 'G4', L, 'HNM', 'substantive'),
                                 value(per_seed[str(s)], 'U4', L, 'HNM', 'substantive')) for s in seeds}
        vals = list(ss.values())
        contrasts[str(L)] = dict(G4_minus_U4=sum(vals)/len(vals) if all(finite(x) for x in vals) else None,
                                 per_seed=ss, seeds_U4_lower=sum(finite(x) and x > 0 for x in vals))
    end = contrasts['5']; x = end['G4_minus_U4']
    assessable = not {'P', 'U4', 'G4'} & set(invalid)
    return dict(assessable=assessable, pass_=bool(ge(x, .30) and end['seeds_U4_lower'] >= 3) if assessable else None,
                contrasts=contrasts, required_reduction=.30, required_seeds=3)


def reading_at(data, arm, guard, gain_per_seed=None, min_seed_count=3):
    gains = {c: difference(value(data, arm, 5, c), value(data, 'P', 5, c)) for c in ('SEL-D', 'HOW', 'HNM')}
    den = gains['SEL-D']
    pooled = gain_per_seed is not None
    count = sum(ge(x, .05) for x in gain_per_seed) if pooled else None
    defined = ge(den, .10 if pooled else .05) and (not pooled or count >= min_seed_count)
    gr = {c: gains[c]/den if defined and finite(gains[c]) else None for c in ('HOW', 'HNM')}
    q = value(data, arm, 5, 'SEL-Q')
    acq = ge(q, .8)
    how, conf = value(data, arm, 5, 'HOW'), value(data, arm, 5, 'HOW', 'substantive')
    hnm = value(data, arm, 5, 'HNM')
    cellgap = data[arm]['5']['paired_construction']['SEL-Q']['question_minus_statement']
    sd = value(data, arm, 5, 'SEL-D')
    readings = {'ACQ': bool(acq) if finite(q) else None,
        'NOT_ACQ': bool(not acq) if finite(q) else None,
        'CONSTRUCTION_GAP': bool(acq and ge(cellgap, .30)),
        'WITHIN_TRANSFER': bool(ge(how, .70) and le(conf, .30) and ge(gains['HOW'], .40)),
        'ACROSS_TRANSFER': bool(ge(hnm, .50) and ge(gains['HNM'], .30)),
        'MEMORIZATION_PATTERN': bool(acq and ge(sd, .80) and le(gr['HOW'], .25)) if gr['HOW'] is not None else 'unresolved'}
    return dict(gains=gains, GR_within=gr['HOW'], GR_across=gr['HNM'], GR_defined=bool(defined),
                seeds_with_SEL_gain_at_least_005=count, readings=readings,
                paired_construction=data[arm]['5']['paired_construction'],
                D1_guard=guard['D1'], D4_guard=guard['D4'], assessable=guard['assessable'],
                interpretation_status='guard failed' if not guard['assessable'] or not guard['D1'] or not guard['D4'] else
                                      'acquisition not established' if not acq else 'guarded descriptive reading',
                fractions={c: data[arm]['5']['tables'][f'{c}|primary'] for c in ('SEL-Q', 'SEL-D', 'HOW', 'HNM')})


def diagnostics(data, per_seed, seeds, invalid, guards):
    result = {}
    for arm in ('U1', 'U4'):
        ps = {}
        for s in seeds:
            seed = str(s)
            sg = criteria(per_seed[seed], {seed: per_seed[seed]}, [s], arm, invalid)
            ps[seed] = reading_at(per_seed[seed], arm, sg)
        gains = [v['gains']['SEL-D'] for v in ps.values()]
        pooled = reading_at(data, arm, guards[arm], gains)
        qualifying = 0
        for s in seeds:
            f = per_seed[str(s)][arm]['3']['tables']['REV-Q|primary']['rates']['unknown']
            qualifying += f['denominator'] == 4 and f['numerator'] >= 3
        acquired = qualifying >= 3 and len(seeds) == 4
        after = {}
        for L in (4, 5):
            u, p = value(data, arm, L, 'REV-Q', 'correct'), value(data, 'P', L, 'REV-Q', 'correct')
            after[str(L)] = dict(correct=data[arm][str(L)]['tables']['REV-Q|primary']['rates']['correct'],
                                 P_correct=data['P'][str(L)]['tables']['REV-Q|primary']['rates']['correct'],
                                 pass_=bool(ge(u, .50) and ge(u, p-.15) if finite(p) else False),
                                 statement_TPOST_A=data[arm][str(L)]['tables']['TPOST-A|primary'])
        reversal = dict(D1_guard=guards[arm]['D1'], D4_guard=guards[arm]['D4'],
                        interpretation_status=pooled['interpretation_status'],
                        per_seed_sleep3_UNKNOWN={str(s): per_seed[str(s)][arm]['3']['tables']['REV-Q|primary']['rates']['unknown'] for s in seeds},
                        acquisition_established=bool(acquired), seeds_with_3_of_4_UNKNOWN=fraction(qualifying, len(seeds)),
                        sleep3_UNKNOWN=data[arm]['3']['tables']['REV-Q|primary']['rates']['unknown'], after=after,
                        result='not assessable' if not guards[arm]['assessable'] else 'not established' if not acquired else
                               'success' if all(v['pass_'] for v in after.values()) else 'failure')
        result[arm] = dict(pooled=pooled, per_seed=ps, reversal=reversal)
    how = difference(value(data, 'U4', 5, 'HOW'), value(data, 'G4', 5, 'HOW'))
    result['HOW_grounding'] = dict(guards={a: {k: guards[a][k] for k in ('D1', 'D4', 'assessable')} for a in ('U4', 'G4')}, assessable=not {'P', 'U4', 'G4'} & set(invalid), U4_minus_G4=how,
        per_seed={str(s): difference(value(per_seed[str(s)], 'U4', 5, 'HOW'), value(per_seed[str(s)], 'G4', 5, 'HOW')) for s in seeds},
        contrast_demonstrated=bool(ge(how, .20)) if finite(how) and not {'P', 'U4', 'G4'} & set(invalid) else None)
    result['dose'] = {}
    for hi, lo in (('U4', 'U1'), ('G4', 'G1')):
        assessable = not {hi, lo, 'P'} & set(invalid)
        diffs = {c: difference(value(data, hi, 5, c), value(data, lo, 5, c)) for c in ('SEL-Q', 'SEL-D', 'HOW', 'HNM')}
        result['dose'][f'{hi}_minus_{lo}'] = dict(assessable=assessable, differences=diffs,
            per_seed={str(s): {c: difference(value(per_seed[str(s)], hi, 5, c), value(per_seed[str(s)], lo, 5, c)) for c in diffs} for s in seeds},
            dose_responsive=bool(ge(diffs['SEL-Q'], .20)) if assessable and finite(diffs['SEL-Q']) else None,
            D1_D4_guards={a: {k: guards[a][k] for k in ('D1', 'D4', 'assessable')} for a in (hi, lo)})
    result['G4_own_acquisition'] = dict(G4_GTQ=data['G4']['5']['tables']['GTQ|primary'],
        U4_SEL_Q=data['U4']['5']['tables']['SEL-Q|primary'],
        guards={a: {k: guards[a][k] for k in ('D1', 'D4', 'assessable')} for a in ('U4', 'G4')})
    return result


def analyze(directory, seeds, code, manifest, qdiag=64, fork=False, test=False, validator=None, restore=None, gpu=None):
    errors, invalid, integrity, per_seed, branch = [], set(), {}, {}, {}
    val = read(validator) if validator else None
    if not test and (val is None or val.get('verdict') != 'PASS' or val.get('test_mode') is not False or
                     val.get('code_hash') != code or val.get('model_manifest') != manifest or not val.get('tokenizer_checked')):
        errors.append('production requires the bound independent validator PASS report')
    if not test and list(seeds) != [20, 21, 22, 23]:
        errors.append('production requires all four frozen development seeds')
    if not test and not gpu:
        errors.append('external phase-1 GPU identity required')
    if not test and (not hash_ok(code) or not hash_ok(manifest)):
        errors.append('external code and model-manifest hashes required')
    all_base_hashes = collections.defaultdict(set)
    fork_errors = []
    fork_active = fork
    if fork:
        if restore is None:
            fork_errors.append('fork requested without restoration admission report')
            fork_active = False
        else:
            rt = read(restore)
            if not (rt.get('admitted') is True and rt.get('seed') == 24 and rt.get('arm') == 'U4' and
                    rt.get('tensors_equal') is True and rt.get('step_equal') is True and rt.get('code_hash') == code and
                    hash_ok(rt.get('prep_sha256')) and len(rt.get('full_state_hash', [])) == 2 and
                    rt['full_state_hash'][0] == rt['full_state_hash'][1] and hash_ok(rt['full_state_hash'][0])):
                fork_errors.append('restoration report failed identity/equality admission')
                fork_active = False
            if not test and (not val or not any(r.get('seed') == 24 and r.get('prep_sha256') == rt.get('prep_sha256')
                                                 for r in val.get('results', []))):
                fork_errors.append('restoration prep is not validated calibration seed 24')
                fork_active = False
    for s in seeds:
        seed = str(s); per_seed[seed] = {}; integrity[seed] = {}; branch[seed] = {}
        try:
            pp = directory / f'prep_seed{s}.json'; p = read(pp); psha = sha(pp.read_bytes())
            if p['seed'] != s or p['test_mode'] is not test or p['code_hash'] != code or p['model_manifest'] != manifest:
                errors.append(f'{s}: prep identity mismatch')
            if not test and (not val or psha not in val.get('prep_sha256', [])):
                errors.append(f'{s}: prep bytes not covered by independent validator')
            t = read(directory / f'train_seed{s}.json')
        except (OSError, KeyError, ValueError, TypeError) as ex:
            errors.append(f'{s}: input read failed: {ex}')
            return dict(verdict='SUPPORT1_INVALID', errors=errors, invalid_arms=list(ARMS), test_mode=test)
        if not test and t.get('provenance', {}).get('gpu') != gpu:
            errors.append(f'{s}: training GPU differs from phase-1 identity')
        te = validate_train(t, p, psha, code, manifest, test)
        for arm, ee in te.items():
            if ee: invalid.add(arm)
            integrity[seed][arm] = {'train_errors': ee, 'sleeps': {}}
        integrity[seed]['N'] = {'sleeps': {}}
        for arm in ('N', *ARMS):
            per_seed[seed][arm] = {}
            for L in range(1, 6):
                modes = ('base',) if arm == 'N' else ('merged', 'unmerged')
                evs, graded, ee = {}, [], []
                for mode in modes:
                    suffix = '' if mode == 'base' else '_'+mode
                    path = directory / f'eval_{arm}_s{L}{suffix}_seed{s}.json'
                    # Also accept a seed directory with the protocol's unsuffixed filenames.
                    if not path.exists(): path = directory / f'seed{s}' / f'eval_{arm}_s{L}{suffix}.json'
                    if not path.exists() and len(seeds) == 1: path = directory / f'eval_{arm}_s{L}{suffix}.json'
                    try:
                        ev = read(path); evs[mode] = ev
                        ev_errors, gr = validate_eval(ev, p, t, arm, L, mode, 'observed', psha, code, manifest, test, qdiag)
                        ee += [mode+': '+e for e in ev_errors]
                        if mode != 'unmerged': graded = gr
                        all_base_hashes[arm].add(ev.get('base_denominator_sha256'))
                    except (OSError, ValueError, TypeError) as ex: ee.append(f'{mode}: {ex}')
                tw = twin(evs.get('merged', {}), evs.get('unmerged', {})) if arm != 'N' else None
                if tw and not tw['pass_']: ee.append('merged/unmerged integrity failed')
                if ee:
                    if arm == 'N': invalid.update(ARMS)
                    else: invalid.add(arm)
                integrity[seed][arm]['sleeps'][str(L)] = dict(errors=ee, twin=tw)
                per_seed[seed][arm][str(L)] = dict(summarize(graded, p), ppl_ratio=evs.get(modes[0], {}).get('ppl_ratio', {}))
        if fork_active:
            for arm in ('U4', 'G4'):
                ee, evs, grs = [], {}, {}
                try:
                    ff = t['fork']
                    if not ff['admitted'] or ff['parent_full_state_hash'][arm] != t['arms'][arm]['3']['full_state_hash']:
                        ee.append('fork admission/parent state mismatch')
                    ww = ff['withheld'][arm]
                    got = {entry_key(r): r['count'] for r in ww['delivered']}
                    if len(got) != len(ww['delivered']) or got != delivered(p['fork_schedule'][arm]):
                        ee.append('fork delivered exposure mismatch')
                    if not hash_ok(ww['ckpt_sha256']) or not hash_ok(ww['full_state_hash']):
                        ee.append('fork checkpoint/state hash malformed')
                    for mode in ('merged', 'unmerged'):
                        path = directory / f'eval_{arm}_s4_withheld_{mode}_seed{s}.json'
                        if not path.exists(): path = directory / f'seed{s}' / f'eval_{arm}_s4_withheld_{mode}.json'
                        if not path.exists() and len(seeds) == 1: path = directory / f'eval_{arm}_s4_withheld_{mode}.json'
                        evs[mode] = read(path)
                        ev_errors, grs[mode] = validate_eval(evs[mode], p, t, arm, 4, mode, 'withheld', psha, code, manifest, test, qdiag)
                        ee += ev_errors
                        if evs[mode]['base_denominator_sha256'] not in all_base_hashes[arm]: ee.append('fork base denominator mismatch')
                    tw = twin(evs['merged'], evs['unmerged'])
                    if not tw['pass_']: ee.append('fork twin integrity failed')
                    mm = summarize(grs['merged'], p)
                    main = per_seed[seed][arm]['4']
                    pairs = {}
                    for cat in ('FORK-H', 'FORK-H-Q'):
                        obs = main['tables'][cat+'|primary']; wh = mm['tables'][cat+'|primary']
                        x, y = obs['rates']['correct']['value'], wh['rates']['unknown']['value']
                        pairs[cat] = dict(observed=obs, withheld=wh,
                            history_sensitive=bool(ge(x, .5) and ge(y, .8)) if finite(x) and finite(y) and not ee and arm not in invalid else None)
                    branch[seed][arm] = dict(errors=ee, twin=tw, pairs=pairs, metrics=mm)
                except (OSError, KeyError, ValueError, TypeError) as ex:
                    branch[seed][arm] = dict(errors=ee+[str(ex)], pairs={})
    # All evals must use the same pinned fp32 base competence denominator.
    n_hashes = all_base_hashes['N']
    for arm, hh in all_base_hashes.items():
        if len(n_hashes) != 1 or hh != n_hashes or not all(hash_ok(h) for h in hh):
            if arm == 'N': invalid.update(ARMS)
            else: invalid.add(arm)
            errors.append(f'{arm}: inconsistent global base denominator')
    if errors: invalid.update(('P', 'U4'))
    pooled = aggregate(per_seed, seeds)
    guards = {a: criteria(pooled, per_seed, seeds, a, invalid) for a in ARMS}
    mech = mechanism(pooled, per_seed, seeds, invalid)
    crit = guards['U4']
    if {'P', 'U4'} & invalid: verdict = 'SUPPORT1_INVALID'
    elif not crit['functional']: verdict = 'SUPPORT1_NO_ADVANCE'
    elif mech['pass_']: verdict = 'SUPPORT1_FUNCTIONAL_AND_GROUNDED'
    else: verdict = 'SUPPORT1_FUNCTIONAL'
    if verdict != 'SUPPORT1_INVALID' and not mech['assessable']: verdict += '_MECHANISM_NOT_ASSESSABLE'
    fork_summary = {}
    if fork_active:
        for arm in ('U4', 'G4'):
            fork_summary[arm] = {}
            for cat in ('FORK-H', 'FORK-H-Q'):
                vals = [branch[str(s)].get(arm, {}).get('pairs', {}).get(cat, {}).get('history_sensitive') for s in seeds]
                fork_summary[arm][cat] = dict(unit='seed branch pair',
                    history_sensitive=fraction(sum(v is True for v in vals), sum(v is not None for v in vals)),
                    unassessable_seed_pairs=sum(v is None for v in vals))
    return dict(analyzer='Sol', schema=SCHEMA, verdict=verdict, test_mode=test, seeds=list(seeds), qdiag=qdiag,
                fork_requested=fork, fork_status='admitted' if fork_active else 'unrun', fork_errors=fork_errors, code_hash=code, model_manifest=manifest, errors=errors,
                invalid_arms=sorted(invalid), per_seed=per_seed, pooled=pooled, integrity=integrity,
                criteria=crit, arm_guards=guards, mechanism=mech,
                diagnostics=diagnostics(pooled, per_seed, seeds, invalid, guards), fork=branch, fork_summary=fork_summary,
                limitations=['Checkpoint/full-state and base-loss hashes are cross-checked as recorded identities; '
                             'tensor/RNG equality and forward passes require the producer restoration/calibration artifacts.',
                             'Test-mode outputs are plumbing only and cannot establish experimental findings.'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--seeds', default='20,21,22,23')
    ap.add_argument('--code-hash', required=True)
    ap.add_argument('--model-manifest-sha', required=True)
    ap.add_argument('--validator-report', type=Path)
    ap.add_argument('--qdiag', type=int, choices=(32, 64), default=64)
    ap.add_argument('--fork', action='store_true')
    ap.add_argument('--gpu', help='recorded phase-1 GPU name; required in production')
    ap.add_argument('--restore-report', type=Path)
    ap.add_argument('--test-mode', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    r = analyze(a.directory, tuple(map(int, a.seeds.split(','))), a.code_hash, a.model_manifest_sha,
                a.qdiag, a.fork, a.test_mode, a.validator_report, a.restore_report, a.gpu)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    def clean(v):
        if isinstance(v, float) and not finite(v): return None
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list): return [clean(x) for x in v]
        return v
    r = clean(r)
    a.out.write_text(json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'test_mode', 'invalid_arms', 'errors')}))
    return 2 if r['verdict'] == 'SUPPORT1_INVALID' else 0


if __name__ == '__main__':
    sys.exit(main())
