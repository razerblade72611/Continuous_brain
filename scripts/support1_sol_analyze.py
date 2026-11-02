#!/usr/bin/env python3
"""Independent stdlib SUPPORT-1 analyzer. Protocol v2.1 amended at ef67261 and A1 at 981f0e7, no producer imports.

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


def analysis_fingerprint():
    parent = Path(__file__).resolve().parent
    return sha(b''.join(sha((parent/name).read_bytes()).encode() for name in
                       ('support1_sol_analyze.py', 'support1_sol_spec.py', 'abst0_sol_probe_schema.json')))


def checked_manifest(directory, name, expected_files, expected_metadata):
    """Hash exact frozen manifest entries; filenames are a closed relative set."""
    path = directory / name
    entries, metadata = {}, {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        first, rest = line.split(' ', 1)
        rest = rest.strip()
        if first in expected_metadata:
            if first in metadata: raise ValueError(f'duplicate {first} in {name}')
            metadata[first] = rest
        else:
            if not hash_ok(first) or rest not in expected_files or rest in entries:
                raise ValueError(f'unexpected/malformed entry in {name}: {line}')
            entries[rest] = first
    if set(entries) != set(expected_files) or set(metadata) != set(expected_metadata):
        raise ValueError(f'incomplete {name}')
    for filename, digest in entries.items():
        if sha((directory/filename).read_bytes()) != digest:
            raise ValueError(f'{name}: hash mismatch for {filename}')
    return dict(sha256=sha(path.read_bytes()), files=entries, metadata=metadata)


def verify_phase1(phase1, phase0, code, manifest, gpu=None, prior_spend=.65, rate=.86,
                  boot_hours=.5, expected_pin=None, bundle=None):
    """Independent raw-answer calibration and phase-1 input verification; stdlib only.

    Rate/prior spend are explicit accounting assumptions, not a billing lookup.
    Tensor equality is checked as a bound restoration report, not rerun here.
    """
    errors = []
    result = dict(verdict='SUPPORT1_PHASE1_HOLD', errors=errors, test_mode=False, auditor='Sol', analyzer_code_hash=analysis_fingerprint())
    def require(ok, msg):
        if not ok: errors.append(msg)
    try:
        require(finite(prior_spend) and prior_spend >= 0 and finite(rate) and rate > 0 and finite(boot_hours) and boot_hours >= 0, 'invalid accounting assumptions')
        p0 = checked_manifest(phase0, 'PHASE0.sha256',
            {f'prep_seed{s}.json' for s in range(20, 25)} | {'validation.json'}, {'CODE_HASH', 'MODEL_MANIFEST'})
        p1 = checked_manifest(phase1, 'PHASE1.sha256',
            {'cal/calcheck.json', 'restoretest.json', 'timings.json', 'projection.json'}, {'PIN', 'CODE_HASH', 'MODEL_MANIFEST', 'GPU'})
        result.update(phase0_manifest=p0, phase1_manifest=p1)
        for m in (p0, p1):
            require(m['metadata']['CODE_HASH'] == code and m['metadata']['MODEL_MANIFEST'] == manifest, 'manifest code/model identity mismatch')
        pm = p1['metadata']; gpu = gpu or pm['GPU']
        require(pm['GPU'] == gpu and bool(gpu) and gpu != 'cpu', 'phase-1 GPU identity mismatch')
        require(re.fullmatch('[0-9a-f]{40}', pm['PIN']) is not None and (expected_pin is None or pm['PIN'] == expected_pin), 'phase-0 PIN identity mismatch')
        val = read(phase0/'validation.json')
        require(val.get('verdict') == 'PASS' and val.get('test_mode') is False and val.get('tokenizer_checked') is True and
                val.get('code_hash') == code and val.get('model_manifest') == manifest, 'phase-0 validator identity/PASS mismatch')
        required_hashes = [p0['files'][f'prep_seed{s}.json'] for s in range(20, 25)]
        require(val.get('prep_sha256') == required_hashes, 'validator does not bind all five frozen prep bytes in order')
        preps = {s: read(phase0/f'prep_seed{s}.json') for s in range(20, 25)}
        for s, p in preps.items():
            require(p.get('seed') == s and p.get('test_mode') is False and p.get('code_hash') == code and
                    p.get('model_manifest') == manifest, f'seed {s}: frozen prep identity mismatch')
        p = preps[24]; ph = p0['files']['prep_seed24.json']
        trpath = phase1/'cal/train_seed24.json'; tr = read(trpath)
        cal = read(phase1/'cal/calcheck.json'); rt = read(phase1/'restoretest.json')
        evpaths = {'merged': phase1/'cal/eval_P_s1_merged.json', 'unmerged': phase1/'cal/eval_P_s1_unmerged.json',
                   'base': phase1/'cal/eval_N_s1.json'}
        evs = {m: read(path) for m, path in evpaths.items()}
        tp = tr.get('provenance', {})
        require((tr.get('schema'), tr.get('experiment'), tr.get('seed')) == (SCHEMA, 'SUPPORT1', 24), 'calibration train identity mismatch')
        for k, v in dict(code_hash=code, prep_sha256=ph, model_manifest=manifest, test_mode=False, gpu=gpu, arms=['P'], sleeps=1, fork=False).items():
            require(type(tp.get(k)) is type(v) and tp.get(k) == v, f'calibration train provenance {k} mismatch')
        require(set(tr['arms']) == {'P'} and set(tr['arms']['P']) == {'1'}, 'calibration train inventory must be P sleep 1 only')
        ts = tr['arms']['P']['1']
        require(hash_ok(ts['ckpt_sha256']) and hash_ok(ts['full_state_hash']), 'invalid calibration checkpoint/state identity')
        require(ts['budget'] == p['budget']['P']['1'], 'calibration delivered budget mismatch')
        counts = {entry_key(r): r['count'] for r in ts['delivered']}
        require(len(counts) == len(ts['delivered']) and counts == delivered(p['schedule']['P']['1']), 'calibration delivered exposures mismatch')
        graded = {}
        for mode, ev in evs.items():
            ee, graded[mode] = validate_eval(ev, p, tr, 'N' if mode == 'base' else 'P', 1, mode,
                                            'observed', ph, code, manifest, False, 64)
            errors.extend(f'calibration {mode}: {e}' for e in ee)
        tw = twin(evs['merged'], evs['unmerged']); require(tw['pass_'], 'calibration twin integrity failed')
        bases = {e['base_denominator_sha256'] for e in evs.values()}
        require(len(bases) == 1 and all(hash_ok(h) for h in bases), 'calibration base denominator mismatch')
        pos = table([r for r in graded['merged'] if r['cat'] == 'POS' and r['kind'] == 'template'])
        fresh = table([r for r in graded['base'] if r['cat'] == 'FRESH'])
        acquisition = pos['rates']['correct']; unknown = fresh['rates']['unknown']
        competence = evs['merged']['ppl_ratio']
        raw_pass = ge(acquisition['value'], .8) and ge(unknown['value'], .9) and all(le(v, 1.02) and v > 0 for v in competence.values())
        require(raw_pass, 'raw calibration does not pass the frozen acquisition/FRESH/competence rule')
        require(acquisition['denominator'] == 10 and unknown['denominator'] == 20, 'calibration population denominator mismatch')
        want_cal = dict(schema=SCHEMA, experiment='SUPPORT1', which='calcheck', seed=24, E=4, pass_=True,
                        prep_sha256=ph, code_hash=code, model_manifest=manifest, test_mode=False)
        for k, v in want_cal.items():
            key = 'pass' if k == 'pass_' else k
            require(type(cal.get(key)) is type(v) and cal.get(key) == v, f'calcheck {key} binding mismatch')
        require(cal.get('errors') == [] and cal.get('train_sha256') == sha(trpath.read_bytes()), 'calcheck error/train-byte binding mismatch')
        require(cal.get('eval_sha256') == {m: sha(path.read_bytes()) for m, path in evpaths.items()}, 'calcheck eval-byte bindings mismatch')
        require(cal.get('n_pos_template') == acquisition['denominator'] and cal.get('n_fresh') == unknown['denominator'] and
                cal.get('pos_template_acquisition') == acquisition['value'] and cal.get('N_fresh_unknown') == unknown['value'], 'calcheck counts/rates disagree with independently graded answers')
        require(cal.get('competence_in_band') is True and cal.get('ppl_ratio') == competence, 'calcheck competence disagrees with raw evaluations')
        require(all(type(rt.get(k)) is bool for k in ('tensors_equal', 'step_equal', 'admitted')), 'restoration equality fields must be booleans')
        rh = rt.get('full_state_hash', [])
        require((rt.get('schema'), rt.get('experiment'), rt.get('which'), rt.get('seed'), rt.get('arm'), rt.get('code_hash'), rt.get('prep_sha256')) ==
                (SCHEMA, 'SUPPORT1', 'restoretest', 24, 'U4', code, ph), 'restoration record provenance mismatch')
        require(len(rh) == 2 and all(hash_ok(h) for h in rh), 'malformed restoration state hashes')
        admitted = rt.get('tensors_equal') is True and rt.get('step_equal') is True and len(rh) == 2 and rh[0] == rh[1]
        require(type(rt.get('admitted')) is bool and rt['admitted'] == admitted, 'restoration admission contradicts its recorded equality checks')
        timing = read(phase1/'timings.json'); projection = read(phase1/'projection.json')
        require((projection.get('schema'), projection.get('experiment'), projection.get('which')) == (SCHEMA, 'SUPPORT1', 'projection'), 'projection record identity mismatch')
        nitems = len(p['schedule']['P']['1']) + sum(len(p['schedule']['U4'][str(L)]) for L in (1, 2, 3)) + 2*len(p['schedule']['U4']['4'])
        nrows = sum(len(ev['rows']) for ev in evs.values())
        require(timing.get('train_items') == nitems and timing.get('eval_rows') == nrows, 'timing denominators disagree with executed inventories')
        require(all(finite(timing.get(k)) and timing[k] > 0 for k in ('train_seconds', 'eval_seconds')), 'invalid measured timings')
        si = timing['train_seconds']/nitems; sr = timing['eval_seconds']/nrows
        require(abs(projection['s_per_item']-si) < 1e-12 and abs(projection['s_per_row']-sr) < 1e-12, 'projection rate mismatch')
        independent_plans = []
        for fork, q in ((True, 64), (False, 64), (True, 32), (False, 32)):
            ntrain = n_eval = legacy_extra = 0
            for s in (20, 21, 22, 23):
                pp = preps[s]
                ntrain += sum(len(rows) for lives in pp['schedule'].values() for rows in lives.values())
                n_eval += 11*sum(len(probes_for(pp, 'observed', L, q)) for L in range(1, 6))
                if fork:
                    ntrain += sum(len(pp['fork_schedule'][a]) for a in ('U4', 'G4'))
                    n_eval += 4*len(probes_for(pp, 'withheld', 4, q))
                    legacy_extra += 4*(len(probes_for(pp, 'withheld', 4, 64))-len(probes_for(pp, 'withheld', 4, q)))
            hours = (si*ntrain+sr*n_eval)/3600+boot_hours
            independent_plans.append(dict(fork=fork, qdiag=q, train_items=ntrain, eval_rows=n_eval, hours=hours,
                usd=hours*rate, legacy_extra_rows=legacy_extra,
                recorded_expected_hours=round(hours+sr*legacy_extra/3600, 2), recorded_expected_usd=round((hours+sr*legacy_extra/3600)*rate, 2)))
        require(len(projection['plans']) == 4, 'projection plan inventory mismatch')
        for actual, expected in zip(projection['plans'], independent_plans):
            require(actual == dict(fork=expected['fork'], qdiag=expected['qdiag'], hours=expected['recorded_expected_hours'], usd=expected['recorded_expected_usd']), 'projection plan differs from independently recomputed counts/rates')
        require(projection.get('code_hash') == code and projection.get('fork_admitted') is admitted and
                projection.get('ceiling_usd') == 30 and finite(projection.get('spent_usd')) and projection['spent_usd'] >= 0, 'projection provenance/accounting mismatch')
        spent = projection['spent_usd']+prior_spend
        choices = [p for p in projection['plans'] if (admitted or not p['fork']) and p['usd']+spent <= 30]
        choice = choices[0] if choices else None
        require(choice is not None and projection.get('choice') == choice and projection.get('stop') is False, 'recorded phase-2 plan does not remain affordable/admissible with prior-attempt spending')
        bundle_claim = (phase1/'BUNDLE.sha256').read_text().split() if (phase1/'BUNDLE.sha256').exists() else []
        if bundle_claim:
            require(len(bundle_claim) == 2 and hash_ok(bundle_claim[0]) and bundle_claim[1] == 'support1_phase1.tar.gz', 'malformed recorded bundle checksum')
        bundle_path = bundle or phase1/'support1_phase1.tar.gz'
        archive_verified = False
        require(bundle is None or bundle_path.is_file(), 'supplied phase-1 archive is missing')
        if bundle_path.is_file():
            require(len(bundle_claim) == 2 and hash_ok(bundle_claim[0]) and sha(bundle_path.read_bytes()) == bundle_claim[0], 'archive checksum mismatch')
            archive_verified = len(bundle_claim) == 2 and sha(bundle_path.read_bytes()) == bundle_claim[0]
        result.update(gpu=gpu, phase0_pin=pm['PIN'], validation_sha256=p0['files']['validation.json'],
            calcheck_sha256=p1['files']['cal/calcheck.json'], restoration_sha256=p1['files']['restoretest.json'],
            base_denominator_sha256=next(iter(bases)), calibration=dict(acquisition=acquisition, N_fresh_unknown=unknown,
                competence=competence, twin=tw, POS_paraphrase=table([r for r in graded['merged'] if r['cat']=='POS' and r['kind'].startswith('para')])),
            restoration=dict(admitted=admitted, full_state_hash=rh, tensor_comparison='record verified; tensor states not re-executed'),
            projection=dict(independent_plans=independent_plans, chosen=choice, recorded_spent=projection['spent_usd'], prior_attempt_spend=prior_spend,
                cumulative_projection=spent+choice['usd'] if choice else None, rate_assumption_usd_per_hour=rate),
            archive=dict(claimed_sha256=bundle_claim[0] if bundle_claim else None, checksum_recomputed=archive_verified,
                note='archive verified' if archive_verified else 'archive not supplied; committed manifest payload verified'))
    except (OSError, KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as e:
        errors.append(f'phase-1 artifact verification failed: {e}')
    result['verdict'] = 'SUPPORT1_PHASE1_GO' if not errors else 'SUPPORT1_PHASE1_HOLD'
    return result


def resolve_artifact(directory, candidates):
    """Accept declared flat/per-seed layouts; conflicting duplicate files fail."""
    paths = [directory/name for name in candidates if (directory/name).is_file()]
    if not paths:
        return directory/candidates[0]
    if len(paths) > 1 and len({sha(p.read_bytes()) for p in paths}) != 1:
        raise ValueError('conflicting copies of artifact: '+', '.join(str(p) for p in paths))
    return paths[0]


def analyze(directory, seeds, code, manifest, qdiag=64, fork=False, test=False, validator=None, restore=None, gpu=None, phase0=None, phase1=None,
            phase0_pin=None, phase1_manifest_sha=None, prior_spend=.65, rate=.86):
    errors, invalid, integrity, per_seed, branch = [], set(), {}, {}, {}
    phase1_binding = None
    if validator is None:
        candidate = phase0/'validation.json' if phase0 is not None else directory/'validation.json'
        if candidate.is_file(): validator = candidate
    if phase1 is not None or not test:
        if phase0 is None or phase1 is None:
            errors.append('production requires --phase0-dir and --phase1-dir for frozen calibration bindings')
        else:
            phase1_binding = verify_phase1(phase1, phase0, code, manifest, gpu, prior_spend=prior_spend,
                                          rate=rate, expected_pin=phase0_pin)
            if phase1_binding['verdict'] != 'SUPPORT1_PHASE1_GO':
                errors.extend(phase1_binding['errors'])
            else:
                gpu = gpu or phase1_binding['gpu']
                if phase1_manifest_sha is not None and phase1_binding['phase1_manifest']['sha256'] != phase1_manifest_sha:
                    errors.append('phase-1 manifest differs from external frozen identity')
                if validator is None or not validator.is_file() or sha(validator.read_bytes()) != phase1_binding['validation_sha256']:
                    errors.append('validator report bytes differ from frozen phase-0 validation.json')
                choice = phase1_binding['projection']['chosen']
                if choice['qdiag'] != qdiag or choice['fork'] is not fork:
                    errors.append('requested QDIAG/fork differs from the predeclared phase-1 plan')
                if restore is None: restore = phase1/'restoretest.json'
                if not restore.is_file() or sha(restore.read_bytes()) != phase1_binding['restoration_sha256']:
                    errors.append('restoration bytes differ from frozen phase-1 report')
        if errors:
            return dict(verdict='SUPPORT1_INVALID', errors=errors, invalid_arms=list(ARMS),
                        test_mode=test, phase1_binding=phase1_binding)
    val = read(validator) if validator else None
    files_read = []
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
            pp = phase0/f'prep_seed{s}.json' if phase0 is not None else resolve_artifact(directory, [f'prep_seed{s}.json', f'seed{s}/prep_seed{s}.json'])
            p = read(pp); psha = sha(pp.read_bytes())
            if p['seed'] != s or p['test_mode'] is not test or p['code_hash'] != code or p['model_manifest'] != manifest:
                errors.append(f'{s}: prep identity mismatch')
            if not test and (not val or psha not in val.get('prep_sha256', [])):
                errors.append(f'{s}: prep bytes not covered by independent validator')
            t = read(resolve_artifact(directory, [f'train_seed{s}.json', f'seed{s}/train_seed{s}.json']))
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
                    names = [f'eval_{arm}_s{L}{suffix}_seed{s}.json', f'seed{s}/eval_{arm}_s{L}{suffix}.json']
                    if len(seeds) == 1: names.append(f'eval_{arm}_s{L}{suffix}.json')
                    try:
                        path = resolve_artifact(directory, names)
                        ev_bytes = path.read_bytes(); ev = json.loads(ev_bytes); evs[mode] = ev
                        files_read.append(dict(file=str(path.relative_to(directory)), sha256=sha(ev_bytes),
                                               seed=s, arm=arm, sleep=L, history='observed', mode=mode))
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
                        names = [f'eval_{arm}_s4_withheld_{mode}_seed{s}.json', f'seed{s}/eval_{arm}_s4_withheld_{mode}.json']
                        if len(seeds) == 1: names.append(f'eval_{arm}_s4_withheld_{mode}.json')
                        path = resolve_artifact(directory, names)
                        ev_bytes = path.read_bytes(); evs[mode] = json.loads(ev_bytes)
                        files_read.append(dict(file=str(path.relative_to(directory)), sha256=sha(ev_bytes),
                                               seed=s, arm=arm, sleep=4, history='withheld', mode=mode))
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
    expected_base = {phase1_binding['base_denominator_sha256']} if phase1_binding else n_hashes
    denominator_integrity = {}
    for arm, hh in all_base_hashes.items():
        good = len(n_hashes) == 1 and hh == n_hashes == expected_base and all(hash_ok(h) for h in hh)
        denominator_integrity[arm] = dict(pass_=bool(good), recorded_hashes=sorted(hh, key=str), expected_hashes=sorted(expected_base, key=str))
        if not good:
            if arm == 'N': invalid.update(ARMS)
            else: invalid.add(arm)
            # An invalid low-dose/G4 arm must not invalidate the P/U4 primary claim.
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
    return dict(analyzer='Sol', analyzer_code_hash=analysis_fingerprint(), schema=SCHEMA, verdict=verdict, test_mode=test, seeds=list(seeds), qdiag=qdiag,
                fork_requested=fork, fork_status=('invalid_secondary' if any(v.get('errors') for d in branch.values() for v in d.values()) else 'admitted') if fork_active else 'unrun', fork_errors=fork_errors, code_hash=code, model_manifest=manifest, errors=errors,
                invalid_arms=sorted(invalid), per_seed=per_seed, pooled=pooled, integrity=integrity,
                criteria=crit, arm_guards=guards, mechanism=mech, phase1_binding=phase1_binding,
                base_denominator_integrity=denominator_integrity,
                file_inventory=dict(expected_main=55*len(seeds), expected_withheld=4*len(seeds) if fork_active else 0, loaded=len(files_read), files=files_read),
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
    ap.add_argument('--phase0-dir', type=Path, help='frozen phase-0 directory, including all five preps and validation.json')
    ap.add_argument('--phase1-dir', type=Path, help='frozen phase-1 directory, including calibration inputs and manifests')
    ap.add_argument('--phase0-pin', help='optional external phase-0 commit identity')
    ap.add_argument('--phase1-manifest-sha', help='optional external PHASE1.sha256 identity')
    ap.add_argument('--prior-spend', type=float, default=.65, help='previous-attempt spending, added to recorded spent_usd')
    ap.add_argument('--usd-per-hour', type=float, default=.86, help='recorded accounting rate, not a billing lookup')
    ap.add_argument('--phase1-bundle', type=Path, help='optional archive for --phase1-only checksum verification')
    ap.add_argument('--phase1-only', action='store_true', help='independently verify phase 1; directory is the phase-1 root')
    ap.add_argument('--test-mode', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.phase1_only:
        if a.phase0_dir is None: ap.error('--phase1-only requires --phase0-dir')
        r = verify_phase1(a.phase1_dir or a.directory, a.phase0_dir, a.code_hash, a.model_manifest_sha,
                          a.gpu, a.prior_spend, a.usd_per_hour, expected_pin=a.phase0_pin, bundle=a.phase1_bundle)
    else:
        r = analyze(a.directory, tuple(map(int, a.seeds.split(','))), a.code_hash, a.model_manifest_sha,
                    a.qdiag, a.fork, a.test_mode, a.validator_report, a.restore_report, a.gpu,
                    a.phase0_dir, a.phase1_dir, a.phase0_pin, a.phase1_manifest_sha, a.prior_spend, a.usd_per_hour)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    def clean(v):
        if isinstance(v, float) and not finite(v): return None
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list): return [clean(x) for x in v]
        return v
    r = clean(r)
    a.out.write_text(json.dumps(r, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'test_mode', 'invalid_arms', 'errors')}))
    return 2 if r['verdict'] in ('SUPPORT1_INVALID', 'SUPPORT1_PHASE1_HOLD') else 0


if __name__ == '__main__':
    sys.exit(main())
