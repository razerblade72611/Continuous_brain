"""Independent PROV-1 contract. No producer modules, torch, or transformers imports.

SUPPORT-1's independently transcribed framing/grading is reused. Historical,
frozen SUPPORT-1 prep JSON is an input, not executable producer analysis.
"""
import random
from support1_sol_spec import (ATTRS, CODE_FILES as S1_CODE_FILES, FRAME, ROOT,
    budget, canon, cell, chash, classify, delivered, finite, fraction, ge, history,
    le, norm, oracle, ordered, pool, question, sha)

SCHEMA = 'PROV1_SCHEMA_V1'
ARMS = ('RP', 'RU', 'RC')
CONDITIONS = ('FAITHFUL', 'ABSENT', 'DEL-Q', 'DEL-X', 'ADD', 'FULL')
B_CONDITIONS = CONDITIONS[:-1]
A_CONDITIONS = tuple(c for c in CONDITIONS if c != 'ABSENT')
S1_HASH = '38b8e70541b53317b3b8ae42ac6bf00e7937549763c52296b5c5355ec0c73fae'
CODE_FILES = ('scripts/prov1.py',) + S1_CODE_FILES


def codehash():
    return sha(b''.join(sha((ROOT / f).read_bytes()).encode() for f in CODE_FILES))


def entries(facts, entity, L):
    values = oracle(facts, L)
    latest = {}
    for f in facts:
        if f['entity'] == entity and f['life'] <= L:
            latest[f['attr']] = max(f['life'], latest.get(f['attr'], 0))
    return {a: latest[a] for a in ATTRS if a in latest}, values


def record(entity, mapping, values=None):
    if not mapping:
        return f'Memory record: {entity} - no experiences.'
    parts = [f'{a} (life {mapping[a]})' + (f': {values[entity, a]}' if values is not None else '')
             for a in ATTRS if a in mapping]
    return f"Memory record: {entity} - experienced: {', '.join(parts)}. Nothing else was experienced about {entity}."


def expand(anchor, h, L):
    facts = history(anchor, h)
    rows = []
    for base in anchor['probes'][h][str(L)]:
        e, a = base['cell']; rs, values = entries(facts, e, L)
        truth = oracle(facts, L).get((e, a), 'UNKNOWN')
        if base['gold_current'] != truth:
            raise ValueError('historical probe gold conflicts with history')
        variants = [('FAITHFUL', record(e, rs), truth), ('ABSENT', None, truth),
                    ('FULL', record(e, rs, values), truth)]
        dx = None
        if a in rs:
            variants.append(('DEL-Q', record(e, {k:v for k,v in rs.items() if k != a}), 'UNKNOWN'))
            others = [k for k in ATTRS if k in rs and k != a]
            if others:
                dx = random.Random(f"PROV1-delx-{anchor['seed']}-{h}-{L}-{e}-{a}").choice(others)
                variants.append(('DEL-X', record(e, {k:v for k,v in rs.items() if k != dx}), truth))
        else:
            added = dict(rs); added[a] = L
            variants.append(('ADD', record(e, added), None))
        for cond, rec, compliance in variants:
            r = dict(probe_id=base['probe_id']+'-'+cond, base_probe_id=base['probe_id'], condition=cond,
                     cat=base['cat'], construction=base['construction'], kind=base['kind'], cell=[e,a],
                     item=(rec+'\n' if rec is not None else '')+base['item'], record=rec,
                     gold_truth=truth, gold_compliance=compliance, supported=a in rs, n_entries=len(rs))
            if cond == 'DEL-X':
                r['delx_attr'] = dx
            rows.append(r)
    return rows


def source_fingerprint(names):
    return chash({n:sha((ROOT/'scripts'/n).read_bytes()) for n in names})
