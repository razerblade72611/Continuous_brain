"""Independent SUPPORT-1 contract helpers. No producer imports; protocol v2.1 amended at ef67261 and A1 at 981f0e7.

Statement stems are the independently frozen ABST-0 file-format inventory. Training
framing is the inherited CONS-0A contract, transcribed here rather than imported.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'SUPPORT1_SCHEMA_V1'
ARMS = ('P', 'U1', 'G1', 'U4', 'G4')
ATTRS = ('city', 'occupation', 'favorite object', 'pet species', 'childhood fear',
         'workplace', 'hobby', 'birth month', 'allergy')
QT = dict(zip(ATTRS, (
    ('Which city does {e} live in?', 'What is the name of the city {e} calls home?'),
    ('What does {e} do for a living?', "What is {e}'s line of work?"),
    ('What object does {e} like best?', 'Which possession does {e} value above all others?'),
    ('What kind of animal does {e} keep as a pet?', 'What species is the pet that {e} looks after?'),
    ('What frightened {e} as a child?', 'What was {e} scared of when young?'),
    ('Where does {e} work?', 'What is the name of the place where {e} is employed?'),
    ('What does {e} do for fun?', 'Which pastime does {e} enjoy?'),
    ('In which month does {e} have a birthday?', 'What month was {e} born in?'),
    ('What is {e} allergic to?', 'What substance causes {e} an allergic reaction?'))))
STEMS = json.loads((ROOT / 'scripts/abst0_sol_probe_schema.json').read_text())
FRAME = ('Answer from your own memory with the exact answer only, or UNKNOWN if you do not know it.\n\n'
         'Statement: The capital city of France is ...\nAnswer: Paris\n\n'
         'Statement: The middle name of a stranger called Pell Arno is ...\nAnswer: UNKNOWN\n\n{item}\nAnswer:')
CODE_FILES = ('scripts/support1.py', 'scripts/abst0.py', 'scripts/cons0a.py', 'scripts/cls0c_b.py',
              'being/v3/lifegen.py', 'being/competence.py', 'data/background.txt',
              'data/competence_eval.txt', 'data/competence_modern.txt', 'lab/support1_norm_vectors.json')
Q_FULL = {'POS-Q': 10, 'SEL-Q': 12, 'HOW-Q': 12, 'HNM-Q': 10, 'REV-Q': 4, 'GTQ': 16}
Q_SMALL = {'POS-Q': 4, 'SEL-Q': 8, 'HOW-Q': 6, 'HNM-Q': 4, 'REV-Q': 4, 'GTQ': 6}
CLASSES = ('correct', 'unknown', 'empty', 'substantive')
POSITIVE = ('POS', 'FIRST', 'RET', 'TPOST-A', 'TPOST-H')
CATEGORIES = (*POSITIVE, 'UPD', 'TPRE-A', 'TPRE-H', 'TPRE-A-UNFAM', 'TPRE-H-UNFAM',
              'SEL-D', 'SEL-P', 'HOW', 'HNM', 'FRESH', *Q_FULL, 'FORK-H', 'FORK-H-Q')


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def chash(x):
    return sha(canon(x))


def codehash(root=ROOT):
    return sha(b''.join(sha((root / f).read_bytes()).encode() for f in CODE_FILES))


def norm(s):
    s = re.sub(r"[^\w\s'-]", ' ', (s or '').lower())
    s = re.sub(r'\s+', ' ', s).strip()
    return re.sub(r'^(a|an|the)\s+', '', s)


def classify(answer, gold):
    s = norm(answer)
    return ('correct' if gold != 'UNKNOWN' and s == norm(gold) else
            'unknown' if s == 'unknown' else 'empty' if s == '' else 'substantive')


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def ge(x, y):
    # Only removes binary-representation error at predeclared boundaries.
    return finite(x) and finite(y) and (x >= y or math.isclose(x, y, rel_tol=0, abs_tol=1e-12))


def le(x, y):
    return finite(x) and finite(y) and (x <= y or math.isclose(x, y, rel_tol=0, abs_tol=1e-12))


def ordered(cells):
    return sorted(cells, key=lambda c: (c[0], ATTRS.index(c[1])))


def cell(f):
    return f['entity'], f['attr']


def oracle(facts, L):
    out = {}
    for f in sorted(facts, key=lambda f: f['life']):
        if f['life'] <= L:
            out[cell(f)] = f['value']
    return out


def history(p, h):
    facts = p['stream']['facts']
    if h == 'observed':
        return facts
    targets = {tuple(c) for c in p['fork']['targets']}
    return [f for f in facts if cell(f) not in targets] + p['fork']['facts']


def question(c, k):
    return 'Question: ' + QT[c[1]][k].format(e=c[0])


def qa(tok, c, k, label):
    prefix = tok.encode(FRAME.format(item=question(c, k)), add_special_tokens=False)
    tail = tok.encode(' ' + label, add_special_tokens=False) + tok.encode('\n', add_special_tokens=False)
    return prefix + tail, len(prefix), len(tail)


def budget(rows):
    return [len(rows), sum(len(r['input_ids']) for r in rows), sum(r['n_loss'] for r in rows)]


def entry_key(r):
    return (r['kind'], tuple(r['cell'] or ()), tuple(r['src_cell'] or ()), r['wording'], r['label'])


def delivered(rows):
    counts = collections.Counter(entry_key(r) for r in rows if r['kind'] != 'bg')
    return {k: v for k, v in counts.items()}


def fraction(n, d):
    return {'numerator': n, 'denominator': d, 'value': n / d if d else None}


def table(rows):
    counts = collections.Counter(r['graded'] for r in rows)
    n = len(rows)
    return {'n': n, 'counts': {c: counts[c] for c in CLASSES},
            'rates': {c: fraction(counts[c], n) for c in CLASSES},
            'answer_rate': fraction(counts['correct'] + counts['substantive'], n),
            'selective_accuracy': fraction(counts['correct'], counts['correct'] + counts['substantive'])}


def pool(fracs):
    vals = [f['value'] for f in fracs.values()]
    return {'value': sum(vals) / len(vals) if vals and all(finite(v) for v in vals) else None,
            'aggregation': 'equal_weight_seed_mean', 'seed_fractions': fracs,
            'descriptive_micro': fraction(sum(f['numerator'] for f in fracs.values()),
                                          sum(f['denominator'] for f in fracs.values()))}


def probes_for(p, h, L, qdiag):
    rows = p['probes'][h][str(L)]
    if qdiag == 64:
        return rows
    # Fallback selection is fixed at prep, never inferred from the observed file.
    keep = set()
    for cat, k in Q_SMALL.items():
        group = sorted((r for r in rows if r['cat'] == cat), key=lambda r: r['probe_id'])
        keep.update(r['probe_id'] for r in group[:k])
    return [r for r in rows if r['cat'] not in Q_SMALL or r['probe_id'] in keep]
