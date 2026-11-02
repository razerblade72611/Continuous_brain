#!/usr/bin/env python3
"""CPU-only independent SUPPORT-1 prep validator, protocol amended at ef67261 and A1 at 981f0e7.

Never imports support1/abst0/cons0a/cls0c_b. Production PASS requires a tokenizer,
matching CODE_FILES and an externally supplied model-manifest hash. Tiny test-mode
preps receive TEST_ONLY even when every applicable invariant passes.
"""
import argparse
import collections
import json
from pathlib import Path
import random
import re
import sys

from support1_sol_spec import (ARMS, ATTRS, CATEGORIES, CODE_FILES, Q_FULL, Q_SMALL, QT,
    ROOT, SCHEMA, STEMS, budget, canon, cell, chash, codehash, history, norm, oracle,
    ordered, qa, question, sha, probes_for)

VERSION = 'SUPPORT1_SOL_VALIDATOR_V3_A1'


class Checks:
    def __init__(self):
        self.errors = []
        self.count = 0

    def require(self, condition, where, message):
        self.count += 1
        if not condition:
            self.errors.append(f'{where}: {message}')


def sets(rows):
    return {tuple(c) for c in rows}


def expected_split(seed, A, pm):
    for r in range(100_000):
        rng = random.Random(f'SUPPORT1-split-{seed}-{r}')
        sel, how = [], []
        for e in sorted(A):
            attrs = [a for a in ATTRS if (e, a) in pm]
            if len(attrs) < 2:
                raise ValueError('fewer than two PM attributes for an A entity')
            rng.shuffle(attrs)
            heads = rng.random() < .5
            h = (len(attrs) + 1) // 2
            se, ho = (attrs[:h], attrs[h:]) if heads else (attrs[h:], attrs[:h])
            sel += [(e, a) for a in se]
            how += [(e, a) for a in ho]
        if all(abs(sum(a == x for _, a in sel) - sum(a == x for _, a in how)) <= 1 for x in ATTRS):
            return r, set(sel), set(how)
    raise ValueError('split retry exhausted')


def expected_blocks(p):
    seed, st = p['seed'], p['stream']
    sel = ordered(sets(p['split']['sel']))
    random.Random(f'SUPPORT1-selcyc-{seed}').shuffle(sel)
    tp = [((t, a), k) for t in sorted(p['tpart']['t_neg']) for a in ATTRS for k in (0, 1)]
    random.Random(f'SUPPORT1-tcyc-{seed}').shuffle(tp)
    si = ti = 0
    visits = collections.Counter()
    out = {}
    for L in range(1, 6):
        n = 2 * sum(f['life'] == L for f in st['facts'])
        aa = [(c, k) for c in ordered(sets(p['strata']['rev']))
              if L < 4 and st['intro'][c[0]] <= L for k in (0, 1)]
        while len(aa) < n:
            if not any(st['intro'][c[0]] <= L for c in sel):
                raise ValueError('empty eligible SEL cycler')
            c = sel[si % len(sel)]
            si += 1
            if st['intro'][c[0]] > L:
                continue
            aa.append((c, visits[c] % 2))
            visits[c] += 1
        tt = [tp[(ti + j) % len(tp)] for j in range(n)]
        ti += n
        out[L] = aa, tt
    return out


def semantic_entry(c, k, label, kind, slot, item=None, source=None, src=None):
    return dict(cell=list(c), wording=k, label=label, kind=kind, slot=slot,
                item_id=item, source_item_id=source, src_cell=list(src) if src else None)


def check_stream(p, C):
    ck = lambda ok, msg: C.require(ok, 'stream', msg)
    st = p['stream']; fs = st['facts']; A, H = set(st['A']), set(st['H'])
    core = A | H
    trans, updates = sets(st['transitions']), sets(st['updates'])
    ck(st['seed'] == p['seed'], 'seed mismatch')
    ck(len(A) == len(H) == 8 and not A & H, 'core partition')
    ck(len(fs) == 53 and [sum(f['life'] == L for f in fs) for L in range(1, 6)] == [10, 10, 10, 13, 10], 'life sizes')
    ck(len({f['fid'] for f in fs}) == len(fs), 'duplicate fact ID')
    ck(len(trans) == 8 and len(updates) == 2 and not trans & updates, 'planned cells')
    ck(sum(c[0] in A for c in trans) == 4 and sum(c[0] in A for c in updates) == 1, 'planned A/H balance')
    for g in (A, H):
        ck([sum(st['intro'][e] == L for e in g) for L in (1, 2)] == [4, 4], 'intro balance')
        ck([sum(e in g and st['intro'][e] == L for e, _ in trans) for L in (1, 2)] == [2, 2], 'transition intro balance')
    bycell = collections.defaultdict(list)
    for f in fs:
        c = cell(f); bycell[c].append(f)
        ck(f['attr'] in ATTRS and isinstance(f['value'], str) and f['value'] != 'UNKNOWN', 'fact attribute/value')
        ck(1 <= f['life'] <= 5 and st['intro'][c[0]] <= f['life'], 'fact before introduction')
        ck(f['fid'].startswith(f"X{p['seed']}L{f['life']}F"), 'fact ID pattern')
    for c, ff in bycell.items():
        ll = sorted(f['life'] for f in ff)
        ck(ll == ([2, 4] if c in updates else [4] if c in trans else [ll[0]]), f'cell collision {c}')
        if c in updates:
            ck(len({f['value'] for f in ff}) == 2, 'unchanged update')
    ck(trans | updates <= set(bycell), 'missing planned facts')
    ck(all(min(f['life'] for f in fs if f['entity'] == e) == st['intro'][e] for e in {f['entity'] for f in fs}), 'exact first introduction')
    ck(len(st['transient']) == len(set(st['transient'])) == 11 and not set(st['transient']) & core, 'transient inventory')
    ck(len(set(st['T'])) == 400 and len(set(st['V'])) == 20, 'name pool sizes')
    pools = [core | set(st['transient']) | set(st['intro']), set(st['T']), set(st['V'])]
    words = [{w.lower() for e in pool for w in e.split()} for pool in pools]
    ck(all(not words[i] & words[j] for i in range(3) for j in range(i)), 'S/T/V word overlap')
    ck(st['fresh'] == [[e, ATTRS[i % 9]] for i, e in enumerate(st['V'])], 'fresh inventory')
    ck(p['stream_sha256'] == chash(st), 'canonical stream hash')
    pm = {(e, a) for e in core for a in ATTRS} - set(bycell) - trans - updates
    rev = {c for c in trans if c[0] in A}
    excluded = updates | {c for c in trans if c[0] in H} | {cell(f) for f in fs if f['life'] in (3, 5) and f['entity'] in core}
    for key, truth in [('pm', pm), ('rev', rev), ('hnm', {c for c in pm if c[0] in H}), ('excluded', excluded)]:
        ck(sets(p['strata'][key]) == truth and len(p['strata'][key]) == len(truth), f'{key} stratum')
    try:
        r, sel, how = expected_split(p['seed'], A, pm)
        ck(p['split']['r'] == r and sets(p['split']['sel']) == sel and sets(p['split']['how']) == how and
           p['split']['sel'] == [list(c) for c in ordered(sel)] and p['split']['how'] == [list(c) for c in ordered(how)],
           'deterministic split/retry')
    except ValueError as e:
        ck(False, str(e))
    t = sorted(st['T']); random.Random(f"SUPPORT1-tpart-{p['seed']}").shuffle(t)
    ck(p['tpart'] == dict(t_neg=t[:200], t_swap=t[200:]), 'global T partition')


def check_schedule(p, C, tok):
    seed, st, test = p['seed'], p['stream'], p['test_mode']
    ck = lambda ok, msg: C.require(ok, 'schedule', msg)
    ck(set(p['schedule']) == set(ARMS), 'arm inventory')
    blocks = expected_blocks(p)
    token_cache = {}
    bg_tokens = None
    if tok:
        lines = (ROOT / 'data/background.txt').read_text().splitlines()
        bg_tokens = tok.encode(' '.join(lines), add_special_tokens=False)
    swap_names = sorted(p['tpart']['t_swap'])
    random.Random(f'SUPPORT1-swap-{seed}').shuffle(swap_names)
    entity_swap, selected_names, signatures = {}, set(), {}
    def signature(name):
        if name not in signatures:
            signatures[name] = tuple((len(ids), loss) for a in ATTRS for k in (0, 1)
                                     for ids, _, loss in [qa(tok, (name, a), k, 'UNKNOWN')])
        return signatures[name]
    for e in sorted(st['A']):
        if test or tok is not None:
            match = next((t for t in swap_names if t not in selected_names and
                          (test or signature(t) == signature(e))), None)
            ck(match is not None, f'no entity-level swap matches all 18 forms: {e}')
        else:
            # Header validation already blocks production PASS without a tokenizer.
            # Still check the recorded entity mapping's structural consistency.
            found = {m['t_name'] for rows in p['swap_map'].values() for m in rows if m['src_cell'][0] == e}
            match = next(iter(found)) if len(found) == 1 else None
        entity_swap[e] = match
        if match is not None:
            ck(match not in selected_names, f'non-injective entity swap: {e}')
            selected_names.add(match)
    prev_pos = []
    for L in range(1, 6):
        tag = f'sleep {L}'; key = str(L)
        now = oracle(st['facts'], L)
        pos = [dict(item_id=f"{f['fid']}-q{k}", cell=list(cell(f)), wording=k, label=f['value'], life=L)
               for f in st['facts'] if f['life'] == L for k in (0, 1)]
        ck(p['positives'][key] == pos, tag + ' positive inventory')
        n = len(pos)
        newest = {(tuple(x['cell']), x['wording']): x for x in prev_pos}
        rp = [newest[k] for k in sorted(newest, key=lambda x: (x[0][0], ATTRS.index(x[0][1]), x[1]))]
        random.Random(f'SUPPORT1-replay-{seed}-{L}').shuffle(rp)
        new_expected = [semantic_entry(x['cell'], x['wording'], x['label'], 'pos', 'new', x['item_id']) for x in pos] * 4
        replay_expected = [semantic_entry(x['cell'], x['wording'], now[tuple(x['cell'])], 'replay', 'replay',
                                          None, x['item_id']) for x in (rp * (4 * n // max(1, len(rp)) + 1))[:4*n]] if rp else []
        aa, tt = blocks[L]
        ck(p['base_blocks'][key] == {'A': [[list(c), k] for c, k in aa], 'T': [[list(c), k] for c, k in tt]}, tag + ' exact base block/cyclers')
        mappings = p['swap_map'][key]
        ck(len(mappings) == len(aa), tag + ' mapping coverage')
        for (c, k), m in zip(aa, mappings):
            ck(m['src_cell'] == list(c) and m['wording'] == k and m['t_name'] in p['tpart']['t_swap'], tag + ' mapping occurrence/source')
            ck(m['t_name'] == entity_swap.get(c[0]), tag + f' fixed entity swap on all 18 forms: {c[0]}')
        base = [semantic_entry(c, k, 'UNKNOWN', 'neg_a', 'third') for c, k in aa] + [semantic_entry(c, k, 'UNKNOWN', 'neg_t', 'third') for c, k in tt]
        base_g = [semantic_entry((m['t_name'], c[1]), k, 'UNKNOWN', 'neg_swap', 'third', src=c)
                  for (c, k), m in zip(aa, mappings)] + base[len(aa):]
        perm = list(range(16*n)); random.Random(f'SUPPORT1-perm-{seed}-{L}').shuffle(perm)
        contents = {}
        for arm in ARMS:
            rows = p['schedule'][arm][key]
            ck(len(rows) == 16*n, tag + f' {arm} item count')
            ck([x['pos'] for x in rows] == list(range(16*n)), tag + f' {arm} positions')
            content = [None] * len(rows)
            for i, x in zip(perm, rows):
                content[i] = x
            contents[arm] = content
            ck(budget(rows) == p['budget'][arm][key], tag + f' {arm} stored budget')
            for i, x in enumerate(content):
                if x is None:
                    continue
                where = f'{arm}/{L}/content{i}'
                slot = 'new' if i < 4*n else 'replay' if i < 8*n else 'third'
                C.require(x['slot'] == slot, where, 'slot/interleave')
                ids, loss_start, n_loss = x['input_ids'], x['loss_start'], x['n_loss']
                C.require(isinstance(ids, list) and all(type(t) is int and t >= 0 for t in ids) and type(loss_start) is int and
                          type(n_loss) is int and 0 <= loss_start < len(ids) and len(ids)-loss_start == n_loss, where, 'token/loss layout')
                C.require(test or len(ids) <= 256, where, 'over 256 tokens')
                if slot == 'new':
                    expected = new_expected[i]
                elif slot == 'replay' and rp:
                    expected = replay_expected[i-4*n]
                elif slot == 'third' and arm != 'P' and (arm.endswith('4') or i < 10*n):
                    expected = dict((base_g if arm.startswith('G') else base)[(i-8*n) % (2*n)])
                else:
                    expected = dict(kind='bg', slot=slot, cell=None, src_cell=None, wording=None,
                                    label=None, item_id=None, source_item_id=None)
                # Replay has null item_id; source_item_id identifies the latest earlier item.
                C.require(set(expected) <= set(x) and all(x.get(k) == v for k, v in expected.items()), where, 'semantic item/source/oracle mismatch')
                if x['kind'] == 'bg':
                    if bg_tokens is not None:
                        j = i-4*n if slot == 'replay' else i-8*n
                        idx = 1_000_000+j if slot == 'replay' else 2_000_000+L*10_000+j
                        start = random.Random(f'CONS0A-bg-SUPPORT1-{seed}-{idx}').randrange(max(1, len(bg_tokens)-len(ids)))
                        C.require(ids == bg_tokens[start:start+len(ids)], where, 'background chunk identity')
                elif tok:
                    c = tuple(x['cell']); token_key = (c, x['wording'], x['label'])
                    if token_key not in token_cache:
                        token_cache[token_key] = qa(tok, c, x['wording'], x['label'])
                    C.require((ids, loss_start, n_loss) == token_cache[token_key], where, 'QA tokenization/frame/label')
            if arm != 'P':
                for i in range(8*n):
                    ck(content[i] == contents['P'][i], f'{tag}/{arm}/{i} identical positive/replay positions')
            if not test or arm in ('P', 'U1', 'U4'):
                ck(budget(rows) == budget(p['schedule']['P'][key]), tag + f' {arm} budget equality')
        for i in range(8*n, 16*n):
            ref = contents['U4'][i]
            for arm in ARMS:
                x = contents[arm][i]
                if not test or arm in ('P', 'U1', 'U4'):
                    ck((len(x['input_ids']), x['n_loss']) == (len(ref['input_ids']), ref['n_loss']), f'{tag}/{arm}/{i} position token/loss match')
                if x['kind'] == 'bg':
                    ck(x == contents['P'][i], f'{tag}/{arm}/{i} same background')
        prev_pos += pos


def check_fork(p, C, tok):
    ck = lambda ok, msg: C.require(ok, 'fork', msg)
    st, fork, seed = p['stream'], p['fork'], p['seed']
    targets = ordered(c for c in sets(st['transitions']) if c[0] in st['H'])
    ck(fork['targets'] == [list(c) for c in targets], 'target inventory')
    ck(len(fork['facts']) == len(set(fork['entities'])) == 4 and fork['entities'] == [f['entity'] for f in fork['facts']], 'distractor inventory')
    source_words = {w.lower() for e in [*st['intro'], *st['T'], *st['V']] for w in e.split()}
    ck(not source_words & {w.lower() for e in fork['entities'] for w in e.split()}, 'fork name-word collision')
    for i, (f, target) in enumerate(zip(fork['facts'], targets)):
        ck(f['fid'] == f'F{seed}L4D{i}' and f['life'] == 4 and f['role'] == 'fork_distractor' and
           f['target'] == list(target) and f['attr'] == target[1], 'distractor identity/attribute')
        ck(f['value'] not in {ff['value'] for ff in st['facts']}, 'distractor reuses stream value')
        if tok and not p['test_mode']:
            for k in (0, 1):
                a = qa(tok, cell(f), k, f['value']); b = qa(tok, target, k, oracle(st['facts'], 4)[target])
                ck((len(a[0]), a[2]) == (len(b[0]), b[2]), 'distractor token/loss mismatch')
    ck(p['fork_schedule']['L'] == 4, 'wrong fork sleep')
    bytarget = {tuple(f['target']): f for f in fork['facts']}
    for arm in ('U4', 'G4'):
        main, branch = p['schedule'][arm]['4'], p['fork_schedule'][arm]
        ck(len(main) == len(branch), f'{arm} fork schedule length')
        changes = 0
        for a, b in zip(main, branch):
            c = tuple(a['cell']) if a['cell'] else None
            if a['kind'] == 'pos' and c in bytarget:
                changes += 1; f = bytarget[c]
                expected = dict(a, cell=list(cell(f)), label=f['value'], item_id=f"{f['fid']}-q{a['wording']}")
                if tok:
                    ids, start, loss = qa(tok, cell(f), a['wording'], f['value'])
                    expected.update(input_ids=ids, loss_start=start, n_loss=loss)
                else:
                    expected.update({k: b[k] for k in ('input_ids', 'loss_start', 'n_loss')})
                ck(b == expected, f'{arm} replacement position/semantic/token')
                if not p['test_mode']:
                    ck((len(a['input_ids']), a['n_loss']) == (len(b['input_ids']), b['n_loss']), f'{arm} replacement budget')
            else:
                ck(a == b, f'{arm} changed non-target presentation')
            ck(not (b['kind'] in ('pos', 'replay') and b['cell'] and tuple(b['cell']) in bytarget), f'{arm} target supervision leaked')
        ck(changes == 32, f'{arm} must replace all 32 target presentations')


def check_probes(p, C):
    seed, st = p['seed'], p['stream']
    ck = lambda ok, msg: C.require(ok, 'probes', msg)
    rev, how, sel = sets(p['strata']['rev']), sets(p['split']['how']), sets(p['split']['sel'])
    updates = sets(st['updates']); trans = sets(st['transitions'])
    ck(set(p['probes']) == {'observed', 'withheld'}, 'history inventory')
    ck(set(p['probes']['observed']) == set(map(str, range(1, 6))) and set(p['probes']['withheld']) == {'4'}, 'history sleep inventory')
    for h, lives in p['probes'].items():
        facts = history(p, h)
        intro = dict(st['intro'])
        if h == 'withheld':
            intro.update({e: 4 for e in p['fork']['entities']})
        for key, rows in lives.items():
            L = int(key); current = oracle(facts, L)
            familiar = lambda c: intro.get(c[0], 99) <= L
            delivered_pairs = {(tuple(x['cell']), x['wording']) for l in range(1, L+1)
                               for x in p['schedule']['U1'][str(l)] if x['kind'] == 'neg_a'}
            delivered_sel = {c for c, _ in delivered_pairs} & sel
            gtq = {(tuple(x['cell']), x['wording']) for l in range(1, L+1)
                   for x in p['schedule']['G1'][str(l)] if x['kind'] == 'neg_swap'}
            expected = collections.defaultdict(set)
            for f in facts:
                c = cell(f)
                if f['life'] > L or c in updates:
                    continue
                expected['POS'].add(c)
                if intro[c[0]] == L: expected['FIRST'].add(c)
                if f['life'] == 1: expected['RET'].add(c)
            for c in trans:
                if L < 4:
                    cohort = 'A' if c[0] in st['A'] else 'H'
                    expected['TPRE-'+cohort+('' if familiar(c) else '-UNFAM')].add(c)
                elif c in current:
                    expected['TPOST-'+('A' if c[0] in st['A'] else 'H')].add(c)
            expected['UPD'] = updates & set(current)
            expected['SEL-D'] = {c for c in delivered_sel if familiar(c)}
            expected['SEL-P'] = {c for c in sel-delivered_sel if familiar(c)}
            expected['HOW'] = {c for c in how if familiar(c)}
            hm = ordered(c for c in sets(p['strata']['hnm']) if familiar(c))
            expected['HNM'] = set(random.Random(f'SUPPORT1-hnm-{seed}-{L}').sample(hm, min(40, len(hm))))
            expected['FRESH'] = sets(st['fresh'])
            if L == 4:
                expected['FORK-H'] = sets(p['fork']['targets'])
                expected['FORK-H-Q'] = sets(p['fork']['targets'])
            groups = collections.defaultdict(list)
            ids = {r['probe_id']: r for r in rows}
            ck(len(ids) == len(rows), f'{h}/{L} duplicate ID')
            seen = set()
            for r in rows:
                cat, c, kind = r['cat'], tuple(r['cell']), r['kind']
                groups[cat].append(r)
                ck(cat in CATEGORIES and c[1] in ATTRS, f'{h}/{L} category/cell')
                ck((cat, c, kind) not in seen, f'{h}/{L} duplicate semantic probe')
                seen.add((cat, c, kind))
                gold = current.get(c, 'UNKNOWN')
                ck(r['gold_current'] == gold, f"{r['probe_id']} forged/incorrect temporal gold")
                if cat == 'UPD':
                    fs = sorted([f for f in facts if cell(f) == c], key=lambda f: f['life'])
                    ck(r['gold_old'] == fs[0]['value'] and r['gold_new'] == (fs[-1]['value'] if L >= 4 else None), 'update old/new temporal mask')
                else:
                    ck(r['gold_old'] is None and r['gold_new'] is None, 'unexpected update gold')
                if r['construction'] == 'statement':
                    ck(cat not in Q_FULL and cat != 'FORK-H-Q' and kind in STEMS[c[1]] and r['item'] == STEMS[c[1]][kind].format(e=c[0]), 'Statement wording/construction')
                    ck(c in expected[cat], f'{cat} cell membership')
                else:
                    ck(r['construction'] == 'question' and kind in ('qt0', 'qt1') and
                       r['item'] == question(c, int(kind[-1])), 'Question wording/construction')
                    k = int(kind[-1])
                    if cat == 'SEL-Q':
                        ck(c in sel and (c, k) in delivered_pairs and r['delivered_eligible'] is True, 'SEL-Q delivery eligibility')
                    elif cat == 'GTQ': ck((c, k) in gtq, 'GTQ not actually delivered')
                    elif cat == 'HOW-Q': ck(c in expected['HOW'], 'HOW-Q membership')
                    elif cat == 'HNM-Q': ck(c in expected['HNM'], 'HNM-Q membership')
                    elif cat == 'POS-Q': ck(c in expected['POS'], 'POS-Q non-update acquired membership')
                    elif cat == 'REV-Q': ck(c in rev and k == 0, 'fixed REV-Q cohort/wording')
                    elif cat == 'FORK-H-Q': ck(c in expected[cat] and k == 0, 'FORK-H-Q cohort')
                    else: ck(False, 'unexpected Question category')
                if cat in ('SEL-Q', 'HOW-Q'):
                    pair = ids.get(r['pair_probe_id'], {})
                    ck(pair.get('cell') == r['cell'] and pair.get('construction') == 'statement' and
                       pair.get('cat') == ('SEL-D' if cat == 'SEL-Q' else 'HOW'), 'construction twin linkage')
                else: ck(r['pair_probe_id'] is None, 'unexpected pair ID')
                if cat != 'SEL-Q': ck(r['delivered_eligible'] is None, 'unexpected eligibility flag')
            for cat, group in groups.items():
                ck([r['probe_id'] for r in group] == [f'{seed}-{h}-{L}-{cat}-{i:04d}' for i in range(len(group))], f'{cat} stable ID numbering')
            for cat, cells in expected.items():
                if cat == 'FORK-H-Q':
                    continue
                truth = {(c, k) for c in cells for k in (STEMS[c[1]] if cat in (*('POS', 'FIRST', 'RET'), 'TPOST-A', 'TPOST-H') else ('template',))}
                got = {(tuple(r['cell']), r['kind']) for r in groups[cat]}
                ck(got == truth and len(groups[cat]) == len(truth), f'{h}/{L}/{cat} exact inventory/kinds')
            sizes = {'POS-Q': len(expected['POS']), 'SEL-Q': sum(c in sel for c, _ in delivered_pairs),
                     'HOW-Q': len(expected['HOW']), 'HNM-Q': len(expected['HNM']), 'REV-Q': 4, 'GTQ': len(gtq)}
            # One RNG, consumed in table order; REV-Q is fixed, not sampled.
            rng = random.Random(f'SUPPORT1-qdiag-{seed}-{L}')
            def sampled_pairs(pop, k, pairs=False):
                pop = sorted(pop, key=lambda x: (x[0][0], ATTRS.index(x[0][1]), x[1])) if pairs else ordered(pop)
                chosen = rng.sample(pop, min(k, len(pop)))
                return chosen if pairs else [(c, i % 2) for i, c in enumerate(chosen)]
            sampled = {
                'POS-Q': sampled_pairs(expected['POS'], 10),
                'SEL-Q': sampled_pairs({(c, k) for c, k in delivered_pairs if c in sel}, 12, True),
                'HOW-Q': sampled_pairs(expected['HOW'], 12),
                'HNM-Q': sampled_pairs(expected['HNM'], 10),
                'REV-Q': [(c, 0) for c in ordered(rev)],
                'GTQ': sampled_pairs(gtq, 16, True)}
            for cat, truth in sampled.items():
                ck([(tuple(r['cell']), int(r['kind'][-1])) for r in groups[cat]] == truth,
                   f'{h}/{L}/{cat} deterministic sample/wording')
            for cat, maximum in Q_FULL.items():
                ck(len(groups[cat]) == min(maximum, sizes[cat]), f'{h}/{L}/{cat} fixed sample size')
            ck({tuple(r['cell']) for r in groups['REV-Q']} == rev, 'REV-Q all four cells')
            if L == 4:
                ck({tuple(r['cell']) for r in groups['FORK-H-Q']} == sets(p['fork']['targets']) and len(groups['FORK-H-Q']) == 4, 'fork question inventory')
    ck(p['qdiag_fallback'] == Q_SMALL, 'fallback allocations')


def validate(p, tok=None, expected_code=None, manifest=None):
    C = Checks()
    try:
        C.require(p['schema'] == SCHEMA and p['experiment'] == 'SUPPORT1' and type(p['seed']) is int and
                  type(p['test_mode']) is bool, 'header', 'schema/experiment/seed/test_mode')
        C.require(expected_code is None or p['code_hash'] == expected_code, 'header', 'CODE_HASH mismatch')
        C.require(manifest is None or p['model_manifest'] == manifest, 'header', 'model manifest mismatch')
        if not p['test_mode']:
            C.require(tok is not None, 'production', 'tokenizer is required')
            C.require(expected_code is not None, 'production', 'external code hash is required')
            C.require(isinstance(manifest, str) and re.fullmatch('[0-9a-f]{64}', manifest) is not None,
                      'production', 'external model-manifest hash is required')
        vecpath = ROOT / 'lab/support1_norm_vectors.json'
        vectors = json.loads(vecpath.read_text())
        C.require(p['normalisation'] == {'code': 'cls0c_b.norm', 'vectors_sha256': sha(vecpath.read_bytes())}, 'normaliser', 'vector file identity')
        C.require(len(vectors) == 15 and all(norm(s) == t for s, t in vectors), 'normaliser', 'frozen vector agreement')
        # Match the inherited word-token Jaccard contract, ignoring the entity placeholder.
        tokens = lambda s: set(re.findall(r'[a-z]+', s.replace('{e}', '').lower()))
        for attr in ATTRS:
            for q in QT[attr]:
                for stems in STEMS.values():
                    for stem in stems.values():
                        x, y = tokens('Question: '+q), tokens(stem)
                        # Construction labels are framing, not part of the stem.
                        x -= {'question'}; y -= {'statement'}
                        C.require(len(x & y)/len(x | y) < .8, 'leak', f'{q} vs {stem}')
        for name, fn in [('stream', check_stream), ('schedule', lambda p, c: check_schedule(p, c, tok)),
                         ('fork', lambda p, c: check_fork(p, c, tok)), ('probes', check_probes)]:
            try:
                fn(p, C)
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
                C.require(False, name, f'malformed artifact: {type(e).__name__}: {e}')
    except (KeyError, TypeError, ValueError) as e:
        C.require(False, 'header', f'malformed artifact: {e}')
    return {'seed': p.get('seed'), 'test_mode': p.get('test_mode'), 'checks': C.count,
            'errors': C.errors, 'applicable_checks_pass': not C.errors}


def project(preps, usd_per_hour=.86):
    """Phase-0 projection from protocol measured ABST-0 rates, not new timings.

    29 training minutes per seed; 2.4 eval minutes per 160 rows; phase 1 1.5 h.
    Phase-1 throughput must replace these estimates before a paid phase-2 run.
    """
    dev = [p for p in preps if p['seed'] in (20, 21, 22, 23)]
    plans = []
    for fork, q in ((True, 64), (False, 64), (True, 32), (False, 32)):
        train_seconds = eval_rows = main_items = fork_items = 0
        for p in dev:
            n = sum(len(rows) for lives in p['schedule'].values() for rows in lives.values())
            f = sum(len(p['fork_schedule'][arm]) for arm in ('U4', 'G4')) if fork else 0
            main_items += n; fork_items += f
            train_seconds += 29*60 * (1+f/n)
            for L in range(1, 6):
                eval_rows += 11*len(probes_for(p, 'observed', L, q))
            if fork:
                eval_rows += 4*len(probes_for(p, 'withheld', 4, q))
        seconds = train_seconds + eval_rows*144/160 + 1.5*3600
        plans.append(dict(fork=fork, qdiag=q, main_training_items=main_items,
                          fork_training_items=fork_items, evaluation_rows=eval_rows,
                          hours=seconds/3600, usd=seconds/3600*usd_per_hour))
    return dict(complete=set(p['seed'] for p in dev) == {20, 21, 22, 23} and len(dev) == 4,
                seeds=[p['seed'] for p in dev], test_mode=any(p['test_mode'] for p in preps),
                assumptions=dict(training_minutes_per_seed=29, eval_seconds_per_row=144/160,
                                 phase1_hours=1.5, usd_per_hour=usd_per_hour),
                plans=plans, ceiling_usd=30, measured_support1_throughput=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prep', nargs='+', required=True, type=Path)
    ap.add_argument('--tokenizer', help='local pinned tokenizer directory; CPU only')
    ap.add_argument('--model-manifest-sha')
    ap.add_argument('--code-root', type=Path, default=ROOT)
    ap.add_argument('--usd-per-hour', type=float, default=.86)
    ap.add_argument('--require-phase0', action='store_true', help='require all five seeds and a complete cost projection')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    results = []
    hashes = []
    preps = []
    for path in a.prep:
        data = path.read_bytes(); p = json.loads(data); preps.append(p)
        r = validate(p, tok, codehash(a.code_root), a.model_manifest_sha)
        results.append(dict(r, path=str(path), prep_sha256=sha(data), code_hash=p.get('code_hash')))
        hashes.append(sha(data))
    projection = project(preps, a.usd_per_hour)
    phase0_errors = []
    if a.require_phase0:
        if sorted(p['seed'] for p in preps) != [20, 21, 22, 23, 24]:
            phase0_errors.append('exactly seeds 20-24 are required')
        if not projection['complete']:
            phase0_errors.append('cost projection needs all four development preps')
        if not projection['plans'][-1]['usd'] <= 30:
            phase0_errors.append('no-fork/32 exceeds the $30 ceiling at the phase-0 rates')
    ok = all(r['applicable_checks_pass'] for r in results) and not phase0_errors
    test = any(r['test_mode'] is not False for r in results)
    verdict = 'FAIL' if not ok else 'TEST_ONLY' if test else 'PASS'
    report = {'validator': VERSION, 'validator_sha256': sha(Path(__file__).read_bytes()),
              'validator_code_hash': sha(b''.join(sha((ROOT/f).read_bytes()).encode() for f in
                  ('scripts/support1_sol_validate.py', 'scripts/support1_sol_spec.py', 'scripts/abst0_sol_probe_schema.json'))),
              'contract_commit': '981f0e77c0c5197b30482d7870af3a5b91cdad85', 'verdict': verdict,
              'prep_sha256': hashes, 'code_hash': codehash(a.code_root), 'test_mode': test,
              'tokenizer_checked': tok is not None, 'model_manifest': a.model_manifest_sha,
              'results': results, 'phase0_errors': phase0_errors, 'projection': projection, 'clarifications': ['unique-cell-wording replay; latest source',
              'A1 fixed one-to-one entity swap; all 18 framed forms', 'fork tokenizer matching', 'fixed QDIAG fallback',
              'background content indices', 'shared content interleave', 'tiny-only relaxations']}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'verdict': verdict, 'checks': sum(r['checks'] for r in results),
                      'errors': sum(len(r['errors']) for r in results)+len(phase0_errors), 'out': str(a.out)}))
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
