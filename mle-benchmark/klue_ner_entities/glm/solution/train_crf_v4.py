"""CRF v4: ensemble of multiple seeds around best params + finer search.

- Search around c1=0.4..0.7, c2=0.1..0.2
- Train multiple CRFs with different random seeds on full data and ensemble
  BIO tags by majority vote at the character level.
"""
import os
import re
import string
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

import sklearn_crfsuite
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(ROOT, 'train.csv')
TEST_CSV = os.path.join(ROOT, 'test.csv')
SUBMISSION_CSV = os.path.join(ROOT, 'outputs', 'submission.csv')

LABELS = ['PS', 'LC', 'OG', 'DT', 'TI', 'QT']

ALL_TAGS = ['O'] + [f'{p}-{t}' for t in LABELS for p in ('B', 'I')]


def parse_entities(entities_str):
    out = []
    if entities_str:
        for p in entities_str.split('|'):
            if p and ':' in p:
                expr, typ = p.rsplit(':', 1)
                out.append((expr, typ))
    return out


def bio_tag(sent, exprs):
    tags = ['O'] * len(sent)
    used = [False] * len(sent)
    pos = 0
    for expr, typ in exprs:
        idx = sent.find(expr, pos)
        if idx == -1:
            idx = sent.find(expr)
        if idx == -1:
            continue
        if any(used[idx:idx + len(expr)]):
            start = idx + 1
            while True:
                idx = sent.find(expr, start)
                if idx == -1:
                    break
                if not any(used[idx:idx + len(expr)]):
                    break
                start = idx + 1
            if idx == -1:
                continue
        for j in range(idx, idx + len(expr)):
            used[j] = True
        tags[idx] = 'B-' + typ
        for j in range(idx + 1, idx + len(expr)):
            tags[j] = 'I-' + typ
        pos = idx + len(expr)
    return tags


HANGUL_JAMO = set(
    'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
    'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
)

def is_hangul_syllable(ch):
    return '가' <= ch <= '힣'

def is_hangul_jamo(ch):
    return ch in HANGUL_JAMO

def char_class(ch):
    if ch.isdigit():
        return 'D'
    if ch.isalpha() and ch.isascii():
        return 'A'
    if is_hangul_syllable(ch) or is_hangul_jamo(ch):
        return 'H'
    if ch.isspace():
        return 'S'
    if ch in string.punctuation or not ch.isalnum():
        return 'P'
    return 'O'


def build_gazetteer(train_df, min_count=1):
    gaz = defaultdict(Counter)
    for ents_str in train_df['entities']:
        for expr, typ in parse_entities(ents_str):
            gaz[expr][typ] += 1
    gaz2 = {}
    for expr, c in gaz.items():
        if sum(c.values()) >= min_count:
            gaz2[expr] = c
    return gaz2


def gazetteer_matches(sent, gaz):
    matches = []
    n = len(sent)
    lengths = sorted({len(e) for e in gaz.keys()})
    for L in lengths:
        if L == 0 or L > n:
            continue
        for i in range(0, n - L + 1):
            sub = sent[i:i + L]
            if sub in gaz:
                c = gaz[sub]
                typ = c.most_common(1)[0][0]
                matches.append((i, i + L, typ, sum(c.values())))
    return matches


def char_features(sent, i, gaz_start, gaz_inside):
    n = len(sent)
    ch = sent[i]
    f = {
        'bias': 1.0,
        'c': ch.lower(),
        'cls': char_class(ch),
        'is_digit': ch.isdigit(),
        'is_alpha': ch.isalpha(),
        'is_upper': ch.isupper(),
        'is_hangul': is_hangul_syllable(ch) or is_hangul_jamo(ch),
        'is_space': ch.isspace(),
        'is_punct': (ch in string.punctuation),
        'pos_bucket': min(i // 10, 14),
    }
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < n:
            f[f'c{off}'] = sent[j].lower()
            f[f'cls{off}'] = char_class(sent[j])
    if i == 0:
        f['BOS'] = True
    if i == n - 1:
        f['EOS'] = True
    classes = []
    for j in range(max(0, i-1), min(n, i+2)):
        classes.append(char_class(sent[j]))
    f['cls_win'] = ''.join(classes)
    f['in_digit_run'] = bool(re.search(r'\d', sent[max(0, i-2):min(n, i+3)]))

    if i in gaz_start:
        for typ, cnt in gaz_start[i].items():
            f['gaz_start_' + typ] = cnt
    if i in gaz_inside:
        f['gaz_inside_any'] = True
        for typ, cnt in gaz_inside[i].items():
            f['gaz_inside_' + typ] = cnt
    return f


def sent_features(sent, gaz):
    matches = gazetteer_matches(sent, gaz)
    gaz_start = defaultdict(Counter)
    gaz_inside = defaultdict(Counter)
    for (s, e, typ, cnt) in matches:
        gaz_start[s][typ] += cnt
        for j in range(s, e):
            gaz_inside[j][typ] += cnt
    return [char_features(sent, i, gaz_start, gaz_inside) for i in range(len(sent))]


def decode_bio(sent, tags):
    ents = []
    i = 0
    n = len(sent)
    while i < n:
        t = tags[i]
        if t.startswith('B-'):
            typ = t[2:]
            start = i
            i += 1
            while i < n and tags[i] == 'I-' + typ:
                i += 1
            expr = sent[start:i]
            ents.append((expr, typ))
        else:
            i += 1
    return ents


def entities_to_str(ents):
    return '|'.join(f'{e}:{t}' for e, t in ents)


def entity_f1(gold_list, pred_list):
    tp = fp = fn = 0
    for g, p in zip(gold_list, pred_list):
        gs = set(parse_entities(g)) if g else set()
        ps = set(parse_entities(p)) if p else set()
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec, tp, fp, fn


def evaluate(crf, X_va, va_sents, va_gold):
    y_va_pred = crf.predict(X_va)
    va_pred_str = [entities_to_str(decode_bio(s, t)) for s, t in zip(va_sents, y_va_pred)]
    return entity_f1(va_gold, va_pred_str)


def vote_tags(list_of_tag_seqs, n):
    """Majority vote per position. Resolve BIO validity softly by picking
    the most common tag; if 'I-X' is chosen at a position with no preceding B,
    decode_bio will skip it (treat as O-equivalent)."""
    out = []
    for i in range(n):
        c = Counter()
        for seq in list_of_tag_seqs:
            if i < len(seq):
                c[seq[i]] += 1
        if c:
            out.append(c.most_common(1)[0][0])
        else:
            out.append('O')
    return out


def main():
    print('Loading data...', flush=True)
    train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    test = pd.read_csv(TEST_CSV)

    print('Building gazetteer...', flush=True)
    gaz = build_gazetteer(train, min_count=1)
    print('Gazetteer size:', len(gaz), flush=True)

    print('Tagging train with BIO + features...', flush=True)
    X_full = []
    y_full = []
    for idx, row in train.iterrows():
        sent = row['sentence']
        ents = parse_entities(row['entities'])
        tags = bio_tag(sent, ents)
        X_full.append(sent_features(sent, gaz))
        y_full.append(tags)

    idxs = list(range(len(train)))
    tr_idx, va_idx = train_test_split(idxs, test_size=0.15, random_state=42)
    X_tr = [X_full[i] for i in tr_idx]
    y_tr = [y_full[i] for i in tr_idx]
    X_va = [X_full[i] for i in va_idx]
    y_va = [y_full[i] for i in va_idx]
    va_sents = [train.iloc[i]['sentence'] for i in va_idx]
    va_gold = [train.iloc[i]['entities'] for i in va_idx]

    best = None
    for c1 in [0.4, 0.5, 0.6, 0.7]:
        for c2 in [0.1, 0.15, 0.2]:
            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs', c1=c1, c2=c2,
                max_iterations=80, all_possible_transitions=True,
            )
            crf.fit(X_tr, y_tr)
            f1, p, r, tp, fp, fn = evaluate(crf, X_va, va_sents, va_gold)
            print(f'c1={c1} c2={c2} -> F1={f1:.4f} P={p:.4f} R={r:.4f}', flush=True)
            if best is None or f1 > best[0]:
                best = (f1, c1, c2, p, r)
    print('Best:', best, flush=True)
    bc1, bc2 = best[1], best[2]

    # ensemble: train 3 full CRFs with different seeds (crfsuite doesn't expose
    # random_state, so we vary subsample of training data via different seeds
    # for bagging effect) and average predictions.
    print('Training ensemble of 3 full CRFs (bagging)...', flush=True)
    rng = np.random.RandomState(123)
    full_models = []
    n_full = len(X_full)
    for m in range(3):
        # bag 90% of full data each time
        sample_idx = rng.choice(n_full, size=int(0.9 * n_full), replace=True)
        Xs = [X_full[i] for i in sample_idx]
        ys = [y_full[i] for i in sample_idx]
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs', c1=bc1, c2=bc2,
            max_iterations=120, all_possible_transitions=True,
        )
        crf.fit(Xs, ys)
        full_models.append(crf)
        print(f'  trained model {m+1}/3', flush=True)

    print('Predicting test with ensemble...', flush=True)
    X_test = [sent_features(s, gaz) for s in test['sentence']]
    pred_rows = []
    for sid, sent, X in zip(test['id'], test['sentence'], X_test):
        seqs = [m.predict([X])[0] for m in full_models]
        voted = vote_tags(seqs, len(sent))
        ents = decode_bio(sent, voted)
        pred_rows.append((sid, entities_to_str(ents)))

    out = pd.DataFrame(pred_rows, columns=['id', 'entities'])
    os.makedirs(os.path.dirname(SUBMISSION_CSV), exist_ok=True)
    out.to_csv(SUBMISSION_CSV, index=False)
    print('Wrote', SUBMISSION_CSV, 'rows=', len(out), flush=True)
    print('Non-empty predictions:', (out['entities'] != '').sum(), flush=True)


if __name__ == '__main__':
    main()
