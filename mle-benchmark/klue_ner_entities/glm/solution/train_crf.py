"""Build BIO tagging and a character-level CRF for KLUE-NER.

Strategy:
- Convert each sentence to character-level BIO tags using greedy left-to-right
  matching of entity expressions (verified to always succeed on train).
- Extract rich per-character features (character, surrounding context, char
  classes like digit/hangul/alpha, prefix/suffix word cues, position).
- Train sklearn-crfsuite CRF.
- Decode test and reconstruct entities with greedy left-to-right grouping.
- Evaluate entity-level micro-F1 with a held-out split.
"""
import os
import sys
import re
import string
import pandas as pd
import numpy as np
from collections import Counter

import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(ROOT, 'train.csv')
TEST_CSV = os.path.join(ROOT, 'test.csv')
SUBMISSION_CSV = os.path.join(ROOT, 'outputs', 'submission.csv')

LABELS = ['PS', 'LC', 'OG', 'DT', 'TI', 'QT']


def parse_entities(entities_str):
    """Return list of (expression, type)."""
    out = []
    if entities_str:
        for p in entities_str.split('|'):
            if p and ':' in p:
                expr, typ = p.rsplit(':', 1)
                out.append((expr, typ))
    return out


def bio_tag(sent, exprs):
    """Greedy left-to-right matching -> BIO tags over characters."""
    tags = ['O'] * len(sent)
    used = [False] * len(sent)
    pos = 0
    for expr, typ in exprs:
        idx = sent.find(expr, pos)
        if idx == -1:
            idx = sent.find(expr)
        if idx == -1:
            continue
        # skip if overlapping
        if any(used[idx:idx + len(expr)]):
            # try later occurrences
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


# ---------- Features ----------

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


def char_features(sent, i):
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
    }
    # bigram of chars
    if i > 0:
        f['prev_c'] = sent[i-1].lower()
        f['prev_cls'] = char_class(sent[i-1])
    else:
        f['BOS'] = True
    if i < n - 1:
        f['next_c'] = sent[i+1].lower()
        f['next_cls'] = char_class(sent[i+1])
    else:
        f['EOS'] = True
    if i > 1:
        f['prev2_c'] = sent[i-2].lower()
    if i < n - 2:
        f['next2_c'] = sent[i+2].lower()
    # trigram window class
    classes = []
    for j in range(max(0, i-1), min(n, i+2)):
        classes.append(char_class(sent[j]))
    f['cls_win'] = ''.join(classes)
    # digit neighborhood: is this part of a number-like run?
    f['in_digit_run'] = bool(re.search(r'\d', sent[max(0, i-2):min(n, i+3)]))
    return f


def sent_features(sent):
    return [char_features(sent, i) for i in range(len(sent))]


# ---------- Decode BIO -> entities ----------

def decode_bio(sent, tags):
    """Convert per-char BIO tags to list of (expression, type)."""
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


# ---------- Entity-level micro-F1 ----------

def entity_f1(gold_list, pred_list):
    """gold_list, pred_list: lists of entity-string (pipe separated)."""
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


def main():
    print('Loading data...', flush=True)
    train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    test = pd.read_csv(TEST_CSV)

    print('Tagging train with BIO...', flush=True)
    X_train_full = []
    y_train_full = []
    for idx, row in train.iterrows():
        sent = row['sentence']
        ents = parse_entities(row['entities'])
        tags = bio_tag(sent, ents)
        X_train_full.append(sent_features(sent))
        y_train_full.append(tags)

    # Split for validation
    idxs = list(range(len(train)))
    tr_idx, va_idx = train_test_split(idxs, test_size=0.15, random_state=42)
    X_tr = [X_train_full[i] for i in tr_idx]
    y_tr = [y_train_full[i] for i in tr_idx]
    X_va = [X_train_full[i] for i in va_idx]
    y_va = [y_train_full[i] for i in va_idx]
    va_sents = [train.iloc[i]['sentence'] for i in va_idx]
    va_gold = [train.iloc[i]['entities'] for i in va_idx]

    print('Training CRF (validation split)...', flush=True)
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1, c2=0.1,
        max_iterations=80,
        all_possible_transitions=True,
    )
    crf.fit(X_tr, y_tr)
    y_va_pred = crf.predict(X_va)
    va_pred_str = []
    for sent, tags in zip(va_sents, y_va_pred):
        ents = decode_bio(sent, tags)
        va_pred_str.append(entities_to_str(ents))
    f1, p, r, tp, fp, fn = entity_f1(va_gold, va_pred_str)
    print(f'Validation entity F1={f1:.4f} P={p:.4f} R={r:.4f} TP={tp} FP={fp} FN={fn}', flush=True)

    print('Training CRF on full data...', flush=True)
    crf_full = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1, c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf_full.fit(X_train_full, y_train_full)

    print('Predicting test...', flush=True)
    X_test = [sent_features(s) for s in test['sentence']]
    y_test = crf_full.predict(X_test)

    rows = []
    for i, (sid, sent, tags) in enumerate(zip(test['id'], test['sentence'], y_test)):
        ents = decode_bio(sent, tags)
        rows.append((sid, entities_to_str(ents)))

    out = pd.DataFrame(rows, columns=['id', 'entities'])
    os.makedirs(os.path.dirname(SUBMISSION_CSV), exist_ok=True)
    out.to_csv(SUBMISSION_CSV, index=False)
    print('Wrote', SUBMISSION_CSV, 'rows=', len(out), flush=True)
    print('Non-empty predictions:', (out['entities'] != '').sum(), flush=True)


if __name__ == '__main__':
    main()
