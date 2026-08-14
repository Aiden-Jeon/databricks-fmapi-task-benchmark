"""Common utilities for KLUE-DP dependency parsing task."""
import json
import pandas as pd

LABELS = ['NP', 'NP_AJT', 'NP_CMP', 'NP_CNJ', 'NP_MOD', 'NP_OBJ', 'NP_SBJ',
          'VP', 'VP_AJT', 'VP_CMP', 'VP_CNJ', 'VP_MOD', 'VP_OBJ', 'VP_SBJ',
          'VNP', 'VNP_AJT', 'VNP_CMP', 'VNP_CNJ', 'VNP_MOD', 'VNP_OBJ', 'VNP_SBJ',
          'AP', 'AP_AJT', 'AP_CMP', 'AP_MOD', 'DP', 'IP', 'X', 'X_AJT', 'X_CMP',
          'X_CNJ', 'X_MOD', 'X_OBJ', 'X_SBJ', 'L', 'R']
LABEL_SET = set(LABELS)

CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
       'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
        'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def jamo_flat(s):
    """Decompose all Hangul syllables in s into a list of jamo (order preserved)."""
    out = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            o -= 0xAC00
            jong = o % 28
            jung = (o % 588) // 28
            cho = o // 588
            out.append(CHO[cho])
            out.append(JUNG[jung])
            if jong:
                out.append(JONG[jong])
        else:
            out.append(ch)
    return out


def load_train(path='train.csv'):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        toks = json.loads(r['tokens'])
        parse = r['parse'].split('|')
        heads, labels = [], []
        for p in parse:
            h, l = p.split(':')
            heads.append(int(h))
            labels.append(l)
        assert len(toks) == len(heads)
        rows.append({'id': r['id'], 'tokens': toks,
                     'heads': heads, 'labels': labels})
    return rows


def load_test(path='test.csv'):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({'id': r['id'], 'tokens': json.loads(r['tokens'])})
    return rows


def las(gold_rows, pred_rows):
    """Micro LAS between gold rows and predictions (dicts with heads/labels)."""
    pred_map = {p['id']: p for p in pred_rows}
    correct = total = 0
    for g in gold_rows:
        p = pred_map[g['id']]
        for gh, gl, ph, pl in zip(g['heads'], g['labels'],
                                  p['heads'], p['labels']):
            total += 1
            if gh == ph and gl == pl:
                correct += 1
    return correct / max(total, 1)


def write_submission(pred_rows, path):
    with open(path, 'w') as f:
        f.write('id,parse\n')
        for p in pred_rows:
            parse = '|'.join(f'{h}:{l}' for h, l in zip(p['heads'], p['labels']))
            f.write(f"{p['id']},\"{parse}\"\n")
