# -*- coding: utf-8 -*-
"""
KorQuAD-style extractive MRC solution (no deep learning / no internet).

Two-stage classical ML:
  Stage 1 (retrieval): pick the context sentences that best match the
      question (jamo-decomposed char 3-5 gram TF-IDF cosine + word overlap
      + char-bigram dice).
  Stage 2 (extraction): candidate answer spans = contiguous eojeol sequences
      (1..MAX_EOJEOL) plus josa-stripped head variants. Each occurrence is
      scored by a LogisticRegression over handcrafted features; features of
      identical candidate strings are max-pooled; a learned question-type
      prior feature helps numeric/date questions. Best candidate wins.

Trains on train.csv with an article-grouped train/valid split for tuning,
then retrains on full data and writes outputs/submission.csv.
"""
import os
import re
import json
import time
import math
import unicodedata
import collections

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
OUT_DIR = os.path.join(BASE, "outputs")
SUB_PATH = os.path.join(OUT_DIR, "submission.csv")
SEED = 42
MAX_EOJEOL = 4
TOPN_RETR = 2

rng = np.random.RandomState(SEED)

JONG = u'ᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'
JOSA_SET = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도',
            '만', '로', '에서', '에게', '께서', '부터', '까지', '보다', '처럼',
            '같이', '으로', '로서', '로써', '이라', '이다', '이며', '이고'}
STOPWORDS = {'것', '수', '등', '및', '또한', '그리고', '하지만', '그러나', '때문',
             '경우', '이것', '그것', '저것', '여기', '거기', '자신', '다른',
             '어떤', '이러한', '그런', '이런', '아주', '매우', '가장'}

RE_YEAR = re.compile(r'^\d{1,4}년$')
EDGE_STRIP = '.,!?;:"\'()[]{}<>《》「」『』“”‘’·-—~'


def has_jong(ch):
    return ch in JONG


def ends_jong(word):
    word = word.strip()
    return bool(word) and has_jong(word[-1])


def decompose(text):
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            code -= 0xAC00
            cho = code // 588
            jung = (code % 588) // 28
            jong = code % 28
            out.append(chr(0x1100 + cho))
            out.append(chr(0x1161 + jung))
            if jong:
                out.append(chr(0x11A8 - 1 + jong))
        else:
            out.append(ch)
    return ''.join(out)


def sent_split(context):
    sents = []
    start = 0
    for m in re.finditer(r'(?<=[.!?])\s+', context):
        sents.append((start, m.start()))
        start = m.end()
    sents.append((start, len(context)))
    return [(s, e) for s, e in sents if e - s > 1]


def tokenize_stripped(sent_text):
    toks = []
    for m in re.finditer(r'\S+', sent_text):
        s, e = m.start(), m.end()
        while s < e and not (sent_text[s].isalnum() or ord(sent_text[s]) >= 0xAC00):
            s += 1
        while e > s and not (sent_text[e - 1].isalnum() or ord(sent_text[e - 1]) >= 0xAC00):
            e -= 1
        if s < e:
            toks.append((sent_text[s:e], s, e))
    return toks


def content_words(text):
    out = []
    for w in re.findall(r'\S+', text):
        w2 = w.strip(EDGE_STRIP)
        if w2 and (len(w2) >= 2 or any(ch.isdigit() for ch in w2)):
            out.append(w2)
    return out


def char_ngrams(text, n=2):
    t = re.sub(r'\s+', '', text)
    return set(t[i:i + n] for i in range(len(t) - n + 1))


def classify_question(q):
    if re.search(r'(언제|몇\s*년|어느\s*해|몇\s*월|몇\s*일|기간|연도)', q):
        return 'when'
    if re.search(r'(누구|누가|어느\s*(사람|인물|작가|화가|예술가|왕|대통령)|제작자|저자)', q):
        return 'who'
    if re.search(r'(어디|어느\s*(곳|도시|나라|국가|지역|장소|대학|학교))', q):
        return 'where'
    if re.search(r'(몇|얼마|수치|비율)', q):
        return 'number'
    if re.search(r'(왜|이유|원인)', q):
        return 'why'
    if re.search(r'(어느|어떤|어떻게|얼마나)', q):
        return 'which'
    return 'what'


def type_bonus(cand, qtype):
    b = 0.0
    has_digit = any(ch.isdigit() for ch in cand)
    is_year = bool(RE_YEAR.match(cand.strip()))
    if qtype == 'when':
        b += 3.0 if is_year else (2.0 if has_digit else 0.0)
    elif qtype == 'number':
        if has_digit:
            b += 2.5
        elif cand.strip().endswith(('명', '개', '대', '번', '회', '세')):
            b += 1.0
    elif qtype in ('who', 'where', 'which'):
        if is_year:
            b -= 2.0
        elif has_digit:
            b -= 1.0
    return b


class JamoRetrieval:
    def __init__(self):
        self.vec = TfidfVectorizer(
            analyzer='char', ngram_range=(3, 5), min_df=2, max_df=0.9,
            preprocessor=self._prep, lowercase=False)

    @staticmethod
    def _prep(t):
        return decompose(unicodedata.normalize('NFC', t))

    def fit(self, texts):
        self.vec.fit(texts)

    def transform(self, texts):
        return self.vec.transform(texts)


def align_tokens_for_gold(sent_text, toks, gold):
    """Split a token so the gold answer becomes exactly recoverable."""
    pos = 0
    while True:
        idx = sent_text.find(gold, pos)
        if idx < 0:
            return toks
        pos = idx + 1
        g0, g1 = idx, idx + len(gold)
        if g0 > 0 and not sent_text[g0 - 1].isspace():
            continue  # gold starts mid-token; skip (rare)
        toks2 = []
        for w, s, e in toks:
            if s < g1 and e > g1:
                # token extends past gold end: split
                toks2.append((sent_text[s:g1], s, g1))
                toks2.append((sent_text[g1:e], g1, e))
            else:
                toks2.append((w, s, e))
        return toks2


def generate_candidates(toks):
    """Candidate spans from stripped tokens: (cand, ti, tj)."""
    cands = []
    n = len(toks)
    for i in range(n):
        for j in range(i, min(i + MAX_EOJEOL, n)):
            cand = ' '.join(t[0] for t in toks[i:j + 1])
            if cand and cand not in STOPWORDS:
                cands.append((cand, i, j))
        w = toks[i][0]
        if len(w) >= 3:
            for k in (2, 1):
                if len(w) - k >= 2 and w[-k:] in JOSA_SET:
                    cands.append((w[:-k], i, i))
                    break
    return cands


def build_rows(context, question, qtype, qv, sents, sent_mat,
               gold=None, gold_stats=None, is_train=True):
    """Return list of (cand, feat, is_gold) pooled per candidate string."""
    q_content = content_words(question)
    q_set = set(q_content)
    q_ngrams = char_ngrams(''.join(q_content), 2)
    qj = re.sub(r'\s+', '', question)

    tf_scores = (sent_mat @ qv.T).toarray().ravel() if sent_mat.shape[0] else np.zeros(0)
    scores = []
    for idx, (st, se) in enumerate(sents):
        stext = context[st:se]
        cw = set(content_words(stext))
        overlap_ratio = len(q_set & cw) / max(1, len(q_set))
        c_ngrams = char_ngrams(''.join(stext.split()), 2)
        dice = 2 * len(q_ngrams & c_ngrams) / max(1, len(q_ngrams) + len(c_ngrams))
        scores.append(2.0 * tf_scores[idx] + 0.8 * overlap_ratio + 1.2 * dice)
    scores = np.asarray(scores)
    topn = TOPN_RETR if is_train else 4
    top = np.argsort(-scores)[:topn]
    if len(top) > 1:
        for k in range(1, len(top)):
            for m in range(k):
                if abs(top[k] - top[m]) == 1:
                    scores[top[k]] += 0.5  # adjacent: likely one sentence split
                    break
    # keep top2 + any additional sentence whose score is close to #2
    keep = list(top[:2])
    for i in top[2:]:
        if scores[i] >= scores[top[1]] - 0.05:
            keep.append(i)
    order = sorted(keep, key=lambda i: -scores[i])

    pooled = {}
    gold_found = False
    for sidx in order:
        st, se = sents[sidx]
        stext = context[st:se]
        toks = tokenize_stripped(stext)
        if not toks:
            continue
        if gold is not None and gold in stext:
            toks = align_tokens_for_gold(stext, toks, gold)
        retr = scores[sidx]
        cw = set(content_words(stext))
        overlap_ratio = len(q_set & cw) / max(1, len(q_set))
        # token positions matching question content words
        qmatch = []
        for ti, (tok, _, _) in enumerate(toks):
            for qw in q_content:
                if tok == qw or (len(qw) >= 2 and qw in tok) or (len(tok) >= 2 and tok in qw):
                    qmatch.append(ti)
                    break
        ntok = len(toks)
        sent1_end = next((i for i, t in enumerate(toks)
                          if t[0][-1] in '.!?' and len(t[0]) > 1), ntok - 1)
        for cand, ti, tj in generate_candidates(toks):
            if qmatch:
                d = min(max(0, ti - m, m - tj) for m in qmatch)
            else:
                d = ntok
            prox = 1.0 / (1.0 + d)
            before = ' '.join(t[0] for t in toks[:ti])
            overlap_before = len(q_set & set(content_words(before))) / max(1, len(q_set))
            c_nospace = cand.replace(' ', '')
            c_ng = char_ngrams(c_nospace, 2)
            dice_q = 2 * len(q_ngrams & c_ng) / max(1, len(q_ngrams) + len(c_ng))
            in_q = 1.0 if c_nospace and c_nospace in qj else 0.0
            has_digit = 1.0 if any(ch.isdigit() for ch in cand) else 0.0
            is_year = 1.0 if RE_YEAR.match(cand.strip()) else 0.0
            neoj = tj - ti + 1
            short_in_q = in_q if neoj <= 2 else 0.0
            first_sent = 1.0 if ti <= sent1_end else 0.0
            if neoj <= 2 and has_digit and is_year and in_q:
                type_pen = -8.0  # short year span straight from question: almost never the answer
            else:
                type_pen = 0.0
            feat = (retr, overlap_ratio, prox, overlap_before,
                    ti / max(1, ntok), dice_q, in_q,
                    1.0 if ends_jong(cand) else 0.0,
                    has_digit, is_year,
                    min(len(cand), 20) / 20.0, neoj / MAX_EOJEOL,
                    type_bonus(cand, qtype),
                    short_in_q, first_sent, type_pen)
            is_gold = 1 if (gold is not None and cand == gold) else 0
            if cand in pooled:
                p = pooled[cand]
                p[0] = tuple(max(a, b) for a, b in zip(p[0], feat))
                p[1] = max(p[1], is_gold)
            else:
                pooled[cand] = [feat, is_gold]
            gold_found = gold_found or bool(is_gold)
    if gold_stats is not None and gold is not None:
        gold_stats['total'] += 1
        gold_stats['found'] += int(gold_found)
    out = []
    for c, v in pooled.items():
        f1g = korquad_f1(c, gold) if (gold is not None and not v[1]) else 0.0
        out.append((c, v[0], v[1], f1g))
    return out


def korquad_f1(pred, gold):
    p = collections.Counter(re.sub(r'\s+', '', str(pred)))
    g = collections.Counter(re.sub(r'\s+', '', str(gold)))
    if not g:
        return 1.0 if not p else 0.0
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    prec = common / max(1, sum(p.values()))
    rec = common / max(1, sum(g.values()))
    return 2 * prec * rec / (prec + rec)


def prepare(df, retr, is_train, gold_stats=None, tag=''):
    all_rows, meta = [], []
    grouped = df.groupby(df['context'], sort=False)
    n_groups = df['context'].nunique()
    t0 = time.time()
    for gi, (ctx, sub) in enumerate(grouped):
        sents = sent_split(ctx)
        sent_texts = [ctx[s:e] for s, e in sents]
        sent_mat = retr.transform(sent_texts) if sent_texts else csr_matrix((0, 4))
        qs = list(sub['question'])
        qmat = retr.transform(qs)
        for qi, (_, r) in enumerate(sub.iterrows()):
            qtype = classify_question(r['question'])
            gold = r['answer'] if is_train else None
            rows = build_rows(ctx, r['question'], qtype, qmat[qi], sents, sent_mat,
                              gold=gold, gold_stats=gold_stats, is_train=is_train)
            all_rows.append(rows)
            meta.append((r['id'], qtype, gold))
        if gi % 500 == 0:
            print(f'  [{tag}] group {gi}/{n_groups} ({time.time()-t0:.0f}s)', flush=True)
    print(f'  [{tag}] done {n_groups} groups in {time.time()-t0:.0f}s', flush=True)
    return all_rows, meta


def rows_to_xy(all_rows, neg_per_row=80, seed=SEED):
    """Keep all positives + best partial-F1 negs + sampled negs."""
    r = np.random.RandomState(seed)
    X, y, w = [], [], []
    for rows in all_rows:
        pos = [t for t in rows if t[2] == 1]
        neg = [t for t in rows if t[2] != 1]
        if not pos:
            continue
        keep = list(pos)
        neg.sort(key=lambda t: -t[3])  # partial-F1 hard negatives first
        keep.extend(neg[:5])
        rest = neg[5:]
        if len(rest) > neg_per_row:
            idx = r.choice(len(rest), neg_per_row, replace=False)
            keep.extend(rest[i] for i in idx)
        else:
            keep.extend(rest)
        for cand, feat, is_gold, f1g in keep:
            X.append(feat)
            y.append(is_gold)
            w.append(5.0 if is_gold else (1.0 + 2.0 * f1g))
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32),
            np.asarray(w, dtype=np.float32))


def predict_all(all_rows, meta, clf, extra_scale=0.0):
    preds = {}
    for i, rows in enumerate(all_rows):
        rid, qtype, _ = meta[i]
        if not rows:
            preds[rid] = ''
            continue
        X = np.asarray([f for _, f, _, _ in rows], dtype=np.float32)
        if hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X)
        else:
            scores = clf.predict_proba(X)[:, 1]
        if extra_scale:
            tb = np.asarray([type_bonus(c, qtype) for c, _, _, _ in rows])
            scores = scores + extra_scale * tb
        preds[rid] = rows[int(np.argmax(scores))][0]
    return preds


def eval_preds(preds, meta):
    return float(np.mean([korquad_f1(preds.get(r[0], ''), r[2]) for r in meta]))


def doc_group_split(tr, valid_frac, seed):
    r = np.random.RandomState(seed)
    doc_ids = tr['id'].str.rsplit('-', n=2).str[0]
    uniq = doc_ids.unique()
    r.shuffle(uniq)
    valid_docs = set(uniq[:int(len(uniq) * valid_frac)])
    return doc_ids.isin(valid_docs).values


def main():
    t_start = time.time()
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    for c in ('context', 'question', 'answer'):
        tr[c] = tr[c].astype(str)
    for c in ('context', 'question'):
        te[c] = te[c].astype(str)
    print(f'train {tr.shape}, test {te.shape}', flush=True)

    is_valid = doc_group_split(tr, 0.08, SEED)
    tr_train = tr[~is_valid].reset_index(drop=True)
    tr_valid = tr[is_valid].reset_index(drop=True)
    print(f'train rows {len(tr_train)}, valid rows {len(tr_valid)}', flush=True)

    print('fitting jamo tfidf...', flush=True)
    retr = JamoRetrieval()
    corpus = []
    for ctx in tr['context'].unique():
        corpus.extend(ctx[s:e] for s, e in sent_split(ctx))
    print('corpus sentences:', len(corpus), flush=True)
    retr.fit(corpus)

    gold_stats = {'total': 0, 'found': 0}
    print('building train features...', flush=True)
    tr_rows, tr_meta = prepare(tr_train, retr, True, gold_stats, 'train')
    print(f'  gold-in-candidates (oracle): '
          f'{gold_stats["found"]/max(1,gold_stats["total"]):.3f}', flush=True)
    print('building valid features...', flush=True)
    va_rows, va_meta = prepare(tr_valid, retr, True, gold_stats, 'valid')

    X, y, w = rows_to_xy(tr_rows)
    print(f'X {X.shape}, positive ratio {y.mean():.4f}', flush=True)

    print('training classifier...', flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=40, random_state=SEED)
    clf.fit(X, y, sample_weight=w)

    print('validating...', flush=True)
    best = (-1.0, 0.0)
    for scale in [0.0, 0.25, 0.5, 1.0]:
        preds = predict_all(va_rows, va_meta, clf, extra_scale=scale)
        f1 = eval_preds(preds, va_meta)
        print(f'  extra type scale {scale}: valid F1 {f1:.4f}', flush=True)
        if f1 > best[0]:
            best = (f1, scale)
    best_f1, best_scale = best
    print(f'best valid F1 {best_f1:.4f} at scale {best_scale}', flush=True)

    print('retraining on full train...', flush=True)
    Xv, yv, wv = rows_to_xy(va_rows)
    clf_full = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=40, random_state=SEED)
    clf_full.fit(np.vstack([X, Xv]), np.concatenate([y, yv]),
                 sample_weight=np.concatenate([w, wv]))

    print('predicting test...', flush=True)
    te_rows, te_meta = prepare(te, retr, False, None, 'test')
    preds = predict_all(te_rows, te_meta, clf_full, extra_scale=best_scale)

    os.makedirs(OUT_DIR, exist_ok=True)
    sub = pd.DataFrame({'id': te['id'],
                        'answer': [preds.get(rid, '') for rid in te['id']]})
    assert len(sub) == len(te) and sub['id'].is_unique
    sub.to_csv(SUB_PATH, index=False)
    print(f'wrote {SUB_PATH} ({time.time()-t_start:.0f}s)', flush=True)

    with open(os.path.join(BASE, 'solution', 'metrics.json'), 'w') as f:
        json.dump({'valid_f1': best_f1, 'extra_type_scale': best_scale}, f)


if __name__ == '__main__':
    main()
