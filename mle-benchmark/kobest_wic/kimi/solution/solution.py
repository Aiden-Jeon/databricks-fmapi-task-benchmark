# -*- coding: utf-8 -*-
"""
KoBEST WiC (Word-in-Context) solution.

Approach
--------
The task: decide whether a target word (marked with [word]) is used with the
SAME sense in two short contexts.  No pretrained models / external data are
allowed, so we build an ensemble of strong classical features over the train
data only:

1. Smoothed per-word label prior (out-of-fold target encoding).
   Words in this dataset carry a strong prior tendency toward same/different
   sense pairings.  We encode it with out-of-fold (OOF) smoothing so the
   training-time cross-validation estimate is honest, and recompute the
   encoding on the FULL train set for test-time prediction.

2. Lexical/context similarity features (TF-IDF cosine similarities):
   - full-context char (2-4) and eojeol TF-IDF cosine similarity,
   - char-window (±3/5/8/12/20 chars around [word]) TF-IDF cosine similarity,
   - eojeol-window (±1/2/3 eojeol around [word]) TF-IDF cosine similarity,
   - eojeol-window Jaccard overlap,
   - right-particle (first syllable after "]") match indicator,
   - context length difference.

3. Distributional embedding similarity: a PPMI + SVD word embedding space is
   learned from the train contexts themselves (no external corpus).  Cosine
   similarity between the two context mean-vectors (with and without the
   target word) captures topical relatedness beyond surface overlap.

Model: average of Logistic Regression and Histogram Gradient Boosting
probabilities (chosen by 5-fold CV on train).

Reproduce:  python solution.py
Reads  train.csv / test.csv from the working directory,
writes outputs/submission.csv.
"""
import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RANDOM_STATE = 0
EMB_DIM = 100
N_SPLITS = 5


# --------------------------------------------------------------------------- #
# feature helpers
# --------------------------------------------------------------------------- #
def cos_sim(A, B):
    """Row-wise cosine similarity of two sparse matrices."""
    num = A.multiply(B).sum(axis=1).A.ravel()
    na = np.sqrt(A.multiply(A).sum(axis=1)).A.ravel()
    nb = np.sqrt(B.multiply(B).sum(axis=1)).A.ravel()
    return num / (na * nb + 1e-9)


def char_window(contexts, words, n):
    out = []
    for c, wi in zip(contexts, words):
        i = c.find('[' + wi + ']')
        if i < 0:
            out.append(c)
            continue
        j = i + len(wi) + 2
        out.append(c[max(0, i - n): j + n])
    return out


def eojeol_window(contexts, words, n):
    out = []
    for c, wi in zip(contexts, words):
        i = c.find('[' + wi + ']')
        if i < 0:
            out.append(c)
            continue
        j = i + len(wi) + 2
        out.append(' '.join(c[:i].split()[-n:] + [wi] + c[j:].split()[:n]))
    return out


def post_particle(contexts, words):
    """First syllable immediately after the closing bracket (조사 hint)."""
    out = []
    for c, wi in zip(contexts, words):
        i = c.find('[' + wi + ']')
        out.append('' if i < 0 else c[i + len(wi) + 2: i + len(wi) + 3])
    return out


def similarity_features(df, fit_corpus=None):
    """TF-IDF based similarity features.  Vectorizers are fitted on
    `fit_corpus` (train contexts) so test-time transformation is consistent."""
    a = df.context_1.tolist()
    b = df.context_2.tolist()
    w = df.word.tolist()

    if fit_corpus is None:
        fit_corpus = dict(
            full=a + b,
            cw={n: char_window(a, w, n) + char_window(b, w, n) for n in (3, 5, 8, 12, 20)},
            ew={n: eojeol_window(a, w, n) + eojeol_window(b, w, n) for n in (1, 2, 3)},
        )
        fit_corpus['vec_full_c'] = TfidfVectorizer(
            analyzer='char', ngram_range=(2, 4), min_df=2,
            sublinear_tf=True).fit(fit_corpus['full'])
        fit_corpus['vec_full_w'] = TfidfVectorizer(
            analyzer='word', token_pattern=r'\S+', min_df=2,
            sublinear_tf=True).fit(fit_corpus['full'])
        for n, texts in fit_corpus['cw'].items():
            fit_corpus[f'vec_cw{n}'] = TfidfVectorizer(
                analyzer='char', ngram_range=(2, 4), min_df=1,
                sublinear_tf=True).fit(texts)
        for n, texts in fit_corpus['ew'].items():
            fit_corpus[f'vec_ew{n}'] = TfidfVectorizer(
                token_pattern=r'\S+', min_df=1, sublinear_tf=True).fit(texts)

    fc = fit_corpus
    F = []
    # full-context similarities
    F.append(cos_sim(fc['vec_full_c'].transform(a), fc['vec_full_c'].transform(b)))
    F.append(cos_sim(fc['vec_full_w'].transform(a), fc['vec_full_w'].transform(b)))
    # char-window similarities
    for n in (3, 5, 8, 12, 20):
        wa, wb = char_window(a, w, n), char_window(b, w, n)
        v = fc[f'vec_cw{n}']
        F.append(cos_sim(v.transform(wa), v.transform(wb)))
    # eojeol-window similarities + Jaccard
    for n in (1, 2, 3):
        ea, eb = eojeol_window(a, w, n), eojeol_window(b, w, n)
        v = fc[f'vec_ew{n}']
        F.append(cos_sim(v.transform(ea), v.transform(eb)))
        F.append(np.array([
            len(set(x.split()) & set(y.split())) /
            (len(set(x.split()) | set(y.split())) + 1e-9)
            for x, y in zip(ea, eb)]))
    # right-particle match
    pa, pb = post_particle(a, w), post_particle(b, w)
    F.append(np.array([x == y and x != '' for x, y in zip(pa, pb)], float))
    # length difference
    F.append(np.abs(np.array([len(x) for x in a]) -
                    np.array([len(x) for x in b])) / 40.0)
    return np.vstack(F).T, fc


def tokenize(ctx):
    return [t.strip('[]').strip('.,;:!?"\'') for t in ctx.split()]


def build_embeddings(texts, dim):
    """PPMI + truncated-SVD word embeddings learned from `texts` only."""
    corpus = [tokenize(c) for c in texts]
    vocab = {}
    for s in corpus:
        for t in s:
            vocab.setdefault(t, len(vocab))
    V = len(vocab)
    co, unig = Counter(), Counter()
    for s in corpus:
        ids = [vocab[t] for t in s]
        unig.update(ids)
        for i in ids:
            for j in ids:
                if i != j:
                    co[(i, j)] += 1
    N = sum(unig.values())
    rows, cols, data = [], [], []
    for (i, j), c in co.items():
        pmi = np.log(c * N / (unig[i] * unig[j]) + 1e-9)
        if pmi > 0:
            rows.append(i); cols.append(j); data.append(pmi)
    M = csr_matrix((data, (rows, cols)), shape=(V, V))
    k = min(dim, V - 1)
    u, s, _ = svds(M, k=k)
    emb = u * np.sqrt(s)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return vocab, emb


def embedding_similarity(df, vocab, emb):
    dim = emb.shape[1]

    def ctx_vec(ctx, skip=None):
        ids = [vocab[t] for t in tokenize(ctx) if t in vocab and t != skip]
        return emb[ids].mean(0) if ids else np.zeros(dim)

    a = df.context_1.tolist(); b = df.context_2.tolist(); w = df.word.tolist()
    va = np.array([ctx_vec(c) for c in a])
    vb = np.array([ctx_vec(c) for c in b])
    s1 = (va * vb).sum(1)
    sa = np.array([ctx_vec(c, wi) for c, wi in zip(a, w)])
    sb = np.array([ctx_vec(c, wi) for c, wi in zip(b, w)])
    s2 = (sa * sb).sum(1)
    return np.vstack([s1, s2]).T


def word_prior(train_df, words, smooth=2.0):
    """Smoothed per-word label mean computed on (full) train."""
    gm = train_df.label.mean()
    stats = train_df.groupby('word')['label'].agg(['mean', 'count'])
    m = words.map(stats['mean']).fillna(gm)
    c = words.map(stats['count']).fillna(0)
    return ((m * c + smooth * gm) / (c + smooth)).values


def oof_word_prior(train_df, n_splits=N_SPLITS, seed=RANDOM_STATE, smooth=2.0):
    """Out-of-fold version (for honest training of the meta classifier)."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    pr = np.zeros(len(train_df))
    gm = train_df.label.mean()
    for tr_idx, te_idx in skf.split(train_df, train_df.label):
        sub = train_df.iloc[tr_idx]
        stats = sub.groupby('word')['label'].agg(['mean', 'count'])
        m = train_df.iloc[te_idx].word.map(stats['mean']).fillna(gm)
        c = train_df.iloc[te_idx].word.map(stats['count']).fillna(0)
        pr[te_idx] = (m * c + smooth * gm) / (c + smooth)
    return pr


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    train = pd.read_csv(os.path.join(root, 'train.csv'))
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    y = train.label.values

    # ---- similarity features (fit on train, applied to both) ----
    Xtr_sim, corpus = similarity_features(train)
    Xte_sim, _ = similarity_features(test, fit_corpus=corpus)

    # ---- distributional embeddings from train contexts ----
    all_train_ctx = pd.concat([train.context_1, train.context_2]).tolist()
    vocab, emb = build_embeddings(all_train_ctx, EMB_DIM)
    Xtr_emb = embedding_similarity(train, vocab, emb)
    Xte_emb = embedding_similarity(test, vocab, emb)

    # ---- word prior ----
    pr_tr = oof_word_prior(train)          # OOF for training
    pr_te = word_prior(train, test.word)   # full-train stats for test

    Xtr = np.hstack([pr_tr[:, None], Xtr_sim, Xtr_emb])
    Xte = np.hstack([pr_te[:, None], Xte_sim, Xte_emb])

    # ---- CV estimate ----
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    lr = LogisticRegression(C=1.0, max_iter=3000)
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=3, random_state=RANDOM_STATE)
    p1 = cross_val_predict(lr, Xtr, y, cv=skf, method='predict_proba')[:, 1]
    p2 = cross_val_predict(hgb, Xtr, y, cv=skf, method='predict_proba')[:, 1]
    acc = ((p1 + p2) / 2 > 0.5).astype(int)
    print(f'5-fold CV accuracy (LR+HGB blend): {(acc == y).mean():.4f}')

    # ---- final fit on all train ----
    lr.fit(Xtr, y)
    hgb.fit(Xtr, y)
    prob = (lr.predict_proba(Xte)[:, 1] + hgb.predict_proba(Xte)[:, 1]) / 2
    pred = (prob > 0.5).astype(int)

    out_dir = os.path.join(root, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    sub = pd.DataFrame({'id': test.id, 'label': pred})
    assert len(sub) == len(test) and sub.id.is_unique
    assert set(sub.id) == set(test.id)
    sub.to_csv(os.path.join(out_dir, 'submission.csv'), index=False)
    print('wrote', os.path.join(out_dir, 'submission.csv'), len(sub), 'rows')
    print('pred label distribution:', sub.label.value_counts().to_dict())


if __name__ == '__main__':
    main()
