"""Final solution for t21_kmmlu: Korean multiple-choice QA via TF-IDF + Logistic Regression.

Approach
--------
Each multiple-choice question is serialized into a single text block that
contains the question followed by all four options prefixed with explicit
position markers ([A], [B], [C], [D]). A multinomial (4-way) Logistic
Regression is trained directly on this representation to predict the label
(1..4) of the correct option.

Two TF-IDF feature sets are concatenated:
  - word 1-2 grams (captures Korean word/term overlap, esp. Hanja-stemmed and
    numbers/English tokens)
  - char 2-4 grams (captures sub-word morphology for Korean Hangul which has
    no whitespace tokenization, and spelling variants)

To reduce variance and improve robustness, predictions are averaged across
several regularization strengths (C) and training seeds, then the argmax
option is taken per question.

This stays strictly within the rules: only train.csv is used, no external
data, no internet, no pretrained weights.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


DATA = '/tmp/kmle/M3_t21_kmmlu_full_20260801_100439/task'
OUT_DIR = os.path.join(DATA, 'outputs')
SEEDS = [0, 42, 7, 2024]
C_VALUES = [0.3, 0.5]


def make_text(row):
    return (
        str(row['question'])
        + ' [A] ' + str(row['A'])
        + ' [B] ' + str(row['B'])
        + ' [C] ' + str(row['C'])
        + ' [D] ' + str(row['D'])
    )


def build_features(train_df, test_df):
    vec_w = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95,
        analyzer='word', token_pattern=r'(?u)\b\w+\b',
    )
    Xw_tr = vec_w.fit_transform(train_df['fulltext'])
    Xw_te = vec_w.transform(test_df['fulltext'])

    vec_c = TfidfVectorizer(
        ngram_range=(2, 4), sublinear_tf=True, min_df=2, max_df=0.95,
        analyzer='char',
    )
    Xc_tr = vec_c.fit_transform(train_df['fulltext'])
    Xc_te = vec_c.transform(test_df['fulltext'])

    Xtr = hstack([Xw_tr, Xc_tr]).tocsr()
    Xte = hstack([Xw_te, Xc_te]).tocsr()
    return Xtr, Xte


def main():
    train = pd.read_csv(os.path.join(DATA, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA, 'test.csv'))

    train['fulltext'] = train.apply(make_text, axis=1)
    test['fulltext'] = test.apply(make_text, axis=1)
    y = train['label'].values

    Xtr, Xte = build_features(train, test)
    print('Train X:', Xtr.shape, 'Test X:', Xte.shape)

    # Quick CV for monitoring (not used for prediction)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    for tr_idx, va_idx in skf.split(Xtr, y):
        clf = LogisticRegression(C=0.3, max_iter=3000, solver='liblinear')
        clf.fit(Xtr[tr_idx], y[tr_idx])
        cv_accs.append(clf.score(Xtr[va_idx], y[va_idx]))
    print('5-fold CV acc (C=0.3): %.4f +/- %.4f' % (np.mean(cv_accs), np.std(cv_accs)))

    # Ensemble: average predicted probabilities across (seed, C) configs
    proba_sum = np.zeros((len(test), 4))
    n_models = 0
    for seed in SEEDS:
        for C in C_VALUES:
            clf = LogisticRegression(
                C=C, max_iter=3000, solver='liblinear', random_state=seed
            )
            clf.fit(Xtr, y)
            proba_sum += clf.predict_proba(Xte)
            n_models += 1
    proba = proba_sum / n_models
    pred = proba.argmax(axis=1) + 1  # labels are 1..4

    out = pd.DataFrame({'id': test['id'].values, 'label': pred})
    out = out.sort_values('id').reset_index(drop=True)

    assert set(out['id']) == set(test['id'])
    assert len(out) == len(test)
    assert out['label'].isin([1, 2, 3, 4]).all()

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, 'submission.csv'), index=False)
    print('Saved submission with', len(out), 'rows')
    print('Pred label dist:')
    print(out['label'].value_counts().sort_index())


if __name__ == '__main__':
    main()
