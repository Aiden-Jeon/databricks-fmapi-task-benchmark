"""KoBEST COPA — final reproducible pipeline.

Formulation
-----------
Each row gives a premise, a question type (원인/결과) and two alternatives.
We learn a scoring function s(premise, question, alt) and predict the
alternative with the higher score.  Because a *linear* score satisfies
    s(a1) - s(a2) = w . (f(a1) - f(a2)),
we train a single logistic regression directly on the **delta feature
vector** f(a1) - f(a2) with target `1 if alternative_1 is correct`.
This is antisymmetric by construction (swapping the alternatives flips the
prediction).  A global intercept plus one non-delta indicator for the
question type absorb the dataset's positional prior
(train: 54.3% label 0 overall, 57.7% for 원인 vs 50.9% for 결과).

Feature blocks (delta-encoded, TF-IDF, L2-normalised)
  * alt_doc   : surface form of the alternative (char 2-4 grams, pseudo-stems,
                eojeol endings, bigrams, final predicate), each also crossed
                with the question type.                       (min_df=1)
  * cross_doc : premise-stem x alternative-predicate interaction tokens
                (crude causal-association memory).            (min_df=2)
  * NUM       : 16 hand-crafted numeric features (lengths, lexical/char
                overlap with the premise, negation cues, ...), standardised.
Plus one item-level (non-delta) feature: 1 if question == 원인.

Validation: 5-fold x 5-seed CV with every vectorizer/scaler fitted on the
training folds only (no leakage) -> accuracy 0.6175 +- 0.0021
(majority / all-zeros baseline = 0.5431).  See cv_honest.py and README.md.

Usage:  cd solution && python run.py
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

import features as F
from cv_honest import fold_matrix
from features import numeric_block

DATA = '..'
OUT = os.path.join(DATA, 'outputs', 'submission.csv')

SPEC = [('alt', F.alt_doc, 1, 1.0),
        ('x', F.cross_doc, 2, 1.0),
        ('NUM', None, 0, 1.0)]
C_LR = 5.0


def load():
    tr = pd.read_csv(f'{DATA}/train.csv')
    te = pd.read_csv(f'{DATA}/test.csv')
    for d in (tr, te):
        d['question'] = d.question.astype(str).str.strip()
    return tr, te


def q_indicator(df):
    return sparse.csr_matrix((df.question == '원인').values.astype(float)[:, None])


def main():
    tr, te = load()
    target = 1 - tr.label.values        # 1 -> alternative_1 is the answer

    num_tr = (numeric_block(tr, 'alternative_1'), numeric_block(tr, 'alternative_2'))
    num_te = (numeric_block(te, 'alternative_1'), numeric_block(te, 'alternative_2'))
    all_idx = np.arange(len(tr))        # fit vectorizers/scaler on train only
    Xtr, Xte = fold_matrix(tr, te, SPEC, all_idx, num_tr, num_te, 'full_tr', 'full_te')
    Xtr = sparse.hstack([Xtr, q_indicator(tr)]).tocsr()
    Xte = sparse.hstack([Xte, q_indicator(te)]).tocsr()
    print('feature matrix:', Xtr.shape)

    clf = LogisticRegression(C=C_LR, max_iter=6000, solver='liblinear')
    clf.fit(Xtr, target)
    p1 = clf.predict_proba(Xte)[:, 1]   # P(alternative_1 is correct)
    pred = (p1 < 0.5).astype(int)       # label 1 == alternative_2

    sub = pd.DataFrame({'id': te.id, 'label': pred})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sub.to_csv(OUT, index=False)

    assert len(sub) == len(te) and sub.id.is_unique and set(sub.id) == set(te.id)
    assert sub.label.isin([0, 1]).all()
    print('train fit acc:', (clf.predict(Xtr) == target).mean())
    print('label distribution:', sub.label.value_counts().to_dict())
    print('wrote', OUT)


if __name__ == '__main__':
    main()
