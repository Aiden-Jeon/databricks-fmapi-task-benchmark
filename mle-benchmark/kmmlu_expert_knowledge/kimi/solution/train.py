"""KMMLU 4-choice MCQ — final training / prediction script.

Approach
--------
No internet / no pretrained models are available, so we rely on classical ML
(scikit-learn) trained only on the provided train.csv.

The task's label signal is largely question-style/topic dependent (questions
asking "which is NOT correct", numeric-answer questions, etc. correlate with
the answer position). The strongest validated model is a multinomial logistic
regression over character 3~5-gram TF-IDF features of the question, blended
50/50 with the same features computed over the full text (question + options),
which adds option-content information.

Validation: 3 seeds x 5-fold stratified CV (averaged out-of-fold accuracy):
  - char_wb(3,5) LR on question   : ~0.334
  - char_wb(3,5) LR on full text  : ~0.330
  - 50/50 probability blend       : ~0.339

Final predictions average the two models each trained via 5-fold
cross-fitting over 5 seeds (25 model fits each) on the full training data.

Usage: python solution/train.py
Reads:  train.csv, test.csv
Writes: outputs/submission.csv
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

LETTERS = ['A', 'B', 'C', 'D']
SEEDS = [0, 1, 2, 3, 4]
N_SPLITS = 5


def make_vectorizer():
    return TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=300000,
    )


def crossfit_predict(texts_tr, texts_te, y, C=0.3, seeds=SEEDS):
    """Train LR with K-fold cross-fitting across several seeds; average
    predicted probabilities on the test set."""
    vect = make_vectorizer()
    X = vect.fit_transform(texts_tr)
    Xte = vect.transform(texts_te)
    te_prob = np.zeros((Xte.shape[0], 4))
    for s in seeds:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
        for itr, _ in skf.split(X, y):
            clf = LogisticRegression(C=C, max_iter=3000)
            clf.fit(X[itr], y[itr])
            te_prob += clf.predict_proba(Xte)
    te_prob /= len(seeds) * N_SPLITS
    return te_prob


def main():
    tr = pd.read_csv('train.csv')
    te = pd.read_csv('test.csv')
    y = tr['label'].values - 1  # 0..3

    q_tr = tr['question'].astype(str).tolist()
    q_te = te['question'].astype(str).tolist()

    def full_text(df):
        return (df['question'].astype(str) + ' ' + df['A'].astype(str) + ' '
                + df['B'].astype(str) + ' ' + df['C'].astype(str) + ' '
                + df['D'].astype(str)).tolist()

    f_tr, f_te = full_text(tr), full_text(te)

    print('training question-only model ...')
    p_q = crossfit_predict(q_tr, q_te, y)
    print('training full-text model ...')
    p_f = crossfit_predict(f_tr, f_te, y)

    prob = 0.5 * p_q + 0.5 * p_f
    pred = prob.argmax(axis=1) + 1  # back to 1..4

    sub = pd.DataFrame({'id': te['id'], 'label': pred.astype(int)})
    assert len(sub) == len(te) and sub['id'].is_unique
    assert set(sub['label']).issubset({1, 2, 3, 4})
    sub.to_csv('outputs/submission.csv', index=False)
    print('wrote outputs/submission.csv')
    print(sub['label'].value_counts().sort_index())


if __name__ == '__main__':
    main()
