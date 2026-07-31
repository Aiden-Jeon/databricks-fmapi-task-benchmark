"""
KoBEST COPA - 인과 추론 (Causal Reasoning)
==========================================
Approach:
  Pairwise ranking. Each row produces two (premise, alternative) candidate
  pairs; the model scores how plausible each alternative is as the
  cause/effect of the premise, and the alternative with the higher score is
  selected as the predicted label (label=0 -> alternative_1, label=1 ->
  alternative_2).

  Features:
    - TF-IDF word n-grams (1-2) on "question premise alternative"
    - TF-IDF character n-grams (2-4, char_wb) on the same text
    - Handcrafted features: lexical overlap, char-bigram overlap, negation
      markers, tense (past/present) markers, length difference.

  Classifier: Logistic Regression (averaged over several seeds for stability).

  Cross-validated accuracy (5-fold group CV): ~0.60.

Constraints honoured: no internet / no external data / no pretrained weights;
only scikit-learn + pandas + numpy (all locally available).
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 0
N_SEEDS = 5
C = 4.0
WORD_NGRAM = (1, 2)
CHAR_NGRAM = (2, 4)
MIN_DF = 2

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)


# ----------------------------------------------------------------------
# Handcrafted feature helpers
# ----------------------------------------------------------------------
def _word_overlap(premise, alt):
    pw = set(re.findall(r'\w+', premise))
    aw = set(re.findall(r'\w+', alt))
    if not pw or not aw:
        return 0.0
    return len(pw & aw) / len(pw | aw)


def _char_bigram_overlap(premise, alt):
    p = premise.replace(' ', '')
    a = alt.replace(' ', '')
    pb = set(p[i:i + 2] for i in range(len(p) - 1))
    ab = set(a[i:i + 2] for i in range(len(a) - 1))
    if not pb or not ab:
        return 0.0
    return len(pb & ab) / len(pb | ab)


def _has_negation(s):
    return 1.0 if re.search(r'안|못|없|아니|않', s) else 0.0


def _is_past(s):
    return 1.0 if re.search(r'었|았|였|졌다|했다', s) else 0.0


def _is_present(s):
    return 1.0 if re.search(r'[^었았였]다\.?$', s) else 0.0


def _len_diff(premise, alt):
    return (len(premise) - len(alt)) / 30.0


EXTRA_COLS = [
    'ov', 'cbov', 'neg_alt', 'neg_pre',
    'past_alt', 'past_pre', 'pres_alt', 'len_diff',
]


def _extra_features(premise, alt):
    return {
        'ov': _word_overlap(premise, alt),
        'cbov': _char_bigram_overlap(premise, alt),
        'neg_alt': _has_negation(alt),
        'neg_pre': _has_negation(premise),
        'past_alt': _is_past(alt),
        'past_pre': _is_past(premise),
        'pres_alt': _is_present(alt),
        'len_diff': _len_diff(premise, alt),
    }


# ----------------------------------------------------------------------
# Build pair dataset
# ----------------------------------------------------------------------
def build_pairs(df, has_label=True):
    """Expand each row into two candidate pairs.

    Returns a DataFrame with one row per (premise, alternative) pair and a
    binary target `y` = 1 if this alternative is the correct one.
    """
    rows = []
    for _, r in df.iterrows():
        q = str(r['question']).strip()
        premise = str(r['premise'])
        alts = [str(r['alternative_1']), str(r['alternative_2'])]
        if has_label:
            labels = [1 if r['label'] == 0 else 0,
                      1 if r['label'] == 1 else 0]
        else:
            labels = [None, None]
        for pos, (alt, y) in enumerate(zip(alts, labels)):
            feats = _extra_features(premise, alt)
            feats.update({
                'gid': r['id'], 'q': q, 'premise': premise, 'alt': alt,
                'pos': pos, 'y': y,
            })
            rows.append(feats)
    pdf = pd.DataFrame(rows)
    pdf['text'] = (pdf['q'].astype(str) + ' ' + pdf['premise'].astype(str)
                  + ' ' + pdf['alt'].astype(str))
    return pdf


# ----------------------------------------------------------------------
# Train / predict
# ----------------------------------------------------------------------
def make_features(pdf_train, pdf_test):
    """Fit vectorizers on the training pairs and return sparse matrices."""
    wv = TfidfVectorizer(sublinear_tf=True, ngram_range=WORD_NGRAM,
                        min_df=MIN_DF).fit(pdf_train['text'])
    cv = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb',
                         ngram_range=CHAR_NGRAM, min_df=MIN_DF).fit(
                             pdf_train['text'])
    Xtr = hstack([wv.transform(pdf_train['text']),
                  cv.transform(pdf_train['text'])]).tocsr()
    Xte = hstack([wv.transform(pdf_test['text']),
                  cv.transform(pdf_test['text'])]).tocsr()
    Etr = pdf_train[EXTRA_COLS].values.astype(float)
    Ete = pdf_test[EXTRA_COLS].values.astype(float)
    Xtr = hstack([Xtr, csr_matrix(Etr)]).tocsr()
    Xte = hstack([Xte, csr_matrix(Ete)]).tocsr()
    return Xtr, Xte


def predict_pairs(pdf_train, pdf_test, n_seeds=N_SEEDS):
    """Return averaged P(correct) for each row in pdf_test."""
    Xtr, Xte = make_features(pdf_train, pdf_test)
    ytr = pdf_train['y'].values
    proba = np.zeros(len(pdf_test))
    for seed in range(n_seeds):
        clf = LogisticRegression(C=C, max_iter=3000,
                                 random_state=RANDOM_STATE + seed)
        clf.fit(Xtr, ytr)
        proba += clf.predict_proba(Xte)[:, 1]
    proba /= n_seeds
    return proba


def predict_rows(train_df, test_df):
    """Predict label (0/1) for each row of test_df."""
    pdf_train = build_pairs(train_df, has_label=True)
    pdf_test = build_pairs(test_df, has_label=False)
    proba = predict_pairs(pdf_train, pdf_test)

    # Group candidates by id; the two pairs were created in order
    # alternative_1 (pos=0) then alternative_2 (pos=1).
    preds = {}
    for gid, group in pdf_test.groupby('gid', sort=False):
        p1 = proba[group.index[0]]
        p2 = proba[group.index[1]]
        preds[gid] = 0 if p1 >= p2 else 1
    return preds


# ----------------------------------------------------------------------
# Optional cross-validation (used during development)
# ----------------------------------------------------------------------
def cross_validate(train_df, n_splits=5):
    pdf = build_pairs(train_df, has_label=True)
    gids = pdf['gid'].values
    ug = pd.unique(gids)
    perm = np.random.RandomState(RANDOM_STATE).permutation(len(ug))
    fold_assign = np.zeros(len(ug), dtype=int)
    for k in range(n_splits):
        fold_assign[perm[k::n_splits]] = k
    gid_to_fold = dict(zip(ug, fold_assign))
    sample_fold = np.array([gid_to_fold[g] for g in gids])
    rfg = train_df.set_index('id')

    accs = []
    for k in range(n_splits):
        tri = np.where(sample_fold != k)[0]
        tei = np.where(sample_fold == k)[0]
        proba = predict_pairs(pdf.iloc[tri], pdf.iloc[tei])
        test_gids = gids[tei]
        idx_by_gid = defaultdict(list)
        for i, g in enumerate(test_gids):
            idx_by_gid[g].append(i)
        correct = 0
        for g, idxs in idx_by_gid.items():
            p1 = proba[idxs[0]]
            p2 = proba[idxs[1]]
            pred = 0 if p1 >= p2 else 1
            if pred == rfg.loc[g, 'label']:
                correct += 1
        accs.append(correct / len(idx_by_gid))
    return np.array(accs)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    train_path = os.path.join(TASK_DIR, 'train.csv')
    test_path = os.path.join(TASK_DIR, 'test.csv')
    out_path = os.path.join(TASK_DIR, 'outputs', 'submission.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df['question'] = train_df['question'].str.strip()
    test_df['question'] = test_df['question'].str.strip()

    preds = predict_rows(train_df, test_df)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = pd.DataFrame({'id': test_df['id'].values,
                        'label': [preds[i] for i in test_df['id'].values]})
    out.to_csv(out_path, index=False)
    print(f'Saved {len(out)} predictions to {out_path}')
    print('Label distribution:', out['label'].value_counts().to_dict())


if __name__ == '__main__':
    main()
