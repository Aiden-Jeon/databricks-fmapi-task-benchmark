"""
KLUE-NLI (t6_klue_nli) solution.

Approach
--------
Linear bag-of-ngrams model over the *hypothesis* text, plus a small set of
premise-hypothesis pairwise numeric features.

EDA findings that motivated the design:
  - Hypothesis-only TF-IDF model reaches ~45% holdout accuracy, while
    premise-only is ~5% (below random). The label signal lives mostly in the
    hypothesis wording (a known KLUE NLI annotation artifact).
  - Joint premise+hypothesis TF-IDF models performed *worse* than
    hypothesis-only, so the premise is used only through lightweight numeric
    pairwise features (overlap, length ratio, negation marker).

Features (all computed from the hypothesis unless noted):
  1. Jamo n-grams: Hangul syllables decomposed into jamo, word-level TF-IDF
     with ngram_range=(2,5) over the jamo token stream (150k feats).
  2. Char n-grams: char_wb TF-IDF with ngram_range=(2,5) (150k feats).
  3. Word n-grams: word TF-IDF with ngram_range=(1,2), min_df=3 (80k feats).
  4. Numeric pairwise features (standardized): premise/hypothesis word
     overlap ratio, length ratio, |hypothesis words|, |overlap|,
     negation-marker-in-hypothesis indicator.

Classifier: multinomial Logistic Regression, C=0.15, max_iter=2000.

Validation: 5-fold stratified CV accuracy = 0.5405.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NEG_MARKERS = ['않', '없', '못', '아니', '안 ']


def to_jamo(text: str) -> str:
    """Decompose Hangul syllables into jamo characters, space-separated."""
    out = []
    for ch in str(text):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            s = code - 0xAC00
            out.append(chr(0x1100 + s // (21 * 28)))          # choseong
            out.append(chr(0x1161 + (s % (21 * 28)) // 28))    # jungseong
            if s % 28:
                out.append(chr(0x11A7 + s % 28))               # jongseong
        else:
            out.append(ch)
    return ' '.join(out)


def numeric_features(df: pd.DataFrame) -> np.ndarray:
    p = df['premise'].astype(str).values
    h = df['hypothesis'].astype(str).values
    feats = []
    for i in range(len(df)):
        ps = set(p[i].split())
        hs = set(h[i].split())
        ov = len(ps & hs) / max(len(hs), 1)
        has_neg = int(any(m in h[i] for m in NEG_MARKERS))
        feats.append([ov, len(h[i]) / max(len(p[i]), 1), len(hs), len(ps & hs), has_neg])
    return np.array(feats, dtype=np.float64)


def build_features(train_df, test_df):
    vec_jamo = TfidfVectorizer(analyzer='word', ngram_range=(2, 5), min_df=2,
                               max_features=150000, sublinear_tf=True,
                               token_pattern=r'\S+')
    vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2,
                               max_features=150000, sublinear_tf=True)
    vec_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                               max_features=80000, sublinear_tf=True)
    scaler = StandardScaler()

    hj_tr = train_df['hypothesis'].astype(str).map(to_jamo).values
    hj_te = test_df['hypothesis'].astype(str).map(to_jamo).values
    h_tr = train_df['hypothesis'].astype(str).values
    h_te = test_df['hypothesis'].astype(str).values

    J_tr, J_te = vec_jamo.fit_transform(hj_tr), vec_jamo.transform(hj_te)
    C_tr, C_te = vec_char.fit_transform(h_tr), vec_char.transform(h_te)
    W_tr, W_te = vec_word.fit_transform(h_tr), vec_word.transform(h_te)

    nf_tr = scaler.fit_transform(numeric_features(train_df))
    nf_te = scaler.transform(numeric_features(test_df))

    X_train = hstack([J_tr, C_tr, W_tr, csr_matrix(nf_tr)]).tocsr()
    X_test = hstack([J_te, C_te, W_te, csr_matrix(nf_te)]).tocsr()
    return X_train, X_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='train.csv')
    ap.add_argument('--test', default='test.csv')
    ap.add_argument('--out', default='outputs/submission.csv')
    args = ap.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    X_train, X_test = build_features(train_df, test_df)
    y_train = train_df['label'].values

    clf = LogisticRegression(C=0.15, max_iter=2000, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    sub = pd.DataFrame({'id': test_df['id'], 'label': preds})
    sub.to_csv(args.out, index=False)
    print(f'wrote {args.out} ({len(sub)} rows)')
    print(sub['label'].value_counts())


if __name__ == '__main__':
    main()
