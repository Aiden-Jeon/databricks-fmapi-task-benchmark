"""KoBEST SentiNeg — negation-aware sentiment classification (accuracy).

Approach
--------
No pretrained weights / internet are available, so the solution is a purely
classical, character-level pipeline that works well for short Korean reviews:

1. Hangul syllables are decomposed into jamo (Unicode NFD).  Negation in Korean
   is expressed through affixes/endings ("안", "못", "-지 않", "-지 마") whose
   surface form differs from the positive counterpart by a single jamo, so
   jamo-level n-grams capture these cues far better than syllable-level ones.
2. TF-IDF n-grams (jamo char_wb 2-6 + jamo char 2-7, syllable char_wb 1-5).
3. Soft ensemble of three diverse linear/kernel margin classifiers; decision
   values are squashed with a sigmoid (rank preserving) and averaged.

Repeated 5-fold CV (4 repeats, 2919 rows):
    jamo-union LinearSVC   0.9588
    jamo RBF SVC           0.9575
    syllable LinearSVC     0.9556
    ensemble (used here)   0.9597

Run:  python solution/train.py   (writes outputs/submission.csv)
"""

import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SEED = 11


def to_jamo(texts):
    """Decompose Hangul syllables into jamo sequences (NFD)."""
    return np.array([unicodedata.normalize("NFD", str(s)) for s in texts])


def identity(texts):
    return np.asarray([str(s) for s in texts])


def jamo_union():
    return make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), sublinear_tf=True, min_df=2),
        TfidfVectorizer(analyzer="char", ngram_range=(2, 7), sublinear_tf=True, min_df=2),
    )


def build_models():
    """(text-transform, estimator) pairs; all expose decision_function."""
    return {
        "jamo_union_svc": (
            to_jamo,
            make_pipeline(jamo_union(), LinearSVC(C=1.0, dual=True, random_state=SEED)),
        ),
        "jamo_rbf_svc": (
            to_jamo,
            make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), sublinear_tf=True, min_df=2),
                SVC(C=10.0, gamma="scale", random_state=SEED),
            ),
        ),
        "syllable_svc": (
            identity,
            make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2),
                LinearSVC(C=1.0, dual=True, random_state=SEED),
            ),
        ),
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensemble_scores(X_train, y_train, X_eval):
    scores = []
    for _, (prep, model) in build_models().items():
        A, B = prep(X_train), prep(X_eval)
        model.fit(A, y_train)
        scores.append(sigmoid(model.decision_function(B)))
    return np.mean(scores, axis=0)


def cross_validate(X, y, n_repeats=4):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    accs = []
    for tr_idx, va_idx in cv.split(X, y):
        p = ensemble_scores(X[tr_idx], y[tr_idx], X[va_idx])
        accs.append((( p > 0.5).astype(int) == y[va_idx]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main(run_cv=True):
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    X, y = train["sentence"].values, train["label"].values

    if run_cv:
        mean, std = cross_validate(X, y)
        print(f"ensemble repeated-CV accuracy: {mean:.4f} +- {std:.4f}")

    scores = ensemble_scores(X, y, test["sentence"].values)
    pred = (scores > 0.5).astype(int)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    sub = pd.DataFrame({"id": test["id"].values, "label": pred})

    sample = pd.read_csv(ROOT / "sample_submission.csv")
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(test) and sub["id"].is_unique
    assert set(sub["id"]) == set(test["id"])
    assert sub["label"].isin([0, 1]).all()

    sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"wrote {out_dir/'submission.csv'} rows={len(sub)}")
    print(sub["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
