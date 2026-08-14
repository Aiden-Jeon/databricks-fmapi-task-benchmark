#!/usr/bin/env python3
"""KoBEST COPA solution (K-MLE-Bench, t13_kobest_copa).

Approach
--------
Pointwise plausibility scoring with swap augmentation:

1. Augment train by swapping the two alternatives (label flips), so the model
   learns order-invariant plausibility. Train size: 2460 -> 4920 pairs.
2. Build a pointwise text for each (premise, alternative) pair:
       "{question} {premise}{connector}{alternative}"
   where connector = " 때문에 " for cause(원인), " 그래서 " for effect(결과).
   Binary target: 1 if the alternative is the correct one.
3. Decompose Korean syllables into jamo (자소) to enlarge the alphabet and
   let char n-grams capture morphological patterns more robustly.
4. TF-IDF over char n-grams (2-5), sublinear tf, min_df=2.
5. LogisticRegression (C=1.0, liblinear) as the plausibility scorer.
6. For test, score both alternatives and pick argmax.

5-fold CV (stratified) accuracy on train: ~0.60.

Only train.csv is used for fitting (vectorizer + classifier). No internet,
no external data, no pretrained weights. Runs on CPU in ~30 seconds.

Reproduce:
    python3 solution/solution.py
writes outputs/submission.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "train.csv")
TEST = os.path.join(BASE, "test.csv")
OUT = os.path.join(BASE, "outputs", "submission.csv")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Hangul jamo decomposition
# ---------------------------------------------------------------------------
CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
       'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
        'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def decompose_jamo(s: str) -> str:
    """Decompose Hangul syllables into choseong/jungseong/jongseong jamo."""
    out = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            i = o - 0xAC00
            out.append(CHO[i // 588])
            out.append(JUNG[(i % 588) // 28])
            j = JONG[i % 28]
            if j:
                out.append(j)
        else:
            out.append(ch)
    return ''.join(out)


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
def norm_question(q: str) -> str:
    q = str(q).strip()
    return "원인" if q.startswith("원인") else "결과"


def swap_augment(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Add each row with alternatives swapped and label flipped, then shuffle."""
    rows = []
    for r in df.itertuples():
        rows.append((r.premise, r.question, r.alternative_1, r.alternative_2,
                     r.label))
        rows.append((r.premise, r.question, r.alternative_2, r.alternative_1,
                     1 - r.label))
    aug = pd.DataFrame(rows, columns=["premise", "question", "alternative_1",
                                      "alternative_2", "label"])
    return aug.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def pointwise_texts(df: pd.DataFrame, with_label: bool = True):
    """Return jamo-decomposed pointwise texts (2 per row) and optional targets."""
    texts, ys = [], []
    for r in df.itertuples():
        q = norm_question(r.question)
        con = " 때문에 " if q == "원인" else " 그래서 "
        for k, alt in enumerate([r.alternative_1, r.alternative_2]):
            texts.append(decompose_jamo(f"{q} {r.premise}{con}{alt}"))
            if with_label:
                ys.append(1 if r.label == k else 0)
    return texts, (np.array(ys) if with_label else None)


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    aug = swap_augment(train, seed=RANDOM_STATE)
    train_texts, train_y = pointwise_texts(aug, with_label=True)
    test_texts, _ = pointwise_texts(test, with_label=False)

    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                          sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    clf = LogisticRegression(C=1.0, max_iter=4000, solver="liblinear",
                             random_state=RANDOM_STATE)
    clf.fit(X_train, train_y)

    scores = np.asarray(clf.decision_function(X_test)).ravel().reshape(-1, 2)
    picks = (scores[:, 1] > scores[:, 0]).astype(int)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sub = pd.DataFrame({"id": test["id"], "label": picks})
    sub.to_csv(OUT, index=False)
    print(f"wrote {OUT}  shape={sub.shape}  "
          f"label_counts={sub['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
