"""Baseline: TF-IDF (char+word) + LogisticRegression, StratifiedKFold OOF + test predict.

Goal: produce a valid submission quickly and establish a CV macro-F1 baseline.
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "train.csv")
TEST = os.path.join(ROOT, "test.csv")
OUT = os.path.join(ROOT, "outputs", "submission.csv")

RANDOM_STATE = 42
N_SPLITS = 5

CLASSES = ["none", "offensive", "hate"]


def build_features(train_text, test_text):
    # Character n-grams (robust to Korean morph variants / typos / spacing)
    char_vec = TfidfVectorizer(
        sublinear_tf=True,
        analyzer="char_wb",
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.95,
        max_features=200000,
    )
    char_vec.fit(list(train_text) + list(test_text))
    Xtr_c = char_vec.transform(train_text)
    Xte_c = char_vec.transform(test_text)

    # Word n-grams (whitespace tokenization — works for Korean without a tokenizer)
    word_vec = TfidfVectorizer(
        sublinear_tf=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=100000,
        token_pattern=r"(?u)\b\w+\b",
    )
    word_vec.fit(list(train_text) + list(test_text))
    Xtr_w = word_vec.transform(train_text)
    Xte_w = word_vec.transform(test_text)

    from scipy.sparse import hstack
    Xtr = hstack([Xtr_c, Xtr_w]).tocsr()
    Xte = hstack([Xte_c, Xte_w]).tocsr()
    return Xtr, Xte


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    train["comment"] = train["comment"].fillna("")
    test["comment"] = test["comment"].fillna("")
    y = train["label"].values
    Xtr, Xte = build_features(train["comment"].values, test["comment"].values)
    print("Train X:", Xtr.shape, "Test X:", Xte.shape)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(train), 3))
    test_proba = np.zeros((len(test), 3))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
        Xtr_f, Xva_f = Xtr[tr_idx], Xtr[va_idx]
        ytr_f = y[tr_idx]
        # class_weight balanced to handle label imbalance
        clf = LogisticRegression(
            C=4.0,
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        clf.fit(Xtr_f, ytr_f)
        oof[va_idx] = clf.predict_proba(Xva_f)
        test_proba += clf.predict_proba(Xte) / N_SPLITS
        fold_pred = oof[va_idx].argmax(1)
        # map indices to class names
        inv = {v: k for k, v in clf.classes_.items()} if isinstance(clf.classes_, dict) else None
        print(f"fold {fold} classes:", clf.classes_)

    # OOF macro-F1
    idx_to_label = {i: c for i, c in enumerate(CLASSES)}
    # ensure class order
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(CLASSES)
    y_num = le.transform(y)
    oof_pred = oof.argmax(1)
    # Need to align clf.classes_ ordering. LogisticRegression.classes_ is sorted label names.
    # Assume alphabetical == CLASSES order (none, offensive, hate -> hate, none, offensive sorted!)
    # So we must map properly:
    sorted_classes = sorted(CLASSES)
    print("Sorted classes (sklearn order):", sorted_classes)
    # Re-map oof/test_proba columns from sorted order to CLASSES order
    oof_named = np.zeros_like(oof)
    test_named = np.zeros_like(test_proba)
    for i, c in enumerate(sorted_classes):
        j = CLASSES.index(c)
        oof_named[:, j] = oof[:, i]
        test_named[:, j] = test_proba[:, i]
    oof_pred_labels = np.array([CLASSES[i] for i in oof_named.argmax(1)])
    macro_f1 = f1_score(y, oof_pred_labels, average="macro")
    print(f"OOF Macro F1: {macro_f1:.4f}")
    print(classification_report(y, oof_pred_labels, digits=4))

    test_pred = np.array([CLASSES[i] for i in test_named.argmax(1)])
    sub = pd.DataFrame({"id": test["id"].values, "label": test_pred})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sub.to_csv(OUT, index=False)
    print("Wrote", OUT, sub.shape)
    print(sub["label"].value_counts())


if __name__ == "__main__":
    main()
