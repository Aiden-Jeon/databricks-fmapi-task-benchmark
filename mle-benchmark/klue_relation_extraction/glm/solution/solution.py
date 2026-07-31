"""KLUE-RE baseline: TF-IDF features + linear classifiers.

No internet / no pretrained weights. Pure sklearn.
"""
import os
import re
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "train.csv")
TEST = os.path.join(ROOT, "test.csv")
SAMPLE = os.path.join(ROOT, "sample_submission.csv")
OUT = os.path.join(ROOT, "outputs", "submission.csv")


def char_tokens(text):
    """Character-level tokens for Korean (no whitespace tokenizer needed)."""
    return re.findall(r"\S", text)


def build_features(train_df, test_df):
    """Build feature matrices combining char/word TF-IDF + entity signals."""
    # Mark entity positions in the sentence so the model knows which spans are subj/obj.
    def mark_entities(row, ent_col):
        s = row["sentence"]
        ent = str(row[ent_col])
        idx = s.find(ent)
        if idx >= 0:
            return s[:idx] + "〈" + ent + "〉" + s[idx + len(ent):]
        return s + " 〈" + ent + "〉"

    # Build augmented sentence with both entities marked, plus an ordered concat.
    def aug(row):
        s = row["sentence"]
        subj = str(row["subject_entity"])
        obj = str(row["object_entity"])
        s = s.replace(subj, "〈SUB〉" + subj + "〈/SUB〉", 1)
        s = s.replace(obj, "〈OBJ〉" + obj + "〈/OBJ〉", 1)
        return s

    train_aug = train_df.apply(aug, axis=1)
    test_aug = test_df.apply(aug, axis=1)

    # Entity pair text (subject + object + order)
    train_pair = train_df["subject_entity"].astype(str) + " [SEP] " + train_df["object_entity"].astype(str)
    test_pair = test_df["subject_entity"].astype(str) + " [SEP] " + test_df["object_entity"].astype(str)

    feats = []

    # Char n-gram TF-IDF on augmented sentence (captures Korean morphology + entity markers)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_df=0.95, sublinear_tf=True)
    Xtr_char = char_vec.fit_transform(train_aug)
    Xte_char = char_vec.transform(test_aug)
    feats.append((Xtr_char, Xte_char))

    # Word n-gram TF-IDF (whitespace) on augmented sentence
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    Xtr_word = word_vec.fit_transform(train_aug)
    Xte_word = word_vec.transform(test_aug)
    feats.append((Xtr_word, Xte_word))

    # Char n-gram on entity pair
    pair_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    Xtr_pair = pair_vec.fit_transform(train_pair)
    Xte_pair = pair_vec.transform(test_pair)
    feats.append((Xtr_pair, Xte_pair))

    # Entity length features
    def hand(df):
        subj = df["subject_entity"].astype(str)
        obj = df["object_entity"].astype(str)
        sent = df["sentence"].astype(str)
        feats = np.column_stack([
            subj.str.len().values,
            obj.str.len().values,
            sent.str.len().values,
            (subj.str.len() - obj.str.len()).values,
        ])
        return feats

    Xtr_hand = hand(train_df)
    Xte_hand = hand(test_df)
    feats.append((csr_matrix(Xtr_hand), csr_matrix(Xte_hand)))

    Xtr = hstack([f[0] for f in feats]).tocsr()
    Xte = hstack([f[1] for f in feats]).tocsr()
    return Xtr, Xte


def main():
    t0 = time.time()
    train_df = pd.read_csv(TRAIN)
    test_df = pd.read_csv(TEST)
    print(f"train {len(train_df)} test {len(test_df)} ({time.time()-t0:.1f}s)")

    Xtr, Xte = build_features(train_df, test_df)
    print(f"features: train {Xtr.shape} test {Xte.shape} ({time.time()-t0:.1f}s)")

    le = LabelEncoder()
    y = le.fit_transform(train_df["label"])
    n_classes = len(le.classes_)
    print(f"classes {n_classes}")

    # Stratified 5-fold OOF to blend LR + LinearSVC-probability
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_test = len(test_df)
    oof = np.zeros((len(train_df), n_classes))
    test_proba = np.zeros((n_test, n_classes))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
        Xtr_f, Xva_f = Xtr[tr_idx], Xtr[va_idx]
        ytr_f, yva_f = y[tr_idx], y[va_idx]

        lr = LogisticRegression(
            C=4.0, max_iter=2000, n_jobs=-1, solver="liblinear", class_weight="balanced"
        )
        lr.fit(Xtr_f, ytr_f)
        oof[va_idx] = lr.predict_proba(Xva_f)
        test_proba += lr.predict_proba(Xte) / skf.n_splits

        print(f"  fold {fold} LR acc {accuracy_score(yva_f, oof[va_idx].argmax(1)):.4f} ({time.time()-t0:.1f}s)")

    oof_acc = accuracy_score(y, oof.argmax(1))
    print(f"OOF LR accuracy: {oof_acc:.4f} ({time.time()-t0:.1f}s)")

    pred = test_proba.argmax(1)
    labels = le.inverse_transform(pred)

    sub = pd.DataFrame({"id": test_df["id"].values, "label": labels})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sub.to_csv(OUT, index=False)
    print(f"saved {OUT} ({time.time()-t0:.1f}s) head:")
    print(sub.head().to_string(index=False))
    print("pred label distribution:")
    print(sub["label"].value_counts())


if __name__ == "__main__":
    main()
