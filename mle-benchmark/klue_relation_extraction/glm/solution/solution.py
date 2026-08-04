#!/usr/bin/env python3
"""
KLUE-RE baseline solution (TF-IDF + LinearSVC with entity-aware features).

No external data / no pretrained weights / no internet.
Uses only train.csv to learn a generalizable classifier.

Usage:
    python3 solution.py
"""
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
TRAIN_PATH = os.path.join(TASK_DIR, "train.csv")
TEST_PATH = os.path.join(TASK_DIR, "test.csv")
OUT_PATH = os.path.join(TASK_DIR, "outputs", "submission.csv")

RANDOM_STATE = 42

# Korean character pattern for tokenization (char + bigram based)
HANGUL = re.compile(r"[가-힣]")


def char_tokenizer(text):
    # Character-level tokens that work well for Korean without morph analyzers
    return list(text)


def word_tokenizer(text):
    # Simple whitespace + punctuation based tokens (fallback for non-Korean)
    return re.findall(r"[가-힣]+|[A-Za-z]+|\d+", text)


def build_text_with_entities(row):
    """Create a text representation that explicitly marks subject/object entities
    and their positions, which is crucial for relation extraction."""
    sentence = row["sentence"]
    subj = str(row["subject_entity"])
    obj = str(row["object_entity"])

    # Find positions of entities in the sentence
    s_pos = sentence.find(subj)
    o_pos = sentence.find(obj)

    # Mark entities with special tokens to give the model entity boundaries.
    # Use markers that the char/word tokenizer will keep as units.
    if s_pos != -1 and o_pos != -1:
        # Decide order based on position
        if s_pos < o_pos:
            # subject first
            # ensure obj search is after subj replacement to avoid overlap issues
            new = (
                sentence[:s_pos]
                + " [SUBJ] " + subj + " [/SUBJ] "
                + sentence[s_pos + len(subj):]
            )
            # recompute obj pos in new string
            o_pos_new = new.find(obj)
            if o_pos_new != -1:
                new = (
                    new[:o_pos_new]
                    + " [OBJ] " + obj + " [/OBJ] "
                    + new[o_pos_new + len(obj):]
                )
        else:
            new = (
                sentence[:o_pos]
                + " [OBJ] " + obj + " [/OBJ] "
                + sentence[o_pos + len(obj):]
            )
            s_pos_new = new.find(subj)
            if s_pos_new != -1:
                new = (
                    new[:s_pos_new]
                    + " [SUBJ] " + subj + " [/SUBJ] "
                    + new[s_pos_new + len(subj):]
                )
        return new
    # Fallback: prepend entities
    return f"[SUBJ] {subj} [/SUBJ] [OBJ] {obj} [/OBJ] {sentence}"


def build_features(df):
    """Build a combined text column with entity markers and extra lexical hints."""
    rows = []
    for _, row in df.iterrows():
        sentence = str(row["sentence"])
        subj = str(row["subject_entity"])
        obj = str(row["object_entity"])

        marked = build_text_with_entities(row)

        # Extra engineered string of hints (entity types, relative position,
        # distance between entities, presence of digits etc.)
        s_pos = sentence.find(subj)
        o_pos = sentence.find(obj)
        if s_pos == -1:
            s_pos = 0
        if o_pos == -1:
            o_pos = 0
        dist = abs(o_pos - s_pos)
        order = "subj_first" if s_pos <= o_pos else "obj_first"

        # Heuristic entity type detection
        subj_type = guess_type(subj)
        obj_type = guess_type(obj)

        # Words between the two entities
        lo, hi = min(s_pos, o_pos), max(s_pos, o_pos)
        if lo >= 0 and hi > lo and hi <= len(sentence):
            between = sentence[lo:hi]
        else:
            between = ""

        # Concatenate all hint tokens at the end so TF-IDF can pick them up
        hint = " ".join([
            f"SUBJ_TYPE={subj_type}",
            f"OBJ_TYPE={obj_type}",
            f"ORDER={order}",
            f"DIST_BUCKET={bucket(dist)}",
            f"BETWEEN:{between}",
        ])
        text = marked + " " + hint
        rows.append(text)
    return rows


def guess_type(entity):
    e = str(entity)
    has_hangul = bool(HANGUL.search(e))
    has_digit = bool(re.search(r"\d", e))
    # Heuristics: ORG often ends with org-suffixes; PER is short hangul name
    org_suffixes = (
        "회사", "주식회사", "기업", "공사", "공단", "청", "부", "위원회",
        "연맹", "연합회", "당", "당", "센터", "대학", "대학교", "학교",
        "은행", "그룹", "보건소", "본부", "사", "그룹", "기관", "연구소",
        "방역대책본부", "도", "시", "군", "구", "리그", "FC", "구단",
        "조직", "협회", "교회", "사령부", "군", "군사령부", "사",
    )
    if any(e.endswith(suf) for suf in org_suffixes):
        return "ORG"
    # Person: typical 2-3 syllable hangul name with no digits
    if has_hangul and not has_digit and 1 <= len(e) <= 6 and " " not in e:
        # could be person; but be lenient
        if re.fullmatch(r"[가-힣·]+", e) and len(e) <= 5:
            return "PER"
    if has_digit:
        return "DATE_OR_NUM"
    if not has_hangul:
        return "FOREIGN"
    return "OTHER"


def bucket(d):
    if d == 0:
        return "0"
    if d < 5:
        return "1-4"
    if d < 15:
        return "5-14"
    if d < 40:
        return "15-39"
    if d < 100:
        return "40-99"
    return "100+"


def main():
    print("Loading data...", flush=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train={train.shape} test={test.shape}", flush=True)

    y = train["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"num classes = {n_classes}", flush=True)

    print("Building features...", flush=True)
    X_train_text = build_features(train)
    X_test_text = build_features(test)

    # TF-IDF: char + word n-grams, robust to Korean
    print("Vectorizing (word n-grams)...", flush=True)
    word_vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=word_tokenizer,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents=None,
        lowercase=False,
    )
    Xtr_w = word_vec.fit_transform(X_train_text)
    Xte_w = word_vec.transform(X_test_text)

    print("Vectorizing (char n-grams)...", flush=True)
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
    )
    Xtr_c = char_vec.fit_transform(X_train_text)
    Xte_c = char_vec.transform(X_test_text)

    from scipy.sparse import hstack
    Xtr = hstack([Xtr_w, Xtr_c]).tocsr()
    Xte = hstack([Xte_w, Xte_c]).tocsr()
    print(f"Xtr={Xtr.shape} Xte={Xte.shape}", flush=True)

    # LogisticRegression with balanced class weights for multi-class
    print("Training LogisticRegression (balanced)...", flush=True)
    lr = LogisticRegression(
        C=4.0,
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    lr.fit(Xtr, y_enc)

    print("Training LinearSVC (class_weight=None, calibrated)...", flush=True)
    svc = LinearSVC(
        C=1.0,
        class_weight=None,
        random_state=RANDOM_STATE,
        max_iter=3000,
    )
    calibrated = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
    calibrated.fit(Xtr, y_enc)

    # Weighted-probability ensemble: best CV at w_svc=0.5 (SVC) / 0.5 (LR).
    proba_lr = lr.predict_proba(Xte)
    proba_svc = calibrated.predict_proba(Xte)
    proba = 0.5 * proba_svc + 0.5 * proba_lr
    pred = np.argmax(proba, axis=1)

    pred_labels = le.inverse_transform(pred)
    print("Predicted label distribution:",
          pd.Series(pred_labels).value_counts().head(10).to_dict(), flush=True)

    # Write submission
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out = pd.DataFrame({"id": test["id"], "label": pred_labels})
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with shape {out.shape}", flush=True)

    # Sanity: ensure all test ids present exactly once
    assert set(out["id"]) == set(test["id"]), "id mismatch!"
    assert out["id"].value_counts().max() == 1, "duplicate ids!"
    print("Sanity checks passed.", flush=True)


if __name__ == "__main__":
    main()
