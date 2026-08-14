"""Train KLUE-RE classifier and write predictions.

Approach: TF-IDF (char n-grams over the entity-marker-wrapped sentence,
word n-grams, word n-grams of the between-entity span, plus binary
subject/object pair identity + order/distance tokens) feeding a LinearSVC.

Validation: 5-fold stratified CV accuracy ~0.703 on train.csv.

Usage:
    python solution/train.py            # train + write outputs/submission.csv
    python solution/train.py --cv       # only run 5-fold CV and report accuracy
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_text

C = 0.3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def between_text(row):
    sent = str(row.sentence)
    s, o = str(row.subject_entity), str(row.object_entity)
    si, oi = sent.find(s), sent.find(o)
    if si < 0 or oi < 0:
        return ""
    if si < oi:
        return sent[si + len(s):oi]
    return sent[oi + len(o):si]


def pair_text(df):
    pairs = []
    for _, r in df.iterrows():
        sent = str(r.sentence)
        s, o = str(r.subject_entity), str(r.object_entity)
        si, oi = sent.find(s), sent.find(o)
        order = "ORDER_SO" if (si >= 0 and oi >= 0 and si < oi) else "ORDER_OS"
        d = abs(oi - si) if si >= 0 and oi >= 0 else 0
        db = "DIST_NEAR" if d < 20 else ("DIST_MID" if d < 60 else "DIST_FAR")
        pairs.append(f"SUBJ={s} OBJ={o} {order} {db}")
    return pairs


def context_text(df, window=3):
    """Words right before the first entity and right after the second one."""
    ctx = []
    for _, r in df.iterrows():
        sent = str(r.sentence)
        s, o = str(r.subject_entity), str(r.object_entity)
        si, oi = sent.find(s), sent.find(o)
        if si < 0 or oi < 0:
            ctx.append("")
            continue
        if si < oi:
            left = sent[:si].split()[-window:]
            right = sent[oi + len(o):].split()[:window]
        else:
            left = sent[:oi].split()[-window:]
            right = sent[si + len(s):].split()[:window]
        ctx.append(" ".join(["L_" + w for w in left] + ["R_" + w for w in right]))
    return ctx


class FeatureExtractor:
    def __init__(self):
        self.vecc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                    min_df=3, sublinear_tf=True)
        self.vecw = TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                                    min_df=4, sublinear_tf=True,
                                    token_pattern=r"(?u)\S+")
        self.vecb = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                    min_df=3, sublinear_tf=True,
                                    token_pattern=r"(?u)\S+")
        self.vecp = TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                                    min_df=3, sublinear_tf=True,
                                    token_pattern=r"(?u)\S+", binary=True)
        self.vecx = TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                                    min_df=3, sublinear_tf=True,
                                    token_pattern=r"(?u)\S+")

    def _blocks(self, df, fit):
        xfull = build_text(df)
        xbet = df.apply(between_text, axis=1).tolist()
        xpair = pair_text(df)
        xctx = context_text(df)
        vecs = [(self.vecc, xfull, 1.0), (self.vecw, xfull, 0.5),
                (self.vecb, xbet, 1.0), (self.vecp, xpair, 1.0),
                (self.vecx, xctx, 0.5)]
        mats = []
        for vec, texts, w in vecs:
            m = vec.fit_transform(texts) if fit else vec.transform(texts)
            mats.append(m * w)
        return sp.hstack(mats).tocsr()

    def fit_transform(self, df):
        return self._blocks(df, fit=True)

    def transform(self, df):
        return self._blocks(df, fit=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="only run 5-fold CV")
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    y = train["label"].values

    fe = FeatureExtractor()
    X = fe.fit_transform(train)
    print(f"features: {X.shape} in {time.time()-t0:.1f}s")

    if args.cv:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for tri, tei in skf.split(X, y):
            clf = LinearSVC(C=C, max_iter=5000, dual="auto")
            clf.fit(X[tri], y[tri])
            accs.append(accuracy_score(y[tei], clf.predict(X[tei])))
        print(f"5-fold CV accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
        return

    clf = LinearSVC(C=C, max_iter=5000, dual="auto")
    clf.fit(X, y)
    print(f"trained in {time.time()-t0:.1f}s")

    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    Xt = fe.transform(test)
    pred = clf.predict(Xt)

    out = pd.DataFrame({"id": test["id"], "label": pred})
    assert out["id"].is_unique and len(out) == len(test)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out_path = os.path.join(ROOT, "outputs", "submission.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)")
    print(out["label"].value_counts().head(10))


if __name__ == "__main__":
    main()
