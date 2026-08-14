"""Train NLI model on KorNLI train.csv and generate predictions for test.csv.

Approach (no internet / no pretrained weights):
  - TF-IDF on word unigrams/bigrams and char n-grams of combined
    (premise + hypothesis) text, plus a diff (s1 - s2) projection to capture
    pair mismatch.
  - TruncatedSVD to dense space.
  - Handcrafted NLI features (token overlap, negation, numerals).
  - LogisticRegression on [SVD(s1+s2), SVD(s1-s2), |SVD(s1-s2)|, hand features].
  - HistogramGradientBoosting on hand + diff-summary features as a second model;
    final prediction = probability-weighted ensemble.
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import hand_features  # noqa: E402

RANDOM_STATE = 42
N_SVD = 300


def build_texts(df):
    s1 = df["sentence1"].fillna("").tolist()
    s2 = df["sentence2"].fillna("").tolist()
    return s1, s2


def main():
    t0 = time.time()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train = pd.read_csv(os.path.join(root, "train.csv"))
    test = pd.read_csv(os.path.join(root, "test.csv"))
    print(f"train={train.shape} test={test.shape}")

    y = train["label"].values
    classes = np.unique(y)

    tr_idx, va_idx = train_test_split(
        np.arange(len(train)), test_size=0.05, random_state=RANDOM_STATE, stratify=y
    )

    # ---------- vectorizers (fit on train split only to measure honest val acc) --
    def make_vec(max_features, ngram_range, analyzer, sublinear=True):
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer=analyzer,
            sublinear_tf=sublinear,
            min_df=2,
        )

    word_vec = make_vec(120_000, (1, 2), "word")
    char_vec = make_vec(150_000, (2, 5), "char_wb")

    s1_all, s2_all = build_texts(train)
    combo_all = [a + " [SEP] " + b for a, b in zip(s1_all, s2_all)]

    print("fitting vectorizers...")
    word_vec.fit([combo_all[i] for i in tr_idx])
    char_vec.fit([combo_all[i] for i in tr_idx])

    def encode(idxs, vec, svd=None, fit=False):
        combo = [combo_all[i] for i in idxs]
        X = vec.transform(combo)
        if svd is not None:
            Xd = svd.transform(X) if not fit else svd.fit_transform(X)
            return Xd
        return X

    # SVD on word and char spaces
    print("SVD fitting...")
    svd_w = TruncatedSVD(n_components=N_SVD, random_state=RANDOM_STATE)
    svd_c = TruncatedSVD(n_components=N_SVD, random_state=RANDOM_STATE)
    Zw_tr = encode(tr_idx, word_vec, svd_w, fit=True)
    Zc_tr = encode(tr_idx, char_vec, svd_c, fit=True)
    Zw_va = encode(va_idx, word_vec, svd_w)
    Zc_va = encode(va_idx, char_vec, svd_c)

    # diff encodings: transform s1 and s2 separately then subtract
    def diff_encode(idxs, vec, svd):
        X1 = svd.transform(vec.transform([s1_all[i] for i in idxs]))
        X2 = svd.transform(vec.transform([s2_all[i] for i in idxs]))
        d = X1 - X2
        return np.hstack([d, np.abs(d)])

    Dw_tr = diff_encode(tr_idx, word_vec, svd_w)
    Dc_tr = diff_encode(tr_idx, char_vec, svd_c)
    Dw_va = diff_encode(va_idx, word_vec, svd_w)
    Dc_va = diff_encode(va_idx, char_vec, svd_c)

    # handcrafted features
    print("handcrafted features...")
    H_tr = hand_features(train.iloc[tr_idx])
    H_va = hand_features(train.iloc[va_idx])

    X_tr = np.hstack([Zw_tr, Zc_tr, Dw_tr, Dc_tr, H_tr])
    X_va = np.hstack([Zw_va, Zc_va, Dw_va, Dc_va, H_va])

    scaler = StandardScaler(with_mean=True)
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    print("training LogisticRegression...")
    lr = LogisticRegression(C=4.0, max_iter=2000, n_jobs=4, random_state=RANDOM_STATE)
    lr.fit(X_tr_s, y[tr_idx])
    va_acc_lr = lr.score(X_va_s, y[va_idx])
    print(f"[val] LogReg acc = {va_acc_lr:.4f}")

    # HGB on hand + diff features (captures non-linear overlap interactions)
    print("training HistGradientBoosting...")
    hgb_feats_tr = np.hstack([H_tr, Dw_tr[:, :50], Dc_tr[:, :50]])
    hgb_feats_va = np.hstack([H_va, Dw_va[:, :50], Dc_va[:, :50]])
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_depth=None,
        random_state=RANDOM_STATE,
    )
    hgb.fit(hgb_feats_tr, y[tr_idx])
    va_acc_hgb = hgb.score(hgb_feats_va, y[va_idx])
    print(f"[val] HGB acc = {va_acc_hgb:.4f}")

    # ensemble probabilities
    p_lr = lr.predict_proba(X_va_s)
    p_hgb = hgb.predict_proba(hgb_feats_va)
    # align class order
    hgb_order = list(hgb.classes_)
    lr_order = list(lr.classes_)
    p_hgb_al = p_hgb[:, [hgb_order.index(c) for c in lr_order]]
    p_ens = 0.7 * p_lr + 0.3 * p_hgb_al
    pred_ens = lr.classes_[p_ens.argmax(axis=1)]
    va_acc_ens = (pred_ens == y[va_idx]).mean()
    print(f"[val] Ensemble acc = {va_acc_ens:.4f}")

    # ------------------ final: retrain on full train, predict test --------------
    print("retraining on full training data...")
    s1_te, s2_te = build_texts(test)
    combo_te = [a + " [SEP] " + b for a, b in zip(s1_te, s2_te)]

    word_vec.fit(combo_all)
    char_vec.fit(combo_all)

    Xw_all = word_vec.transform(combo_all)
    Xc_all = char_vec.transform(combo_all)
    svd_w = TruncatedSVD(n_components=N_SVD, random_state=RANDOM_STATE).fit(Xw_all)
    svd_c = TruncatedSVD(n_components=N_SVD, random_state=RANDOM_STATE).fit(Xc_all)

    Zw_all = svd_w.transform(Xw_all)
    Zc_all = svd_c.transform(Xc_all)
    Zw_te = svd_w.transform(word_vec.transform(combo_te))
    Zc_te = svd_c.transform(char_vec.transform(combo_te))

    def diff_full(vec, svd, s1_list, s2_list):
        return svd.transform(vec.transform(s1_list)) - svd.transform(vec.transform(s2_list))

    d_w = diff_full(word_vec, svd_w, s1_all, s2_all)
    d_c = diff_full(char_vec, svd_c, s1_all, s2_all)
    d_w_te = diff_full(word_vec, svd_w, s1_te, s2_te)
    d_c_te = diff_full(char_vec, svd_c, s1_te, s2_te)

    Dw_all = np.hstack([d_w, np.abs(d_w)])
    Dc_all = np.hstack([d_c, np.abs(d_c)])
    Dw_te = np.hstack([d_w_te, np.abs(d_w_te)])
    Dc_te = np.hstack([d_c_te, np.abs(d_c_te)])

    H_all = hand_features(train)
    H_te = hand_features(test)

    X_all = np.hstack([Zw_all, Zc_all, Dw_all, Dc_all, H_all])
    X_te = np.hstack([Zw_te, Zc_te, Dw_te, Dc_te, H_te])
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    X_te_s = scaler.transform(X_te)

    lr_full = LogisticRegression(C=4.0, max_iter=2000, n_jobs=4, random_state=RANDOM_STATE)
    lr_full.fit(X_all_s, y)

    hgb_all = np.hstack([H_all, Dw_all[:, :50], Dc_all[:, :50]])
    hgb_te_f = np.hstack([H_te, Dw_te[:, :50], Dc_te[:, :50]])
    hgb_full = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, random_state=RANDOM_STATE
    )
    hgb_full.fit(hgb_all, y)

    p_lr_te = lr_full.predict_proba(X_te_s)
    p_hgb_te = hgb_full.predict_proba(hgb_te_f)
    hgb_order = list(hgb_full.classes_)
    lr_order = list(lr_full.classes_)
    p_hgb_te_al = p_hgb_te[:, [hgb_order.index(c) for c in lr_order]]
    p_te = 0.7 * p_lr_te + 0.3 * p_hgb_te_al
    pred_te = lr_full.classes_[p_te.argmax(axis=1)]

    out = pd.DataFrame({"id": test["id"], "label": pred_te})
    os.makedirs(os.path.join(root, "outputs"), exist_ok=True)
    out_path = os.path.join(root, "outputs", "submission.csv")
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}  shape={out.shape}")
    print(out["label"].value_counts())
    print(f"elapsed {time.time()-t0:.1f}s")

    # persist models for reproducibility
    joblib.dump(
        {
            "word_vec": word_vec, "char_vec": char_vec,
            "svd_w": svd_w, "svd_c": svd_c,
            "scaler": scaler, "lr": lr_full, "hgb": hgb_full,
            "classes": classes,
        },
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib"),
    )
    print("model saved to solution/model.joblib")


if __name__ == "__main__":
    main()
