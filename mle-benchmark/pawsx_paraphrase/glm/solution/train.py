"""PAWS-X Korean paraphrase detection - final solution.

Approach (PAWS-X is adversarial: high lexical overlap with swapped entities):
  1. Build TF-IDF (word 1-2gram + char_wb 2-5gram) on each sentence independently.
  2. Construct comparative sparse features: |tfidf(s1)-tfidf(s2)| (abs diff),
     tfidf(s1)*tfidf(s2) (element-wise product), plus cosine similarities.
  3. Add hand-crafted pair numeric features (token/char Jaccard & overlap,
     length ratios, counts).
  4. Train LogisticRegression (C=0.5) and calibrated LinearSVC via 5-fold OOF.
  5. Add a GradientBoostingClassifier on the small numeric+cosine feature set.
  6. Ensemble: weighted average of the three OOF probability vectors.

Reproducible (SEED=42), no internet / no external data, train.csv only.
"""
import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "train.csv")
TEST = os.path.join(BASE, "test.csv")
SUB = os.path.join(BASE, "outputs", "submission.csv")
SEED = 42
N_SPLITS = 5


def normalize(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def char_tokens(s, n=3):
    if not s:
        return []
    s = re.sub(r"\s+", "", s)
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def word_tokens(s):
    return re.findall(r"\S+", s) if s else []


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_coeff(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def pair_numeric(df):
    rows = []
    for a, b in zip(df["sentence1"], df["sentence2"]):
        wa, wb = word_tokens(a), word_tokens(b)
        ca, cb = char_tokens(a, 3), char_tokens(b, 3)
        la, lb = len(a), len(b)
        nwa, nwb = len(wa), len(wb)
        rows.append({
            "w_jaccard": jaccard(wa, wb),
            "w_overlap": overlap_coeff(wa, wb),
            "c3_jaccard": jaccard(ca, cb),
            "c3_overlap": overlap_coeff(ca, cb),
            "len1": la, "len2": lb, "len_diff": abs(la - lb),
            "len_ratio": (la / lb) if lb else 0.0,
            "len_min": min(la, lb), "len_max": max(la, lb),
            "nw1": nwa, "nw2": nwb, "nw_diff": abs(nwa - nwb),
            "nw_ratio": (nwa / nwb) if nwb else 0.0,
            "char_len_diff_ratio": abs(la - lb) / max(la, lb, 1),
        })
    return pd.DataFrame(rows).values.astype(np.float32)


def cos_sparse(v1, v2):
    dot = np.asarray(v1.multiply(v2).sum(axis=1)).ravel()
    n1 = np.asarray(v1.multiply(v1).sum(axis=1)).ravel()
    n2 = np.asarray(v2.multiply(v2).sum(axis=1)).ravel()
    denom = np.sqrt(n1) * np.sqrt(n2)
    out = np.zeros_like(dot, dtype=float)
    nz = denom > 0
    out[nz] = dot[nz] / denom[nz]
    return out


def main():
    t0 = time.time()
    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(TEST)
    for df in (tr, te):
        df["sentence1"] = df["sentence1"].map(normalize)
        df["sentence2"] = df["sentence2"].map(normalize)
    y = tr["label"].values.astype(int)

    # Hand-crafted pair numeric features
    nf_tr = pair_numeric(tr)
    nf_te = pair_numeric(te)

    # TF-IDF on each sentence independently (fit on all sentences)
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95,
        sublinear_tf=True, max_features=40000, dtype=np.float32,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_df=0.95,
        sublinear_tf=True, max_features=60000, dtype=np.float32,
    )
    all_s1 = pd.concat([tr["sentence1"], te["sentence1"]]).values
    all_s2 = pd.concat([tr["sentence2"], te["sentence2"]]).values
    word_vec.fit(list(all_s1) + list(all_s2))
    char_vec.fit(list(all_s1) + list(all_s2))

    s1w = word_vec.transform(tr["sentence1"]); s2w = word_vec.transform(tr["sentence2"])
    s1c = char_vec.transform(tr["sentence1"]); s2c = char_vec.transform(tr["sentence2"])
    te_s1w = word_vec.transform(te["sentence1"]); te_s2w = word_vec.transform(te["sentence2"])
    te_s1c = char_vec.transform(te["sentence1"]); te_s2c = char_vec.transform(te["sentence2"])

    # Comparative sparse features
    def absdiff(a, b):
        d = (a - b)
        d.data = np.abs(d.data)
        return d

    tr_wd = absdiff(s1w, s2w); tr_wm = s1w.multiply(s2w); tr_wc = cos_sparse(s1w, s2w)
    tr_cd = absdiff(s1c, s2c); tr_cc = cos_sparse(s1c, s2c)
    te_wd = absdiff(te_s1w, te_s2w); te_wm = te_s1w.multiply(te_s2w); te_wc = cos_sparse(te_s1w, te_s2w)
    te_cd = absdiff(te_s1c, te_s2c); te_cc = cos_sparse(te_s1c, te_s2c)

    cos_tr = np.column_stack([tr_wc, tr_cc]).astype(np.float32)
    cos_te = np.column_stack([te_wc, te_cc]).astype(np.float32)

    X_tr = hstack([csr_matrix(nf_tr), csr_matrix(cos_tr),
                   tr_wd, tr_wm, tr_cd]).tocsr()
    X_te = hstack([csr_matrix(nf_te), csr_matrix(cos_te),
                   te_wd, te_wm, te_cd]).tocsr()
    print("X_tr", X_tr.shape, "X_te", X_te.shape, "time", round(time.time() - t0, 1))

    # Dense small feature set for GBM
    Z_tr = np.hstack([nf_tr, cos_tr]).astype(np.float32)
    Z_te = np.hstack([nf_te, cos_te]).astype(np.float32)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Model A: LogisticRegression on full sparse comparative features
    oof_lr = np.zeros(len(tr)); te_lr = np.zeros(len(te))
    for fold, (i_tr, i_va) in enumerate(skf.split(X_tr, y)):
        clf = LogisticRegression(C=0.5, max_iter=5000, solver="liblinear",
                                 random_state=SEED, dual=False)
        clf.fit(X_tr[i_tr], y[i_tr])
        oof_lr[i_va] = clf.predict_proba(X_tr[i_va])[:, 1]
        te_lr += clf.predict_proba(X_te)[:, 1] / N_SPLITS
    print("LR oof acc:", accuracy_score(y, (oof_lr > 0.5).astype(int)))

    # Model B: Calibrated LinearSVC on full sparse comparative features
    oof_svc = np.zeros(len(tr)); te_svc = np.zeros(len(te))
    for fold, (i_tr, i_va) in enumerate(skf.split(X_tr, y)):
        base = LinearSVC(C=1.0, max_iter=8000, random_state=SEED, dual="auto")
        clf = CalibratedClassifierCV(base, cv=3)
        clf.fit(X_tr[i_tr], y[i_tr])
        oof_svc[i_va] = clf.predict_proba(X_tr[i_va])[:, 1]
        te_svc += clf.predict_proba(X_te)[:, 1] / N_SPLITS
    print("SVC oof acc:", accuracy_score(y, (oof_svc > 0.5).astype(int)))

    # Model C: GradientBoosting on small dense numeric+cosine features
    oof_gb = np.zeros(len(tr)); te_gb = np.zeros(len(te))
    for fold, (i_tr, i_va) in enumerate(skf.split(Z_tr, y)):
        clf = GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.85, random_state=SEED,
        )
        clf.fit(Z_tr[i_tr], y[i_tr])
        oof_gb[i_va] = clf.predict_proba(Z_tr[i_va])[:, 1]
        te_gb += clf.predict_proba(Z_te)[:, 1] / N_SPLITS
    print("GB oof acc:", accuracy_score(y, (oof_gb > 0.5).astype(int)))

    # Search ensemble weights among the three OOFs
    best = (0, None)
    from itertools import product
    grid = [round(w, 2) for w in np.arange(0, 1.01, 0.1)]
    for a, b in product(grid, grid):
        c = 1 - a - b
        if c < -1e-6 or c > 1 + 1e-6:
            continue
        c = max(0.0, c)
        oof = a * oof_lr + b * oof_svc + c * oof_gb
        acc = accuracy_score(y, (oof > 0.5).astype(int))
        if acc > best[0]:
            best = (acc, (a, b, c))
    print("Best ensemble oof acc:", best[0], "weights (lr,svc,gb):", best[1])

    a, b, c = best[1]
    te_ens = a * te_lr + b * te_svc + c * te_gb
    pred = (te_ens > 0.5).astype(int)

    out = pd.DataFrame({"id": te["id"], "label": pred})
    os.makedirs(os.path.dirname(SUB), exist_ok=True)
    out.to_csv(SUB, index=False)
    print("Saved:", SUB, out.shape, out["label"].value_counts().to_dict())
    print("Elapsed:", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
