"""Reproducible solution for KoBEST BoolQ (spec.md).

Task: Given a Korean `paragraph` and a yes/no `question`, predict `label` in {0,1}.
Metric: accuracy.

Constraints: no internet / no pretrained weights / only train.csv. Only scikit-learn,
numpy, scipy, pandas are available locally.

Approach (ensemble of linear models on TF-IDF + hand features):
1. Word (1,2)-gram TF-IDF on `paragraph` and `question` separately (stacked).
2. char_wb (2,5)-gram TF-IDF on `paragraph` and `question` separately.
3. Hand-crafted dense features: token-overlap ratios, Jaccard, negation/hedge cue
   counts, log length ratio, etc.
4. Stratified 5-fold OOF + averaged test predictions for each base learner:
   - LogisticRegression on text-only and on text+hand.
   - CalibratedClassifierCV(LinearSVC) on text-only.
5. Final prediction = weighted blend of the three learners (weights chosen from OOF).
6. Threshold at 0.5, write `outputs/submission.csv` matching sample id order.

Usage: `python run.py`  (from the task dir, or directly: python solution/run.py)
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

SEED = 42
HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent
TRAIN_PATH = TASK_DIR / "train.csv"
TEST_PATH = TASK_DIR / "test.csv"
SAMPLE_PATH = TASK_DIR / "sample_submission.csv"
OUT_PATH = TASK_DIR / "outputs" / "submission.csv"

# Korean negation / hedge cue substrings (rough heuristic; no internet used).
NEG_CUES = ["없", "않", "못", "아니", "부정", "반대", "거짓", "틀리", "불가", "거의 없", "전혀", "절대"]
HEDGE_CUES = ["가능성", "수 있다", "수도", "어쩌면", "아마", "추정", "추측", "불확실", "경우가", "있을 수"]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", str(text)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def count_cues(text: str, cues) -> int:
    s = str(text)
    return sum(1 for c in cues if c in s)


def build_hand(df: pd.DataFrame, p_col: str, q_col: str) -> np.ndarray:
    feats = []
    for i in range(len(df)):
        p = df.iloc[i][p_col]
        q = df.iloc[i][q_col]
        tp, tq = tokens(p), tokens(q)
        inter = len(tp & tq)
        ov_p = inter / max(1, len(tp))
        ov_q = inter / max(1, len(tq))
        jac = jaccard(tp, tq)
        neg_p = count_cues(p, NEG_CUES)
        neg_q = count_cues(q, NEG_CUES)
        hedge_p = count_cues(p, HEDGE_CUES)
        hedge_q = count_cues(q, HEDGE_CUES)
        pl, ql = len(str(p)), len(str(q))
        ratio = pl / max(1, ql)
        feats.append(
            [
                np.log1p(pl),
                np.log1p(ql),
                float(len(tp)),
                float(len(tq)),
                ov_p,
                ov_q,
                jac,
                float(neg_p),
                float(neg_q),
                float(neg_p + neg_q),
                float(hedge_p),
                float(hedge_q),
                ratio,
                float(neg_q > 0),
                float(hedge_p > 0),
            ]
        )
    return np.array(feats, dtype=np.float32)


def cv_predict(factory, X, y, X_test, n_splits=5, seed=SEED):
    """OOF + averaged test predictions for a model factory."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(X_test.shape[0], dtype=float)
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        clf = factory()
        clf.fit(X[tr_idx], y[tr_idx])
        if hasattr(clf, "predict_proba"):
            oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
            test_pred += clf.predict_proba(X_test)[:, 1] / n_splits
        else:
            oof[va_idx] = clf.decision_function(X[va_idx])
            test_pred += clf.decision_function(X_test) / n_splits
    return oof, test_pred


def main():
    print("Loading data ...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    y = train["label"].astype(int).values

    p_tr = train["paragraph"].astype(str).values
    q_tr = train["question"].astype(str).values
    p_te = test["paragraph"].astype(str).values
    q_te = test["question"].astype(str).values

    print("Building TF-IDF features ...")
    vec_p = TfidfVectorizer(
        ngram_range=(1, 2), max_features=25000, sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b", min_df=2,
    )
    vec_q = TfidfVectorizer(
        ngram_range=(1, 2), max_features=15000, sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b", min_df=1,
    )
    vec_cp = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), max_features=20000,
        sublinear_tf=True, min_df=2,
    )
    vec_cq = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), max_features=8000,
        sublinear_tf=True, min_df=2,
    )
    Xp_tr = vec_p.fit_transform(p_tr);   Xp_te = vec_p.transform(p_te)
    Xq_tr = vec_q.fit_transform(q_tr);   Xq_te = vec_q.transform(q_te)
    Xcp_tr = vec_cp.fit_transform(p_tr); Xcp_te = vec_cp.transform(p_te)
    Xcq_tr = vec_cq.fit_transform(q_tr); Xcq_te = vec_cq.transform(q_te)

    X_tr = hstack([Xp_tr, Xq_tr, Xcp_tr, Xcq_tr]).tocsr()
    X_te = hstack([Xp_te, Xq_te, Xcp_te, Xcq_te]).tocsr()

    print("Building hand features ...")
    H_tr = build_hand(train, "paragraph", "question")
    H_te = build_hand(test, "paragraph", "question")
    scaler = StandardScaler()
    H_tr_s = scaler.fit_transform(H_tr)
    H_te_s = scaler.transform(H_te)
    Xf_tr = hstack([X_tr, csr_matrix(H_tr_s)]).tocsr()
    Xf_te = hstack([X_te, csr_matrix(H_te_s)]).tocsr()

    print("Cross-validating base learners ...")

    def mk_lr(C=2.0):
        return LogisticRegression(
            C=C, solver="liblinear", max_iter=3000, random_state=SEED
        )

    def mk_svc():
        return CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=SEED, max_iter=5000), cv=3
        )

    # Learner 1: LR on text+hand
    oof1, tp1 = cv_predict(lambda: mk_lr(2.0), Xf_tr, y, Xf_te)
    print("  LR(text+hand)  oof acc =", round(accuracy_score(y, (oof1 > 0.5).astype(int)), 4))

    # Learner 2: SVC on text
    oof2, tp2 = cv_predict(mk_svc, X_tr, y, X_te)
    print("  SVC(text)       oof acc =", round(accuracy_score(y, (oof2 > 0.5).astype(int)), 4))

    # Learner 3: LR on text (different regularization)
    oof3, tp3 = cv_predict(lambda: mk_lr(4.0), X_tr, y, X_te)
    print("  LR(text)        oof acc =", round(accuracy_score(y, (oof3 > 0.5).astype(int)), 4))

    # Grid-search a small blend weight on OOF
    best = (0.0, 0.0)
    for w1 in np.linspace(0.0, 1.0, 21):
        w2 = (1.0 - w1) * 0.5
        w3 = (1.0 - w1) * 0.5
        ens = w1 * oof1 + w2 * oof2 + w3 * oof3
        acc = accuracy_score(y, (ens > 0.5).astype(int))
        if acc > best[1]:
            best = (float(w1), float(acc))
    w1 = best[0]
    w2 = (1.0 - w1) * 0.5
    w3 = (1.0 - w1) * 0.5
    print(f"  best blend  w(LR_hand)={w1:.3f}  w(SVC)={w2:.3f}  w(LR_text)={w3:.3f}  oof acc={best[1]:.4f}")

    final_proba = w1 * tp1 + w2 * tp2 + w3 * tp3
    preds = (final_proba > 0.5).astype(int)

    # Build submission preserving sample id order
    pred_df = pd.DataFrame({"id": test["id"].values, "label": preds})
    sub = sample[["id"]].merge(pred_df, on="id", how="left")
    sub["label"] = sub["label"].astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    print("Wrote", OUT_PATH)
    print(sub.head())
    print("label distribution:")
    print(sub["label"].value_counts())
    assert len(sub) == len(sample), "submission length mismatch"
    assert set(sub["id"]) == set(sample["id"]), "id set mismatch"
    assert list(sub.columns) == ["id", "label"], "columns wrong"
    print("OK")


if __name__ == "__main__":
    main()
