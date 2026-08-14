# -*- coding: utf-8 -*-
"""
Spooky Author Identification (Kaggle / MLE-bench) - TF-IDF + Linear Models
제출: 3-class softmax 확률, multi-class log loss 최소화 목표.

전략:
1) 다양한 TF-IDF 피처(word uni/bi, word 1-3, char 3-5)를 만들고
2) 로지스틱 회귀 / Multinomial NB / Calibrated LinearSVC 등 확률 출력 모델을
   Stratified 5-fold OOF로 평가 (log loss 기준)
3) 가장 좋은 모델(또는 단순 앙상블)로 전체 학습 후 test 예측
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

RANDOM_STATE = 42
N_SPLITS = 5

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE, "train.csv")
TEST_PATH = os.path.join(BASE, "test.csv")
SUB_PATH = os.path.join(BASE, "outputs", "submission.csv")

CLASSES = ["EAP", "HPL", "MWS"]


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def build_features(train_text, test_text):
    """여러 TF-IDF 뷰를 만들어 반환. 각 뷰는 (Xtr, Xte) 튜플 리스트."""
    views = []

    # word 1-2, sublinear
    v1 = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2,
        sublinear_tf=True, strip_accents="unicode",
        max_features=120000,
    )
    views.append((v1, "word12"))

    # word 1-3
    v2 = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3), min_df=3,
        sublinear_tf=True, strip_accents="unicode",
        max_features=160000,
    )
    views.append((v2, "word123"))

    # char 3-5 (word boundaries 유지)
    v3 = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3,
        sublinear_tf=True, max_features=120000,
    )
    views.append((v3, "char35"))

    out = []
    for vec, name in views:
        t0 = time.time()
        Xtr = vec.fit_transform(train_text)
        Xte = vec.transform(test_text)
        print(f"  view {name}: train{Xtr.shape} test{Xte.shape} ({time.time()-t0:.1f}s)")
        out.append((name, Xtr, Xte))
    return out


def evaluate_oof(make_model, X, y, name):
    """make_model() -> fresh estimator with predict_proba. OOF log loss 반환."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((X.shape[0], len(CLASSES)))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = make_model()
        model.fit(X[tr_idx], y[tr_idx])
        p = model.predict_proba(X[va_idx])
        # 클래스 순서 정렬
        order = [list(model.classes_).index(c) for c in range(len(CLASSES))]
        p = p[:, order]
        oof[va_idx] = p
    ll = log_loss(y, oof, labels=list(range(len(CLASSES))))
    print(f"  [OOF] {name}: logloss = {ll:.5f}")
    return ll, oof


def main():
    t_start = time.time()
    train, test = load_data()
    print(f"train: {train.shape}, test: {test.shape}")

    label_map = {c: i for i, c in enumerate(CLASSES)}
    y = train["author"].map(label_map).values

    print("Building TF-IDF features...")
    views = build_features(train["text"].values, test["text"].values)

    # 뷰 결합: word12 + char35 가 일반적으로 강력. word123도 추가.
    Xtr_all = hstack([v[1] for v in views]).tocsr()
    Xte_all = hstack([v[2] for v in views]).tocsr()
    print(f"combined: train{Xtr_all.shape} test{Xte_all.shape}")

    results = {}

    # --- 1) Logistic Regression (saga, multinomial) on combined ---
    def make_lr():
        return LogisticRegression(
            C=6.0, solver="saga", multi_class="multinomial",
            max_iter=400, n_jobs=4, random_state=RANDOM_STATE,
        )
    results["lr_combined"] = evaluate_oof(make_lr, Xtr_all, y, "LR(combined)")

    # --- 2) Logistic Regression per-view (char35만) ---
    char35_tr = dict((n, (a, b)) for n, a, b in views)["char35"][0]
    char35_te = dict((n, (a, b)) for n, a, b in views)["char35"][1]
    results["lr_char35"] = evaluate_oof(make_lr, char35_tr, y, "LR(char35)")

    # --- 3) Multinomial NB on word12 ---
    word12_tr = dict((n, (a, b)) for n, a, b in views)["word12"][0]
    word12_te = dict((n, (a, b)) for n, a, b in views)["word12"][1]

    def make_mnb():
        return MultinomialNB(alpha=0.1)
    results["mnb_word12"] = evaluate_oof(make_mnb, word12_tr, y, "MNB(word12)")

    # --- 4) Calibrated LinearSVC on combined ---
    def make_svc():
        base = LinearSVC(C=0.5, random_state=RANDOM_STATE, max_iter=3000)
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")
    results["svc_combined"] = evaluate_oof(make_svc, Xtr_all, y, "CalSVC(combined)")

    # 가중 평균 앙상블 (OOF log loss 역수 기반 가중)
    oofs = {k: v[1] for k, v in results.items()}
    lls = {k: v[0] for k, v in results.items()}
    print("\nOOF summary:")
    for k, v in sorted(lls.items(), key=lambda x: x[1]):
        print(f"  {k}: {v:.5f}")

    # 단순 평균 + 가중 평균 앙상블도 OOF로 평가
    keys = list(oofs.keys())
    avg_oof = np.mean([oofs[k] for k in keys], axis=0)
    avg_ll = log_loss(y, avg_oof, labels=[0, 1, 2])
    print(f"  ensemble(simple avg): {avg_ll:.5f}")

    inv = {k: 1.0 / (lls[k] ** 4) for k in keys}  # log loss 차이 강조
    s = sum(inv.values())
    w = {k: inv[k] / s for k in keys}
    wavg_oof = sum(oofs[k] * w[k] for k in keys)
    wavg_ll = log_loss(y, wavg_oof, labels=[0, 1, 2])
    print(f"  ensemble(weighted avg): {wavg_ll:.5f}  weights={ {k: round(v,3) for k,v in w.items()} }")

    # 최종 선택: 가장 낮은 OOF log loss
    candidates = dict(lls)
    candidates["ens_avg"] = (avg_ll, None)
    candidates["ens_wavg"] = (wavg_ll, None)
    best = min(candidates.items(), key=lambda x: x[1][0] if isinstance(x[1], tuple) else x[1])
    # 위 형식 통일 위해 재구성
    cand = {k: v[0] for k, v in results.items()}
    cand["ens_avg"] = avg_ll
    cand["ens_wavg"] = wavg_ll
    best_name = min(cand, key=cand.get)
    print(f"\nBest approach: {best_name} (OOF logloss={cand[best_name]:.5f})")

    # --- 전체 학습 후 test 예측 ---
    test_preds = {}

    # 각 단일 모델 전체 학습
    lr_full = make_lr(); lr_full.fit(Xtr_all, y)
    p = lr_full.predict_proba(Xte_all)
    order = [list(lr_full.classes_).index(c) for c in range(3)]
    test_preds["lr_combined"] = p[:, order]

    lr_c = make_lr(); lr_c.fit(char35_tr, y)
    p = lr_c.predict_proba(char35_te)
    order = [list(lr_c.classes_).index(c) for c in range(3)]
    test_preds["lr_char35"] = p[:, order]

    mnb = make_mnb(); mnb.fit(word12_tr, y)
    p = mnb.predict_proba(word12_te)
    order = [list(mnb.classes_).index(c) for c in range(3)]
    test_preds["mnb_word12"] = p[:, order]

    svc = make_svc(); svc.fit(Xtr_all, y)
    p = svc.predict_proba(Xte_all)
    order = [list(svc.classes_).index(c) for c in range(3)]
    test_preds["svc_combined"] = p[:, order]

    if best_name == "ens_avg":
        final = np.mean([test_preds[k] for k in keys], axis=0)
    elif best_name == "ens_wavg":
        final = sum(test_preds[k] * w[k] for k in keys)
    else:
        final = test_preds[best_name]

    # 안전: 확률 클리핑 + 정규화 (log loss 폭발 방지)
    eps = 1e-6
    final = np.clip(final, eps, 1.0)
    final = final / final.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({
        "id": test["id"],
        "EAP": final[:, 0],
        "HPL": final[:, 1],
        "MWS": final[:, 2],
    })
    # id 순서는 test.csv 그대로 (모든 id 정확히 한 번)
    assert sub["id"].is_unique
    assert set(sub["id"]) == set(test["id"])
    os.makedirs(os.path.dirname(SUB_PATH), exist_ok=True)
    sub.to_csv(SUB_PATH, index=False)
    print(f"\nSaved: {SUB_PATH}  shape={sub.shape}")
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
