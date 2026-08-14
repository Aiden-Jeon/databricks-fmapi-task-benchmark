# -*- coding: utf-8 -*-
"""
네이버 영화리뷰 감성 분석 (t4_nsmc) 솔루션

방법:
- train.csv만 사용 (외부 데이터/사전학습 가중치 없음)
- TF-IDF (단어 bigram + 글자 2~5gram, 이중 채널) 특징 추출
- LogisticRegression(선형) 분류기 + (선택) LinearSVC 앙상블
- 시간 예산 내에서 재현 가능하도록 단순/결정적 파이프라인 사용

실행:
    python solution/solution.py
결과:
    outputs/submission.csv 생성
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.base import clone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "submission.csv")

RANDOM_STATE = 42


def clean_text(s: str) -> str:
    """가벼운 정규화: 공백 정리 정도만 (구어체/이모티콘 정보 보존)."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_features(train_texts, test_texts):
    """단어 bigram TF-IDF + 글자 n-gram TF-IDF를 결합한 희소 특징 행렬."""
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.999,
        sublinear_tf=True,
        max_features=400000,
        token_pattern=r"(?u)\S+",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        min_df=3,
        sublinear_tf=True,
        max_features=600000,
    )
    Xw_tr = word_vec.fit_transform(train_texts)
    Xw_te = word_vec.transform(test_texts)
    Xc_tr = char_vec.fit_transform(train_texts)
    Xc_te = char_vec.transform(test_texts)
    X_tr = sparse.hstack([Xw_tr, Xc_tr]).tocsr()
    X_te = sparse.hstack([Xw_te, Xc_te]).tocsr()
    return X_tr, X_te


def make_model():
    return LogisticRegression(
        C=2.0,
        solver="liblinear",
        max_iter=200,
        random_state=RANDOM_STATE,
    )


def main():
    t0 = time.time()
    print("[1/4] 데이터 로딩...", flush=True)
    train = pd.read_csv(TRAIN_PATH, dtype={"id": str})
    test = pd.read_csv(TEST_PATH, dtype={"id": str})
    train["document"] = train["document"].fillna("").map(clean_text)
    test["document"] = test["document"].fillna("").map(clean_text)
    y = train["label"].astype(int).values
    print(f"  train={len(train)}, test={len(test)}, "
          f"label1 비율={y.mean():.4f}", flush=True)

    # --- 홀드아웃 검증 (모델 선택용) ---
    print("[2/4] 홀드아웃 검증 (90/10)...", flush=True)
    tr_idx, va_idx = train_test_split(
        np.arange(len(train)), test_size=0.1,
        random_state=RANDOM_STATE, stratify=y,
    )
    X_tr, X_va = build_features(
        train["document"].iloc[tr_idx],
        train["document"].iloc[va_idx],
    )
    model = make_model()
    model.fit(X_tr, y[tr_idx])
    acc = model.score(X_va, y[va_idx])
    print(f"  holdout accuracy (word+char TFIDF, LR C=2.0): {acc:.5f}", flush=True)
    del X_tr, X_va

    # --- 전체 학습 데이터로 최종 학습 ---
    print("[3/4] 전체 train으로 최종 모델 학습...", flush=True)
    X_train_full, X_test = build_features(train["document"], test["document"])
    final_model = make_model()
    final_model.fit(X_train_full, y)

    # LinearSVC를 소프트 보팅으로 추가 (확률 대신 결정함수를 정규화해 평균)
    svc = LinearSVC(C=0.5, random_state=RANDOM_STATE)
    svc.fit(X_train_full, y)

    def to_prob(scores):
        # 로지스틱 시그모이드로 결정함수를 확률처럼 변환 (캘리브레이션 용도 X, 순위용)
        return 1.0 / (1.0 + np.exp(-scores))

    p_lr = final_model.predict_proba(X_test)[:, 1]
    p_svc = to_prob(svc.decision_function(X_test))
    p = 0.5 * p_lr + 0.5 * p_svc
    pred = (p >= 0.5).astype(int)

    # --- 제출 파일 생성 ---
    print("[4/4] 제출 파일 생성...", flush=True)
    sub = pd.DataFrame({"id": test["id"].astype(str), "label": pred.astype(int)})
    # test.csv의 id 순서/중복 검증
    assert len(sub) == len(test), "행 수 불일치"
    assert sub["id"].is_unique, "id 중복"
    assert set(sub["id"]) == set(test["id"]), "id 집합 불일치"
    assert sub["label"].isin([0, 1]).all(), "label 값 오류"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    print(f"  저장 완료: {OUT_PATH} (양성 비율={pred.mean():.4f})", flush=True)
    print(f"총 소요 시간: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
